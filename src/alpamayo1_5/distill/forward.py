# SPDX-FileCopyrightText: Copyright (c) 2026 YSH-research
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The student forward: one velocity from ``(x, t_start, t_target)`` and a cache.

This is the model's own ``step_fn`` -- projection, expert, output projection
-- taken out of the sampler's closure so it can be called with a chosen
``x``, a chosen pair of times, and a chosen conditioning, with gradients.

The one thing it must not do is what the closure does: hand the model a cache
object that is shared across calls and ``crop`` it afterwards. The cache
update inside the expert concatenates the 64 new positions onto the prompt,
producing a *new* tensor; ``crop`` then leaves the layer holding a view of
that new tensor, which under autograd carries a ``grad_fn``. A second call
concatenates onto a graph whose saved tensors the first backward freed, and
the second backward fails. So every call here wraps the conditioning tensors
in a **fresh** cache object that references them without copying, lets the
expert append into that throwaway, and never crops. The conditioning is never
written to; several draws against one conditioning in one step are
independent; and the object dies with the call's graph.

``DynamicCache(ddp_cache_data=...)`` would be the obvious constructor and is
the wrong one: it initialises each layer by *updating* an empty tensor with
the given one, which is a copy of 455 MB per layer set.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


def make_cache(keys: list[torch.Tensor], values: list[torch.Tensor]) -> DynamicCache:
    """A cache object that *references* the given tensors. No copy.

    Each layer's ``keys``/``values`` is the tensor passed in; the expert's
    attention appends to it by concatenation, which allocates a new tensor for
    that layer inside this object and leaves the input untouched.
    """
    layers = []
    for k, v in zip(keys, values):
        layer = DynamicLayer()
        layer.keys, layer.values = k, v
        layer.dtype, layer.device = k.dtype, k.device
        layer.is_initialized = True
        layers.append(layer)
    cache = DynamicCache()
    cache.layers = layers
    cache.layer_class_to_replicate = None
    return cache


def expert_inputs(model, offset: torch.Tensor, rope_delta: torch.Tensor, kv_len: int,
                  batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Position ids ``(3, B, 64)`` and the float attention mask ``(B, 1, 64, kv_len + 64)``.

    Delegates to the model's own builder so the student attends exactly the
    way inference does: positions continue from ``offset + rope_delta``, and
    everything from ``offset`` to the end of the cache is masked.
    """
    n = model.action_space.get_action_space_dims()[0]
    return model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_delta, kv_cache_seq_len=kv_len,
        n_diffusion_tokens=n, b_star=batch, device=device, prefix_mask=None,
    )


def student_velocity(
    model,
    x: torch.Tensor,
    t_start: torch.Tensor,
    t_target: torch.Tensor | None,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """``v = F(x, t_start, t_target | conditioning)``, shape ``(B, 64, 2)`` in fp32.

    ``x`` is ``(B, 64, 2)``; ``t_start`` and ``t_target`` are ``(B,)``. With
    ``t_target`` None the head receives the upstream ``(B, 1, 1)`` shape, so the
    call is the pretrained forward regardless of which head class is loaded.
    """
    B = x.shape[0]
    n = model.action_space.get_action_space_dims()[0]
    proj_dtype = next(model.action_in_proj.parameters()).dtype
    if t_target is None:
        t = t_start.view(B, 1, 1)
    else:
        t = torch.stack([t_target.view(B, 1), t_start.view(B, 1)], dim=-1)     # (B, 1, 2)
    emb = model.action_in_proj(x.to(proj_dtype), t.to(proj_dtype))
    if emb.dim() == 2:
        emb = emb.view(B, n, -1)
    cache = make_cache(keys, values)
    kwargs = {"is_causal": False} if model.config.expert_non_causal_attention else {}
    out = model.expert(inputs_embeds=emb, position_ids=position_ids, past_key_values=cache,
                       attention_mask=attention_mask, use_cache=True, **kwargs)
    hidden = out.last_hidden_state[:, -n:]
    return model.action_out_proj(hidden).to(torch.float32).view(B, n, -1)
