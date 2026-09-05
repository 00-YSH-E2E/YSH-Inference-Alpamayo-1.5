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

"""What the action head attends to, computed once and kept as tensors.

The head cross-attends the frozen VLM's key/value cache: the prompt (four
camera frames plus text, 3086 tokens for the Cam-4 spec) followed by the
reasoning the VLM sampled for that particular trajectory sample and the
trajectory-start marker. Producing that cache costs vision 632 ms plus
prefill 1056 ms per clip, against 48 ms for one expert forward -- so a
training loop that rebuilt it per step would spend 95% of its time on a
frozen network. This module builds it once per clip and hands back plain
tensors the pool can hold and the student forward can wrap.

Two stages, mirroring what inference does implicitly:

1. :func:`build_prompt_kv` -- the prompt alone, one VLM forward at batch 1.
   This is the expensive part and it is shared by every reasoning sample of
   the clip, because they all saw the same frames.
2. :func:`build_cot_tails` -- the K reasoning sequences, forwarded against the
   prompt cache with an explicit ``cache_position`` so RoPE continues from
   where the prompt ended. Cheap, and it yields only the tail positions.

The precedent for stage 2 is the model's own classifier-free-guidance path
(``alpamayo1_5.py``, "Vision encoder runs only once"), which does exactly
this for its unguided prefix. What is new here is only that the reasoning
comes from a table rather than from a ``generate`` call.

Everything is under ``no_grad`` and the VLM stays in eval mode. The
quantization recipe's teacher-forced forward, which this adapts, runs the
VLM in whatever grad context it inherits -- under a training loop that is an
8-billion-parameter autograd graph, and it does not fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from alpamayo1_5.distill.forward import make_cache


@dataclass
class PromptKV:
    """One clip's prompt cache: 36 layers of ``(1, kv_heads, P, head_dim)``."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    rope_delta: torch.Tensor          # (1, 1) long -- the VLM's per-sequence RoPE offset
    prompt_len: int                   # P; 3086 for Cam-4, asserted constant by the caller
    x1: torch.Tensor | None           # (1, 64, 2) fp32 GT action, None when not training


@dataclass
class TailKV:
    """K reasoning tails against one prompt: 36 layers of ``(K, kv_heads, t_tail, head_dim)``."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    offset: torch.Tensor              # (K,) long -- P + n_generated_k, first position after the marker
    n_generated: torch.Tensor         # (K,) long
    t_tail: int


@torch.no_grad()
def build_prompt_kv(model, data: dict, *, with_gt: bool = True) -> PromptKV:
    """The prompt's KV cache, RoPE delta, and (optionally) the GT action.

    ``data`` is what ``load_physical_aiavdataset`` plus the processor produce,
    already on the model's device: ``tokenized_data`` (input_ids, pixel
    values, image grid, attention mask), ``ego_history_xyz``/``_rot``, and --
    for training -- ``ego_future_xyz``/``_rot``.

    Unlike the quantization recipe this does **not** append the
    trajectory-start marker: the marker belongs to the reasoning tail, after
    the reasoning, where inference puts it.
    """
    tokenized = dict(data["tokenized_data"])
    input_ids = tokenized.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids, {"ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"]},
    )
    out = model.vlm(input_ids=input_ids, use_cache=True, return_dict=True,
                    logits_to_keep=1, **tokenized)
    cache = out.past_key_values
    rope_delta = model.vlm.model.rope_deltas.detach().clone()
    del out
    prompt_len = cache.get_seq_length()
    keys = [layer.keys.detach() for layer in cache.layers]
    values = [layer.values.detach() for layer in cache.layers]

    x1 = None
    if with_gt:
        x1 = model.action_space.traj_to_action(
            traj_history_xyz=data["ego_history_xyz"][:, 0],
            traj_history_rot=data["ego_history_rot"][:, 0],
            traj_future_xyz=data["ego_future_xyz"][:, 0],
            traj_future_rot=data["ego_future_rot"][:, 0],
        ).to(dtype=torch.float32)
    return PromptKV(keys=keys, values=values, rope_delta=rope_delta,
                    prompt_len=int(prompt_len), x1=x1)


@torch.no_grad()
def build_cot_tails(model, prompt: PromptKV, cot_ids: list, t_tail: int = 128) -> TailKV:
    """Forward K reasoning sequences against the prompt cache; keep only the tails.

    ``cot_ids[k]`` is the generated sequence exactly as the runner recorded it
    -- reasoning tokens, then the reasoning-end and trajectory-start markers.
    Sequences are right-padded to ``t_tail``; the padding positions do produce
    cache entries, and the expert's attention mask blocks them from ``offset``
    onward, which is the same thing inference does with the gap after the
    marker.

    The prompt tensors are expanded to K along the batch dimension without a
    copy; the concatenation inside the VLM's cache update materialises them,
    as it would for K generated sequences.
    """
    device = prompt.keys[0].device
    K = len(cot_ids)
    P = prompt.prompt_len
    pad = model.tokenizer.pad_token_id
    lengths = torch.tensor([len(c) for c in cot_ids], device=device, dtype=torch.long)
    if int(lengths.max()) > t_tail:
        raise ValueError(f"a reasoning sequence has {int(lengths.max())} tokens; t_tail={t_tail}")

    tails = torch.full((K, t_tail), pad, dtype=torch.long, device=device)
    for k, ids in enumerate(cot_ids):
        tails[k, : len(ids)] = torch.tensor(list(map(int, ids)), dtype=torch.long, device=device)
    mask = torch.cat([torch.ones(K, P, dtype=torch.long, device=device),
                      (tails != pad).long()], dim=1)

    cache = make_cache([k.expand(K, -1, -1, -1) for k in prompt.keys],
                       [v.expand(K, -1, -1, -1) for v in prompt.values])
    # The VLM reads its RoPE delta off the module, not off the call. Set it to
    # the prompt's, since the tail continues that prompt.
    model.vlm.model.rope_deltas = prompt.rope_delta.expand(K, -1)
    cache_position = torch.arange(P, P + t_tail, device=device, dtype=torch.long)
    out = model.vlm(input_ids=tails, attention_mask=mask, past_key_values=cache,
                    cache_position=cache_position, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    del out
    keys = [layer.keys[:, :, P:, :].contiguous() for layer in cache.layers]
    values = [layer.values[:, :, P:, :].contiguous() for layer in cache.layers]
    return TailKV(keys=keys, values=values, offset=P + lengths, n_generated=lengths,
                  t_tail=t_tail)


@torch.no_grad()
def build_single_forward_kv(model, data: dict, cot_ids) -> tuple[PromptKV, torch.Tensor]:
    """Reference for tests: prompt and one reasoning sequence in a single forward.

    Returns the full cache as a :class:`PromptKV` whose ``prompt_len`` is the
    combined length, plus the offset. Two-stage construction must reproduce
    this to within bf16 noise, or the tail forward is positioned wrong.
    """
    tokenized = dict(data["tokenized_data"])
    input_ids = tokenized.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids, {"ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"]},
    )
    device = input_ids.device
    extra = torch.as_tensor(cot_ids, dtype=input_ids.dtype, device=device).view(1, -1)
    input_ids = torch.cat([input_ids, extra], dim=1)
    if tokenized.get("attention_mask") is not None:
        am = tokenized["attention_mask"]
        tokenized["attention_mask"] = torch.cat(
            [am, torch.ones((am.shape[0], extra.shape[1]), dtype=am.dtype, device=device)], 1)
    out = model.vlm(input_ids=input_ids, use_cache=True, return_dict=True,
                    logits_to_keep=1, **tokenized)
    cache = out.past_key_values
    rope_delta = model.vlm.model.rope_deltas.detach().clone()
    del out
    full = PromptKV(keys=[x.keys.detach() for x in cache.layers],
                    values=[x.values.detach() for x in cache.layers],
                    rope_delta=rope_delta, prompt_len=int(cache.get_seq_length()), x1=None)
    offset = torch.tensor([full.prompt_len], device=device, dtype=torch.long)
    return full, offset
