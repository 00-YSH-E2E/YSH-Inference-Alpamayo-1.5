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

"""A GPU-resident, rotating pool of conditionings.

The frozen VLM's cache is 455 MB per clip and costs 1.7 s to build; the
expert forward the student trains through costs 48 ms. Rebuilding the cache
every step makes a 30k-step run take two days; caching it to disk takes 592
GB of a disk that does not survive a recycle. The pool keeps a bounded set of
conditionings on the GPU, serves batches from it, and rotates: each entry is
used several times, then evicted for a fresh clip.

Two levels, because the expensive part is shared. All K reasoning samples of
a clip saw the same frames, so their caches agree on the first 3086
positions and differ only in a tail of at most 128. A prompt slot holds the
shared part once; a tail slot holds one sample's reasoning and points at its
prompt. At the default sizes that is 64 prompts (29 GB) and 384 tails
(7 GB) -- 384 resident conditionings, against 80 if each carried its own
copy of the prompt.

A batch is gathered by concatenating prompt and tail along the sequence
axis, once per step, and reused by every forward of that step. The pool
tensors themselves are never handed to the model: the student forward wraps
the gathered copy in a throwaway cache, so nothing here is ever written to
by attention.

Bookkeeping lives on the CPU; only the caches and the per-slot values the
model needs (RoPE delta, GT action, offset) live on the device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from alpamayo1_5.distill.conditioning import PromptKV, TailKV


@dataclass
class ConditioningBatch:
    """What one training step attends to. ``B'`` is ``B * duplicate``.

    ``keys``/``values`` are lists over layers of ``(B', kv_heads, L, head_dim)``
    with ``L = prompt_len + t_tail``; rows are tiled so that row ``j`` is
    conditioning ``j mod B``. ``x1`` and ``tail_slots`` are untiled, ``(B, ...)``.
    """

    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    offset: torch.Tensor          # (B',) long
    rope_delta: torch.Tensor      # (B', 1) long
    x1: torch.Tensor              # (B, 64, 2) fp32
    tail_slots: torch.Tensor      # (B,) long, CPU
    kv_len: int


class ConditioningPool:
    def __init__(
        self,
        *,
        n_prompts: int,
        n_tails: int,
        n_layers: int,
        kv_heads: int,
        prompt_len: int,
        t_tail: int,
        head_dim: int,
        action_shape: tuple[int, int] = (64, 2),
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.P, self.S, self.n_layers = n_prompts, n_tails, n_layers
        self.prompt_len, self.t_tail = prompt_len, t_tail
        self.device = torch.device(device)
        p_shape = (n_prompts, kv_heads, prompt_len, head_dim)
        t_shape = (n_tails, kv_heads, t_tail, head_dim)
        self.prompt_k = [torch.empty(p_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.prompt_v = [torch.empty(p_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.tail_k = [torch.empty(t_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.tail_v = [torch.empty(t_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        # per prompt, on device
        self.prompt_rope = torch.zeros(n_prompts, 1, dtype=torch.long, device=device)
        self.prompt_x1 = torch.zeros(n_prompts, *action_shape, dtype=torch.float32, device=device)
        # per tail, on device
        self.tail_offset = torch.zeros(n_tails, dtype=torch.long, device=device)
        # bookkeeping, on CPU
        self.prompt_valid = torch.zeros(n_prompts, dtype=torch.bool)
        self.prompt_refs = torch.zeros(n_prompts, dtype=torch.long)
        self.prompt_ident: list[Any] = [None] * n_prompts
        self.tail_valid = torch.zeros(n_tails, dtype=torch.bool)
        self.tail_prompt = torch.full((n_tails,), -1, dtype=torch.long)
        self.tail_uses = torch.zeros(n_tails, dtype=torch.long)
        self.tail_born = torch.zeros(n_tails, dtype=torch.long)
        self.tail_ident: list[Any] = [None] * n_tails
        self.step = 0

    # -- capacity ----------------------------------------------------------
    @property
    def bytes_per_prompt(self) -> int:
        return 2 * self.n_layers * self.prompt_k[0][0].numel() * self.prompt_k[0].element_size()

    @property
    def bytes_per_tail(self) -> int:
        return 2 * self.n_layers * self.tail_k[0][0].numel() * self.tail_k[0].element_size()

    def free_prompt_slots(self) -> list[int]:
        return (~self.prompt_valid).nonzero().flatten().tolist()

    def free_tail_slots(self) -> list[int]:
        return (~self.tail_valid).nonzero().flatten().tolist()

    def n_valid(self) -> tuple[int, int]:
        return int(self.prompt_valid.sum()), int(self.tail_valid.sum())

    # -- insertion ---------------------------------------------------------
    def put_prompt(self, slot: int, kv: PromptKV, ident: Any = None) -> None:
        if self.prompt_valid[slot]:
            raise ValueError(f"prompt slot {slot} is occupied; evict first")
        if kv.prompt_len != self.prompt_len:
            raise ValueError(f"prompt_len {kv.prompt_len} != pool {self.prompt_len}")
        if kv.x1 is None:
            raise ValueError("training conditionings need the GT action (x1)")
        for layer in range(self.n_layers):
            self.prompt_k[layer][slot].copy_(kv.keys[layer][0])
            self.prompt_v[layer][slot].copy_(kv.values[layer][0])
        self.prompt_rope[slot].copy_(kv.rope_delta[0])
        self.prompt_x1[slot].copy_(kv.x1[0])
        self.prompt_valid[slot] = True
        self.prompt_refs[slot] = 0
        self.prompt_ident[slot] = ident

    def put_tail(self, slot: int, prompt_slot: int, tail: TailKV, k: int, ident: Any = None) -> None:
        if self.tail_valid[slot]:
            raise ValueError(f"tail slot {slot} is occupied; evict first")
        if not self.prompt_valid[prompt_slot]:
            raise ValueError(f"prompt slot {prompt_slot} is empty")
        if tail.t_tail != self.t_tail:
            raise ValueError(f"t_tail {tail.t_tail} != pool {self.t_tail}")
        for layer in range(self.n_layers):
            self.tail_k[layer][slot].copy_(tail.keys[layer][k])
            self.tail_v[layer][slot].copy_(tail.values[layer][k])
        self.tail_offset[slot] = tail.offset[k]
        self.tail_valid[slot] = True
        self.tail_prompt[slot] = prompt_slot
        self.prompt_refs[prompt_slot] += 1
        self.tail_uses[slot] = 0
        self.tail_born[slot] = self.step
        self.tail_ident[slot] = ident

    # -- serving -----------------------------------------------------------
    def sample_batch(self, B: int, generator: torch.Generator | None = None,
                     duplicate: int = 1) -> ConditioningBatch:
        """``B`` tails drawn uniformly without replacement, tiled ``duplicate`` times.

        ``generator`` is a **CPU** generator: slot bookkeeping lives on the
        host, and the noise generator on the device is a different stream.
        """
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("sample_batch draws slot indices on the CPU; pass a CPU generator")
        valid = self.tail_valid.nonzero().flatten()
        if len(valid) == 0:
            raise RuntimeError("the pool is empty")
        if len(valid) >= B:
            pick = valid[torch.randperm(len(valid), generator=generator)[:B]]
        else:
            pick = valid[torch.randint(len(valid), (B,), generator=generator)]
        return self.gather(pick, duplicate)

    def gather(self, tail_slots: torch.Tensor, duplicate: int = 1) -> ConditioningBatch:
        tail_slots = torch.as_tensor(tail_slots, dtype=torch.long)
        prompt_slots = self.tail_prompt[tail_slots]
        t_idx = tail_slots.repeat(duplicate).to(self.device)
        p_idx = prompt_slots.repeat(duplicate).to(self.device)
        keys = [torch.cat([self.prompt_k[layer][p_idx], self.tail_k[layer][t_idx]], dim=2)
                for layer in range(self.n_layers)]
        values = [torch.cat([self.prompt_v[layer][p_idx], self.tail_v[layer][t_idx]], dim=2)
                  for layer in range(self.n_layers)]
        return ConditioningBatch(
            keys=keys, values=values,
            offset=self.tail_offset[t_idx],
            rope_delta=self.prompt_rope[p_idx],
            x1=self.prompt_x1[prompt_slots.to(self.device)],
            tail_slots=tail_slots,
            kv_len=self.prompt_len + self.t_tail,
        )

    def mark_used(self, tail_slots: torch.Tensor) -> None:
        self.tail_uses[torch.as_tensor(tail_slots)] += 1
        self.step += 1

    # -- rotation ----------------------------------------------------------
    def evict_tails(self, n: int) -> tuple[list[int], list[int]]:
        """Free the ``n`` most-used tails (oldest first among ties), and any
        prompt left with no tails. Returns ``(tail_slots, prompt_slots)`` freed."""
        valid = self.tail_valid.nonzero().flatten()
        if len(valid) == 0 or n <= 0:
            return [], []
        # sort by uses desc, then born asc
        key = self.tail_uses[valid] * (self.step + 2) + (self.step - self.tail_born[valid])
        order = torch.argsort(key, descending=True)
        victims = valid[order[:n]].tolist()
        freed_prompts: list[int] = []
        for slot in victims:
            p = int(self.tail_prompt[slot])
            self.tail_valid[slot] = False
            self.tail_prompt[slot] = -1
            self.tail_ident[slot] = None
            self.prompt_refs[p] -= 1
            if self.prompt_refs[p] == 0:
                self.prompt_valid[p] = False
                self.prompt_ident[p] = None
                freed_prompts.append(p)
        return victims, freed_prompts

    def evict_prompts(self, n: int) -> tuple[list[int], list[int]]:
        """Free ``n`` whole clips: the prompts whose tails have been used most
        (oldest first among ties), with every tail that points at them.

        Refill works a clip at a time -- one prompt plus its K tails -- so
        eviction has to free a prompt slot or refill has nothing to fill.
        Evicting the most-used *tails* does not: they spread across prompts,
        every prompt keeps at least one tail, no slot frees, and the pool
        bleeds down until a prompt happens to drain. Observed: 36 -> 12 tails
        in fifty steps with no clip added.
        """
        valid = self.prompt_valid.nonzero().flatten().tolist()
        if not valid or n <= 0:
            return [], []
        def score(p: int) -> tuple[float, float]:
            tails = (self.tail_prompt == p) & self.tail_valid
            uses = self.tail_uses[tails].float().mean().item() if tails.any() else 0.0
            born = self.tail_born[tails].float().min().item() if tails.any() else 0.0
            return (uses, -born)                       # most used, then oldest
        victims = sorted(valid, key=score, reverse=True)[:n]
        freed_tails: list[int] = []
        for p in victims:
            for t in ((self.tail_prompt == p) & self.tail_valid).nonzero().flatten().tolist():
                self.tail_valid[t] = False
                self.tail_prompt[t] = -1
                self.tail_ident[t] = None
                freed_tails.append(t)
            self.prompt_refs[p] = 0
            self.prompt_valid[p] = False
            self.prompt_ident[p] = None
        return freed_tails, victims

    def stats(self) -> dict[str, float]:
        v = self.tail_valid
        n_p, n_t = self.n_valid()
        return {
            "prompts": n_p, "tails": n_t,
            "tail_uses_mean": float(self.tail_uses[v].float().mean()) if n_t else 0.0,
            "tail_age_mean": float((self.step - self.tail_born[v]).float().mean()) if n_t else 0.0,
            "resident_gb": (n_p * self.bytes_per_prompt + n_t * self.bytes_per_tail) / 1e9,
        }
