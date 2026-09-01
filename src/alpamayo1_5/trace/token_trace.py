# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Capture what only exists while the model is generating.

Two things vanish the moment inference returns and cannot be recovered:

* **Per-token confidence.** A compressed model tends to lose confidence before
  its text visibly degrades -- it keeps emitting fluent driving language while
  the distribution behind each token flattens. Reading twenty samples will not
  catch that; mean entropy will, and unlike a hallucination rate it is a
  continuous number that goes straight onto a comparison curve.
* **Where the time went.** One end-to-end latency number cannot say whether a
  change helped vision encoding, the autoregressive loop, or the trajectory
  head. Those respond differently to compression.

Everything here is additive -- no upstream file is modified. The model's
``generate`` is wrapped on the instance (the repo does the same thing itself in
``diffusion_expert_cuda_graph.py:368``) and timing comes from module hooks.

Notes on details that are easy to get wrong, all verified against this
checkpoint:

* ``vlm_outputs.logits`` is **already masked** over the trajectory-token band:
  ``ExpertLogitsProcessor`` writes ``-inf`` in place and HF stores that same
  tensor. It is however **pre-temperature and pre-top-p**, since those warpers
  are out-of-place. That combination is exactly right for a quality metric --
  unreachable tokens are excluded, and the number does not move when sampling
  settings change.
* Generation stops only once **every** row has emitted the end marker, so rows
  finish at different steps and accumulate junk afterwards. Averaging over the
  full batch length would mix that junk into every statistic.
* The prompt is left-padded with the same id used for post-EOS padding, so any
  count must slice the generated region off first.
* Hooks fire when the Python call returns, not when the CUDA work completes.
  Timing uses events and one synchronize at the end; a synchronize inside a
  hook would serialize the pipeline and inflate the decode segment.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Resolved from nvidia/Alpamayo-1.5-10B. Overridable because a fine-tuned
# checkpoint may extend the vocabulary.
DEFAULT_SPECIAL_IDS = {
    "pad": 151643,
    "im_end": 151645,
    "cot_start": 155677,
    "cot_end": 155678,
    "meta_action_start": 155679,
    "meta_action_end": 155680,
    "traj_future_start": 155681,
}


@dataclass
class TokenTrace:
    """Per-token record for one generate call, shape ``[K, T]`` unless noted."""

    token_ids: np.ndarray
    logprob: np.ndarray
    entropy: np.ndarray
    n_generated: np.ndarray  # [K] tokens through and including the end marker
    n_cot: np.ndarray  # [K] tokens inside the reasoning span
    eos_missing: np.ndarray  # [K] bool -- ran to max_new_tokens without the marker
    prompt_len: int

    def sample(self, k: int) -> dict[str, Any]:
        """Row ``k``, trimmed to its own valid length."""
        n = int(self.n_generated[k])
        return {
            "token_ids": self.token_ids[k, :n],
            "token_logprob": self.logprob[k, :n],
            "token_entropy": self.entropy[k, :n],
            "n_generated_tokens": n,
            "n_cot_tokens": int(self.n_cot[k]),
            "eos_missing": bool(self.eos_missing[k]),
            # Same for every row of a batch, but it has to travel on the row:
            # the writer reads it back off row 0 to record how far into the
            # padded sequence generation started. Leaving it off meant every
            # run.json written so far carries prompt_len 0, which reads as a
            # real measurement and slices an offline reader into the padding.
            "prompt_len": int(self.prompt_len),
        }


@dataclass
class SegmentTiming:
    """Wall-clock split of one inference, in milliseconds."""

    vision_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    postgen_ms: float = 0.0
    expert_ms: float = 0.0
    total_ms: float = 0.0
    #: ``total_ms`` minus the five named segments. Vision, prefill and decode all
    #: run inside the generate span, so the named parts never summed to the total
    #: and nothing said by how much. Recording the remainder makes the split
    #: self-checking: the six add up, or the instrumentation is wrong.
    other_ms: float = 0.0
    n_vision_calls: int = 0
    n_decode_steps: int = 0
    n_expert_calls: int = 0
    #: False means no timing was taken at all (no CUDA, or no events recorded).
    #: Without this, an unmeasured run reports 0.0 ms and reads as "took no
    #: time" rather than "never measured" -- a plausible number is worse than
    #: a missing one, because it gets averaged into a comparison.
    measured: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "t_vision_ms": self.vision_ms,
            "t_prefill_ms": self.prefill_ms,
            "t_decode_ms": self.decode_ms,
            "t_postgen_ms": self.postgen_ms,
            "t_expert_ms": self.expert_ms,
            "t_total_ms": self.total_ms,
            "t_other_ms": self.other_ms,
            # Decode is batched over K rows and runs until the *last* row stops,
            # so a per-row token count is the wrong denominator for it. This is
            # the right one, and it was being computed and thrown away.
            "n_decode_steps": self.n_decode_steps,
            "n_vision_calls": self.n_vision_calls,
            "n_expert_calls": self.n_expert_calls,
            "timing_measured": self.measured,
        }


def _reduce_logits(logits_step: torch.Tensor, chosen: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """One step of logits to (logprob of chosen token, entropy). Never keeps the logits.

    ``logits_step`` carries ``-inf`` over tokens the model is forbidden to emit.
    Those contribute zero probability, and zero probability must contribute zero
    to the entropy -- computing ``p * log p`` directly would produce ``nan``.
    """
    logp = torch.log_softmax(logits_step.float(), dim=-1)
    p = logp.exp()
    contrib = torch.where(p > 0, p * logp, torch.zeros_like(p))
    entropy = -contrib.sum(dim=-1)
    chosen_logp = logp.gather(1, chosen.view(-1, 1)).squeeze(1)
    return chosen_logp, entropy


class InferenceTracer:
    """Records token statistics and segment timing for calls made inside the block.

    Use as a context manager around one inference; read ``trace`` and ``timing``
    afterwards. Installing and removing is idempotent, and the model is left
    exactly as it was found.
    """

    def __init__(self, model: Any, special_token_ids: dict[str, int] | None = None) -> None:
        self.model = model
        self.ids = dict(DEFAULT_SPECIAL_IDS)
        if special_token_ids:
            self.ids.update(special_token_ids)
        self.trace: TokenTrace | None = None
        self.timing = SegmentTiming()
        self._handles: list[Any] = []
        self._saved: dict[str, Any] = {}
        self._events: dict[str, list] = {}
        self._enabled = False

    # -- installation ------------------------------------------------------
    def __enter__(self) -> "InferenceTracer":
        self._install()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._remove()

    def _mark(self, bucket: str, kind: str) -> None:
        if not torch.cuda.is_available():
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events.setdefault(bucket, []).append((kind, event))

    def _install(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        self._events.clear()
        self.timing = SegmentTiming()

        vlm = self.model.vlm
        inner = getattr(vlm, "model", None)

        # Vision tower: runs once per generate (the sampler nulls pixel_values
        # after the first step), so this isolates image encoding.
        visual = getattr(inner, "visual", None)
        if visual is not None:
            self._handles.append(
                visual.register_forward_pre_hook(lambda *_: self._mark("vision", "start"))
            )
            self._handles.append(
                visual.register_forward_hook(lambda *_: self._mark("vision", "end"))
            )

        # Language model: call #1 is the prompt prefill, the rest are decode
        # steps. The call count is also the exact number of generated tokens.
        language = getattr(inner, "language_model", None)
        if language is not None:
            self._handles.append(
                language.register_forward_pre_hook(lambda *_: self._mark("lm", "start"))
            )
            self._handles.append(
                language.register_forward_hook(lambda *_: self._mark("lm", "end"))
            )

        # Trajectory head: one call per Euler step.
        expert = getattr(self.model, "expert", None)
        if expert is not None:
            self._handles.append(
                expert.register_forward_pre_hook(lambda *_: self._mark("expert", "start"))
            )
            self._handles.append(
                expert.register_forward_hook(lambda *_: self._mark("expert", "end"))
            )

        self._saved["generate"] = vlm.generate
        vlm.generate = self._wrap_generate(vlm.generate)

        diffusion = getattr(self.model, "diffusion", None)
        if diffusion is not None and hasattr(diffusion, "sample"):
            self._saved["sample"] = diffusion.sample
            self._saved["diffusion"] = diffusion
            diffusion.sample = self._wrap_sample(diffusion.sample)

    def _remove(self) -> None:
        if not self._enabled:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if "generate" in self._saved:
            self.model.vlm.generate = self._saved.pop("generate")
        if "sample" in self._saved:
            self._saved.pop("diffusion").sample = self._saved.pop("sample")
        self._enabled = False

    # -- wrappers ----------------------------------------------------------
    def _wrap_generate(self, original: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._mark("generate", "start")
            out = original(*args, **kwargs)
            self._mark("generate", "end")
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            try:
                self._consume(out, int(input_ids.shape[1]))
            except Exception as exc:  # instrumentation must not break the run
                print(f"[token_trace] could not read logits: {exc}")
            # These are large (roughly 0.6MB per row-step) and this method never
            # frees them; dropping them here is a memory win, not just cleanup.
            if hasattr(out, "logits"):
                try:
                    del out.logits
                except Exception:
                    pass
            return out

        return wrapper

    def _wrap_sample(self, original: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._mark("diffusion", "start")
            out = original(*args, **kwargs)
            self._mark("diffusion", "end")
            return out

        return wrapper

    # -- consumption -------------------------------------------------------
    def _consume(self, out: Any, prompt_len: int) -> None:
        logits = getattr(out, "logits", None)
        sequences = out.sequences
        gen = sequences[:, prompt_len:].detach()
        rows, steps = gen.shape

        logprob = np.full((rows, steps), np.nan, dtype=np.float32)
        entropy = np.full((rows, steps), np.nan, dtype=np.float32)
        if logits is not None:
            for t in range(min(steps, len(logits))):
                lp, ent = _reduce_logits(logits[t], gen[:, t])
                logprob[:, t] = lp.detach().cpu().numpy()
                entropy[:, t] = ent.detach().cpu().numpy()

        end_id = self.ids["traj_future_start"]
        cot_end_id = self.ids["cot_end"]
        pad_id = self.ids["pad"]
        gen_cpu = gen.cpu().numpy()

        n_generated = np.zeros(rows, dtype=np.int32)
        n_cot = np.zeros(rows, dtype=np.int32)
        eos_missing = np.zeros(rows, dtype=bool)
        for r in range(rows):
            row = gen_cpu[r]
            hit = np.flatnonzero(row == end_id)
            if hit.size:
                # Through and including the end marker.
                n_generated[r] = int(hit[0]) + 1
            else:
                # No marker: the row either emitted the chat end token and was
                # padded out, or ran to max_new_tokens. Count non-pad tokens --
                # only valid on the generated slice, since the prompt is
                # left-padded with this same id.
                eos_missing[r] = True
                nonpad = np.flatnonzero(row != pad_id)
                n_generated[r] = int(nonpad[-1]) + 1 if nonpad.size else 0
            # The prompt ends at the reasoning-start marker, so the reasoning
            # span begins at index 0 of the generated region.
            close = np.flatnonzero(row[: n_generated[r]] == cot_end_id)
            n_cot[r] = int(close[0]) if close.size else int(n_generated[r])

        self.trace = TokenTrace(
            token_ids=gen_cpu.astype(np.int32),
            logprob=logprob,
            entropy=entropy,
            n_generated=n_generated,
            n_cot=n_cot,
            eos_missing=eos_missing,
            prompt_len=prompt_len,
        )

    # -- results -----------------------------------------------------------
    def finalize(self) -> SegmentTiming:
        """Resolve recorded events into segment times. Call once, after inference."""
        timing = SegmentTiming()
        if not torch.cuda.is_available() or not self._events:
            self.timing = timing
            return timing
        torch.cuda.synchronize()  # the only sync -- inside a hook it would distort decode

        def spans(bucket: str) -> list[tuple[Any, Any]]:
            marks = self._events.get(bucket, [])
            out = []
            pending = None
            for kind, event in marks:
                if kind == "start":
                    pending = event
                elif pending is not None:
                    out.append((pending, event))
                    pending = None
            return out

        def total(pairs: list[tuple[Any, Any]]) -> float:
            return float(sum(a.elapsed_time(b) for a, b in pairs))

        vision = spans("vision")
        lm = spans("lm")
        expert = spans("expert")
        diffusion = spans("diffusion")
        generate = spans("generate")

        timing.n_vision_calls = len(vision)
        timing.vision_ms = total(vision)
        if lm:
            # The first language-model call is the prefill; the rest are decode.
            timing.prefill_ms = total(lm[:1])
            timing.decode_ms = total(lm[1:])
            timing.n_decode_steps = len(lm) - 1
        timing.n_expert_calls = len(expert)
        timing.expert_ms = total(diffusion) if diffusion else total(expert)

        # Between generate returning and the trajectory head starting there is
        # real host work -- per-sample Python loops that force device syncs.
        # Split into three and it lands in the wrong bucket.
        if generate and diffusion:
            gap = generate[0][1].elapsed_time(diffusion[0][0])
            timing.postgen_ms = float(max(gap, 0.0))
        if generate:
            timing.total_ms = total(generate) + timing.postgen_ms + timing.expert_ms
            # What the named segments do not account for: logits processors,
            # sampling, stopping criteria, KV-cache management. Clamped at zero
            # because a negative remainder means the spans overlap, which is a
            # bug in the marks rather than negative time spent.
            named = (timing.vision_ms + timing.prefill_ms + timing.decode_ms
                     + timing.postgen_ms + timing.expert_ms)
            timing.other_ms = float(max(timing.total_ms - named, 0.0))
        timing.measured = True
        self.timing = timing
        return timing


@contextlib.contextmanager
def trace_inference(model: Any, special_token_ids: dict[str, int] | None = None):
    """Convenience wrapper: yields a tracer and resolves timings on exit."""
    tracer = InferenceTracer(model, special_token_ids)
    with tracer:
        yield tracer
    tracer.finalize()
