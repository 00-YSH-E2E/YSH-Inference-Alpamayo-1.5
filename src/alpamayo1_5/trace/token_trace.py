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
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from alpamayo1_5.trace import timing_math
from alpamayo1_5.trace.timing_math import TimingResult

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
    #: ``[K, T, 2]`` -- the flow head's initial condition, as it entered the
    #: first Euler step (so already multiplied by the diffusion temperature).
    #: The other irrecoverable input: the ten-step teacher is a deterministic
    #: map from this tensor to the trajectory, and a one-step student is judged
    #: on reproducing that map point by point. Without x0 there is nothing to
    #: pair student and teacher on except the clip, which is far too coarse.
    x0: np.ndarray | None = None

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
            "x0": self.x0[k] if self.x0 is not None else None,
        }


#: The timing record. Kept under its old name for callers written against it.
SegmentTiming = TimingResult


class _EventPool:
    """CUDA timing events, reused across marks and across passes.

    Creating an event per mark is an allocation on the decode path, and the
    deeper trace levels place hundreds of marks per step. Events are taken in
    order and handed back all at once when the next pass installs, which is
    safe because ``finalize`` has read every event of a pass before the next
    one starts. A pass that never finalized (it raised) loses its events to the
    next one, and says so.
    """

    def __init__(self) -> None:
        self._events: list[Any] = []
        self._next = 0
        self.owner: Any = None

    def acquire(self, owner: Any) -> None:
        if self.owner is not None and self.owner is not owner:
            print("[token_trace] a previous pass never finalized; its events are reused")
        self.owner = owner
        self._next = 0

    def release(self, owner: Any) -> None:
        if self.owner is owner:
            self.owner = None

    def take(self) -> Any:
        if self._next == len(self._events):
            self._events.append(torch.cuda.Event(enable_timing=True))
        event = self._events[self._next]
        self._next += 1
        return event


_POOL = _EventPool()


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
        self.timing = TimingResult()
        self._x0: Any = None
        self._handles: list[Any] = []
        self._saved: dict[str, Any] = {}
        #: ``(bucket, kind, event, perf_counter)`` in the order the host saw them.
        self._marks: list[tuple[str, str, Any, float]] = []
        self._enabled = False
        self._wall_start: float | None = None
        self._start_unix: float | None = None
        self._alloc0: tuple[int, int] | None = None
        self._graph0: dict[str, int] | None = None
        self._runner: Any = None
        self._capture_ms = 0.0
        self._in_capture = False

    # -- installation ------------------------------------------------------
    def __enter__(self) -> "InferenceTracer":
        self._install()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._remove()

    def _mark(self, bucket: str, kind: str) -> None:
        # A mark inside a graph capture would record a timing event into the
        # captured graph. Capture brackets its own work; nothing inside it is
        # a span of this pass.
        if self._in_capture or not torch.cuda.is_available():
            return
        event = _POOL.take()
        event.record()
        # Stamped after record() so both clocks bracket the same instant.
        # About 50ns, which is why it can sit on the decode path without
        # becoming the thing it is measuring.
        self._marks.append((bucket, kind, event, time.perf_counter()))

    def _set_attr(self, obj: Any, name: str, value: Any) -> None:
        """Shadow ``obj.name`` on the instance, remembering how to undo it.

        Restoring by assignment would leave an instance attribute behind that
        was never there -- a bound method pinning its object -- so the undo
        deletes what was added and reassigns only what was already present.
        """
        had = name in vars(obj)
        self._saved.setdefault("attrs", []).append((obj, name, had, vars(obj).get(name)))
        setattr(obj, name, value)

    def _install(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        self._marks.clear()
        self.timing = TimingResult()
        if torch.cuda.is_available():
            _POOL.acquire(self)
        self._wall_start = time.perf_counter()
        self._start_unix = time.time()
        # Deltas, not absolutes. Peak memory is deliberately *not* reset
        # here: the runner resets it once per run and reads the global peak
        # for vram_peak_gb, so resetting per clip would quietly redefine
        # that metric as the peak of the last clip.
        self._alloc0 = self._alloc_counters()
        self._runner = self._graph_runner()
        self._graph0 = self._graph_counters()
        self._capture_ms = 0.0
        if self._runner is not None and hasattr(self._runner, "_capture"):
            self._set_attr(self._runner, "_capture", self._wrap_capture(self._runner._capture))

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

        # The flow head's initial condition. step_fn calls action_in_proj(x, t)
        # once per Euler step, and on the first call x is the noise the sampler
        # just drew. Captured here rather than by changing the sampler, so the
        # upstream file stays untouched and a run made without the tracer is
        # simply missing the column rather than behaving differently.
        proj = getattr(self.model, "action_in_proj", None)
        if proj is not None:
            self._handles.append(proj.register_forward_pre_hook(self._capture_x0))

        # Trajectory head: one call per Euler step. With graphs on, the hooks
        # still fire: the runner replaces forward, and module hooks wrap the
        # call rather than the function.
        expert = getattr(self.model, "expert", None)
        if expert is not None:
            self._handles.append(
                expert.register_forward_pre_hook(lambda *_: self._mark("expert", "start"))
            )
            self._handles.append(
                expert.register_forward_hook(lambda *_: self._mark("expert", "end"))
            )

        self._set_attr(vlm, "generate", self._wrap_generate(vlm.generate))

        diffusion = getattr(self.model, "diffusion", None)
        if diffusion is not None and hasattr(diffusion, "sample"):
            self._set_attr(diffusion, "sample", self._wrap_sample(diffusion.sample))

    # -- counters ----------------------------------------------------------
    @staticmethod
    def _alloc_counters() -> tuple[int, int] | None:
        """Allocator totals ``(allocations, retries)``, or None when unreadable.

        Both are monotonic over the process, so only a delta across one
        inference means anything. Retries matter more than the count: each one
        is the allocator failing, flushing its cache and trying again, and that
        stall lands in whichever segment was unlucky. None rather than zeros
        when there is nothing to read -- zero allocations is a plausible
        measurement, and a plausible number is worse than a missing one.
        """
        if not torch.cuda.is_available():
            return None
        try:
            stats = torch.cuda.memory_stats()
        except Exception:
            return None
        return (
            int(stats.get("allocation.all.allocated", 0)),
            int(stats.get("num_alloc_retries", 0)),
        )

    def _graph_runner(self) -> Any:
        """The expert's CUDA-graph runner, if one is installed.

        The attribute is set on both the model and the expert, and which one
        is present depends on how the graph was enabled. The model is checked
        first: that is the documented entry point.
        """
        runner = getattr(self.model, "_diffusion_expert_cuda_graph", None)
        if runner is None:
            runner = getattr(
                getattr(self.model, "expert", None), "_diffusion_expert_cuda_graph", None
            )
        return runner

    def _graph_counters(self) -> dict[str, int] | None:
        """The runner's tallies, or None when no runner is installed.

        The runner kept these since it was written and nothing ever read them,
        so a run could not say whether its steps replayed or fell back to eager
        -- and a mean over both describes neither.
        """
        runner = self._runner if self._runner is not None else self._graph_runner()
        if runner is None:
            return None
        try:
            return dict(runner.stats)
        except Exception:
            return None

    def _wrap_capture(self, original: Any) -> Any:
        """Time the runner's capture and keep marks out of it."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            self._in_capture = True
            try:
                return original(*args, **kwargs)
            finally:
                self._in_capture = False
                self._capture_ms += (time.perf_counter() - started) * 1000.0

        return wrapper

    def _remove(self) -> None:
        if not self._enabled:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for obj, name, had, value in reversed(self._saved.pop("attrs", [])):
            if had:
                setattr(obj, name, value)
            else:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
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

    def _capture_x0(self, module: Any, args: tuple) -> None:
        """Keep the first action_in_proj input of a sample() call; ignore the rest."""
        if self._x0 is None and args:
            self._x0 = args[0].detach()

    def _wrap_sample(self, original: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._x0 = None
            self._mark("diffusion", "start")
            out = original(*args, **kwargs)
            self._mark("diffusion", "end")
            # generate() ran first and built self.trace; the noise only exists now.
            if self.trace is not None and self._x0 is not None:
                self.trace.x0 = self._x0.float().cpu().numpy()
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
    def finalize(self) -> TimingResult:
        """Resolve recorded events into segment times. Call once, after inference."""
        alloc1 = self._alloc_counters()
        graph1 = self._graph_counters()
        records: list[timing_math.Record] = []
        wall_end = None
        if torch.cuda.is_available() and self._marks:
            torch.cuda.synchronize()  # the only sync -- inside a hook it would distort decode
            # An end anchor on the device. Every event's time is taken relative
            # to it, which puts all marks on one device axis, and the host time
            # read right after it completes is what later maps device times onto
            # the host clock the board sampler uses.
            anchor = torch.cuda.Event(enable_timing=True)
            anchor.record()
            anchor.synchronize()
            wall_end = time.perf_counter()
            records = [(bucket, kind, -float(event.elapsed_time(anchor)), host)
                       for bucket, kind, event, host in self._marks]
        self.timing = timing_math.resolve(
            records,
            wall_start_s=self._wall_start if records else None,
            wall_end_s=wall_end,
            start_unix_s=self._start_unix,
            alloc_before=self._alloc0,
            alloc_after=alloc1,
            graph_before=self._graph0,
            graph_after=graph1,
            capture_ms=self._capture_ms,
        )
        _POOL.release(self)
        return self.timing


@contextlib.contextmanager
def trace_inference(model: Any, special_token_ids: dict[str, int] | None = None):
    """Convenience wrapper: yields a tracer and resolves timings on exit."""
    tracer = InferenceTracer(model, special_token_ids)
    with tracer:
        yield tracer
    tracer.finalize()
