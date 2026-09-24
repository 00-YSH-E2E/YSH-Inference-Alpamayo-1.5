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
import functools
import inspect
import os
import resource
import time
import warnings
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

#: The allocator's peak over the whole run. Each pass reads and resets the peak
#: at its segment boundaries, which would otherwise leave the run-level
#: ``vram_peak_gb`` describing only the last segment of the last clip. Every
#: read is folded in here first, so the run's peak keeps its meaning.
_RUN_PEAK = {"bytes": 0}


def run_peak_bytes() -> int:
    """Peak allocated bytes since the run's reset, across every pass and between them."""
    peak = _RUN_PEAK["bytes"]
    if torch.cuda.is_available():
        peak = max(peak, int(torch.cuda.max_memory_allocated()))
    return peak

#: How deep the instrumentation goes. ``off`` installs no hook at all -- only a
#: pair of marks around the call, for the wall clock -- and exists so the cost
#: of ``basic`` can be measured against it on the same clip. Deeper levels are
#: added by the commits that implement them.
TRACE_LEVELS = ("off", "basic", "step")

#: What torch's sync debug mode says when an operation synchronizes the host
#: with the device. Checked against libc10_cuda in the pinned torch.
_SYNC_MESSAGE = "called a synchronizing CUDA operation"

#: Where the step level attributes a synchronization, by the mark that opens
#: the phase. Decode steps alternate between "decode" (the forward) and
#: "gen_loop" (everything generate does between forwards).
_PHASE_AFTER = {
    ("generate", "start"): "gen_loop", ("vision", "start"): "vision",
    ("vision", "end"): "prefill", ("lm", "end"): "gen_loop",
    ("generate", "end"): "postgen", ("consume", "start"): "consume",
    ("consume", "end"): "postgen", ("diffusion", "start"): "expert",
    ("diffusion", "end"): "tail",
}


@functools.lru_cache(maxsize=1)
def _timed_lists() -> tuple[type, type]:
    """generate's processor and stopping lists, with every member marked.

    generate looks ``__call__`` up on the class, so an instance attribute
    cannot intercept it; a subclass returned from the instance-wrapped
    ``_get_logits_processor`` can. The loop mirrors LogitsProcessorList's own,
    signature check included, so what runs is unchanged.
    """
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList

    class TimedProcessors(LogitsProcessorList):
        tracer: Any = None

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            tracer = self.tracer
            tracer._mark("lp", "start")
            for processor in self:
                bucket = f"lp:{type(processor).__name__}"
                tracer._mark(bucket, "start")
                if len(inspect.signature(processor.__call__).parameters) > 2:
                    scores = processor(input_ids, scores, **kwargs)
                else:
                    scores = processor(input_ids, scores)
                tracer._mark(bucket, "end")
            tracer._mark("lp", "end")
            return scores

    class TimedStopping(StoppingCriteriaList):
        tracer: Any = None

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            self.tracer._mark("stop", "start")
            try:
                return super().__call__(input_ids, scores, **kwargs)
            finally:
                self.tracer._mark("stop", "end")

    return TimedProcessors, TimedStopping


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


def _process_stats() -> dict[str, Any] | None:
    """Process CPU, the main thread's context switches, and resident memory.

    Read outside the timed spans (at install and finalize), never at a mark:
    a getrusage and a /proc read are cheap, but not free, and the point of the
    marks is to cost next to nothing.
    """
    try:
        times = os.times()
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        with open("/proc/self/statm") as handle:
            resident_pages = int(handle.read().split()[1])
        available = None
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
        return {
            "cpu_s": times.user + times.system,
            "nvcsw": usage.ru_nvcsw,
            "nivcsw": usage.ru_nivcsw,
            "rss_bytes": resident_pages * os.sysconf("SC_PAGE_SIZE"),
            "mem_available": available,
        }
    except (OSError, ValueError, AttributeError):
        return None


def _process_delta(before: dict[str, Any] | None,
                   after: dict[str, Any] | None) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    return {
        "proc_cpu_ms": (after["cpu_s"] - before["cpu_s"]) * 1000.0,
        "ctx_vol": max(after["nvcsw"] - before["nvcsw"], 0),
        "ctx_invol": max(after["nivcsw"] - before["nivcsw"], 0),
        "rss_bytes": after["rss_bytes"],
        "host_mem_avail_min_bytes": (
            min(before["mem_available"], after["mem_available"])
            if before.get("mem_available") is not None
            and after.get("mem_available") is not None else None),
    }


class InferenceTracer:
    """Records token statistics and segment timing for calls made inside the block.

    Use as a context manager around one inference; read ``trace`` and ``timing``
    afterwards. Installing and removing is idempotent, and the model is left
    exactly as it was found.
    """

    def __init__(self, model: Any, special_token_ids: dict[str, int] | None = None,
                 level: str = "basic") -> None:
        if level not in TRACE_LEVELS:
            raise ValueError(f"trace level {level!r} is not one of {TRACE_LEVELS}")
        self.model = model
        self.level = level
        self.ids = dict(DEFAULT_SPECIAL_IDS)
        if special_token_ids:
            self.ids.update(special_token_ids)
        self.trace: TokenTrace | None = None
        self.timing = TimingResult()
        self._x0: Any = None
        self._handles: list[Any] = []
        self._saved: dict[str, Any] = {}
        #: ``(bucket, kind, event, perf_counter, thread_time)`` in host order.
        self._marks: list[tuple[str, str, Any, float, float]] = []
        #: Host milliseconds inside wrapped host-side functions, by name.
        self._host_ms: dict[str, float] = {}
        self._process0: dict[str, Any] | None = None
        #: Allocator reads at segment boundaries, in call order.
        self._memory: list[dict[str, Any]] = []
        #: Shapes read off hook arguments: the KV cache the head attends to, the images.
        self._shapes: dict[str, Any] = {}
        self._lm_calls = 0
        # Step level only.
        self._phase_now = "pre_generate"
        self._kv_bytes: dict[str, int] = {}
        self._kv_calls = 0
        self._host_lists: dict[str, list[float]] = {}
        self._sync_counts: dict[str, int] = {}
        self._sync_sites: dict[str, int] = {}
        self._sync_prev_mode: Any = None
        self._warn_ctx: Any = None
        self._seen_caches: set[int] = set()
        self._enabled = False
        self._wall_start: float | None = None
        self._start_unix: float | None = None
        self._alloc0: tuple[int, int] | None = None
        self._graph0: dict[str, int] | None = None
        self._runner: Any = None
        self._capture_ms = 0.0
        self._in_capture = False
        #: Host seconds spent inside _mark over the pass: the instrument's own cost.
        self._hook_s = 0.0

    # -- installation ------------------------------------------------------
    def __enter__(self) -> "InferenceTracer":
        self._install()
        return self

    def __exit__(self, *exc: Any) -> None:
        # Brackets the call at every level: the spans before generate and after
        # the head, and the time to the first trajectory, are measured from it.
        self._mark("call", "end")
        self._remove()

    def _mark(self, bucket: str, kind: str) -> None:
        # A mark inside a graph capture would record a timing event into the
        # captured graph. Capture brackets its own work; nothing inside it is
        # a span of this pass.
        if self._in_capture or not torch.cuda.is_available():
            return
        entered = time.perf_counter()
        event = _POOL.take()
        event.record()
        # Stamped after record() so both clocks bracket the same instant. The
        # thread's CPU time rides along: against the host clock it separates a
        # busy host from one that was waiting or descheduled.
        stamp = time.perf_counter()
        self._marks.append((bucket, kind, event, stamp, time.thread_time()))
        if self.level == "step":
            if bucket == "lm" and kind == "start":
                self._phase_now = "prefill" if self._lm_calls == 0 else "decode"
            else:
                self._phase_now = _PHASE_AFTER.get((bucket, kind), self._phase_now)
        # What the mark itself cost the host. Summed per pass, it is the
        # instrument's own share of the wall clock -- a number the recording
        # rules require and that no end-to-end comparison can recover later.
        self._hook_s += stamp - entered

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
        self._hook_s = 0.0
        self._host_ms = {}
        self._memory = []
        self._shapes = {}
        self._lm_calls = 0
        self._phase_now = "pre_generate"
        self._kv_bytes, self._kv_calls = {}, 0
        self._host_lists = {}
        self._sync_counts, self._sync_sites = {}, {}
        self._seen_caches = set()
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
        self._process0 = _process_stats()
        self._mem_mark("start")
        self._runner = self._graph_runner()
        self._graph0 = self._graph_counters()
        self._capture_ms = 0.0
        if self.level == "off":
            # Nothing inside the call is observed: two marks bracket it, so the
            # wall clock is taken exactly as at every other level, and nothing
            # else is added to what is being timed.
            self._mark("call", "start")
            return
        if self._runner is not None and hasattr(self._runner, "_capture"):
            self._set_attr(self._runner, "_capture", self._wrap_capture(self._runner._capture))

        vlm = self.model.vlm
        inner = getattr(vlm, "model", None)

        # Vision tower: runs once per generate (the sampler nulls pixel_values
        # after the first step), so this isolates image encoding.
        visual = getattr(inner, "visual", None)
        if visual is not None:
            self._handles.append(
                visual.register_forward_pre_hook(self._on_vision_start, with_kwargs=True)
            )
            self._handles.append(visual.register_forward_hook(self._on_vision_end))

        # Language model: call #1 is the prompt prefill, the rest are decode
        # steps. The call count is also the exact number of generated tokens.
        language = getattr(inner, "language_model", None)
        if language is not None:
            self._handles.append(
                language.register_forward_pre_hook(lambda *_: self._mark("lm", "start"))
            )
            self._handles.append(language.register_forward_hook(self._on_lm_end))

        # The whole VLM forward, once per generate step, and the vocabulary
        # projection inside it. lm_head runs after the language model returns,
        # so the decode span never contained it; together these two split the
        # generate remainder into preamble, lm_head, forward glue and the loop.
        # generate calls the model through __call__ for prefill and decode alike,
        # so module hooks fire on every step.
        if isinstance(vlm, torch.nn.Module):
            self._handles.append(
                vlm.register_forward_pre_hook(lambda *_: self._mark("vlm", "start"))
            )
            self._handles.append(vlm.register_forward_hook(lambda *_: self._mark("vlm", "end")))
        head = getattr(vlm, "lm_head", None)
        if isinstance(head, torch.nn.Module):
            self._handles.append(
                head.register_forward_pre_hook(lambda *_: self._mark("lm_head", "start"))
            )
            self._handles.append(
                head.register_forward_hook(lambda *_: self._mark("lm_head", "end"))
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
                expert.register_forward_pre_hook(self._on_expert_start, with_kwargs=True)
            )
            self._handles.append(
                expert.register_forward_hook(lambda *_: self._mark("expert", "end"))
            )

        self._set_attr(vlm, "generate", self._wrap_generate(vlm.generate))

        diffusion = getattr(self.model, "diffusion", None)
        if diffusion is not None and hasattr(diffusion, "sample"):
            self._set_attr(diffusion, "sample", self._wrap_sample(diffusion.sample))

        # Host-side work around generate that no module hook sees: every one is
        # called through its instance, so an instance attribute intercepts it.
        # fuse_traj_tokens runs before generate, the K-fold input copy and the
        # rope index inside it, action_to_traj after the head.
        for obj, name, key in (
            (self.model, "fuse_traj_tokens", "fuse_traj"),
            (vlm, "_expand_inputs_for_generation", "expand_inputs"),
            (inner, "get_rope_index", "rope_index"),
        ):
            if obj is not None and callable(getattr(obj, name, None)):
                self._set_attr(obj, name, self._wrap_host(getattr(obj, name), key))
        space = getattr(self.model, "action_space", None)
        if space is not None and callable(getattr(space, "action_to_traj", None)):
            self._set_attr(space, "action_to_traj",
                           self._wrap_host(space.action_to_traj, "action_to_traj", "a2t"))

        if self.level == "step":
            self._install_step(vlm, inner)

        # Last, so the call's span starts where the model's work does.
        self._mark("call", "start")

    # -- step level -----------------------------------------------------------
    def _hook_pair(self, module: Any, bucket: str) -> None:
        if isinstance(module, torch.nn.Module):
            self._handles.append(
                module.register_forward_pre_hook(lambda *_: self._mark(bucket, "start")))
            self._handles.append(module.register_forward_hook(lambda *_: self._mark(bucket, "end")))

    def _install_step(self, vlm: Any, inner: Any) -> None:
        """Inside the steps: the head's projections and KV concatenation, the decode
        loop's processors and stopping criteria, postgen's parts, the vision
        tower's parts, and every host-device synchronization.

        Around 150 marks per decode step: this level is for attribution, and
        the overhead probe measures what it costs.
        """
        self._hook_pair(getattr(self.model, "action_in_proj", None), "in_proj")
        self._hook_pair(getattr(self.model, "action_out_proj", None), "out_proj")
        # The KV cache is created inside generate; it is wrapped the first time
        # the language model or the head is handed it.
        for module in (getattr(inner, "language_model", None), getattr(self.model, "expert", None)):
            if isinstance(module, torch.nn.Module):
                self._handles.append(
                    module.register_forward_pre_hook(self._see_cache, with_kwargs=True))
        try:
            timed_processors, timed_stopping = _timed_lists()
        except Exception:
            timed_processors = timed_stopping = None
        if timed_processors is not None and callable(getattr(vlm, "_get_logits_processor", None)):
            self._set_attr(vlm, "_get_logits_processor",
                           self._wrap_list(vlm._get_logits_processor, timed_processors))
        if timed_stopping is not None and callable(getattr(vlm, "_get_stopping_criteria", None)):
            self._set_attr(vlm, "_get_stopping_criteria",
                           self._wrap_list(vlm._get_stopping_criteria, timed_stopping))
        for name, key in (("prepare_inputs_for_generation", "prep_inputs"),
                          ("_update_model_kwargs_for_generation", "update_kwargs")):
            if callable(getattr(vlm, name, None)):
                self._set_attr(vlm, name, self._wrap_host_list(getattr(vlm, name), key))
        for name, bucket in (("_find_eos_offset", "find_eos"),
                             ("_build_expert_pos_ids_and_attn_mask", "build_mask")):
            if callable(getattr(self.model, name, None)):
                self._set_attr(self.model, name,
                               self._wrap_host(getattr(self.model, name), bucket, bucket))
        visual = getattr(inner, "visual", None)
        if visual is not None:
            self._hook_pair(getattr(visual, "patch_embed", None), "v_patch")
            blocks = getattr(visual, "blocks", None)
            if blocks is not None and len(blocks):
                self._handles.append(blocks[0].register_forward_pre_hook(
                    lambda *_: self._mark("v_blocks", "start")))
                self._handles.append(blocks[-1].register_forward_hook(
                    lambda *_: self._mark("v_blocks", "end")))
            for merger in getattr(visual, "deepstack_merger_list", None) or []:
                self._hook_pair(merger, "v_deepstack")
            self._hook_pair(getattr(visual, "merger", None), "v_merger")
        self._start_sync_audit()

    def _wrap_list(self, original: Any, timed_class: type) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            built = original(*args, **kwargs)
            try:
                timed = timed_class(built)
                timed.tracer = self
                return timed
            except Exception:
                return built

        return wrapper

    def _wrap_host_list(self, original: Any, key: str) -> Any:
        """Host time of every call, kept as a list -- one entry per step."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._host_lists.setdefault(key, []).append(
                    (time.perf_counter() - started) * 1000.0)

        return wrapper

    def _see_cache(self, module: Any, args: tuple, kwargs: dict) -> None:
        cache = kwargs.get("past_key_values")
        if cache is None or id(cache) in self._seen_caches:
            return
        self._seen_caches.add(id(cache))
        if callable(getattr(cache, "update", None)):
            self._set_attr(cache, "update", self._wrap_kv_update(cache.update))
        if callable(getattr(cache, "crop", None)):
            self._set_attr(cache, "crop", self._wrap_host_list(cache.crop, "crop"))

    def _wrap_kv_update(self, original: Any) -> Any:
        """Time each layer's cache update and count what it wrote.

        DynamicLayer.update concatenates the whole cache with the new keys and
        values on every call: the returned tensors are the new cache, and their
        size is what was written (about as much again was read).
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._mark("kv_cat", "start")
            out = original(*args, **kwargs)
            self._mark("kv_cat", "end")
            try:
                phase = self._phase_now if self._phase_now in ("prefill", "decode",
                                                               "expert") else "other"
                self._kv_bytes[phase] = (self._kv_bytes.get(phase, 0)
                                         + int(out[0].nbytes + out[1].nbytes))
                self._kv_calls += 1
            except Exception:
                pass
            return out

        return wrapper

    def _start_sync_audit(self) -> None:
        """Count every host-device synchronization, by phase and by code site.

        torch's sync debug mode warns on each one; the warnings are caught here
        rather than shown. A synchronization in the decode loop stalls the host
        until the device drains, which is what makes a step launch-bound.
        """
        try:
            self._sync_prev_mode = torch.cuda.get_sync_debug_mode()
            self._warn_ctx = warnings.catch_warnings()
            self._warn_ctx.__enter__()
            warnings.filterwarnings("always", message=f".*{_SYNC_MESSAGE}.*")
            warnings.filterwarnings("ignore", message=".*Synchronization debug mode.*")
            shown = warnings.showwarning

            def show(message: Any, category: Any, filename: str, lineno: int,
                     file: Any = None, line: Any = None) -> None:
                if _SYNC_MESSAGE in str(message):
                    phase = self._phase_now
                    self._sync_counts[phase] = self._sync_counts.get(phase, 0) + 1
                    site = "/".join(str(filename).split("/")[-2:]) + f":{lineno}"
                    self._sync_sites[site] = self._sync_sites.get(site, 0) + 1
                    return
                shown(message, category, filename, lineno, file, line)

            warnings.showwarning = show
            torch.cuda.set_sync_debug_mode("warn")
        except Exception:
            self._stop_sync_audit()

    def _stop_sync_audit(self) -> None:
        try:
            if self._sync_prev_mode is not None:
                torch.cuda.set_sync_debug_mode(self._sync_prev_mode)
        finally:
            self._sync_prev_mode = None
            if self._warn_ctx is not None:
                self._warn_ctx.__exit__(None, None, None)
                self._warn_ctx = None

    # -- boundaries and shapes ---------------------------------------------
    def _mem_mark(self, label: str) -> None:
        """Read the allocator at a segment boundary, then reset its peak.

        The peak read here is the peak since the previous boundary, which is
        what makes it a per-segment number. Host-side bookkeeping only: no
        sync, and about a tenth of a millisecond.
        """
        if not torch.cuda.is_available():
            return
        try:
            stats = torch.cuda.memory_stats()
            peak = int(stats.get("allocated_bytes.all.peak", 0))
            _RUN_PEAK["bytes"] = max(_RUN_PEAK["bytes"], peak)
            self._memory.append({
                "label": label,
                "current": int(stats.get("allocated_bytes.all.current", 0)),
                "peak": peak,
                "reserved_peak": int(stats.get("reserved_bytes.all.peak", 0)),
                "ooms": int(stats.get("num_ooms", 0)),
            })
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _on_vision_start(self, module: Any, args: tuple, kwargs: dict) -> None:
        self._mem_mark("vision_start")
        self._mark("vision", "start")
        if "n_images" in self._shapes:
            return
        # Qwen3-VL calls the tower as visual(pixel_values, grid_thw=...): one row
        # of patch features per patch, one grid row per image. generate has
        # already copied the images K times, so these count K x the frames.
        pixels = args[0] if args else kwargs.get("hidden_states")
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        try:
            if pixels is not None:
                self._shapes["n_vision_patches"] = int(pixels.shape[0])
                merge = getattr(module, "spatial_merge_unit", None)
                if merge:
                    self._shapes["n_vision_tokens"] = int(pixels.shape[0]) // int(merge)
            if grid is not None:
                self._shapes["n_images"] = int(grid.shape[0])
        except Exception:
            pass

    def _on_vision_end(self, *_: Any) -> None:
        self._mark("vision", "end")
        self._mem_mark("vision_end")

    def _on_lm_end(self, *_: Any) -> None:
        self._mark("lm", "end")
        self._lm_calls += 1
        if self._lm_calls == 1:
            self._mem_mark("prefill_end")

    def _on_expert_start(self, module: Any, args: tuple, kwargs: dict) -> None:
        if "kv_bytes" not in self._shapes:
            # The prompt cache the head attends to at every Euler step: the
            # generated prompt's keys and values, K rows, every layer. Metadata
            # only -- nbytes and shapes -- so reading it costs no sync.
            try:
                cache = kwargs.get("past_key_values")
                layers = [layer for layer in getattr(cache, "layers", [])
                          if getattr(layer, "keys", None) is not None]
                if layers:
                    self._shapes["kv_bytes"] = int(sum(layer.keys.nbytes + layer.values.nbytes
                                                       for layer in layers))
                    self._shapes["kv_layers"] = len(layers)
                    self._shapes["kv_rows"] = int(layers[0].keys.shape[0])
                    self._shapes["kv_final_tokens"] = int(layers[0].keys.shape[-2])
                embeds = kwargs.get("inputs_embeds")
                if embeds is not None and embeds.dim() == 3:
                    self._shapes["expert_tokens"] = int(embeds.shape[1])
            except Exception:
                pass
        self._mark("expert", "start")

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

    def _wrap_host(self, original: Any, key: str, bucket: str | None = None) -> Any:
        """Time a host-side function; optionally mark it on the device as well."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if bucket is not None:
                self._mark(bucket, "start")
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._host_ms[key] = (self._host_ms.get(key, 0.0)
                                      + (time.perf_counter() - started) * 1000.0)
                if bucket is not None:
                    self._mark(bucket, "end")

        return wrapper

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
        self._stop_sync_audit()
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
            self._mem_mark("generate_end")
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            # Marked because it runs inside the postgen window and would
            # otherwise be counted as the model's own time.
            self._mark("consume", "start")
            try:
                self._consume(out, int(input_ids.shape[1]))
            except Exception as exc:  # instrumentation must not break the run
                print(f"[token_trace] could not read logits: {exc}")
            finally:
                self._mark("consume", "end")
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
            self._mem_mark("diffusion_start")
            self._mark("diffusion", "start")
            out = original(*args, **kwargs)
            self._mark("diffusion", "end")
            self._mem_mark("diffusion_end")
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
        self._mem_mark("end")
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
            records = [(bucket, kind, -float(event.elapsed_time(anchor)), host, cpu)
                       for bucket, kind, event, host, cpu in self._marks]
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
            hook_ms=self._hook_s * 1000.0,
            host_ms=self._host_ms,
            process=_process_delta(self._process0, _process_stats()),
            memory=self._memory,
            shapes=self._shapes,
            step={"kv_bytes": self._kv_bytes, "kv_calls": self._kv_calls,
                  "host_lists": self._host_lists, "sync_counts": self._sync_counts,
                  "sync_sites": self._sync_sites} if self.level == "step" else None,
        )
        _POOL.release(self)
        return self.timing


@contextlib.contextmanager
def trace_inference(model: Any, special_token_ids: dict[str, int] | None = None,
                    level: str = "basic"):
    """Convenience wrapper: yields a tracer and resolves timings on exit."""
    tracer = InferenceTracer(model, special_token_ids, level=level)
    with tracer:
        yield tracer
    tracer.finalize()
