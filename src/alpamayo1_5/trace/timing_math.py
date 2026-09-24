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

"""Turn the tracer's marks into the numbers a timing row carries.

The tracer does as little as possible while the model runs: it records a CUDA
event and a host timestamp at each mark and nothing else. After one final
synchronize it converts every event to a device time on a common axis, and
hands the result here as plain floats.

That split is deliberate. Everything below is arithmetic -- pairing starts
with ends, attributing spans, subtracting remainders -- and it is where the
bugs in a timing split live. As plain Python it runs in CI, where torch is not
installed, against synthetic marks whose right answer is known.

A record is ``(bucket, kind, device_ms, host_s[, cpu_s])``:

* ``bucket`` names what was marked (``vision``, ``lm``, ``expert``,
  ``diffusion``, ``generate``);
* ``kind`` is ``start`` or ``end``;
* ``device_ms`` is the event's device time relative to a common anchor, so the
  difference of two is the device time between them;
* ``host_s`` is ``perf_counter`` when the mark was recorded -- which is when
  the work was *enqueued*, not when it ran;
* ``cpu_s``, optional, is the main thread's CPU time at the mark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

Record = tuple  # (bucket, kind, device_ms, host_s) or with cpu_s appended
Span = tuple[float, float, float, float]  # device start, device end, host start, host end


@dataclass
class TimingResult:
    """Everything one traced pass measured. ``None`` means "not measured"."""

    # The legacy split. Same definitions as every earlier run.
    vision_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    postgen_ms: float | None = None
    expert_ms: float | None = None
    total_ms: float | None = None
    other_ms: float | None = None
    n_vision_calls: int = 0
    n_decode_steps: int = 0
    n_expert_calls: int = 0
    measured: bool = False

    wall_ms: float | None = None
    compute_span_ms: float | None = None
    start_unix_s: float | None = None
    start_host_s: float | None = None

    vision_call_ms: list[float] = field(default_factory=list)
    decode_step_ms: list[float] = field(default_factory=list)
    decode_step_host_ms: list[float] = field(default_factory=list)
    expert_step_ms: list[float] = field(default_factory=list)
    expert_step_host_ms: list[float] = field(default_factory=list)

    n_cuda_allocs: int | None = None
    n_alloc_retries: int | None = None

    graph_captures: int | None = None
    graph_replays: int | None = None
    graph_fallbacks: int | None = None
    graph_n_graphs: int | None = None
    graph_capture_ms: float | None = None
    graph_mode: str | None = None

    trace_consume_ms: float | None = None
    trace_consume_host_ms: float | None = None
    postgen_model_ms: float | None = None
    trace_n_marks: int | None = None
    trace_hook_host_ms: float | None = None

    pre_generate_ms: float | None = None
    pre_generate_host_ms: float | None = None
    tail_ms: float | None = None
    action_to_traj_ms: float | None = None
    first_traj_ms: float | None = None
    fuse_traj_host_ms: float | None = None
    expand_inputs_host_ms: float | None = None
    rope_index_host_ms: float | None = None
    cpu_ms: dict[str, float] = field(default_factory=dict)
    proc_cpu_ms: float | None = None
    ctx_vol: int | None = None
    ctx_invol: int | None = None
    rss_bytes: int | None = None

    memory: dict[str, Any] = field(default_factory=dict)
    shapes: dict[str, Any] = field(default_factory=dict)
    #: Host-clock intervals of the pass and its segments, for attributing the
    #: board sampler's readings. Not a column: the energy computed from them is.
    windows: dict[str, tuple[float, float]] = field(default_factory=dict)
    anchor_lag_ms: float | None = None
    #: Trace level step only: columns, and the sync sites (not a column).
    step: dict[str, Any] = field(default_factory=dict)
    sync_sites: dict[str, int] = field(default_factory=dict)

    gen_preamble_ms: float | None = None
    lm_head_ms: float | None = None
    vlm_glue_ms: float | None = None
    gen_loop_ms: float | None = None
    ttft_ms: float | None = None
    n_vlm_forwards: int | None = None
    lm_head_step_ms: list[float] = field(default_factory=list)
    decode_gap_ms: list[float] = field(default_factory=list)
    span_violations: int | None = None

    def legacy(self) -> dict[str, Any]:
        """The keys ``predictions.parquet`` carries, with their schema-3 meaning."""
        return {
            "t_vision_ms": self.vision_ms,
            "t_prefill_ms": self.prefill_ms,
            "t_decode_ms": self.decode_ms,
            "t_postgen_ms": self.postgen_ms,
            "t_expert_ms": self.expert_ms,
            "t_other_ms": self.other_ms,
            "t_total_ms": self.total_ms,
            "n_vision_calls": self.n_vision_calls,
            "n_decode_steps": self.n_decode_steps,
            "n_expert_calls": self.n_expert_calls,
            "timing_measured": self.measured,
        }

    def row(self) -> dict[str, Any]:
        """Everything for the timing row. Lists are copied: a reader that trims
        its copy must not trim the record."""
        out = self.legacy()
        out.update({
            "t_wall_ms": self.wall_ms,
            "t_compute_span_ms": self.compute_span_ms,
            "t_start_unix_s": self.start_unix_s,
            "t_start_host_s": self.start_host_s,
            "vision_call_ms": list(self.vision_call_ms),
            "decode_step_ms": list(self.decode_step_ms),
            "decode_step_host_ms": list(self.decode_step_host_ms),
            "expert_step_ms": list(self.expert_step_ms),
            "expert_step_host_ms": list(self.expert_step_host_ms),
            "n_cuda_allocs": self.n_cuda_allocs,
            "n_alloc_retries": self.n_alloc_retries,
            "graph_captures": self.graph_captures,
            "graph_replays": self.graph_replays,
            "graph_fallbacks": self.graph_fallbacks,
            "graph_n_graphs": self.graph_n_graphs,
            "graph_capture_ms": self.graph_capture_ms,
            "graph_mode": self.graph_mode,
            "t_trace_consume_ms": self.trace_consume_ms,
            "t_trace_consume_host_ms": self.trace_consume_host_ms,
            "t_postgen_model_ms": self.postgen_model_ms,
            "trace_n_marks": self.trace_n_marks,
            "trace_hook_host_ms": self.trace_hook_host_ms,
            "t_gen_preamble_ms": self.gen_preamble_ms,
            "t_lm_head_ms": self.lm_head_ms,
            "t_vlm_glue_ms": self.vlm_glue_ms,
            "t_gen_loop_ms": self.gen_loop_ms,
            "t_ttft_ms": self.ttft_ms,
            "n_vlm_forwards": self.n_vlm_forwards,
            "lm_head_step_ms": list(self.lm_head_step_ms),
            "decode_gap_ms": list(self.decode_gap_ms),
            "trace_span_violations": self.span_violations,
            "t_pre_generate_ms": self.pre_generate_ms,
            "t_pre_generate_host_ms": self.pre_generate_host_ms,
            "t_tail_ms": self.tail_ms,
            "t_action_to_traj_ms": self.action_to_traj_ms,
            "t_first_traj_ms": self.first_traj_ms,
            "t_fuse_traj_host_ms": self.fuse_traj_host_ms,
            "t_expand_inputs_host_ms": self.expand_inputs_host_ms,
            "t_rope_index_host_ms": self.rope_index_host_ms,
            **{f"cpu_{name}_ms": self.cpu_ms.get(name) for name in CPU_SEGMENTS},
            "proc_cpu_ms": self.proc_cpu_ms,
            "ctx_vol": self.ctx_vol,
            "ctx_invol": self.ctx_invol,
            "rss_bytes": self.rss_bytes,
            **self.memory,
            **self.shapes,
            "anchor_lag_ms": self.anchor_lag_ms,
            **self.step,
        })
        return out


#: Which segment a memory boundary closes: the peak read at a boundary is the
#: peak since the previous one, because each read is followed by a reset.
_MEMORY_SEGMENTS = {
    "vision_start": "pre",
    "vision_end": "vision",
    "prefill_end": "prefill",
    "generate_end": "decode",
    "diffusion_start": "postgen",
    "diffusion_end": "expert",
}


def memory_segments(boundaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-segment allocator peaks from the boundaries of one pass, as timing columns.

    ``boundaries`` are ``{label, current, peak, reserved_peak, ooms}`` in call
    order, starting with ``start`` and ending with ``end``. The ``start``
    boundary's peak is what happened *before* the pass -- between clips -- and
    is left out of the pass's own peak. ``end`` closes the tail only when the
    head's end was seen; at trace level off there are no inner boundaries and
    the pass gets its overall peak alone.
    """
    if not boundaries:
        return {}
    out: dict[str, Any] = {}
    first, last = boundaries[0], boundaries[-1]
    if first.get("label") == "start":
        out["mem_start_bytes"] = first.get("current")
    if last.get("label") == "end":
        out["mem_end_bytes"] = last.get("current")
    inner = [b for b in boundaries if b.get("label") != "start"]
    labels = [b.get("label") for b in boundaries]
    for i, b in enumerate(boundaries):
        segment = _MEMORY_SEGMENTS.get(b.get("label"))
        if b.get("label") == "end" and i > 0 and labels[i - 1] == "diffusion_end":
            segment = "tail"
        if segment is not None and b.get("peak") is not None:
            out[f"mem_peak_{segment}_bytes"] = int(b["peak"])
    peaks = [int(b["peak"]) for b in inner if b.get("peak") is not None]
    if peaks:
        out["mem_peak_clip_bytes"] = max(peaks)
    reserved = [int(b["reserved_peak"]) for b in inner if b.get("reserved_peak") is not None]
    if reserved:
        out["mem_reserved_peak_bytes"] = max(reserved)
    if first.get("ooms") is not None and last.get("ooms") is not None:
        out["n_ooms"] = max(int(last["ooms"]) - int(first["ooms"]), 0)
    return out


#: The segments main-thread CPU time is split into.
CPU_SEGMENTS = ("pass", "pre_generate", "vision", "prefill", "decode", "postgen", "expert",
                "tail")


def span_pairs(records: Iterable[Record], bucket: str) -> list[Span]:
    """Pair each ``start`` with the next ``end`` of the same bucket, in order.

    An ``end`` with no open ``start`` is ignored, and a ``start`` that never
    closes is dropped: either means a mark was lost, and a half span has no
    duration worth reporting.
    """
    out: list[Span] = []
    pending: tuple[float, float] | None = None
    for b, kind, device_ms, host_s, *_ in records:
        if b != bucket:
            continue
        if kind == "start":
            pending = (device_ms, host_s)
        elif kind == "end" and pending is not None:
            out.append((pending[0], device_ms, pending[1], host_s))
            pending = None
    return out


def _device(spans: list[Span]) -> list[float]:
    return [float(end - start) for start, end, _, _ in spans]


def _host(spans: list[Span]) -> list[float]:
    return [float((end - start) * 1000.0) for _, _, start, end in spans]


def graph_mode(captures: int | None, replays: int | None, fallbacks: int | None) -> str | None:
    """Which path a pass's expert steps took. ``None`` when no runner was installed.

    A capture step is also counted as a replay by the runner, so ``capture``
    means "captured at least once, never fell back".
    """
    if captures is None or replays is None or fallbacks is None:
        return None
    if fallbacks and (replays or captures):
        return "mixed"
    if fallbacks:
        return "fallback"
    if captures:
        return "capture"
    if replays:
        return "replay"
    return "none"


def _delta(after: Mapping[str, int] | None, before: Mapping[str, int] | None,
           key: str) -> int | None:
    if after is None or before is None:
        return None
    return max(int(after.get(key, 0)) - int(before.get(key, 0)), 0)


def _cpu_pairs(records: Iterable[Record], bucket: str) -> list[tuple[float, float]]:
    """Main-thread CPU time at each start and end of ``bucket``, paired in order."""
    out: list[tuple[float, float]] = []
    pending: float | None = None
    for r in records:
        if r[0] != bucket or len(r) < 5 or r[4] is None:
            continue
        if r[1] == "start":
            pending = r[4]
        elif pending is not None:
            out.append((pending, r[4]))
            pending = None
    return out


def _split_host(result: TimingResult, records: list[Record], generate: list[Span],
                diffusion: list[Span]) -> None:
    """The call outside generate and the head, and main-thread CPU per segment.

    The call is bracketed by its own marks, so the spans before generate
    (deep copy, trajectory tokenization) and after the head (action to
    trajectory, text extraction) are measured rather than left inside wall.
    """
    call = span_pairs(records, "call")
    a2t = span_pairs(records, "a2t")
    if call and generate:
        result.pre_generate_ms = float(max(generate[0][0] - call[0][0], 0.0))
        result.pre_generate_host_ms = float(max((generate[0][2] - call[0][2]) * 1000.0, 0.0))
    if call and diffusion:
        result.tail_ms = float(max(call[-1][1] - diffusion[-1][1], 0.0))
    if a2t:
        result.action_to_traj_ms = float(sum(_device(a2t)))
        if call:
            result.first_traj_ms = float(a2t[-1][1] - call[0][0])

    cpu = {name: _cpu_pairs(records, name)
           for name in ("call", "generate", "vision", "lm", "diffusion")}
    ms: dict[str, float] = {}

    def total(pairs: list[tuple[float, float]]) -> float:
        return float(sum(end - start for start, end in pairs) * 1000.0)

    if cpu["call"]:
        ms["pass"] = total(cpu["call"])
    if cpu["vision"]:
        ms["vision"] = total(cpu["vision"])
    if cpu["lm"]:
        ms["prefill"] = total(cpu["lm"][:1])
        ms["decode"] = total(cpu["lm"][1:])
    if cpu["diffusion"]:
        ms["expert"] = total(cpu["diffusion"])
    # The gaps between spans: call start to generate start, generate end to
    # the head's start, the head's end to the call's end.
    if cpu["call"] and cpu["generate"]:
        ms["pre_generate"] = max((cpu["generate"][0][0] - cpu["call"][0][0]) * 1000.0, 0.0)
    if cpu["generate"] and cpu["diffusion"]:
        ms["postgen"] = max((cpu["diffusion"][0][0] - cpu["generate"][0][1]) * 1000.0, 0.0)
    if cpu["diffusion"] and cpu["call"]:
        ms["tail"] = max((cpu["call"][-1][1] - cpu["diffusion"][-1][1]) * 1000.0, 0.0)
    result.cpu_ms = ms


def _host_windows(result: TimingResult, records: list[Record], lm: list[Span],
                  vision: list[Span], diffusion: list[Span], wall_start_s: float | None,
                  wall_end_s: float | None) -> None:
    """Map the device spans onto the host clock, where the board sampler lives.

    Device times are relative to the end anchor, which completed just before
    the host read ``wall_end_s``; so a device time ``d`` (negative ms)
    happened at ``wall_end_s + d / 1000`` on the host clock. The error is the
    few microseconds between the anchor completing and the host reading.

    ``anchor_lag_ms`` is how late the call's first event ran on the device
    after the host enqueued it. Near zero means the GPU was idle when the pass
    began; large means earlier work was still queued, and the start of the
    pass's energy window is less certain.
    """
    if wall_end_s is None or wall_start_s is None:
        return

    def host(device_ms: float) -> float:
        return wall_end_s + device_ms / 1000.0

    windows = {"call": (wall_start_s, wall_end_s)}
    if vision:
        windows["vision"] = (host(vision[0][0]), host(vision[-1][1]))
    if lm:
        windows["prefill"] = (host(lm[0][0]), host(lm[0][1]))
    if len(lm) > 1:
        windows["decode"] = (host(lm[1][0]), host(lm[-1][1]))
    if diffusion:
        windows["expert"] = (host(diffusion[0][0]), host(diffusion[-1][1]))
    result.windows = windows
    call = span_pairs(records, "call")
    if call:
        result.anchor_lag_ms = float((host(call[0][0]) - call[0][2]) * 1000.0)


#: Logits processor class -> column stem.
_PROCESSORS = {"ExpertLogitsProcessor": "mask", "TemperatureLogitsWarper": "temperature",
               "TopPLogitsWarper": "top_p"}
SYNC_PHASES = ("pre_generate", "vision", "prefill", "decode", "gen_loop", "postgen", "consume",
               "expert", "tail")


def _within(spans: list[Span], outer: Span) -> float:
    """Device time of the spans that start inside ``outer``."""
    return float(sum(end - start for start, end, _, _ in spans if outer[0] <= start <= outer[1]))


def _split_step(result: TimingResult, records: list[Record], lm: list[Span],
                expert: list[Span], diffusion: list[Span], step: Mapping[str, Any]) -> None:
    """Trace level step: what each Euler step and each decode step is made of.

    KV concatenations are attributed to the forward they ran inside by device
    time: the prefill's, each decode step's, each Euler step's.
    """
    out: dict[str, Any] = {}
    kv = span_pairs(records, "kv_cat")
    if lm:
        out["kv_cat_ms_prefill"] = _within(kv, lm[0])
        decode = [_within(kv, s) for s in lm[1:]]
        out["decode_kv_cat_ms"] = decode
        out["kv_cat_ms_decode"] = float(sum(decode))
    per_euler = [_within(kv, s) for s in expert]
    out["expert_kv_cat_ms"] = per_euler
    out["kv_cat_ms_expert"] = float(sum(per_euler))
    for phase in ("prefill", "decode", "expert"):
        out[f"kv_cat_bytes_{phase}"] = int(step.get("kv_bytes", {}).get(phase, 0))
    out["n_kv_cat_calls"] = int(step.get("kv_calls", 0))

    in_proj, out_proj = span_pairs(records, "in_proj"), span_pairs(records, "out_proj")
    out["expert_in_proj_ms"] = _device(in_proj)
    out["expert_out_proj_ms"] = _device(out_proj)
    rest = []
    for i, done in enumerate(out_proj):
        following = in_proj[i + 1][0] if i + 1 < len(in_proj) else (
            diffusion[-1][1] if diffusion else None)
        if following is not None:
            rest.append(float(max(following - done[1], 0.0)))
    out["expert_euler_rest_ms"] = rest

    out["decode_logits_proc_ms"] = _device(span_pairs(records, "lp"))
    out["decode_stop_ms"] = _device(span_pairs(records, "stop"))
    totals = {stem: 0.0 for stem in (*_PROCESSORS.values(), "other")}
    for bucket in sorted({r[0] for r in records if str(r[0]).startswith("lp:")}):
        stem = _PROCESSORS.get(bucket[3:], "other")
        totals[stem] += float(sum(_device(span_pairs(records, bucket))))
    for stem, value in totals.items():
        out[f"lp_{stem}_ms"] = value

    lists = step.get("host_lists", {})
    out["decode_prep_inputs_host_ms"] = list(lists.get("prep_inputs", []))
    out["decode_update_kwargs_host_ms"] = list(lists.get("update_kwargs", []))
    out["expert_crop_host_ms"] = list(lists.get("crop", []))

    find_eos = float(sum(_device(span_pairs(records, "find_eos"))))
    build_mask = float(sum(_device(span_pairs(records, "build_mask"))))
    out["t_find_eos_ms"], out["t_build_mask_ms"] = find_eos, build_mask
    if result.postgen_ms is not None:
        out["t_postgen_rest_ms"] = float(max(
            result.postgen_ms - (result.trace_consume_ms or 0.0) - find_eos - build_mask, 0.0))

    parts = {name: float(sum(_device(span_pairs(records, bucket))))
             for name, bucket in (("patch_embed", "v_patch"), ("blocks", "v_blocks"),
                                  ("deepstack", "v_deepstack"), ("merger", "v_merger"))}
    for name, value in parts.items():
        out[f"t_vision_{name}_ms"] = value
    if result.vision_ms is not None:
        out["t_vision_rest_ms"] = float(max(result.vision_ms - sum(parts.values()), 0.0))

    counts = step.get("sync_counts", {})
    out["n_syncs_total"] = int(sum(counts.values()))
    for phase in SYNC_PHASES:
        out[f"n_syncs_{phase}"] = int(counts.get(phase, 0))
    result.step = out
    result.sync_sites = dict(step.get("sync_sites", {}))


def _split_generate(result: TimingResult, records: list[Record], generate: list[Span],
                    wall_start_s: float | None) -> None:
    """Break the generate span's remainder into the four things it was made of.

    Every VLM forward contains the vision tower (prefill only), the language
    model and lm_head, plus glue between them. Around the forwards, generate
    spends time before the first (preamble) and between and after them (the
    loop). With the legacy definitions this is exact:

        t_other = gen - vision - lm = preamble + lm_head + glue + loop

    A negative part means a nesting assumption broke (a forward outside the
    generate span, a hook that fired twice); it is floored and counted rather
    than allowed to push another part up by the same amount unnoticed.
    """
    vlm = span_pairs(records, "vlm")
    head = span_pairs(records, "lm_head")
    if not generate or not vlm:
        return
    gen_ms = float(sum(_device(generate)))
    sum_vlm = float(sum(_device(vlm)))
    lm_head = float(sum(_device(head)))
    lm_total = (result.prefill_ms or 0.0) + (result.decode_ms or 0.0)
    parts = {
        "preamble": vlm[0][0] - generate[0][0],
        "glue": sum_vlm - (result.vision_ms or 0.0) - lm_total - lm_head,
        "loop": gen_ms - (vlm[0][0] - generate[0][0]) - sum_vlm,
    }
    result.span_violations = sum(1 for v in parts.values() if v < 0.0)
    result.gen_preamble_ms = float(max(parts["preamble"], 0.0))
    result.vlm_glue_ms = float(max(parts["glue"], 0.0))
    result.gen_loop_ms = float(max(parts["loop"], 0.0))
    result.lm_head_ms = lm_head
    result.lm_head_step_ms = _device(head)
    result.decode_gap_ms = [float(vlm[i + 1][0] - vlm[i][1]) for i in range(len(vlm) - 1)]
    result.n_vlm_forwards = len(vlm)
    if wall_start_s is not None:
        # The loop syncs on each sampled token before it can enqueue the next
        # forward, so the host stamp of forward #2 is when token #1 existed.
        ready = vlm[1][2] if len(vlm) > 1 else generate[0][3]
        result.ttft_ms = float((ready - wall_start_s) * 1000.0)


def resolve(
    records: list[Record],
    *,
    wall_start_s: float | None = None,
    wall_end_s: float | None = None,
    start_unix_s: float | None = None,
    alloc_before: tuple[int, int] | None = None,
    alloc_after: tuple[int, int] | None = None,
    graph_before: Mapping[str, int] | None = None,
    graph_after: Mapping[str, int] | None = None,
    capture_ms: float | None = None,
    hook_ms: float | None = None,
    host_ms: Mapping[str, float] | None = None,
    process: Mapping[str, Any] | None = None,
    memory: list[Mapping[str, Any]] | None = None,
    shapes: Mapping[str, Any] | None = None,
    step: Mapping[str, Any] | None = None,
) -> TimingResult:
    """Attribute the marks of one pass.

    The legacy definitions are kept exactly, so ``predictions.parquet`` goes on
    meaning what it meant:

    * the first language-model call is prefill, every later one is decode;
    * expert time is the diffusion span when it was marked, else the sum of
      the Euler steps;
    * postgen is the device gap from generate's end to the diffusion start;
    * total is the generate span plus postgen plus expert, and ``other`` is
      what the named spans leave of it, floored at zero -- a negative
      remainder means overlapping marks, not negative time.
    """
    result = TimingResult(start_unix_s=start_unix_s, start_host_s=wall_start_s)

    runner = graph_after is not None and graph_before is not None
    if runner:
        result.graph_captures = _delta(graph_after, graph_before, "captures")
        result.graph_replays = _delta(graph_after, graph_before, "replays")
        result.graph_fallbacks = _delta(graph_after, graph_before, "eager_fallbacks")
        result.graph_n_graphs = int(graph_after.get("graphs", 0))
        result.graph_capture_ms = float(capture_ms or 0.0)
    result.graph_mode = graph_mode(
        result.graph_captures, result.graph_replays, result.graph_fallbacks)

    if alloc_before is not None and alloc_after is not None:
        result.n_cuda_allocs = max(alloc_after[0] - alloc_before[0], 0)
        result.n_alloc_retries = max(alloc_after[1] - alloc_before[1], 0)

    if memory:
        result.memory = memory_segments(memory)
    if process and process.get("host_mem_avail_min_bytes") is not None:
        result.memory["host_mem_avail_min_bytes"] = process["host_mem_avail_min_bytes"]
    if shapes:
        result.shapes = dict(shapes)
    if process:
        result.proc_cpu_ms = process.get("proc_cpu_ms")
        result.ctx_vol = process.get("ctx_vol")
        result.ctx_invol = process.get("ctx_invol")
        result.rss_bytes = process.get("rss_bytes")
    if host_ms:
        result.fuse_traj_host_ms = host_ms.get("fuse_traj")
        result.expand_inputs_host_ms = host_ms.get("expand_inputs")
        result.rope_index_host_ms = host_ms.get("rope_index")

    if not records:
        return result
    result.trace_n_marks = len(records)
    result.trace_hook_host_ms = hook_ms

    vision = span_pairs(records, "vision")
    lm = span_pairs(records, "lm")
    expert = span_pairs(records, "expert")
    diffusion = span_pairs(records, "diffusion")
    generate = span_pairs(records, "generate")

    # A span that was never marked is absent, not zero. At trace level off
    # nothing inside the call is marked, and a row of 0.0 ms spans would read
    # as a pass that took no time in any of them.
    result.vision_call_ms = _device(vision)
    result.n_vision_calls = len(vision)
    result.vision_ms = float(sum(result.vision_call_ms)) if vision else None
    if lm:
        result.prefill_ms = float(sum(_device(lm[:1])))
        result.decode_step_ms = _device(lm[1:])
        result.decode_step_host_ms = _host(lm[1:])
        result.decode_ms = float(sum(result.decode_step_ms))
    result.n_decode_steps = max(len(lm) - 1, 0)

    result.expert_step_ms = _device(expert)
    result.expert_step_host_ms = _host(expert)
    result.n_expert_calls = len(expert)
    # The diffusion span also covers the sampler's arithmetic between steps,
    # which the per-step sum does not. Prefer it for the aggregate; keep the
    # steps for the shape.
    if diffusion:
        result.expert_ms = float(sum(_device(diffusion)))
    elif expert:
        result.expert_ms = float(sum(result.expert_step_ms))

    if generate and diffusion:
        result.postgen_ms = float(max(diffusion[0][0] - generate[0][1], 0.0))
    if generate:
        result.total_ms = (float(sum(_device(generate))) + (result.postgen_ms or 0.0)
                           + (result.expert_ms or 0.0))
        named = sum(v or 0.0 for v in (result.vision_ms, result.prefill_ms, result.decode_ms,
                                       result.postgen_ms, result.expert_ms))
        result.other_ms = float(max(result.total_ms - named, 0.0))

    # The tracer's own logits pass sits inside the postgen window: the generate
    # wrapper marks generate's end, then runs it, then returns to the model. It
    # stays in t_postgen_ms so that column keeps its schema-3 meaning, and is
    # taken out here to show what the model's code alone cost.
    consume = span_pairs(records, "consume")
    if consume:
        result.trace_consume_ms = float(sum(_device(consume)))
        result.trace_consume_host_ms = float(sum(_host(consume)))
        if generate and diffusion:
            result.postgen_model_ms = float(max(result.postgen_ms - result.trace_consume_ms,
                                                0.0))

    _split_generate(result, records, generate, wall_start_s)
    _split_host(result, records, generate, diffusion)
    _host_windows(result, records, lm, vision, diffusion, wall_start_s, wall_end_s)
    if step is not None:
        _split_step(result, records, lm, expert, diffusion, step)

    spans = [v for v in (result.vision_ms, result.prefill_ms, result.decode_ms,
                         result.expert_ms) if v is not None]
    result.compute_span_ms = float(sum(spans)) if spans else None
    if wall_start_s is not None and wall_end_s is not None:
        result.wall_ms = float((wall_end_s - wall_start_s) * 1000.0)
    result.measured = True
    return result
