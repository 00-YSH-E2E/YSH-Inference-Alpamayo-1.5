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

"""Kernel-level numbers from a profile pass's trace, without torch.

A profile pass runs the inference once more under torch.profiler (CPU and
CUDA) and exports its Chrome trace. Read here, the trace answers what the
tracer's CUDA events cannot:

* when the GPU actually sat idle -- the gaps between kernels, not the span
  from a segment's first mark to its last -- and what the host was doing
  while it did;
* how many kernels each decode and Euler step launched, and of what kind;
* how long the host spent in launch calls, and blocked in synchronizations;
* which attention backend each phase ran, from the op SDPA dispatched to.

The segments are the tracer's spans, entered as profiler ranges named
``trace::<bucket>``. A kernel belongs to the segment whose code launched it,
found through its launch call's correlation id; a stretch of GPU idle belongs
to the segment the host was in while it lasted. Kineto puts host and device
timestamps on one clock, so the two can be compared directly.

Pure python over the trace's event dicts, so it is tested without a GPU.
"""

from __future__ import annotations

import bisect
import functools
import gzip
import json
import re
import statistics
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

#: The host segments of a pass, in the order they happen. ``gen_other`` is
#: generate's work outside the language model, its head and the vision tower
#: (sampling, processors, the loop); ``trace`` the tracer's own logits pass;
#: ``head_other`` the sampler's work outside the Euler steps.
SEGMENTS = ("pre", "vision", "prefill", "decode", "lm_head", "gen_other", "postgen", "trace",
            "expert", "head_other", "tail")

#: The tracer buckets the segments are cut from.
_BUCKETS = ("vision", "lm", "lm_head", "generate", "consume", "expert", "diffusion")

#: What a device event did. ``memory`` is copy-engine work: memcpy and memset.
CATEGORIES = ("attention", "gemm", "conv", "cat", "copy", "norm", "softmax", "reduction",
              "elementwise", "memory", "other")

#: The phases whose attention backend is reported.
SDPA_PHASES = ("vision", "prefill", "decode", "expert")

RANGE_PREFIX = "trace::"
_DEVICE_KINDS = {"kernel": "kernel", "gpu_memcpy": "memcpy", "gpu_memset": "memset"}
_API_CATS = frozenset({"cuda_runtime", "cuda_driver"})
_SYNC_CALLS = frozenset({
    "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize", "cudaMemcpy",
    "cuCtxSynchronize", "cuStreamSynchronize", "cuEventSynchronize",
})
_GRAPH_LAUNCHES = frozenset({"cudaGraphLaunch", "cuGraphLaunch"})

#: The op scaled_dot_product_attention dispatched to, and the backend it names.
SDPA_BACKENDS = {
    "aten::_scaled_dot_product_flash_attention": "flash",
    "aten::_scaled_dot_product_efficient_attention": "efficient",
    "aten::_scaled_dot_product_cudnn_attention": "cudnn",
    "aten::_scaled_dot_product_attention_math": "math",
    "aten::_scaled_dot_product_fused_attention_overrideable": "fused",
    "aten::_scaled_dot_product_flash_attention_for_cpu": "cpu",
}
_SDPA_OPS = frozenset(SDPA_BACKENDS) | {"aten::scaled_dot_product_attention"}

# A kernel's category comes first from the op that launched it -- inside SDPA
# it is attention whatever the op (the math backend's bmm included) -- and
# only without an op from its own name. Order matters in both lists.
_OP_CATEGORIES = (
    ("gemm", re.compile(r"^aten::(mm|addmm|bmm|baddbmm|addbmm|matmul|linear|_addmm_activation"
                        r"|_scaled_mm|_int_mm)$")),
    ("conv", re.compile(r"^aten::(_?convolution|conv\w*|cudnn_convolution\w*)$")),
    ("cat", re.compile(r"^aten::(cat|_cat|stack)$")),
    ("copy", re.compile(r"^aten::(copy_|clone|contiguous|_to_copy|to|index_copy_?|index_select"
                        r"|gather|scatter_?|index|index_put_|_index_put_impl_|masked_scatter_?"
                        r"|masked_select|narrow_copy|repeat|embedding)$")),
    ("norm", re.compile(r"norm", re.I)),
    ("softmax", re.compile(r"softmax", re.I)),
    ("reduction", re.compile(r"^aten::(sum|mean|max|min|amax|amin|argmax|argmin|sort|topk|cumsum"
                             r"|cumprod|var|std|var_mean|any|all|nonzero|multinomial|logsumexp"
                             r"|prod|count_nonzero)$")),
)
_NAME_CATEGORIES = (
    ("attention", re.compile(r"flash|fmha|attention|sdpa", re.I)),
    ("gemm", re.compile(r"gemm|gemv|nvjet|xmma|cublas|cutlass", re.I)),
    ("cat", re.compile(r"CatArrayBatchedCopy")),
    ("copy", re.compile(r"copy", re.I)),
    ("norm", re.compile(r"norm", re.I)),
    ("softmax", re.compile(r"softmax", re.I)),
    ("reduction", re.compile(r"reduce|sort|topk|scan|argmax", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled", re.I)),
)


def load_events(path: Path | str) -> list[dict[str, Any]]:
    """The ``traceEvents`` of an exported Chrome trace, gzipped or not."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f).get("traceEvents", [])


@functools.lru_cache(maxsize=8192)
def category(kind: str, op: str | None, in_sdpa: bool, name: str) -> str:
    """What a device event did: memory, attention, the launching op's kind, or its name's."""
    if kind != "kernel":
        return "memory"
    if in_sdpa:
        return "attention"
    if op is not None and op.startswith("aten::"):
        for label, pattern in _OP_CATEGORIES:
            if pattern.search(op):
                return label
        return "elementwise"
    for label, pattern in _NAME_CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


class _Intervals:
    """Sorted, merged host intervals; answers "is t inside one" by bisection."""

    def __init__(self, spans: list[tuple[float, float]]) -> None:
        merged = _union(spans)
        self.starts = [s for s, _ in merged]
        self.ends = [e for _, e in merged]

    def __contains__(self, t: float) -> bool:
        i = bisect.bisect_right(self.starts, t) - 1
        return i >= 0 and t < self.ends[i]


class _Calls:
    """The n-th range of one bucket: which call of a stack a host instant is in."""

    def __init__(self, spans: list[tuple[float, float]]) -> None:
        self.starts = [s for s, _ in spans]
        self.ends = [e for _, e in spans]

    def index(self, t: float) -> int | None:
        i = bisect.bisect_right(self.starts, t) - 1
        return i if i >= 0 and t < self.ends[i] else None


def segment_of(is_open: Callable[[str], bool], lm_call: int | None,
               started: Callable[[str], bool], ended: Callable[[str], bool]) -> str:
    """Which segment the host is in, from the tracer spans open around it.

    ``lm_call`` is the index of the language-model call in progress, if one is;
    ``started`` and ``ended`` say whether a bucket's first span has begun and
    its last has finished. The innermost span wins; outside every span, the
    position relative to generate and the sampler decides.
    """
    if is_open("vision"):
        return "vision"
    if lm_call is not None:
        return "prefill" if lm_call == 0 else "decode"
    if is_open("lm_head"):
        return "lm_head"
    if is_open("generate"):
        return "gen_other"
    if is_open("consume"):
        return "trace"
    if is_open("expert"):
        return "expert"
    if is_open("diffusion"):
        return "head_other"
    if not started("generate") and not started("diffusion"):
        return "pre"
    if ended("generate") and not started("diffusion"):
        return "postgen"
    return "tail"


def _label(t: float, calls: Mapping[str, _Calls]) -> str:
    """The segment the host is in at ``t``, given the call's ranges."""
    return segment_of(
        lambda b: calls[b].index(t) is not None, calls["lm"].index(t),
        lambda b: bool(calls[b].starts) and t >= calls[b].starts[0],
        lambda b: bool(calls[b].ends) and t >= calls[b].ends[-1])


def _partition(ranges: Mapping[str, list[tuple[float, float]]], c0: float,
               c1: float) -> tuple[list[float], list[str], dict[str, _Calls]]:
    """Cut the call into elementary host intervals, each wholly in one segment."""
    calls = {b: _Calls(sorted(ranges.get(b, []))) for b in _BUCKETS}
    points = {c0, c1}
    for spans in ranges.values():
        for start, end in spans:
            points.update(p for p in (start, end) if c0 < p < c1)
    ordered = sorted(points)
    starts: list[float] = []
    labels: list[str] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        label = _label((a + b) / 2.0, calls)
        if labels and labels[-1] == label:
            continue
        starts.append(a)
        labels.append(label)
    return starts, labels, calls


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def analyze(events: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The profile pass's timing-row columns, and one row per device event.

    Returns ``({}, [])`` when the trace has no ``trace::call`` range -- a pass
    the tracer did not bracket cannot be cut into segments.
    """
    ranges: dict[str, list[tuple[float, float]]] = {}
    device: list[tuple[str, Mapping[str, Any]]] = []
    api: list[Mapping[str, Any]] = []
    ops: dict[Any, str] = {}
    sdpa_spans: list[tuple[float, float]] = []
    backends: list[tuple[float, str]] = []
    n_events = 0
    for e in events:
        n_events += 1
        if e.get("ph") != "X":
            continue
        cat, name = e.get("cat"), str(e.get("name", ""))
        if cat in _DEVICE_KINDS:
            device.append((_DEVICE_KINDS[cat], e))
        elif cat in _API_CATS:
            api.append(e)
        elif cat == "cpu_op":
            ops[(e.get("args") or {}).get("External id")] = name
            if name in _SDPA_OPS:
                ts = float(e["ts"])
                sdpa_spans.append((ts, ts + float(e.get("dur", 0.0))))
                if name in SDPA_BACKENDS:
                    backends.append((ts, SDPA_BACKENDS[name]))
        elif cat == "user_annotation" and name.startswith(RANGE_PREFIX):
            ts = float(e["ts"])
            ranges.setdefault(name[len(RANGE_PREFIX):], []).append(
                (ts, ts + float(e.get("dur", 0.0))))
    if not ranges.get("call"):
        return {}, []
    c0, c1 = min(ranges["call"])
    starts, labels, calls = _partition(ranges, c0, c1)

    def segment(t: float) -> str:
        if t >= c1:
            return "tail"
        return labels[max(bisect.bisect_right(starts, t) - 1, 0)]

    launches = {(a.get("args") or {}).get("correlation"): a for a in api}
    in_sdpa = _Intervals(sdpa_spans)
    stack_of = {"vision": "vision", "prefill": "lm", "decode": "lm", "expert": "expert"}

    kernels: list[dict[str, Any]] = []
    busy: list[tuple[float, float]] = []
    linked: set[Any] = set()
    # A copy to pageable host memory returns only once the device has caught
    # up to it: its "launch" is a synchronization, and is counted as one.
    blocking: set[Any] = set()
    leads: dict[str, list[float]] = {s: [] for s in SEGMENTS}
    n_unlinked = 0
    for kind, e in device:
        args = e.get("args") or {}
        ts, dur = float(e["ts"]), float(e.get("dur", 0.0))
        launch = launches.get(args.get("correlation"))
        at = float(launch["ts"]) if launch is not None else ts
        if not c0 <= at < c1:
            continue
        if launch is None:
            n_unlinked += 1
        else:
            linked.add(args.get("correlation"))
            name = str(e.get("name", ""))
            if kind == "memcpy" and "DtoH" in name and "Pageable" in name:
                blocking.add(args.get("correlation"))
        seg = segment(at)
        stack = stack_of.get(seg)
        step = calls[stack].index(at) if stack else None
        name = str(e.get("name", ""))
        op = ops.get(args.get("External id"))
        launch_us = float(launch.get("dur", 0.0)) if launch is not None else None
        lead_us = ts - (float(launch["ts"]) + launch_us) if launch is not None else None
        if kind == "kernel" and lead_us is not None:
            leads[seg].append(lead_us)
        grid, block = args.get("grid"), args.get("block")
        kernels.append({
            "kind": kind, "segment": seg, "step_index": step, "name": name, "op": op,
            "category": category(kind, op, at in in_sdpa, name),
            "stream": args.get("stream"), "start_us": ts - c0, "dur_us": dur,
            "launch_us": launch_us, "lead_us": lead_us,
            "grid_size": _product(grid), "block_size": _product(block),
            "regs_per_thread": args.get("registers per thread"),
            "smem_bytes": args.get("shared memory"),
        })
        busy.append((ts, ts + dur))

    row: dict[str, Any] = {"prof_n_events": n_events}
    only = [k for k in kernels if k["kind"] == "kernel"]
    row["prof_n_kernels"] = len(only)
    row["prof_n_memory_ops"] = len(kernels) - len(only)
    row["prof_n_unlinked"] = n_unlinked
    row["prof_kernel_ms"] = sum(k["dur_us"] for k in only) / 1000.0

    # Busy is the union of every device interval; idle is the rest of the
    # window, from the call's start to whichever ends later, the call or its
    # last kernel -- and each stretch of it is cut by the host segments.
    union = _union(busy)
    w1 = max([c1] + [end for _, end in union])
    row["prof_window_ms"] = (w1 - c0) / 1000.0
    row["prof_gpu_busy_ms"] = sum(min(end, w1) - max(start, c0) for start, end in union) / 1000.0
    idle = {s: 0.0 for s in SEGMENTS}
    cursor = c0
    for start, end in union + [(w1, w1)]:
        if start > cursor:
            _cut(cursor, min(start, w1), starts, labels, c1, idle)
        cursor = max(cursor, end)
    row["prof_gpu_idle_ms"] = sum(idle.values()) / 1000.0

    launch_ms = {s: 0.0 for s in SEGMENTS}
    sync_ms = {s: 0.0 for s in SEGMENTS}
    launch_durs: list[float] = []
    n_syncs = n_graph = 0
    for a in api:
        at = float(a["ts"])
        if not c0 <= at < c1:
            continue
        name = str(a.get("name", ""))
        dur = float(a.get("dur", 0.0))
        correlation = (a.get("args") or {}).get("correlation")
        if correlation in linked and correlation not in blocking:
            launch_durs.append(dur)
            launch_ms[segment(at)] += dur
        if name in _SYNC_CALLS or correlation in blocking:
            n_syncs += 1
            sync_ms[segment(at)] += dur
        if name in _GRAPH_LAUNCHES:
            n_graph += 1
    row["prof_n_launches"] = len(launch_durs)
    row["prof_launch_api_ms"] = sum(launch_ms.values()) / 1000.0
    row["prof_launch_us_p50"] = statistics.median(launch_durs) if launch_durs else None
    row["prof_n_syncs"] = n_syncs
    row["prof_sync_api_ms"] = sum(sync_ms.values()) / 1000.0
    row["prof_n_graph_launches"] = n_graph
    pooled = [lead for values in leads.values() for lead in values]
    row["prof_lead_ms_p50"] = statistics.median(pooled) / 1000.0 if pooled else None

    count = {s: 0 for s in SEGMENTS}
    kernel_us = {s: 0.0 for s in SEGMENTS}
    category_us = {c: 0.0 for c in CATEGORIES}
    for k in kernels:
        category_us[k["category"]] += k["dur_us"]
        if k["kind"] == "kernel":
            count[k["segment"]] += 1
            kernel_us[k["segment"]] += k["dur_us"]
    for s in SEGMENTS:
        row[f"prof_kernels_{s}"] = count[s]
        row[f"prof_kernel_ms_{s}"] = kernel_us[s] / 1000.0
        row[f"prof_idle_ms_{s}"] = idle[s] / 1000.0
        row[f"prof_launch_api_ms_{s}"] = launch_ms[s] / 1000.0
        row[f"prof_sync_api_ms_{s}"] = sync_ms[s] / 1000.0
        row[f"prof_lead_ms_p50_{s}"] = (statistics.median(leads[s]) / 1000.0
                                        if leads[s] else None)
    for c in CATEGORIES:
        row[f"prof_cat_ms_{c}"] = category_us[c] / 1000.0

    n_decode = max(len(calls["lm"].starts) - 1, 0)
    n_expert = len(calls["expert"].starts)
    row["prof_kernels_per_decode_step"] = (
        row["prof_kernels_decode"] / n_decode if n_decode else None)
    row["prof_kernels_per_expert_step"] = (
        row["prof_kernels_expert"] / n_expert if n_expert else None)

    seen: dict[str, set[str]] = {p: set() for p in SDPA_PHASES}
    for ts, backend in backends:
        if c0 <= ts < c1 and segment(ts) in seen:
            seen[segment(ts)].add(backend)
    for p in SDPA_PHASES:
        row[f"prof_sdpa_{p}"] = "+".join(sorted(seen[p])) or None
    row["prof_n_sdpa_calls"] = sum(1 for ts, _ in backends if c0 <= ts < c1)
    return row, kernels


def _cut(a: float, b: float, starts: list[float], labels: list[str], c1: float,
         into: dict[str, float]) -> None:
    """Add the host interval [a, b) to ``into``, split by segment."""
    if a >= c1:
        into["tail"] += b - a
        return
    if b > c1:
        into["tail"] += b - c1
        b = c1
    i = max(bisect.bisect_right(starts, a) - 1, 0)
    while a < b:
        stop = min(b, starts[i + 1] if i + 1 < len(starts) else c1)
        into[labels[i]] += stop - a
        a = stop
        i += 1


def _product(dims: Any) -> int | None:
    if not isinstance(dims, (list, tuple)) or not dims:
        return None
    out = 1
    for d in dims:
        out *= int(d)
    return out
