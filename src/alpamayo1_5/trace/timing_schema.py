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

"""What ``timing.parquet`` holds, and the one place that says so.

Measurement lives in its own table rather than in ``predictions.parquet``, for
two reasons that are about comparison, not tidiness:

* **Latency and accuracy are compared under different rules.** Accuracy parity
  may cross machines -- a Thor run and a Pro 6000 run of the same checkpoint on
  the same clips are supposed to agree, and ``compare.py`` deliberately leaves
  ``machine`` out of its pairing. Latency may not: a number is only comparable
  with one taken on the same board, power mode, driver, torch build and depth
  of instrumentation. One table cannot carry both rules.
* **The measurement schema moves faster.** Every new instrument adds columns.
  Had they gone into the predictions table, each one would have cut every new
  run off from the schema-3 corpus it is meant to be compared with.

So ``predictions.parquet`` stays frozen at schema 3 -- its legacy ``t_*``
columns keep their names and definitions -- and this table carries everything
else, versioned on its own.

The registry below is the single source of truth. The writer builds the
parquet schema from it, the runner derives the MLflow names from it, the
comparison code reads directions from it and the documentation is generated
from it. A column that is not declared here does not exist.

Nothing in this module imports torch: CI installs numpy, pandas and pyarrow
only, and everything here is tested there.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

#: Bump in every change that adds, removes or redefines a column. Tables with
#: different versions refuse to concatenate: a column that exists in one run
#: and not another would come back as a silent NaN in a comparison.
TIMING_SCHEMA_VERSION = 4

#: Bump when the hooks that produce the basic-level numbers change. Instrument
#: cost moves with them, so two runs measured by different tracers are not
#: comparable on latency even when every other condition matches.
TRACER_VERSION = 3

CHANGELOG = {
    1: "Initial table: legacy spans, host wall clock, per-call arrays, allocator and "
    "CUDA-graph counters, run conditions.",
    2: "The tracer's own cost: t_trace_consume_ms and its host twin, t_postgen_model_ms, "
    "trace_n_marks, trace_hook_host_ms. Tracer 2 marks its logits pass.",
    3: "generate split: t_gen_preamble_ms, t_lm_head_ms, t_vlm_glue_ms, t_gen_loop_ms, "
    "t_ttft_ms, n_vlm_forwards, lm_head_step_ms, decode_gap_ms, trace_span_violations. "
    "Tracer 3 hooks the VLM forward and lm_head.",
    4: "Measurement protocol: overhead_probe, timing_repeats and repeat_clips conditions, "
    "pass_output_match. Warmup, probe and repeat passes are rows of their own kind.",
}

#: What a row can be. Only ``main`` rows feed predictions and latency
#: aggregates; the others are extra passes over the same clip whose purpose is
#: to measure something else (the tracer's own cost, noise, kernels, memory).
ROW_KINDS = ("main", "warmup", "probe", "repeat", "profile", "memsnap")

_DTYPES = ("f64", "f32", "i16", "i32", "i64", "b", "s", "lf32")
_DIRECTIONS = {"L": "lower", "H": "higher", "N": "neutral"}


@dataclass(frozen=True)
class Col:
    """One column of ``timing.parquet``.

    ``direction`` is what a comparison should call better: L (lower), H
    (higher) or N (neutral -- a count or a condition, where better has no
    meaning and ranking it would be a mistake).
    """

    name: str
    dtype: str
    unit: str
    direction: str
    group: str
    description: str
    since: int = 1


def _cols(group: str, specs: Iterable[tuple[str, str, str, str, str]],
          since: int = 1) -> tuple[Col, ...]:
    return tuple(Col(name, dtype, unit, direction, group, text, since)
                 for name, dtype, unit, direction, text in specs)


IDENTITY = _cols("identity", (
    ("timing_schema_version", "i16", "", "N", "Version of this table's column set."),
    ("tracer_version", "i16", "", "N",
     "Version of the basic-level instrumentation. Latency is compared only within one."),
    ("run_id", "s", "", "N", "MLflow run id, or a local id when the run was not tracked."),
    ("variant", "s", "", "N", "The run's comparison label."),
    ("git_commit", "s", "", "N", "Commit of the code that produced the row."),
    ("clip_id", "s", "", "N", "Clip the pass ran on."),
    ("t0_us", "i64", "us", "N", "Sample timestamp within the clip."),
    ("clip_index", "i32", "", "N", "Position of the clip in the run's clip list."),
    ("row_kind", "s", "", "N",
     "main | warmup | probe | repeat | profile | memsnap. Only main rows feed predictions "
     "and latency aggregates."),
    ("pass_index", "i16", "", "N",
     "0 for a clip's main pass; extra passes over the same clip count up from 1."),
    ("t_start_unix_s", "f64", "s", "N", "Wall-clock time the pass started, Unix seconds."),
    ("t_start_host_s", "f64", "s", "N",
     "perf_counter when the pass started. The board sampler stamps the same clock, which is "
     "what lets its readings be attributed to a pass."),
))

CONDITIONS = _cols("condition", (
    ("machine", "s", "", "N", "Where the run happened. Latency never crosses machines."),
    ("power_mode", "s", "", "N", "nvpmodel mode name."),
    ("torch_version", "s", "", "N", "torch build; its kernels are part of the measurement."),
    ("transformers_version", "s", "", "N", "transformers version; it owns the decode loop."),
    ("cuda_runtime", "s", "", "N", "CUDA runtime torch was built against."),
    ("attn_impl", "s", "", "N", "Attention implementation requested at load."),
    ("dtype", "s", "", "N", "Compute dtype."),
    ("model", "s", "", "N", "Model coordinate as given on the command line."),
    ("data_spec", "s", "", "N", "Camera and sensor configuration."),
    ("num_traj_samples", "i16", "", "N", "K, trajectories per clip."),
    ("inference_step", "i16", "", "N", "Euler steps requested; null means the default."),
    ("max_new_tokens", "i16", "", "N", "Generation cap actually passed to generate."),
    ("cuda_graph", "b", "", "N", "Whether the expert was replayed through CUDA graphs."),
    ("cuda_graph_max_graphs", "i16", "", "N",
     "Captured signatures kept; null when graphs are off."),
    ("trace_level", "s", "", "N",
     "How deep the instrumentation went. Deeper levels cost more; never compare latency "
     "across levels."),
    ("sample_hz", "f32", "Hz", "N", "Board sampler rate while the pass ran."),
    ("warmup", "i16", "", "N", "Untimed inferences run before the first main pass."),
))

#: The spans ``predictions.parquet`` has always carried, under the same names
#: and definitions. ``t_other_ms`` is the remainder of the named spans, so the
#: seven are checkable: vision + prefill + decode + postgen + expert + other
#: equals total.
LEGACY = _cols("segment", (
    ("t_vision_ms", "f64", "ms", "L", "Vision tower span, device clock."),
    ("t_prefill_ms", "f64", "ms", "L", "First language-model call, device clock."),
    ("t_decode_ms", "f64", "ms", "L", "Every later language-model call, device clock."),
    ("t_postgen_ms", "f64", "ms", "L",
     "From generate returning to the trajectory head starting. Host work lives here."),
    ("t_expert_ms", "f64", "ms", "L", "The diffusion sampler, all Euler steps."),
    ("t_other_ms", "f64", "ms", "L",
     "The generate span minus vision, prefill and decode: lm_head, logits processing, "
     "sampling, stopping criteria."),
    ("t_total_ms", "f64", "ms", "L", "Generate span + postgen + expert."),
    ("n_vision_calls", "i16", "", "N", "Vision tower calls in the pass."),
    ("n_decode_steps", "i16", "", "N",
     "Language-model calls after prefill -- decode's denominator, since the batch decodes "
     "until its last row finishes."),
    ("n_expert_calls", "i16", "", "N", "Euler steps that actually executed."),
    ("timing_measured", "b", "", "N", "False: every time in the row is absent, not zero."),
))

CLOCKS = _cols("clock", (
    ("t_wall_ms", "f64", "ms", "L",
     "Host wall clock from tracer install to after the final synchronize: what the caller "
     "waited. The device spans cannot see host gaps between them; this can."),
    ("t_compute_span_ms", "f64", "ms", "N",
     "Device time inside the vision, prefill, decode and expert spans. Idle inside a span "
     "is included, so this is not GPU busy time -- that comes from the kernel profile."),
))

PER_CALL = _cols("per_call", (
    ("vision_call_ms", "lf32", "ms", "L", "Each vision tower call, device clock."),
    ("decode_step_ms", "lf32", "ms", "L", "Each decode step, device clock."),
    ("decode_step_host_ms", "lf32", "ms", "N",
     "Each decode step, host clock: how long Python took to enqueue it."),
    ("expert_step_ms", "lf32", "ms", "L",
     "Each Euler step of the trajectory head, device clock. The first carries warmup and, "
     "with graphs on, the capture."),
    ("expert_step_host_ms", "lf32", "ms", "N", "Each Euler step, host clock."),
))

ALLOC = _cols("alloc", (
    ("n_cuda_allocs", "i64", "", "L",
     "Allocator allocations during the pass. A KV cache rebuilt by concatenation on every "
     "layer and step lands here in the hundreds."),
    ("n_alloc_retries", "i64", "", "L",
     "Allocator retries: each is a cache flush and a stall in whichever span was unlucky."),
))

GRAPH = _cols("graph", (
    ("graph_captures", "i32", "", "N", "Expert CUDA graphs captured; null when graphs are off."),
    ("graph_replays", "i32", "", "N",
     "Expert steps replayed. A capture step also counts as a replay."),
    ("graph_fallbacks", "i32", "", "L",
     "Expert steps that ran eager with a graph runner installed. Never average latency over "
     "rows that fell back together with rows that did not."),
    ("graph_n_graphs", "i32", "", "N", "Graphs held by the runner after the pass."),
    ("graph_capture_ms", "f64", "ms", "L", "Host time spent capturing during the pass."),
    ("graph_mode", "s", "", "N",
     "none | replay | capture | fallback | mixed: which path the pass's expert steps took."),
))

TRACE = _cols("trace", (
    ("t_trace_consume_ms", "f64", "ms", "L",
     "Device time of the tracer's own logits pass. It runs inside the postgen window, so "
     "t_postgen_ms includes it -- kept there so the legacy column keeps its meaning."),
    ("t_trace_consume_host_ms", "f64", "ms", "L", "The same pass, host clock."),
    ("t_postgen_model_ms", "f64", "ms", "L",
     "postgen minus the tracer's logits pass: what the model's own code between generate "
     "and the trajectory head cost."),
    ("trace_n_marks", "i32", "", "N", "Marks recorded in the pass."),
    ("trace_hook_host_ms", "f64", "ms", "L",
     "Host time spent inside the tracer's marks. The instrument's own cost, per pass."),
), since=2)

#: What t_other_ms was made of. By construction
#: t_other_ms = t_gen_preamble_ms + t_lm_head_ms + t_vlm_glue_ms + t_gen_loop_ms,
#: so the remainder that used to be one opaque number is now four checkable ones.
GENERATE = _cols("generate", (
    ("t_gen_preamble_ms", "f64", "ms", "L",
     "generate's start to the first forward: K-fold input expansion, cache and processor "
     "setup."),
    ("t_lm_head_ms", "f64", "ms", "L",
     "The vocabulary projection, every step. It runs after the language model returns, so "
     "the decode span never included it -- 1.27 GB of weights read per step."),
    ("t_vlm_glue_ms", "f64", "ms", "L",
     "Inside each VLM forward but outside vision, the language model and lm_head: "
     "embeddings, image-token scatter, rope index, deepstack."),
    ("t_gen_loop_ms", "f64", "ms", "L",
     "generate outside every forward: logits processors, sampling, stopping criteria and "
     "the host syncs between steps."),
    ("t_ttft_ms", "f64", "ms", "L",
     "Host clock from the model call to the second forward -- the first reasoning token is "
     "on the host by then, since the loop syncs on it."),
    ("n_vlm_forwards", "i16", "", "N", "VLM forward calls: prefill plus decode steps."),
    ("lm_head_step_ms", "lf32", "ms", "L", "Each lm_head call, device clock."),
    ("decode_gap_ms", "lf32", "ms", "L",
     "Device time between consecutive forwards: the host loop each decode step waits on."),
    ("trace_span_violations", "i16", "", "L",
     "Derived spans that came out negative and were floored. Nonzero means a nesting "
     "assumption broke and the split is suspect."),
), since=3)

PROTOCOL = _cols("condition", (
    ("overhead_probe", "i16", "", "N",
     "Clips that got an extra off/on pass pair to measure the tracer's own cost."),
    ("timing_repeats", "i16", "", "N", "Extra passes per repeated clip, for the noise band."),
    ("repeat_clips", "i16", "", "N", "How many clips, from the start, got the repeats."),
    ("pass_output_match", "b", "", "N",
     "Extra passes only: whether the trajectories came out bit-identical to the main pass. "
     "False means the passes were not replicates, and their timing differences are not noise."),
), since=4)

COLUMNS: tuple[Col, ...] = (IDENTITY + CONDITIONS + LEGACY + CLOCKS + PER_CALL + ALLOC + GRAPH
                            + TRACE + GENERATE + PROTOCOL)

#: The keys ``predictions.parquet`` reads off a pass, unchanged since schema 3.
LEGACY_KEYS = tuple(c.name for c in LEGACY)

_BY_NAME = {c.name: c for c in COLUMNS}


def column(name: str) -> Col:
    return _BY_NAME[name]


def names() -> tuple[str, ...]:
    return tuple(c.name for c in COLUMNS)


def directions() -> dict[str, str]:
    """Column name to ``lower`` / ``higher`` / ``neutral``, for comparison tables."""
    return {c.name: _DIRECTIONS[c.direction] for c in COLUMNS}


# -- rows -------------------------------------------------------------------
def is_number(value: Any) -> bool:
    """A finite real number, numpy scalars included, bools excluded.

    ``isinstance(v, (int, float))`` is the obvious test and it is wrong twice:
    it rejects ``np.float32``, so a metric computed with numpy is silently
    skipped, and it accepts ``True``, so a flag gets averaged as a measurement.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def _coerce(col: Col, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    kind = col.dtype
    if kind in ("f64", "f32"):
        return float(value)
    if kind in ("i16", "i32", "i64"):
        return int(value)
    if kind == "b":
        return bool(value)
    if kind == "s":
        return str(value)
    if kind == "lf32":
        return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]
    raise ValueError(f"unknown dtype {kind!r} for {col.name}")


def unknown_keys(row: Mapping[str, Any]) -> list[str]:
    """Keys a row carries that the registry does not declare."""
    return sorted(k for k in row if k not in _BY_NAME)


def coerce_row(row: Mapping[str, Any], strict: bool = False) -> dict[str, Any]:
    """Exactly the declared columns, in declared order, with declared types.

    Missing columns become nulls: a row from a pass that did not run an
    instrument says so by absence, never by a plausible zero. Undeclared keys
    raise under ``strict`` (the tests) and are dropped otherwise (a run), since
    a typo in instrumentation must not cost a run.
    """
    if strict:
        extra = unknown_keys(row)
        if extra:
            raise ValueError(f"undeclared timing columns: {extra}")
    return {c.name: _coerce(c, row.get(c.name)) for c in COLUMNS}


def pa_schema() -> Any:
    """The parquet schema. Explicit, so a column that is null in every row of a
    run keeps its type and the run still concatenates with one where it is not."""
    import pyarrow as pa

    types = {
        "f64": pa.float64(), "f32": pa.float32(),
        "i16": pa.int16(), "i32": pa.int32(), "i64": pa.int64(),
        "b": pa.bool_(), "s": pa.string(), "lf32": pa.list_(pa.float32()),
    }
    return pa.schema([pa.field(c.name, types[c.dtype]) for c in COLUMNS])


# -- aggregation ------------------------------------------------------------
#: Every MLflow key :func:`aggregate` can emit. Declared rather than derived
#: from whatever a run happened to compute, for the same reason the runner
#: declares its per-clip metrics: otherwise the experiment's metric namespace is
#: decided by the last code change, and keys appear and vanish between runs
#: without anyone choosing that.
AGGREGATE_KEYS = (
    # The seven legacy spans, under the names every earlier run used.
    "t_vision_ms", "t_prefill_ms", "t_decode_ms", "t_postgen_ms",
    "t_expert_ms", "t_other_ms", "t_total_ms",
    # Tails. latency_p95 is the recording rules' name for t_total_ms_p95.
    "t_total_ms_p50", "t_total_ms_p95", "t_total_ms_p99", "latency_p95",
    "t_wall_ms", "t_wall_ms_p50", "t_wall_ms_p95", "t_wall_ms_p99",
    "compute_span_ratio",
    "expert_step_first_ms", "expert_step_steady_ms", "expert_step_p95_ms",
    "expert_step_spread_ms", "expert_launch_ratio",
    "decode_step_p95_ms", "decode_launch_ratio",
    "alloc.n_cuda_allocs", "alloc.n_alloc_retries_sum",
    "graph.captures_sum", "graph.replays_sum", "graph.fallbacks_sum",
    "graph.fallback_clip_frac", "graph.capture_ms_sum", "graph.t_expert_ms_replay_only",
    "timing.n_main_rows",
    "t_postgen_model_ms", "trace.consume_ms", "trace.hook_host_ms", "trace.n_marks",
    "t_gen_preamble_ms", "t_lm_head_ms", "t_vlm_glue_ms", "t_gen_loop_ms",
    "t_ttft_ms", "t_ttft_ms_p95", "decode_gap_ms", "decode_gap_ms_p95", "lm_head_step_ms",
    "trace.span_violations_sum",
    "trace.overhead_pct", "trace.overhead_lo", "trace.overhead_hi", "trace.overhead_n",
    "cold_start_ms", "cold_start_excess_ms", "latency_cv", "latency_cv_n",
    "timing.n_extra_rows", "pass.output_mismatch_sum",
)


def main_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows a latency number may rest on: main passes that were measured."""
    return [r for r in rows
            if r.get("row_kind", "main") == "main" and r.get("timing_measured") is True]


def _values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if is_number(r.get(key))]


def _arrays(rows: Iterable[Mapping[str, Any]], key: str) -> list[list[float]]:
    out = []
    for r in rows:
        value = r.get(key)
        if value is None:
            continue
        array = [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]
        if array:
            out.append(array)
    return out


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _ratio_of_sums(rows: Iterable[Mapping[str, Any]], num: str, den: str) -> list[float]:
    out = []
    for r in rows:
        a, b = r.get(num), r.get(den)
        if a is None or b is None:
            continue
        a_sum = float(np.sum(np.asarray(a, dtype=np.float64)))
        b_sum = float(np.sum(np.asarray(b, dtype=np.float64)))
        if b_sum > 0.0:
            out.append(a_sum / b_sum)
    return out


def overhead(rows: Iterable[Mapping[str, Any]], n_boot: int = 2000,
             seed: int = 0) -> dict[str, float] | None:
    """The tracer's own cost, from probe passes: off and on over the same clip.

    Per clip, ``100 * (wall_on - wall_off) / wall_off``; the run's number is the
    median over clips, with a bootstrap interval over clips. Paired per clip
    because clips differ by seconds and the cost is milliseconds -- an
    unpaired difference of means would be all clip variance. The probe order
    alternates by clip, so a slow drift (the board warming) does not load onto
    one side.
    """
    by_clip: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r.get("row_kind") != "probe" or not is_number(r.get("t_wall_ms")):
            continue
        side = "off" if r.get("trace_level") == "off" else "on"
        by_clip.setdefault(str(r.get("clip_id")), {"off": [], "on": []})[side].append(
            float(r["t_wall_ms"]))
    pcts = [100.0 * (np.mean(v["on"]) - np.mean(v["off"])) / np.mean(v["off"])
            for v in by_clip.values() if v["on"] and v["off"] and np.mean(v["off"]) > 0.0]
    if not pcts:
        return None
    values = np.asarray(pcts, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(values, size=(n_boot, values.size), replace=True), axis=1)
    return {
        "trace.overhead_pct": float(np.median(values)),
        "trace.overhead_lo": float(np.percentile(boot, 2.5)),
        "trace.overhead_hi": float(np.percentile(boot, 97.5)),
        "trace.overhead_n": float(values.size),
    }


def _protocol(rows: list[Mapping[str, Any]]) -> dict[str, float | None]:
    """Numbers that need the extra passes, not only the main ones."""
    out: dict[str, float | None] = {}
    out.update(overhead(rows) or {})
    extra = [r for r in rows if r.get("row_kind", "main") != "main"]
    if extra:
        out["timing.n_extra_rows"] = float(len(extra))
        # Extra passes are re-seeded replicates of the main one. A mismatch
        # means they were not, and everything measured on them is suspect.
        out["pass.output_mismatch_sum"] = float(
            sum(1 for r in extra if r.get("pass_output_match") is False))

    # The first pass the process made, whatever its kind: cuDNN autotuning,
    # allocator growth and lazy initialisation all land on it. It is a real
    # deployment number -- the first plan after boot -- and a contaminant
    # anywhere else, which is why warmup passes exist.
    stamped = [r for r in rows if is_number(r.get("t_wall_ms"))]
    if stamped:
        first = min(stamped, key=lambda r: (float(r.get("t_start_host_s") or 0.0),
                                            r.get("clip_index") or 0, r.get("pass_index") or 0))
        out["cold_start_ms"] = float(first["t_wall_ms"])
        steady = _values(main_rows(rows), "t_wall_ms")
        if steady:
            out["cold_start_excess_ms"] = out["cold_start_ms"] - float(np.median(steady))

    # Run-to-run noise on one clip: the main pass and its repeats. Only passes
    # that reproduced the main pass's output count -- otherwise the spread
    # includes a different amount of work, not just noise.
    groups: dict[str, list[float]] = {}
    for r in rows:
        kind = r.get("row_kind", "main")
        if kind not in ("main", "repeat") or not is_number(r.get("t_wall_ms")):
            continue
        if kind == "repeat" and r.get("pass_output_match") is False:
            continue
        groups.setdefault(str(r.get("clip_id")), []).append(float(r["t_wall_ms"]))
    cvs = [float(np.std(v, ddof=1) / np.mean(v)) for v in groups.values()
           if len(v) >= 2 and np.mean(v) > 0.0]
    if cvs:
        out["latency_cv"] = float(np.mean(cvs))
        out["latency_cv_n"] = float(len(cvs))
    return out


def aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Run-level numbers for MLflow, from the main measured rows only.

    Restricting to measured rows is a fix, not a nicety: the averaging this
    replaces took the legacy spans off every row, and an unmeasured row carried
    0.0 -- so a run where timing failed on some clips reported them as having
    taken no time. The protocol numbers (tracer overhead, cold start, noise
    band) are the exception: they exist only in the extra passes.
    """
    rows = list(rows)
    protocol = _protocol(rows)
    rows = main_rows(rows)
    out: dict[str, float | None] = {"timing.n_main_rows": float(len(rows)), **protocol}
    if not rows:
        return {k: v for k, v in out.items() if v is not None}

    for col in LEGACY:
        if col.unit == "ms":
            out[col.name] = _mean(_values(rows, col.name))
    for key in ("t_total_ms", "t_wall_ms"):
        values = _values(rows, key)
        if values:
            for pct in (50, 95, 99):
                out[f"{key}_p{pct}"] = float(np.percentile(values, pct))
    out["latency_p95"] = out.get("t_total_ms_p95")
    out["t_wall_ms"] = _mean(_values(rows, "t_wall_ms"))

    spans = [float(r["t_compute_span_ms"]) / float(r["t_wall_ms"]) for r in rows
             if is_number(r.get("t_compute_span_ms")) and is_number(r.get("t_wall_ms"))
             and float(r["t_wall_ms"]) > 0.0]
    out["compute_span_ratio"] = _mean(spans)

    steps = _arrays(rows, "expert_step_ms")
    # The first Euler step carries warmup and, with graphs on, the capture.
    # Folded into one mean it describes a step that runs once per clip as
    # though it ran every time.
    out["expert_step_first_ms"] = _mean([s[0] for s in steps])
    out["expert_step_steady_ms"] = _mean([float(np.mean(s[1:])) for s in steps if len(s) > 1])
    out["expert_step_p95_ms"] = _mean([float(np.percentile(s, 95)) for s in steps])
    # Steps are the same shapes doing the same work, so a wide spread is the
    # instrumentation, the allocator or a capture -- not the model.
    out["expert_step_spread_ms"] = _mean([max(s) - min(s) for s in steps if len(s) > 1])
    # Host enqueue time over device time. Near 1: the device waited on the
    # host, the step is launch-bound, and removing launches has something to
    # win. Well below 1: the device is the bottleneck. Above 1 only when work
    # queued earlier hid a slow host.
    out["expert_launch_ratio"] = _mean(
        _ratio_of_sums(rows, "expert_step_host_ms", "expert_step_ms"))

    decode = _arrays(rows, "decode_step_ms")
    out["decode_step_p95_ms"] = _mean([float(np.percentile(s, 95)) for s in decode])
    out["decode_launch_ratio"] = _mean(
        _ratio_of_sums(rows, "decode_step_host_ms", "decode_step_ms"))

    out["alloc.n_cuda_allocs"] = _mean(_values(rows, "n_cuda_allocs"))
    retries = _values(rows, "n_alloc_retries")
    out["alloc.n_alloc_retries_sum"] = float(sum(retries)) if retries else None

    graphed = [r for r in rows if r.get("graph_captures") is not None]
    if graphed:
        # Totals, not means: one fallback anywhere makes every latency mean a
        # blend of two configurations, which is a yes-or-no question.
        for name in ("captures", "replays", "fallbacks"):
            out[f"graph.{name}_sum"] = float(sum(_values(graphed, f"graph_{name}")))
        out["graph.fallback_clip_frac"] = float(
            np.mean([float((r.get("graph_fallbacks") or 0) > 0) for r in graphed]))
        out["graph.capture_ms_sum"] = float(sum(_values(graphed, "graph_capture_ms")))
        replay_only = [r for r in graphed if r.get("graph_mode") == "replay"]
        out["graph.t_expert_ms_replay_only"] = _mean(_values(replay_only, "t_expert_ms"))

    out["t_postgen_model_ms"] = _mean(_values(rows, "t_postgen_model_ms"))
    out["trace.consume_ms"] = _mean(_values(rows, "t_trace_consume_ms"))
    out["trace.hook_host_ms"] = _mean(_values(rows, "trace_hook_host_ms"))
    out["trace.n_marks"] = _mean(_values(rows, "trace_n_marks"))

    for key in ("t_gen_preamble_ms", "t_lm_head_ms", "t_vlm_glue_ms", "t_gen_loop_ms",
                "t_ttft_ms"):
        out[key] = _mean(_values(rows, key))
    ttft = _values(rows, "t_ttft_ms")
    out["t_ttft_ms_p95"] = float(np.percentile(ttft, 95)) if ttft else None
    # Per step, pooled over clips: a gap is a property of one step, and pooling
    # weights each step equally however many steps its clip decoded.
    gaps = [g for a in _arrays(rows, "decode_gap_ms") for g in a]
    out["decode_gap_ms"] = _mean(gaps)
    out["decode_gap_ms_p95"] = float(np.percentile(gaps, 95)) if gaps else None
    out["lm_head_step_ms"] = _mean([h for a in _arrays(rows, "lm_head_step_ms") for h in a])
    violations = _values(rows, "trace_span_violations")
    out["trace.span_violations_sum"] = float(sum(violations)) if violations else None

    return {k: float(v) for k, v in out.items() if v is not None and math.isfinite(v)}
