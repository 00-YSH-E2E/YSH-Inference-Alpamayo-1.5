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

"""docs/MEASUREMENTS.md, rendered from the registry.

    python -m alpamayo1_5.trace.measurements_doc > docs/MEASUREMENTS.md

A column list written by hand is out of date by the next commit. This one
is rendered from what the code declares -- timing_schema's columns, groups,
MLflow keys and changelog, the writer's long tables -- and a test fails
until the file on disk matches it. The prose around the tables is here too,
so the page changes in the same commit as the thing it describes.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

from alpamayo1_5.trace import profile_parse as PP
from alpamayo1_5.trace import roofline as RL
from alpamayo1_5.trace import thermal as TH
from alpamayo1_5.trace import timing_schema as TS
from alpamayo1_5.trace import writer as W

#: One line per column group, in registry order.
GROUP_TEXT = {
    "identity": "Which run, clip and pass the row is.",
    "condition": "What the pass was measured under, repeated on every row so runs "
                 "concatenate. Latency is compared only between rows that agree on these "
                 "(`compare.TIMING_PAIRING`).",
    "segment": "The legacy split, on the device clock, with the definitions "
               "`predictions.parquet` has carried since schema 1.",
    "clock": "Both clocks around the pass: CUDA events for the device, `perf_counter` for "
             "the host.",
    "per_call": "Every decode step and Euler step, on both clocks.",
    "alloc": "The CUDA caching allocator over the pass.",
    "graph": "The expert's CUDA graph runner, when one is installed.",
    "trace": "What the tracer itself cost: its marks, its hooks' host time, its logits pass.",
    "generate": "`generate` split into its preamble, lm_head, forward glue and loop; the "
                "first token.",
    "host": "Host stages around the model call: data, messages, preprocessing, H2D, results.",
    "pass_host": "Inside the call on the host clock: before generate, the tail, CPU time by "
                 "segment, the process.",
    "memory": "Allocator peaks per segment, retries and OOMs, host memory.",
    "shape": "The shapes the pass had: the KV cache, the images and patches.",
    "energy": "Energy per clip and per segment, integrated from the board's rails.",
    "board": "Clocks, over-current events, throttle state and temperature during the pass.",
    "step": "Trace level step: inside the decode and Euler steps, and every synchronization.",
    "layer": "Trace level layer: attention and MLP time per phase.",
    "profile": "Profile passes: kernels, true GPU idle, launches and syncs per host segment.",
    "flops": "Counted passes: FLOPs and operand bytes per host segment.",
}

FILES = (
    ("predictions.parquet", "sample", "HF", f"schema {W.SCHEMA_VERSION}, frozen"),
    ("per_clip.parquet", "clip", "HF", f"schema {W.SCHEMA_VERSION}, frozen"),
    ("timing.parquet", "pass", "HF", f"timing schema {TS.TIMING_SCHEMA_VERSION}"),
    ("thermal.parquet", "board reading", "HF", f"thermal schema {TH.THERMAL_SCHEMA_VERSION}"),
    ("layers.parquet", "layer span of a main pass (`--trace-level layer`)", "HF",
     f"layers schema {TS.LAYERS_SCHEMA_VERSION}"),
    ("kernels.parquet", "device event of a profile pass (`--profile-clips`)", "HF",
     f"kernels schema {TS.KERNELS_SCHEMA_VERSION}"),
    ("run.json", "run", "MLflow artifact and HF", "the run's config, params, work model, "
     "probe peaks, calibration"),
    ("memsnap_*.pickle, profile_*.json.gz", "--", "local only", "paths in run.json"),
)

ROW_KINDS = (
    ("main", "--", "The clip's pass. The only rows latency means and `predictions.parquet` "
                   "use."),
    ("warmup", "`--warmup N`", "Before the first clip, on it. Recorded, never averaged."),
    ("probe", "`--overhead-probe N`", "The first N clips again at level off and at the run's "
                                      "level, order alternating: `trace.overhead_pct`."),
    ("repeat", "`--timing-repeats R`", "The same clip again: `latency_cv`, `cold_start_ms`."),
    ("memsnap", "`--memory-snapshot N`", "Under the allocator's history recorder."),
    ("flops", "`--flop-count N`", "FLOPs and bytes counted by segment. Never timed."),
    ("profile", "`--profile-clips N`", "Under torch.profiler, at level basic. Never timed."),
)

LEVELS = (
    ("off", "No hook. The wall clock only: the probe's baseline."),
    ("basic", "Segment spans, per-call arrays, token statistics. The default."),
    ("step", "Also inside the steps: projections, KV concatenation, processors, stopping, "
             "and a sync audit. About 150 marks a decode step."),
    ("layer", "Also every layer of all three stacks: the attention -- its Q/K/V and output "
              "projections and its cache update apart -- and the MLP: `layers.parquet`. "
              "648 marks a language-model or head call."),
)

#: run.sh's variable, the runner's flag, what it measures.
FLAGS = (
    ("TRACE_LEVEL", "--trace-level", "How deep the instrumentation goes (above)."),
    ("WARMUP", "--warmup", "Untimed passes before the first clip (default 2)."),
    ("OVERHEAD_PROBE", "--overhead-probe", "Clips that get the off/on probe pair (default 4)."),
    ("TIMING_REPEATS", "--timing-repeats", "Repeat passes per clip on the first REPEAT_CLIPS."),
    ("REPEAT_CLIPS", "--repeat-clips", "How many clips get the repeats."),
    ("MEMORY_SNAPSHOT", "--memory-snapshot", "Clips that get an allocator-history pass."),
    ("SAMPLE_HZ", "--sample-hz", "Board sampler rate (default 10)."),
    ("CUDA_GRAPH", "--cuda-graph", "Replay the expert through CUDA graphs."),
    ("CUDA_GRAPH_MAX_GRAPHS", "--cuda-graph-max-graphs", "Captured signatures kept."),
    ("PROFILE_CLIPS", "--profile-clips", "Clips that get a torch.profiler pass."),
    ("PROFILE_TRACE", "--profile-trace", "Keep those passes' Chrome traces (local)."),
    ("FLOP_COUNT", "--flop-count", "Clips that get a counted pass."),
    ("ROOFLINE_PROBE", "--roofline-probe", "Measure the board's peaks before the first clip."),
    ("DEADLINE_MS", "--deadline-ms", "A latency budget to report misses against."),
    ("STEADY_SKIP", "--steady-skip", "Clips left out of the steady-state numbers."),
    ("MODEL_REVISION", "--model-revision", "The model snapshot loaded and recorded."),
)

_DTYPE = {"f64": "float64", "f32": "float32", "i16": "int16", "i32": "int32", "i64": "int64",
          "b": "bool", "s": "string", "lf32": "list<float32>"}
_BETTER = {"L": "lower", "H": "higher", "N": ""}


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: Iterable[str], rows: Iterable[Iterable[str]]) -> list[str]:
    header = list(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def _long(name: str, columns: Iterable[tuple[str, str, str]]) -> list[str]:
    return [f"### {name}", ""] + _table(
        ("column", "type", ""), ((f"`{c}`", _DTYPE[t], text) for c, t, text in columns)) + [""]


def render() -> str:
    lines = [
        "# Measurements",
        "",
        "<!-- Generated by `python -m alpamayo1_5.trace.measurements_doc > "
        "docs/MEASUREMENTS.md`. Do not edit by hand: tests/test_measurements_doc.py fails "
        "until this file matches the code. -->",
        "",
        "What a run measures, where each number lands, and what it means. Rendered from "
        "`src/alpamayo1_5/trace/timing_schema.py` and the writer, so it is as current as "
        "the code.",
        "",
        "Raw measurements go to parquet and on to Hugging Face; MLflow gets run-level "
        "aggregates and params only. Anything that can be computed from the parquet is not "
        "stored twice.",
        "",
        "## Files",
        "",
        *_table(("file", "one row per", "goes to", "version"),
                ((f"`{f}`", per, dest, v) for f, per, dest, v in FILES)),
        "",
        "## Passes",
        "",
        "Every inference is a row of `timing.parquet`, told apart by `row_kind`.",
        "",
        *_table(("row_kind", "made by", ""), ((f"`{k}`", flag, text)
                                             for k, flag, text in ROW_KINDS)),
        "",
        "## Trace levels",
        "",
        "Recorded on every row as `trace_level`. Latency is never compared across levels.",
        "",
        *_table(("level", ""), ((f"`{k}`", text) for k, text in LEVELS)),
        "",
        "## Settings",
        "",
        "Set in `scripts/run.sh` (or `OVERRIDE_<NAME>=... ./scripts/run.sh`).",
        "",
        *_table(("run.sh", "flag", ""), ((f"`{v}`", f"`{f}`", text) for v, f, text in FLAGS)),
        "",
        f"## timing.parquet (schema {TS.TIMING_SCHEMA_VERSION}, tracer "
        f"{TS.TRACER_VERSION})",
        "",
        "`better` is what a comparison calls an improvement; blank for counts and "
        "conditions, where ranking would be a mistake. `since` is the schema version that "
        "added the column.",
        "",
    ]
    groups: dict[str, list[TS.Col]] = {}
    for col in TS.COLUMNS:
        groups.setdefault(col.group, []).append(col)
    for group, cols in groups.items():
        lines += [f"### {group}", "", GROUP_TEXT[group], ""]
        lines += _table(("column", "type", "unit", "better", "since", ""),
                        ((f"`{c.name}`", _DTYPE[c.dtype], c.unit, _BETTER[c.direction],
                          str(c.since), c.description) for c in cols))
        lines.append("")
    lines += [
        "## Long tables",
        "",
        "One row per reading, span or device event; each joins its timing row on "
        "`(clip_id, row_kind, pass_index)`. Profile segments are "
        + ", ".join(f"`{s}`" for s in PP.SEGMENTS) + "; kernel categories "
        + ", ".join(f"`{c}`" for c in PP.CATEGORIES) + ".",
        "",
        *_long("thermal.parquet", W._THERMAL_COLUMNS),
        *_long("layers.parquet", W._LAYER_COLUMNS),
        *_long("kernels.parquet", W._KERNEL_COLUMNS),
        "## MLflow metrics",
        "",
        "Logged once, in one batch, at the end of the run: aggregates over the main passes, "
        "and the extra passes' own numbers (probe, profile, counted). "
        f"The efficiency numbers need run.json's `work_model`; the roof fractions also "
        "need `--roofline-probe`. A FLOP calibration outside "
        f"[{RL.CALIBRATION_BAND[0]}, {RL.CALIBRATION_BAND[1]}] is reported as a wrong "
        "formula.",
        "",
    ]
    prefixes: dict[str, list[str]] = {}
    for key in TS.AGGREGATE_KEYS:
        prefixes.setdefault(key.split(".")[0] + "." if "." in key else "", []).append(key)
    for prefix, keys in prefixes.items():
        label = f"`{prefix}*`" if prefix else "headline (no prefix)"
        lines += [f"- {label}: " + ", ".join(f"`{k}`" for k in keys)]
    lines += ["", "## Changelog", ""]
    lines += _table(("timing schema", ""), ((str(v), text) for v, text in
                                             sorted(TS.CHANGELOG.items())))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.write(render())
