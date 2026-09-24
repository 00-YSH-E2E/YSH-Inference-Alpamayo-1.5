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

"""The run directory: what a finished inference run leaves behind.

Layout, one directory per run (see :func:`run_dir_name`)::

    out/Alpamayo-1.5_Cam-4_Vanilla_100clip_k6-temp0.6_thor_26.09.01_39581f9b/
    ├── predictions.parquet   one row per (clip_id, t0_us, sample_k): raw output
    ├── per_clip.parquet      one row per clip: situation label and its metrics
    ├── timing.parquet        one row per inference pass: where the time went
    ├── thermal.parquet       the board over the run: one row per sensor reading
    ├── layers.parquet        trace level layer only: one row per layer span of a main pass
    ├── kernels.parquet       profile passes only: one row per kernel, memcpy or memset
    ├── run.json              run-level metadata and the constants needed to
    │                         recompute anything offline
    ├── gt.parquet            logged future -- local only, never uploaded
    └── samples/<clip_id>.png

The name carries the machine, and the trailing eight characters are the MLflow
run id -- so a directory names both where it ran and which run it was, and a
run's ``output_uri`` names its directory.

Two decisions drive the whole schema.

**Long format, one row per sampled trajectory.** The config columns repeat on
every row, which looks wasteful until you try to compare runs: because the
schema is identical across runs, every run's parquet concatenates into one
dataframe and ``groupby("variant")`` produces the comparison table directly.
A schema that drifts between runs breaks that, and then the parquet has no
reason to exist.

**Raw only.** Predictions, ego history and token traces are stored because they
cannot be recovered without re-running the model. ADE, jerk, scene labels and
token statistics are not stored, because they are cheap to recompute and their
definitions will change -- freezing them here would mean re-running inference
whenever a definition moves, and would leave stale columns that still look
authoritative.

``timing.parquet`` is versioned apart from the other two (see
``timing_schema``). Latency is compared under stricter rules than accuracy --
never across machines, power modes or instrumentation depth -- and its columns
grow with every new instrument. Kept in the predictions table, each new
instrument would have cut new runs off from every older one.

The logged future is the one deliberate exception: it is recoverable from
``(clip_id, t0_us)``, so it goes to ``gt.parquet`` for local convenience and is
excluded from upload.

That exclusion is about size and recoverability, not licensing. It would be
convenient to also call it "gated content stays out of the shared repo", but
that is not true of what does get uploaded: every ``samples/*.png`` embeds the
source dataset's camera frames, and they are the bulk of the upload. The evals
repo is therefore private, and has to stay private for as long as the source
dataset is gated -- turning it public is one click and cannot be undone for
anything already cloned.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from alpamayo1_5.trace import timing_schema as TS

# Bump when a column is added, removed or changes meaning. Concatenating runs
# with different schema versions is the failure this exists to make visible.
#
# 2: prompt_len, t_other_ms, n_decode_steps, timing_measured.
#    Version 1 runs are not merely missing these columns -- their prompt_len was
#    written as 0 into run.json by a reader that asked the row for a key the row
#    never carried, so refusing to concatenate v1 with v2 is the right outcome
#    rather than an inconvenience.
# 3: x0, seed_scheme, diffusion_temperature. Version 2 runs re-seeded every
#    clip with the bare seed, which made their K samples non-exchangeable
#    (sample 0 sat 0.5 m off the others); version 3 runs XOR a clip hash. The
#    samples in the two are drawn by different procedures, so a comparison
#    across the boundary would attribute that difference to whatever axis it
#    was grouped on. Refusing to concatenate them is, again, the point.
SCHEMA_VERSION = 3

# Columns whose values are flat float32 arrays. Stored as lists; the trailing
# shape is recorded in run.json so a reader can reshape without guessing.
_ARRAY_SHAPES = {
    "pred_xy": ("T", 2),
    "x0": ("T", 2),
    "pred_rot": ("T", 3, 3),
    "hist_xy": ("N_hist", 2),
    "hist_rot": ("N_hist", 3, 3),
}


def machine_name(explicit: str | None = None) -> str:
    """Short label for the machine a run happened on.

    Defaults to the hostname's first component, which is what ``env.host``
    records and what ``config/experiments.yaml`` matches on -- so the directory
    name, the MLflow tag and the hub's machine rule all say the same word.
    Pass ``explicit`` when the hostname is not the name you think of the machine
    by (a rented box, a container).
    """
    if explicit:
        return explicit.strip()
    return socket.gethostname().split(".")[0] or "unknown"


def run_dir_name(
    variant: str,
    date: str,
    run_id: str,
    model: str = "Alpamayo-1.5",
    data: str = "Cam-4",
    machine: str | None = None,
    label: str | None = None,
) -> str:
    """Directory name: ``{model}_{data}_{variant}[_{label}]_{machine}_{date}_{run_id[:8]}``.

    Example: ``Alpamayo-1.5_Cam-4_Vanilla_thor_26.09.01_39581f9b``, and with a
    label: ``Alpamayo-1.5_Cam-4_Vanilla_k6-t0.9_thor_26.09.01_39581f9b``.

    ``label`` exists for sweeps. A sweep over sampling settings holds the
    variant fixed and varies K or temperature, so without it every directory in
    the batch reads identically except for the run id -- unique, but not
    identifiable, and you cannot pick the one you want by eye. The label names
    only what actually varied in that sweep, so a sweep over one axis does not
    inherit clutter from axes that stayed still.

    It is deliberately not folded into ``variant``: variant is the comparison
    axis and has to stay the same string across runs for a grouping to hold.
    Settings that change per run belong beside it, not inside it.

    The name answers, in order, the questions asked when looking at an old
    result: which model, on what data, run how, **where** -- then when. It
    matches the layout of the data directories (``Alpamayo-1.5_Cam-4_Vanilla``)
    with the machine and date appended.

    Machine is in the name rather than only in the tags because latency is
    meaningless without it. The same checkpoint on a Thor and on a Pro 6000
    produces two directories that are otherwise identical in every visible
    field, and the numbers inside are not comparable. It also matters for runs
    made off the tailnet, where the MLflow record does not exist yet and the
    directory name is the only thing saying where the work happened.

    Args:
        variant: How this run differs -- ``Vanilla``, ``Pruned-24L``, ``INT8``.
            This is also the ``variant`` column in the parquet, so the directory
            name and the comparison table always agree.
        date: ``YY.MM.DD``.
        run_id: MLflow run id. Only the first 8 characters are used.
        model: Model identity.
        data: What the model was fed -- camera count, sensor set.
        machine: Where it ran. Defaults to the short hostname, matching
            ``env.host``.

    The run id is part of the name for two reasons. Two runs of the same variant
    on the same day would otherwise collide, which is not hypothetical -- the
    same configuration gets re-run constantly while something is being debugged.
    And it makes the link bidirectional: from a directory you can find the
    MLflow run, and from a run you can find its files, without searching.
    """

    def clean(text: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "-" for c in str(text)).strip("-")

    parts = [clean(model), clean(data), clean(variant) or "run",
             clean(label) if label else "",
             clean(machine_name(machine)), clean(date), run_id[:8]]
    return "_".join(p for p in parts if p)


def _flat(array: Any) -> list[float]:
    """Flatten to a plain float32 list. Parquet stores these as list columns."""
    return np.asarray(array, dtype=np.float32).reshape(-1).tolist()


def build_rows(samples: Iterable[dict], config: dict) -> pd.DataFrame:
    """Assemble the long-format table.

    Args:
        samples: One dict per (clip, t0, sample_k). Expected keys are listed in
            ``REQUIRED_SAMPLE_KEYS``; missing optional keys become nulls rather
            than dropping the row, because a partially-instrumented run is still
            worth keeping.
        config: Values repeated on every row -- model variant, sampling
            settings, git commit. Repetition is what makes cross-run
            concatenation work.

    Returns:
        A DataFrame with a stable column order.
    """
    rows = []
    for s in samples:
        row = {
            # identity
            "schema_version": SCHEMA_VERSION,
            "run_id": config.get("run_id"),
            "variant": config.get("variant"),
            "git_commit": config.get("git_commit"),
            "clip_id": s["clip_id"],
            "t0_us": int(s["t0_us"]),
            "sample_k": int(s["sample_k"]),
            # model output -- irrecoverable
            "pred_xy": _flat(s["pred_xy"]),
            # model input to the flow head -- equally irrecoverable, and the
            # thing a step-distilled student is paired against. None when the
            # tracer did not run.
            "x0": _flat(s["x0"]) if s.get("x0") is not None else None,
            "pred_rot": _flat(s["pred_rot"]) if s.get("pred_rot") is not None else None,
            # model input -- without this the kinematics cannot be recomputed
            "hist_xy": _flat(s["hist_xy"]),
            "hist_rot": _flat(s["hist_rot"]) if s.get("hist_rot") is not None else None,
            # reasoning
            "cot": s.get("cot"),
            "meta_action": s.get("meta_action"),
            "token_ids": (
                np.asarray(s["token_ids"], dtype=np.int32).tolist()
                if s.get("token_ids") is not None
                else None
            ),
            "token_logprob": (
                _flat(s["token_logprob"]) if s.get("token_logprob") is not None else None
            ),
            "token_entropy": (
                _flat(s["token_entropy"]) if s.get("token_entropy") is not None else None
            ),
            "n_generated_tokens": s.get("n_generated_tokens"),
            "n_cot_tokens": s.get("n_cot_tokens"),
            "eos_missing": s.get("eos_missing"),
            "prompt_len": s.get("prompt_len"),
            # timing. The six t_* values sum to t_total_ms; t_other_ms is what
            # the named segments leave over, so the sum is checkable.
            "t_vision_ms": s.get("t_vision_ms"),
            "t_prefill_ms": s.get("t_prefill_ms"),
            "t_decode_ms": s.get("t_decode_ms"),
            "t_postgen_ms": s.get("t_postgen_ms"),
            "t_expert_ms": s.get("t_expert_ms"),
            "t_other_ms": s.get("t_other_ms"),
            "t_total_ms": s.get("t_total_ms"),
            # Decode's own denominator. Per-row token counts are not it: the
            # batch decodes until the last row finishes.
            "n_decode_steps": s.get("n_decode_steps"),
            "n_vision_calls": s.get("n_vision_calls"),
            # The number of diffusion Euler steps that actually executed. The
            # --inference-step parameter records None when unset, and the
            # sampler then silently uses the checkpoint's own default, so this
            # is the only place the executed count is visible.
            "n_expert_calls": s.get("n_expert_calls"),
            # False -> the t_* values above are absent, not zero.
            "timing_measured": s.get("timing_measured"),
        }
        row.update(config.get("columns", {}))
        rows.append(row)
    return pd.DataFrame(rows)


#: Per-clip values that are *not* raw model output and so are excluded from
#: per_clip.parquet: the trajectory arrays, the ground truth, and the frames.
_PER_CLIP_DROP = frozenset({"pred_xy", "gt_xy", "data"})


def write_per_clip(
    out_dir: Path, per_clip: list[dict], config: dict | None = None
) -> Path | None:
    """One row per clip: its situation label and every metric computed for it.

    This is what makes a breakdown possible at all. ``predictions.parquet``
    holds raw model output by design, so min_ade, the scene label and the rest
    are nowhere in it -- without this file the only way to ask "how did the
    straight clips do" is to re-fetch the gated dataset and recompute.

    Deliberately derived rather than raw, which contradicts the rule the
    parquet follows. The justification is different: these are cheap to
    recompute *if you still have the inputs*, and the inputs are gated. A few
    kilobytes here buys every pivot anyone will want later, and the file names
    its schema_version so a stale definition is visible rather than silent.

    ``config`` stamps run identity onto every row, exactly as
    :func:`build_rows` does, and for the same reason: **a sweep writes one of
    these files per run, and without identity they concatenate into a pile with
    no axis to group on.** The pivot this file promises is a comparison across
    runs, so the run has to be a column. ``t0_us`` comes from the row rather
    than from ``config["columns"]`` because it varies per clip once anyone
    sweeps the sample timestamp -- putting it in the config would silently
    overwrite the real value with a single scalar.
    """
    identity: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    columns: dict[str, Any] = {}
    if config is not None:
        identity["run_id"] = config.get("run_id")
        identity["variant"] = config.get("variant")
        identity["git_commit"] = config.get("git_commit")
        columns = dict(config.get("columns", {}))

    rows = []
    for clip in per_clip:
        if not clip.get("clip_id"):
            continue
        row = dict(identity)
        row.update({k: v for k, v in clip.items() if k not in _PER_CLIP_DROP})
        # Same precedence as build_rows: config wins a name collision, so the
        # two files can never disagree about what the run was.
        row.update(columns)
        rows.append(row)
    if not rows:
        return None

    path = Path(out_dir) / "per_clip.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    return path


def write_timing(out_dir: Path, rows: list[dict]) -> Path | None:
    """One row per inference pass, typed by the registry in ``timing_schema``.

    Written through pyarrow with the explicit schema rather than through
    pandas: pandas infers a column that is null in every row as null-typed, and
    a null-typed column refuses to concatenate with the same column typed in
    another run -- which is exactly the situation of an instrument that was off
    in one run and on in the next.
    """
    if not rows:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    unknown = sorted({k for r in rows for k in TS.unknown_keys(r)})
    if unknown:
        # Dropped rather than raised: a typo in instrumentation must not cost a
        # run. The tests build rows strictly, so it cannot reach a commit.
        print(f"[timing] undeclared columns dropped: {', '.join(unknown)}")
    table = pa.Table.from_pylist([TS.coerce_row(r) for r in rows], schema=TS.pa_schema())
    path = Path(out_dir) / "timing.parquet"
    pq.write_table(table, path, compression="zstd")
    return path


def write_thermal(out_dir: Path, thermal: Any, run_id: str | None = None) -> Path | None:
    """The board sampler's full series, long format: one row per sensor reading.

    Kept whole rather than summarised, because the questions asked of it come
    later: which clip ran hot, whether the clock sagged inside one segment,
    when the over-current counter moved. Every row carries the perf_counter
    stamp the tracer also uses, the clip index and the phase of the clip.
    """
    table = thermal.table() if thermal is not None else None
    if not table or not table.get("tick"):
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    from alpamayo1_5.trace import thermal as TH

    n = len(table["tick"])
    schema = pa.schema([
        ("thermal_schema_version", pa.int16()), ("run_id", pa.string()),
        ("tick", pa.int32()), ("t_host_s", pa.float64()), ("t_unix_s", pa.float64()),
        ("read_ms", pa.float32()), ("clip_index", pa.int32()), ("phase", pa.string()),
        ("sensor", pa.string()), ("value", pa.float64()), ("unit", pa.string()),
    ])
    columns = {"thermal_schema_version": [TH.THERMAL_SCHEMA_VERSION] * n,
               "run_id": [run_id] * n, **table}
    path = Path(out_dir) / "thermal.parquet"
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), path, compression="zstd")
    return path


_LAYER_COLUMNS = (
    ("layers_schema_version", "i16"), ("run_id", "s"), ("clip_id", "s"), ("clip_index", "i32"),
    ("row_kind", "s"), ("pass_index", "i16"), ("timing_schema_version", "i16"),
    ("tracer_version", "i16"), ("stack", "s"), ("phase", "s"), ("call_index", "i16"),
    ("layer", "i16"), ("part", "s"), ("device_ms", "f32"), ("host_ms", "f32"),
)


def write_layers(out_dir: Path, rows: list[dict]) -> Path | None:
    """Every layer span of the main passes of a trace-level-layer run, long format.

    Kept whole because the question asked of it is a shape -- which layers,
    which part, whether decode steps drift as the cache grows -- and a sum
    per phase in the timing row cannot answer it. ``(clip_id, row_kind,
    pass_index)`` joins a span to its timing row.
    """
    return _write_long(Path(out_dir) / "layers.parquet", _LAYER_COLUMNS,
                       TS.LAYERS_SCHEMA_VERSION, rows)


_KERNEL_COLUMNS = (
    ("kernels_schema_version", "i16"), ("run_id", "s"), ("clip_id", "s"), ("clip_index", "i32"),
    ("row_kind", "s"), ("pass_index", "i16"), ("timing_schema_version", "i16"),
    ("tracer_version", "i16"), ("kind", "s"), ("segment", "s"), ("step_index", "i16"),
    ("name", "s"), ("op", "s"), ("category", "s"), ("stream", "i32"), ("start_us", "f64"),
    ("dur_us", "f32"), ("launch_us", "f32"), ("lead_us", "f32"), ("grid_size", "i64"),
    ("block_size", "i32"), ("regs_per_thread", "i16"), ("smem_bytes", "i32"),
)


def write_kernels(out_dir: Path, rows: list[dict]) -> Path | None:
    """Every device event of the profile passes, long format.

    ``start_us`` is from the call's start, so gaps can be recomputed; the
    segment is where the launch happened and the category comes from the
    launching op (profile_parse). ``(clip_id, row_kind, pass_index)`` joins a
    kernel to its timing row.
    """
    return _write_long(Path(out_dir) / "kernels.parquet", _KERNEL_COLUMNS,
                       TS.KERNELS_SCHEMA_VERSION, rows)


def _write_long(path: Path, columns: tuple[tuple[str, str], ...], version: int,
                rows: list[dict]) -> Path | None:
    """A long table with an explicit schema; its first column is its version."""
    if not rows:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    types = {"i16": pa.int16(), "i32": pa.int32(), "i64": pa.int64(), "f32": pa.float32(),
             "f64": pa.float64(), "s": pa.string()}
    schema = pa.schema([(name, types[kind]) for name, kind in columns])
    first, rest = columns[0][0], [name for name, _ in columns[1:]]
    table = pa.Table.from_pylist([{first: version, **{k: r.get(k) for k in rest}} for r in rows],
                                 schema=schema)
    pq.write_table(table, path, compression="zstd")
    return path


def write_run(
    out_dir: Path,
    samples: list[dict],
    config: dict,
    meta: dict,
    gt: list[dict] | None = None,
    per_clip: list[dict] | None = None,
    timing: list[dict] | None = None,
    thermal: Any = None,
    layers: list[dict] | None = None,
    kernels: list[dict] | None = None,
) -> Path:
    """Write predictions/per_clip/timing parquet, run.json and (locally) gt.parquet.

    ``meta`` carries what an offline reader needs and cannot derive:
    action-space normalization constants (they differ per checkpoint, and using
    the wrong ones makes recomputed kinematics quietly wrong), the special-token
    id map and vocab layout (needed to re-segment the token stream), and the
    prompt length (needed to slice generated tokens off the padded sequence).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = build_rows(samples, config)
    frame.to_parquet(out_dir / "predictions.parquet", index=False, compression="zstd")
    if per_clip:
        write_per_clip(out_dir, per_clip, config)
    if timing:
        write_timing(out_dir, timing)
    if thermal is not None:
        write_thermal(out_dir, thermal, config.get("run_id"))
    if layers:
        write_layers(out_dir, layers)
    if kernels:
        write_kernels(out_dir, kernels)

    payload = dict(meta)
    payload["schema_version"] = SCHEMA_VERSION
    if timing:
        payload["timing"] = {
            **dict(payload.get("timing", {})),
            "schema_version": TS.TIMING_SCHEMA_VERSION,
            "tracer_version": TS.TRACER_VERSION,
            "n_rows": len(timing),
        }
    payload["array_shapes"] = {k: list(v) for k, v in _ARRAY_SHAPES.items()}
    payload["n_rows"] = int(len(frame))
    payload["columns"] = list(frame.columns)
    (out_dir / "run.json").write_text(json.dumps(payload, indent=2, default=str))

    if gt:
        # Local only. Recoverable from (clip_id, t0_us), and keeping gated
        # dataset content out of the shared repo costs nothing here.
        pd.DataFrame(
            [{"clip_id": g["clip_id"], "t0_us": int(g["t0_us"]), "gt_xy": _flat(g["gt_xy"])}
             for g in gt]
        ).to_parquet(out_dir / "gt.parquet", index=False, compression="zstd")
    return out_dir


def upload_paths(out_dir: Path) -> list[Path]:
    """Files that go to Hugging Face. ``gt.parquet`` is deliberately absent."""
    out_dir = Path(out_dir)
    paths = [out_dir / "predictions.parquet", out_dir / "per_clip.parquet",
             out_dir / "timing.parquet", out_dir / "thermal.parquet",
             out_dir / "layers.parquet", out_dir / "kernels.parquet", out_dir / "run.json"]
    samples = out_dir / "samples"
    if samples.is_dir():
        paths.extend(sorted(samples.glob("*.png")))
    return [p for p in paths if p.exists()]


def load_runs(root: Path | str, pattern: str = "*/predictions.parquet") -> pd.DataFrame:
    """Concatenate every run under ``root`` into one frame.

    This is the payoff of the fixed schema: many runs become one table that can
    be grouped by ``variant``. A schema mismatch raises here rather than
    producing a silently misaligned comparison.
    """
    files = sorted(Path(root).glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    versions = {int(f["schema_version"].iloc[0]) for f in frames if len(f)}
    if len(versions) > 1:
        raise ValueError(f"schema versions differ across runs: {sorted(versions)}")
    return pd.concat(frames, ignore_index=True)


def load_timing(root: Path | str, pattern: str = "*/timing.parquet") -> pd.DataFrame:
    """Concatenate every run's timing table under ``root``.

    Refuses to mix table versions, for the same reason :func:`load_runs` does.
    Whether two runs may be *compared* on latency is a stricter question --
    same machine, power mode, driver and tracer -- and is answered by the
    comparison gate, not here.
    """
    files = sorted(Path(root).glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    versions = {int(f["timing_schema_version"].dropna().iloc[0])
                for f in frames if len(f) and f["timing_schema_version"].notna().any()}
    if len(versions) > 1:
        raise ValueError(f"timing schema versions differ across runs: {sorted(versions)}")
    return pd.concat(frames, ignore_index=True)
