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

    out/Alpamayo-1.5_Cam-4_Vanilla_26.09.01_39581f9b/
    ├── predictions.parquet   one row per (clip_id, t0_us, sample_k)
    ├── run.json              run-level metadata and the constants needed to
    │                         recompute anything offline
    ├── gt.parquet            logged future -- local only, never uploaded
    └── samples/<clip_id>.png

The trailing eight characters are the MLflow run id, which is what makes the
link bidirectional: a directory names its run, and a run's ``output_uri``
names its directory.

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
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Bump when a column is added, removed or changes meaning. Concatenating runs
# with different schema versions is the failure this exists to make visible.
#
# 2: prompt_len, t_other_ms, n_decode_steps, timing_measured.
#    Version 1 runs are not merely missing these columns -- their prompt_len was
#    written as 0 into run.json by a reader that asked the row for a key the row
#    never carried, so refusing to concatenate v1 with v2 is the right outcome
#    rather than an inconvenience.
SCHEMA_VERSION = 2

# Columns whose values are flat float32 arrays. Stored as lists; the trailing
# shape is recorded in run.json so a reader can reshape without guessing.
_ARRAY_SHAPES = {
    "pred_xy": ("T", 2),
    "pred_rot": ("T", 3, 3),
    "hist_xy": ("N_hist", 2),
    "hist_rot": ("N_hist", 3, 3),
}


def run_dir_name(
    variant: str,
    date: str,
    run_id: str,
    model: str = "Alpamayo-1.5",
    data: str = "Cam-4",
) -> str:
    """Directory name for one run: ``{model}_{data}_{variant}_{date}_{run_id}``.

    The name answers, in order, the three questions asked when looking at an old
    result: which model, on what data, run how -- then when. It matches the
    layout of the data directories (``Alpamayo-1.5_Cam-4_Vanilla``) with the
    date appended.

    Args:
        variant: How this run differs -- ``Vanilla``, ``Pruned-24L``, ``INT8``.
            This is also the ``variant`` column in the parquet, so the directory
            name and the comparison table always agree.
        date: ``YY.MM.DD``.
        run_id: MLflow run id. Only the first 8 characters are used.
        model: Model identity.
        data: What the model was fed -- camera count, sensor set.

    The run id is part of the name for two reasons. Two runs of the same variant
    on the same day would otherwise collide, which is not hypothetical -- the
    same configuration gets re-run constantly while something is being debugged.
    And it makes the link bidirectional: from a directory you can find the
    MLflow run, and from a run you can find its files, without searching.
    """

    def clean(text: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "-" for c in str(text)).strip("-")

    parts = [clean(model), clean(data), clean(variant) or "run", clean(date), run_id[:8]]
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
            # False -> the t_* values above are absent, not zero.
            "timing_measured": s.get("timing_measured"),
        }
        row.update(config.get("columns", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def write_run(
    out_dir: Path,
    samples: list[dict],
    config: dict,
    meta: dict,
    gt: list[dict] | None = None,
) -> Path:
    """Write predictions.parquet, run.json and (locally) gt.parquet.

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

    payload = dict(meta)
    payload["schema_version"] = SCHEMA_VERSION
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
    paths = [out_dir / "predictions.parquet", out_dir / "run.json"]
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
