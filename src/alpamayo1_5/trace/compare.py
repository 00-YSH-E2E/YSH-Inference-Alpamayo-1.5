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

"""Comparing runs to each other -- the half the writer left undone.

:mod:`writer` records one run exactly. It does not compare two, and
``load_runs`` is not that comparison: it concatenates and stops, which is the
easy third of the job. The hard two thirds are deciding **whether** two runs
may be compared at all, and computing differences in a way that survives the
question "is that bigger than the noise".

The shape of the answer here is a **paired** comparison. Every arm sees the
same clips, at the same seed, with the same sampling parameters -- so a clip's
own difficulty cancels, and what is left is the effect of the axis. That is
worth a great deal statistically, and it is fragile: one arm run on a different
clip list, or with a different seed, silently converts a paired comparison into
an unpaired one that still produces numbers. Hence :func:`gate`, which runs
before any arithmetic and refuses rather than warns.

Three things live here that a reader might expect to find in the plotting code,
because they are arithmetic and arithmetic is testable:

* :func:`shared_limits` -- axis ranges. A per-panel autoscale is the standard
  way to erase a collapse: the spread shrinks by 5x, the axis shrinks with it,
  and every panel looks identical.
* :func:`spread_is_degenerate` -- whether a distribution still has width. A KDE
  that silently skips a zero-variance arm removes exactly the arm the figure
  is about.
* :func:`arm_colors` / :func:`arm_markers` -- stable per-arm styling, so the
  same arm is the same colour in every figure.

The dividing line is not "statistics versus pictures". It is that a wrong
artist call produces a visibly wrong figure, while a wrong axis limit produces
an invisibly wrong one -- and everything in the second category is arithmetic.

**No torch, and no import of** :mod:`~alpamayo1_5.trace.metrics`, which imports
torch at module scope. CI installs numpy, pandas, pyarrow and pytest and
nothing else, so anything importing torch from here would take the whole
module out of the tested set. :mod:`writer` is safe and is imported for
``SCHEMA_VERSION``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import writer as W

# Files a finished run leaves behind. ``gt.parquet`` is local-only and
# ``samples/`` is optional, so neither is required for an arm to be usable.
_RUN_FILES = {
    "per_clip": "per_clip.parquet",
    "predictions": "predictions.parquet",
    "run_json": "run.json",
    "gt": "gt.parquet",
}

# Config columns that must agree across arms for the pairing to mean anything.
# `machine` is deliberately absent: it changes latency, not trajectories, and
# an arm re-run on another box is still a valid arm for accuracy comparison.
PAIRING_CONFIG = (
    "seed", "num_traj_samples", "temperature", "top_p", "model",
    "attn_impl", "data_spec", "dtype", "max_new_tokens", "variant",
    "conditioning_source",
)

# What identifies a clip across arms. `t0_us` is a constant today -- one sample
# per clip -- but it is part of the key so that the day a sweep varies it, the
# pairing does not silently collapse two rows into one.
CLIP_KEY = ("clip_id", "t0_us")

# The metrics this module knows how to compare. Anything else can still be
# passed explicitly; this is the default set and the order figures use.
DEFAULT_METRICS = (
    "min_ade", "mean_ade", "min_fde", "mean_fde",
    "diversity_final_m", "diversity_mean_m", "sample_gain",
)

# Columns that only exist when K >= 2. `metrics.diversity` returns {} for a
# single sample, so these are absent -- not zero -- in a K=1 run.
DIVERSITY_COLUMNS = ("diversity_mean_m", "diversity_final_m", "diversity_max_m")


class GateError(RuntimeError):
    """A comparison was refused. Carries the gate table for reporting."""

    def __init__(self, message: str, table: pd.DataFrame) -> None:
        super().__init__(message)
        self.table = table


# --------------------------------------------------------------------------
#  Discovery
# --------------------------------------------------------------------------

def _read_run_json(path: Path) -> dict[str, Any]:
    """``run.json`` as a dict, or ``{}`` if it is missing or unreadable.

    Unreadable is treated as missing on purpose. A run interrupted mid-write
    leaves a truncated file, and the useful outcome is a row saying the run is
    incomplete -- not a traceback that hides the eleven runs after it.
    """
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def discover_runs(root: Path | str, match: str = "*") -> pd.DataFrame:
    """One row per run directory under ``root``: what it has and what it lacks.

    This exists instead of :func:`writer.load_runs` for one reason that matters
    more than the rest: **an interrupted run must be visible.** ``load_runs``
    globs for ``*/predictions.parquet``, so a run that died before writing one
    is not an error, not a warning, and not a row -- it is simply absent, and
    the arm it was going to be quietly drops out of the comparison. A sweep
    that loses its baseline that way still produces a full set of tables, all
    of them measured against the wrong arm.

    So the contract here is the opposite: every directory becomes a row, and
    ``has_per_clip`` says whether it is usable. The caller decides.

    Directories whose name starts with ``_`` or ``.`` are skipped. That is not
    tidiness -- the analysis output goes to ``_analysis/`` under the same root,
    and without this a second run of the tool discovers its own output as an
    arm.

    Args:
        root: Directory holding run directories -- ``OUT_ROOT`` from ``run.sh``.
        match: Glob applied to directory names, for selecting one sweep out of
            a root holding several.

    Returns:
        A frame with one row per directory, sorted by name. Empty (with the
        right columns, so callers can filter without a special case) when
        nothing matches.
    """
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(match)):
        if not path.is_dir() or path.name[:1] in {"_", "."}:
            continue
        present = {k: (path / f).exists() for k, f in _RUN_FILES.items()}
        meta = _read_run_json(path / _RUN_FILES["run_json"])
        params = meta.get("params", {}) or {}
        rows.append({
            "run_dir": path.name,
            "path": str(path),
            **{f"has_{k}": v for k, v in present.items()},
            "run_id": meta.get("run_id"),
            "variant": meta.get("variant"),
            "machine": meta.get("machine"),
            "date": meta.get("date"),
            "schema_version": meta.get("schema_version"),
            "step_declared": params.get("inference_step"),
            "seed": params.get("seed"),
            "num_traj_samples": params.get("num_traj_samples"),
            "temperature": params.get("temperature"),
            "n_clips_declared": params.get("n_clips"),
            "n_clips_listed": len(meta.get("clips") or []),
        })
    columns = [
        "run_dir", "path", "has_per_clip", "has_predictions", "has_run_json",
        "has_gt", "run_id", "variant", "machine", "date", "schema_version",
        "step_declared", "seed", "num_traj_samples", "temperature",
        "n_clips_declared", "n_clips_listed",
    ]
    return pd.DataFrame(rows, columns=columns)


def executed_steps(runs: pd.DataFrame) -> pd.DataFrame:
    """How many expert calls each run actually made, read from predictions.

    This is the only honest record of what ran. ``flow_matching.py`` resolves
    its step count as ``inference_step or self.num_inference_steps``, so a
    falsy value -- ``None``, ``0``, an empty string from an unset shell
    variable -- does not fail and does not warn: it runs the default ten. An
    arm labelled ``s1`` in that state produces a beautiful null result, and
    nothing else in the pipeline contradicts it.

    Reads two columns out of forty-two. ``pred_xy`` and ``pred_rot`` are most
    of the file's bytes and are not touched.

    Returns:
        One row per run with ``n_expert_calls_min`` / ``_max`` (equal unless
        something varied within the run, which is itself a finding) and
        ``n_rows``. Runs without ``predictions.parquet`` get NaN rather than
        being dropped.
    """
    rows: list[dict[str, Any]] = []
    for r in runs.itertuples():
        rec: dict[str, Any] = {"run_dir": r.run_dir, "n_expert_calls_min": np.nan,
                               "n_expert_calls_max": np.nan, "n_rows": 0}
        if getattr(r, "has_predictions", False):
            try:
                df = pd.read_parquet(Path(r.path) / _RUN_FILES["predictions"],
                                     columns=["n_expert_calls"])
            except (OSError, ValueError, KeyError):
                df = pd.DataFrame()
            if len(df):
                calls = pd.to_numeric(df["n_expert_calls"], errors="coerce").dropna()
                if len(calls):
                    rec["n_expert_calls_min"] = float(calls.min())
                    rec["n_expert_calls_max"] = float(calls.max())
                rec["n_rows"] = int(len(df))
        rows.append(rec)
    return pd.DataFrame(rows, columns=["run_dir", "n_expert_calls_min",
                                       "n_expert_calls_max", "n_rows"])


# --------------------------------------------------------------------------
#  Loading
# --------------------------------------------------------------------------

def load_per_clip(
    runs: pd.DataFrame,
    axis: str = "inference_step",
    executed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Concatenate the usable runs, with the arm axis resolved and non-null.

    Four columns are added to what the parquet already carries:

    ``run_dir``      which directory the row came from -- ``load_runs`` drops
                     this, and without it two arms with the same declared step
                     are indistinguishable.
    ``step``         the axis value as a nullable ``Int64``.
    ``step_source``  ``"declared"`` if the run recorded it, ``"executed"`` if
                     it had to be recovered from ``n_expert_calls``.
    ``arm``          display name, ``s<step>``.

    The null-axis case is not hypothetical and not benign. A run whose
    ``inference_step`` column is ``None`` (object dtype) breaks two things at
    once: ``sorted(set(steps))`` raises ``TypeError`` comparing ``int`` to
    ``NoneType``, and -- worse, because it is silent -- ``groupby("step")``
    drops the arm entirely, since pandas excludes NaN keys by default. Both
    were observed on disk. The fix is not to drop the run but to recover the
    value from what actually executed, and to say so in ``step_source``.

    Args:
        runs: Output of :func:`discover_runs`. Rows with ``has_per_clip``
            false are skipped -- they have nothing to load.
        axis: Column holding the comparison axis.
        executed: Output of :func:`executed_steps`, if already computed.
            Otherwise it is computed for the runs that need it, and only those.

    Raises:
        ValueError: No usable run; or a run whose axis value cannot be
            established even from ``n_expert_calls``. Guessing there would mean
            inventing the label the entire comparison is grouped by.
    """
    usable = runs[runs["has_per_clip"]] if len(runs) else runs
    if not len(usable):
        raise ValueError(
            "no run directory has per_clip.parquet -- nothing to compare.\n"
            f"    looked at {len(runs)} director{'y' if len(runs) == 1 else 'ies'}"
            + ("" if not len(runs) else ":\n      " + "\n      ".join(
                f"{r.run_dir}  (per_clip {'yes' if r.has_per_clip else 'no'},"
                f" predictions {'yes' if r.has_predictions else 'no'})"
                for r in runs.itertuples()))
        )

    frames: list[pd.DataFrame] = []
    need_fallback: list[str] = []
    for r in usable.itertuples():
        df = pd.read_parquet(Path(r.path) / _RUN_FILES["per_clip"])
        df = df.copy()
        df["run_dir"] = r.run_dir
        raw = df[axis] if axis in df.columns else pd.Series([None] * len(df))
        df["step"] = pd.to_numeric(raw, errors="coerce").astype("Int64")
        df["step_source"] = "declared"
        if df["step"].isna().any():
            need_fallback.append(r.run_dir)
        frames.append(df)

    if need_fallback:
        if executed is None:
            executed = executed_steps(usable[usable["run_dir"].isin(need_fallback)])
        by_dir = executed.set_index("run_dir")
        for df in frames:
            name = df["run_dir"].iloc[0]
            if name not in need_fallback or name not in by_dir.index:
                continue
            lo = by_dir.at[name, "n_expert_calls_min"]
            hi = by_dir.at[name, "n_expert_calls_max"]
            if pd.isna(lo) or lo != hi:
                continue
            df["step"] = df["step"].fillna(int(lo))
            df["step_source"] = "executed"

    still_null = [df["run_dir"].iloc[0] for df in frames if df["step"].isna().any()]
    if still_null:
        raise ValueError(
            f"`{axis}` is null and could not be recovered for: {', '.join(still_null)}.\n"
            "    The column is written by the runner; a null one means the shell\n"
            "    variable was unset, in which case the model silently ran its\n"
            "    default step count. Set INFERENCE_STEP in run.sh and re-run --\n"
            "    the value cannot be guessed here without inventing the label\n"
            "    the whole comparison groups by."
        )

    out = pd.concat(frames, ignore_index=True)
    out["arm"] = "s" + out["step"].astype(int).astype(str)
    return out


def arm_order(per_clip: pd.DataFrame, baseline: int | None = None) -> list[str]:
    """Arms with the baseline first, then descending by step.

    Descending because the axis is a reduction: ten steps is the reference and
    one step is the extreme, so reading left to right follows the experiment.
    The baseline is pulled to the front rather than assumed to be the largest,
    since a sweep may compare against something other than its maximum.
    """
    pairs = (per_clip[["arm", "step"]].drop_duplicates()
             .sort_values("step", ascending=False))
    arms = list(pairs["arm"])
    if baseline is not None:
        want = f"s{int(baseline)}"
        if want not in arms:
            raise ValueError(
                f"baseline {want} is not among the arms: {', '.join(arms) or '(none)'}"
            )
        arms.remove(want)
        arms.insert(0, want)
    return arms


# --------------------------------------------------------------------------
#  The pairing gate
# --------------------------------------------------------------------------

def _row(check: str, status: str, detail: str, arms: str = "", n: int = 0) -> dict[str, Any]:
    return {"check": check, "status": status, "detail": detail, "arms": arms, "n": n}


def _hash_sequence(value: Any) -> int:
    """Order-sensitive hash of an array-like cell, for equality across arms."""
    try:
        return hash(tuple(int(v) for v in value))
    except TypeError:
        return hash(value)


def load_token_hashes(runs: pd.DataFrame) -> pd.DataFrame:
    """``(run_dir, clip_id, sample_k)`` -> hash of that sample's token ids.

    Reads three columns of forty-two; the trajectory arrays are two thirds of
    the file and are not touched.
    """
    frames: list[pd.DataFrame] = []
    for r in runs.itertuples():
        if not getattr(r, "has_predictions", False):
            continue
        try:
            df = pd.read_parquet(Path(r.path) / _RUN_FILES["predictions"],
                                 columns=["clip_id", "sample_k", "token_ids"])
        except (OSError, ValueError, KeyError):
            continue
        frames.append(pd.DataFrame({
            "run_dir": r.run_dir,
            "clip_id": df["clip_id"],
            "sample_k": df["sample_k"],
            "token_hash": [_hash_sequence(v) for v in df["token_ids"]],
        }))
    if not frames:
        return pd.DataFrame(columns=["run_dir", "clip_id", "sample_k", "token_hash"])
    return pd.concat(frames, ignore_index=True)


def gate(
    per_clip: pd.DataFrame,
    runs: pd.DataFrame,
    executed: pd.DataFrame | None = None,
    tokens: pd.DataFrame | None = None,
    baseline: int | None = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Everything that must be true before a paired difference means anything.

    One row per check, **including the ones that pass**. A gate that prints
    nothing when it is happy is indistinguishable from a gate that did not
    run, and the whole point of this table is that it goes in the output
    directory next to the numbers it licensed.

    Statuses are ``ok``, ``warn`` and ``fail``. Any ``fail`` should stop the
    caller before the bootstrap: the checks are ordered so that the cheap
    structural ones come first and the expensive cross-file ones last, and a
    failure in the first few makes the rest meaningless anyway.

    The check that earns its keep is ``executed_steps``. ``flow_matching``
    resolves ``inference_step or self.num_inference_steps``, so an arm whose
    step never reached the model runs the default and reports success. Nothing
    else in the pipeline notices, and the resulting table reads as "reducing
    steps costs nothing" -- a confident wrong answer, which is worse than a
    crash. ``n_expert_calls`` is compared as a **ratio** rather than an
    equality, because classifier-free guidance would make it two calls per
    step; a constant factor is fine, a varying one is not.

    ``token_ids`` is checked because the pairing rests on it. ``_euler`` draws
    its noise once before the loop and the loop consumes no randomness, so
    identical seed plus identical token stream implies identical initial
    noise -- which is what makes a per-clip difference attributable to the
    step count. It is compared with a tolerance rather than exactly: bfloat16
    reduction order can flip a near-tie under top-p sampling, and a handful of
    clips diverging that way is a different thing from a systematic mismatch.
    """
    rows: list[dict[str, Any]] = []
    arms = sorted(per_clip["arm"].unique(), key=lambda a: -int(a[1:]))
    by_arm = {a: g for a, g in per_clip.groupby("arm", dropna=False)}

    # 1 -- schema version
    versions = sorted({int(v) for v in per_clip["schema_version"].unique()})
    if len(versions) > 1:
        rows.append(_row("schema_version", "fail",
                         f"runs were written by different schemas: {versions}. "
                         "Columns may not mean the same thing.", ", ".join(arms)))
    else:
        rows.append(_row("schema_version", "ok",
                         f"all arms at v{versions[0]}"
                         + ("" if versions[0] == W.SCHEMA_VERSION
                            else f" (this code expects v{W.SCHEMA_VERSION})"),
                         ", ".join(arms), len(arms)))

    # 2 -- enough arms, distinct, baseline present
    steps = sorted({int(s) for s in per_clip["step"].dropna().unique()})
    if len(arms) < 2:
        rows.append(_row("arms", "fail",
                         f"only {len(arms)} arm ({', '.join(arms) or 'none'}) -- "
                         "nothing to compare it against", ", ".join(arms), len(arms)))
    elif len(steps) != len(arms):
        rows.append(_row("arms", "fail",
                         f"{len(arms)} arms but {len(steps)} distinct steps -- "
                         "two runs share a step and would be averaged together",
                         ", ".join(arms), len(arms)))
    elif baseline is not None and f"s{int(baseline)}" not in arms:
        rows.append(_row("arms", "fail",
                         f"baseline s{int(baseline)} is not present; have {', '.join(arms)}",
                         ", ".join(arms), len(arms)))
    else:
        rows.append(_row("arms", "ok", f"{len(arms)} arms at steps {steps}",
                         ", ".join(arms), len(arms)))

    # 3 -- what actually executed
    if executed is None:
        executed = executed_steps(runs[runs["has_predictions"]])
    dir_to_arm = per_clip[["run_dir", "arm", "step"]].drop_duplicates().set_index("run_dir")
    ratios: dict[str, float] = {}
    varied: list[str] = []
    missing: list[str] = []
    for e in executed.itertuples():
        if e.run_dir not in dir_to_arm.index:
            continue
        a = dir_to_arm.at[e.run_dir, "arm"]
        step = int(dir_to_arm.at[e.run_dir, "step"])
        if pd.isna(e.n_expert_calls_min):
            missing.append(a)
        elif e.n_expert_calls_min != e.n_expert_calls_max:
            varied.append(f"{a}: {e.n_expert_calls_min:g}..{e.n_expert_calls_max:g}")
        else:
            ratios[a] = float(e.n_expert_calls_min) / step
    distinct = sorted(set(round(v, 6) for v in ratios.values()))
    if varied:
        rows.append(_row("executed_steps", "fail",
                         "n_expert_calls is not constant within an arm: "
                         + "; ".join(varied), ", ".join(sorted(set(varied)))))
    elif len(distinct) > 1:
        worst = ", ".join(f"{a}={ratios[a]:g} calls/step" for a in arms if a in ratios)
        rows.append(_row("executed_steps", "fail",
                         "calls-per-step differs across arms, so at least one arm did "
                         f"not run the step count it claims: {worst}. `inference_step "
                         "or num_inference_steps` in flow_matching means a falsy value "
                         "silently runs the default.", ", ".join(sorted(ratios))))
    elif not ratios:
        rows.append(_row("executed_steps", "warn",
                         "no predictions.parquet to read n_expert_calls from -- the "
                         "step counts are taken on trust", ", ".join(missing)))
    else:
        note = f"{distinct[0]:g} expert call(s) per declared step, same in every arm"
        rows.append(_row("executed_steps", "ok" if not missing else "warn",
                         note + ("" if not missing
                                 else f"; unverified for {', '.join(missing)}"),
                         ", ".join(sorted(ratios)), len(ratios)))

    # 4 -- same clips
    sets = {a: set(map(tuple, g[list(CLIP_KEY)].itertuples(index=False, name=None)))
            for a, g in by_arm.items()}
    common = set.intersection(*sets.values()) if sets else set()
    ragged = {a: len(s) - len(common) for a, s in sets.items() if len(s) != len(common)}
    if ragged:
        detail = ("arms do not cover the same clips; "
                  + ", ".join(f"{a} has {n} the others lack" for a, n in ragged.items())
                  + f". {len(common)} clips are common to all.")
        rows.append(_row("clip_sets", "warn" if allow_partial else "fail",
                         detail + (" Restricted to the common set."
                                   if allow_partial else
                                   " An arm that died on the hard clips looks best."),
                         ", ".join(sorted(ragged)), len(common)))
    else:
        rows.append(_row("clip_sets", "ok", f"all arms cover the same {len(common)} clips",
                         ", ".join(arms), len(common)))

    # 5 -- identical sampling configuration
    disagree: list[str] = []
    absent: list[str] = []
    for col in PAIRING_CONFIG:
        if col not in per_clip.columns:
            absent.append(col)
            continue
        vals = {a: sorted({repr(v) for v in g[col].unique()}) for a, g in by_arm.items()}
        flat = {v for vs in vals.values() for v in vs}
        if len(flat) > 1:
            disagree.append(f"{col}: " + ", ".join(f"{a}={'/'.join(v)}"
                                                   for a, v in sorted(vals.items())))
    if disagree:
        rows.append(_row("config", "fail",
                         "arms were not run at the same settings, so a difference "
                         "between them is not attributable to the axis -- "
                         + "; ".join(disagree), ", ".join(arms)))
    else:
        rows.append(_row("config", "ok",
                         f"{len(PAIRING_CONFIG) - len(absent)} settings identical across arms"
                         + (f" ({', '.join(absent)} not in this schema)" if absent else ""),
                         ", ".join(arms), len(PAIRING_CONFIG) - len(absent)))

    # 6 -- strata are a property of the clip, not of the arm
    strat_bad: list[str] = []
    for col in ("scene", "speed_profile"):
        if col not in per_clip.columns:
            continue
        wide = per_clip.pivot_table(index=list(CLIP_KEY), columns="arm", values=col,
                                    aggfunc="first")
        n_diff = int((wide.nunique(axis=1) > 1).sum())
        if n_diff:
            strat_bad.append(f"{col}: {n_diff} clips labelled differently between arms")
    if strat_bad:
        rows.append(_row("strata", "fail",
                         "; ".join(strat_bad) + ". These come from the logged future, "
                         "so they cannot legitimately depend on the arm -- a mid-sweep "
                         "change to the thresholds would re-partition silently.",
                         ", ".join(arms)))
    else:
        rows.append(_row("strata", "ok",
                         "scene and speed_profile agree across arms for every clip",
                         ", ".join(arms), len(common)))

    # 7 -- same code
    commits = sorted({str(c)[:8] for c in per_clip.get("git_commit", pd.Series(dtype=object))
                      .dropna().unique()})
    if len(commits) > 1:
        rows.append(_row("git_commit", "warn",
                         f"arms were run from different commits: {', '.join(commits)}. "
                         "Fine if the diff does not touch inference; not fine otherwise.",
                         ", ".join(arms), len(commits)))
    else:
        rows.append(_row("git_commit", "ok",
                         f"one commit across all arms: {commits[0] if commits else 'unknown'}",
                         ", ".join(arms), 1))

    # 8 -- diversity was measurable everywhere
    no_div = sorted(a for a, g in by_arm.items()
                    if any(c not in g.columns or g[c].isna().all()
                           for c in DIVERSITY_COLUMNS))
    if no_div:
        rows.append(_row("diversity", "warn",
                         f"diversity is absent for {', '.join(no_div)} (K<2 records no "
                         "spread). Those metrics are dropped rather than zero-filled -- "
                         "zero would read as a total collapse that was never measured.",
                         ", ".join(no_div), len(no_div)))
    else:
        rows.append(_row("diversity", "ok", "diversity present in every arm",
                         ", ".join(arms), len(arms)))

    # 9 & 10 -- internal identities, true by construction
    for name, lhs, rhs, why in (
        ("sample_gain", per_clip.get("sample_gain"),
         per_clip.get("mean_ade", pd.Series(dtype=float))
         - per_clip.get("min_ade", pd.Series(dtype=float)),
         "sample_gain is defined as mean_ade - min_ade"),
        ("full_horizon", per_clip.get("ade_6.4s"), per_clip.get("min_ade"),
         "ade_6.4s covers the whole 64-waypoint horizon, so it is minADE"),
    ):
        if lhs is None or rhs is None or not len(lhs):
            rows.append(_row(name, "warn", f"columns missing; cannot check that {why}"))
            continue
        worst = float((lhs - rhs).abs().max())
        rows.append(_row(name, "ok" if worst < 1e-6 else "fail",
                         f"{why}; largest disagreement {worst:.2e}",
                         ", ".join(arms), len(lhs)))

    # 11 -- the same CoT, hence the same initial noise
    if tokens is None:
        tokens = load_token_hashes(runs[runs["has_predictions"]])
    tok = tokens.merge(dir_to_arm.reset_index()[["run_dir", "arm"]], on="run_dir", how="inner")
    if tok["arm"].nunique() < 2:
        rows.append(_row("token_ids", "warn",
                         "fewer than two arms have predictions.parquet; the assumption "
                         "that arms share their initial noise is untested"))
    else:
        wide = tok.pivot_table(index=["clip_id", "sample_k"], columns="arm",
                               values="token_hash", aggfunc="first")
        full = wide.dropna()
        n_diff = int((full.nunique(axis=1) > 1).sum())
        frac = n_diff / max(len(full), 1)
        detail = (f"{n_diff}/{len(full)} sampled CoTs ({frac:.2%}) differ between arms. "
                  "Identical seed and identical token stream is what makes the arms "
                  "share their initial noise; bf16 reduction order can flip a near-tie "
                  "under top-p, so a few are expected and many are not.")
        rows.append(_row("token_ids", "ok" if frac == 0 else
                         ("warn" if frac <= 0.01 else "fail"), detail,
                         ", ".join(arms), n_diff))

    return pd.DataFrame(rows, columns=["check", "status", "detail", "arms", "n"])


# --------------------------------------------------------------------------
#  Statistics
# --------------------------------------------------------------------------

def paired_matrix(
    per_clip: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    arms: Sequence[str] | None = None,
    key: Sequence[str] = CLIP_KEY,
    strata: Sequence[str] = ("scene", "speed_profile"),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Rearrange the long frame into ``X[clip, arm, metric]``.

    The pairing lives in the shape. Once the data is an array whose first axis
    is the clip, every later operation indexes axis 0 and cannot accidentally
    compare arm A's clip 7 with arm B's clip 8 -- there is no join left to get
    wrong. Anything that needs a subset (a stratum, a bootstrap replicate)
    selects along that one axis and stays aligned by construction.

    A row with a NaN anywhere is dropped whole rather than per metric. Dropping
    per metric would give each metric its own denominator, and then a table of
    seven metrics silently describes seven different clip sets.

    Strata are read from the **first** arm only. They are a property of the
    logged future and cannot legitimately differ between arms -- :func:`gate`
    checks exactly that -- so taking them from one arm makes the assumption
    explicit rather than averaging over a disagreement that should not exist.

    Returns:
        ``(X, index)``. ``X`` has shape ``[n_clips, n_arms, n_metrics]``.
        ``index`` has one row per surviving clip, in the same order as axis 0,
        carrying the key columns and the stratum labels.
    """
    arms = list(arms) if arms is not None else arm_order(per_clip)
    key = list(key)
    metrics = list(metrics)

    blocks: list[pd.DataFrame] = []
    for a in arms:
        g = per_clip[per_clip["arm"] == a]
        if g.duplicated(subset=key).any():
            n = int(g.duplicated(subset=key).sum())
            raise ValueError(
                f"arm {a} has {n} duplicate {tuple(key)} rows. One row per clip is "
                "what makes the pairing a pairing; two would be silently averaged."
            )
        missing = [m for m in metrics if m not in g.columns]
        if missing:
            raise ValueError(f"arm {a} has no column(s) {missing}")
        blocks.append(g.set_index(key).sort_index())

    common = blocks[0].index
    for b in blocks[1:]:
        common = common.intersection(b.index)
    common = common.sort_values()

    X = np.stack([b.loc[common, metrics].to_numpy(dtype=float) for b in blocks], axis=1)
    keep = ~np.isnan(X).any(axis=(1, 2))
    X = X[keep]

    index = blocks[0].loc[common].reset_index()[key].loc[keep].reset_index(drop=True)
    for col in strata:
        if col in blocks[0].columns:
            index[col] = blocks[0].loc[common, col].to_numpy()[keep]
    return X, index


def ratio_deltas(means: np.ndarray, base: int = 0) -> np.ndarray:
    """Percent change of each arm against ``base``, as a **ratio of means**.

    Broadcasts over any leading axes, so the point estimate ``[arm, metric]``
    and a bootstrap block ``[replicate, group, arm, metric]`` go through the
    identical arithmetic. That is deliberate: a confidence interval computed by
    a different expression than the estimate it brackets is a bug waiting for
    an argument about which one is right.

    The mean of per-clip ratios would be the wrong statistic here, not merely a
    different one. A clip whose baseline minADE is 0.02 m contributes a
    +2000% ratio that no amount of averaging over the other eleven hundred
    clips can damp, so the summary ends up describing one easy clip.
    """
    ref = np.take(means, [base], axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (means / ref - 1.0)


def divergence(deltas: np.ndarray, i_min: int, i_mean: int) -> np.ndarray:
    """``delta%(minADE) - delta%(meanADE)`` -- positive when the two disagree.

    Positive means the arm got **worse** on minADE while getting **better** on
    meanADE, which is the signature of a sampler that has stopped covering the
    modes and is averaging them instead. Jensen's inequality is why the two can
    move in opposite directions at all: shrinking the samples toward their own
    mean can only reduce the mean distance and can only increase the minimum.

    The operand order is the one thing here that cannot be caught by eye. Swap
    it and every number keeps its magnitude, every stratum keeps its rank, and
    the conclusion reverses.
    """
    return np.take(deltas, i_min, axis=-1) - np.take(deltas, i_mean, axis=-1)


def stratum_masks(
    index: pd.DataFrame,
    columns: Sequence[str] = ("scene", "speed_profile"),
    min_clips: int = 20,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    """Boolean masks over the clip axis: one for ``all``, then one per stratum.

    The two label columns are **overlapping axes, not one partition.** A clip
    that is turning while braking is counted under ``scene=curve`` and under
    ``speed_profile=decel`` both -- ``run_inference_tracked`` says so where it
    assigns them. So the masks carry which axis they came from, and a caller
    that lays ``straight``, ``decel``, ``lane_change`` and ``curve`` along one
    axis of a chart is mixing two partitions and double-counting the overlap.

    Strata below ``min_clips`` are excluded and **returned** rather than
    dropped in silence: a table that omits a stratum without saying so reads
    as a stratum that does not exist.

    Returns:
        ``(labels, masks, dropped)`` -- ``masks`` has shape
        ``[n_groups, n_clips]`` and ``labels[i]`` describes ``masks[i]``.
    """
    labels: list[dict[str, Any]] = [{"axis": "all", "stratum": "all", "n": len(index)}]
    masks: list[np.ndarray] = [np.ones(len(index), dtype=bool)]
    dropped: list[dict[str, Any]] = []
    for col in columns:
        if col not in index.columns:
            continue
        for value in sorted(str(v) for v in index[col].dropna().unique()):
            mask = (index[col].astype(str) == value).to_numpy()
            entry = {"axis": col, "stratum": value, "n": int(mask.sum())}
            if entry["n"] < min_clips:
                dropped.append({**entry, "min_clips": min_clips})
                continue
            labels.append(entry)
            masks.append(mask)
    return labels, np.asarray(masks), dropped


def bootstrap_group_means(
    X: np.ndarray,
    masks: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    chunk: int = 512,
) -> np.ndarray:
    """Bootstrap means for every group, sharing one resample per replicate.

    Returns the **means**, not intervals, with shape
    ``[n_boot, n_groups, n_arms, n_metrics]``. Handing back the replicates is
    the point: any derived quantity -- a percent delta, a divergence, a
    difference between two strata -- has to be computed *inside* the replicate
    to keep its correlations. Subtracting two independently-derived intervals
    ignores that the two quantities move together and inflates the result
    enough to turn a real effect into "not significant", with no symptom.

    One resample per replicate is shared by every group, arm and metric for the
    same reason. Resampling separately per arm would break the pairing that the
    experiment was designed around.

    Implementation note, and it is load-bearing at this size: the obvious
    ``X[idx]`` builds ``[n_boot, n_clips, n_arms, n_metrics]``, which is 2.5 GB
    for 10k replicates over 1181 clips. A bootstrap mean is a multinomially
    weighted mean, exactly, so the whole thing collapses to one matmul per
    group per chunk and the working set stays in the low megabytes.

    Args:
        X: ``[n_clips, n_arms, n_metrics]`` from :func:`paired_matrix`.
        masks: ``[n_groups, n_clips]`` booleans; defaults to a single all-clips
            group.
        n_boot: Replicates.
        seed: Fixed, so a rerun of the analysis reproduces the intervals.
        chunk: Replicates per batch. Trades peak memory against loop overhead.
    """
    n, n_arms, n_metrics = X.shape
    if masks is None:
        masks = np.ones((1, n), dtype=bool)
    masks = np.asarray(masks, dtype=bool)
    flat = X.reshape(n, -1)
    out = np.empty((n_boot, len(masks), n_arms * n_metrics), dtype=float)

    rng = np.random.default_rng(seed)
    probs = np.full(n, 1.0 / n)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        counts = rng.multinomial(n, probs, size=b).astype(float)   # [b, n]
        for g, mask in enumerate(masks):
            w = counts[:, mask]                                    # [b, n_g]
            denom = w.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[done:done + b, g] = (w @ flat[mask]) / denom
        done += b
    return out.reshape(n_boot, len(masks), n_arms, n_metrics)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval over axis 0 of a bootstrap block."""
    lo = 100.0 * (1.0 - level) / 2.0
    return (np.percentile(values, lo, axis=0), np.percentile(values, 100.0 - lo, axis=0))


def _log10_of_int(value: int) -> float:
    """``log10`` of an arbitrarily large integer, without overflowing to inf."""
    if value <= 0:
        return -math.inf
    shift = max(0, value.bit_length() - 512)
    return math.log10(value >> shift) + shift * math.log10(2.0)


def sign_test(baseline: np.ndarray, arm: np.ndarray, tol: float = 0.0) -> dict[str, Any]:
    """Exact two-sided sign test on a paired vector. No scipy on the runner.

    Counts clips where ``arm`` beats ``baseline`` (smaller is better for every
    metric here) against clips where it loses, and asks how surprising that
    split is under a fair coin. It makes no assumption about the shape of the
    per-clip differences, which matters because they are heavily skewed: a few
    clips move by metres while most move by centimetres.

    Ties are **excluded**, not split. They are real and not rare -- when the
    sampler collapses in bfloat16, several samples become bit-identical and the
    difference is exactly zero -- and splitting them halfway would manufacture
    evidence out of the one outcome that carries none.

    ``p_value`` underflows to ``0.0`` below about 1e-308. ``p_log10`` is
    computed from the exact integers and stays meaningful there, so report that
    rather than printing a p of zero.
    """
    d = np.asarray(arm, dtype=float) - np.asarray(baseline, dtype=float)
    d = d[~np.isnan(d)]
    n_better = int((d < -tol).sum())
    n_worse = int((d > tol).sum())
    n_tie = int(len(d) - n_better - n_worse)
    m = n_better + n_worse
    if m == 0:
        return {"n_better": 0, "n_worse": 0, "n_tie": n_tie,
                "p_value": 1.0, "p_log10": 0.0}
    tail = sum(math.comb(m, k) for k in range(0, min(n_better, n_worse) + 1))
    p_log10 = min(0.0, math.log10(2.0) + _log10_of_int(tail) - m * math.log10(2.0))
    return {"n_better": n_better, "n_worse": n_worse, "n_tie": n_tie,
            "p_value": min(1.0, 10.0 ** p_log10) if p_log10 > -300 else 0.0,
            "p_log10": p_log10}
