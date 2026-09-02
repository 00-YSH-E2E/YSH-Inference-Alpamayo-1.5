#!/usr/bin/env python
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

"""Turn a directory of finished runs into the tables that compare them.

    python scripts/compare_sweep.py --runs-root /workspace/runs \\
        --match 'Alpamayo-1.5_Cam-4_Vanilla_1181clip_*' --baseline 10

Reads only; the run directories are never modified. Output goes to
``<runs-root>/_analysis/<name>_<date>_<hash>/`` -- under the runs root for the
same reason ``run.sh`` puts runs there, outside the repo so results are not
commit candidates. The leading underscore is what stops a second invocation
from discovering the first one's output as an arm.

The output directory is named by a hash of the inputs, so re-running the same
analysis overwrites its own results instead of accumulating near-duplicates,
while changing any parameter lands somewhere new.

Everything statistical lives in ``trace.compare``, which imports numpy and
pandas and nothing else. This file is argument parsing, preflight and file
writing. The split is not tidiness: CI installs four packages and runs the
tests, so anything holding arithmetic has to stay on the importable side.

The preflight follows ``run.sh``: three states, every problem collected before
anything is reported, and a refusal ends with the fact that nothing was
written yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5.trace import compare as C   # noqa: E402

OK, WARN, STOP = "OK  ", "note", "STOP"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="/workspace/runs",
                   help="Directory holding run directories (run.sh's OUT_ROOT).")
    p.add_argument("--match", default="*", help="Glob over run directory names.")
    p.add_argument("--name", default="sweep", help="Name for the output directory.")
    p.add_argument("--axis", default="inference_step",
                   help="per_clip column separating the arms.")
    p.add_argument("--baseline", type=int, default=None,
                   help="Axis value everything is compared against. "
                        "Defaults to the largest.")
    p.add_argument("--metrics", default=",".join(C.DEFAULT_METRICS))
    p.add_argument("--strata", default="scene,speed_profile",
                   help="Overlapping axes, kept separate in the tables.")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-stratum-clips", type=int, default=20)
    p.add_argument("--allow-partial", action="store_true",
                   help="Compare the clips the arms share, instead of refusing "
                        "when they differ. An arm that died on the hard clips "
                        "scores best on what it finished, so this is a "
                        "deliberate choice, not a convenience.")
    p.add_argument("--out", default=None, help="Override the output directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="Preflight and the gate, then stop before the bootstrap.")
    return p.parse_args()


def output_dir(args: argparse.Namespace, run_dirs: list[str]) -> Path:
    if args.out:
        return Path(args.out)
    key = json.dumps({"runs": sorted(run_dirs), "metrics": args.metrics,
                      "strata": args.strata, "n_boot": args.n_boot,
                      "seed": args.seed, "floor": args.min_stratum_clips,
                      "baseline": args.baseline, "partial": args.allow_partial},
                     sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    stamp = date.today().strftime("%y.%m.%d")
    return Path(args.runs_root) / "_analysis" / f"{args.name}_{stamp}_{digest}"


def preflight(args: argparse.Namespace, runs: pd.DataFrame,
              out: Path) -> list[tuple[str, str]]:
    """Everything checkable before reading a parquet. Collects, never exits."""
    notes: list[tuple[str, str]] = []
    root = Path(args.runs_root)

    if not root.is_dir():
        notes.append((STOP, f"runs root does not exist: {root}"))
        return notes

    usable = runs[runs["has_per_clip"]] if len(runs) else runs
    if not len(runs):
        notes.append((STOP, f"no run directory under {root} matches {args.match!r}"))
    elif len(usable) < 2:
        listing = "\n         ".join(
            f"{r.run_dir}  per_clip={'yes' if r.has_per_clip else 'NO'}"
            f"  predictions={'yes' if r.has_predictions else 'NO'}"
            for r in runs.itertuples())
        notes.append((STOP, f"{len(usable)} of {len(runs)} matched directories are "
                            f"usable; two arms are the minimum.\n         {listing}"))
    else:
        notes.append((OK, f"{len(usable)} usable run"
                          f"{'s' if len(usable) != 1 else ''} of {len(runs)} matched"))
        dead = runs[~runs["has_per_clip"]]
        if len(dead):
            notes.append((WARN, f"{len(dead)} matched directory(ies) have no "
                                "per_clip.parquet and are excluded -- an "
                                "interrupted run, not a missing arm: "
                                + ", ".join(dead["run_dir"])))

    missing_pred = usable[~usable["has_predictions"]] if len(usable) else usable
    if len(missing_pred):
        notes.append((WARN, "no predictions.parquet for "
                            + ", ".join(missing_pred["run_dir"])
                            + " -- the step counts and the shared CoT cannot be "
                              "verified for those arms"))

    for parent in [out, *out.parents]:
        if (parent / ".git").exists():
            notes.append((STOP, f"output would land inside a git repository "
                                f"({parent}). Results are not commit candidates; "
                                f"leave --out under {root}."))
            break
    else:
        notes.append((OK, f"output: {out}"
                          + ("  (exists, will be overwritten)" if out.exists() else "")))

    try:
        import pyarrow                                            # noqa: F401
        notes.append((OK, "pyarrow present"))
    except ImportError:
        notes.append((STOP, "pyarrow is not installed; the run directory is parquet"))

    return notes


def show(notes: list[tuple[str, str]]) -> bool:
    print("\n-- preflight " + "-" * 58)
    for status, message in notes:
        print(f"{status}  {message}")
    blocked = any(s == STOP for s, _ in notes)
    if blocked:
        print("\nNothing has been written.")
    print("-" * 71)
    return blocked


def report(tables: dict, gate_table: pd.DataFrame) -> None:
    """The half-page a person actually reads. The parquet holds the rest."""
    metrics = tables["metrics"]
    overall = metrics[metrics["axis"] == "all"]
    print("\n-- headline (all clips) " + "-" * 47)
    for metric in ("min_ade", "mean_ade", "diversity_final_m"):
        rows = overall[overall["metric"] == metric]
        if not len(rows):
            continue
        arrow = {"lower": "lower is better", "higher": "higher is better",
                 "neutral": "no good direction"}[rows["direction"].iloc[0]]
        print(f"  {metric}  ({arrow})")
        for r in rows.itertuples():
            ci = f"[{r.delta_lo:+6.1f}, {r.delta_hi:+6.1f}]"
            win = (f"{int(r.n_better):5d} better / {int(r.n_worse):5d} worse"
                   if pd.notna(r.n_better)
                   else f"{r.n_lower:5d} lower  / {r.n_higher:5d} higher")
            print(f"    {r.arm:<5} {r.mean:8.4f}  {r.delta_pct:+7.2f}%  {ci}  {win}")

    div = tables["divergence"]
    if len(div):
        print("\n-- divergence: delta%(minADE) - delta%(meanADE) " + "-" * 24)
        print("   positive = losing coverage while the average improves")
        for axis in ["all"] + sorted(set(div["axis"]) - {"all"}):
            block = div[div["axis"] == axis]
            for r in block.itertuples():
                print(f"  {r.axis:<14}{r.stratum:<12}n={r.n:<5}{r.arm:<5}"
                      f"{r.divergence_pct:+7.2f}%  [{r.lo:+6.1f}, {r.hi:+6.1f}]"
                      f"   naive [{r.naive_lo:+6.1f}, {r.naive_hi:+6.1f}]")

    order = tables["order"]
    if len(order):
        print("\n-- does the divergence follow the scene? " + "-" * 31)
        print("   P(a diverges more than b), one resample shared by both")
        for r in order.itertuples():
            print(f"  {r.axis:<14}{r.arm:<5}{r.stratum_a:<12}({r.divergence_a:+6.1f}%)"
                  f"  vs {r.stratum_b:<12}({r.divergence_b:+6.1f}%)"
                  f"   P = {r.p_a_greater:.3f}")

    if tables["dropped"]:
        print("\n-- strata below the floor, excluded " + "-" * 36)
        for d in tables["dropped"]:
            print(f"  {d['axis']}={d['stratum']}  n={d['n']} < {d['min_clips']}")

    warned = gate_table[gate_table["status"] == "warn"]
    if len(warned):
        print(f"\n{len(warned)} gate warning(s) -- see gate.parquet")


def main() -> int:
    args = parse_args()
    runs = C.discover_runs(args.runs_root, args.match)
    out = output_dir(args, list(runs[runs["has_per_clip"]]["run_dir"]) if len(runs) else [])
    if show(preflight(args, runs, out)):
        return 2

    per_clip = C.load_per_clip(runs, axis=args.axis)
    arms = C.arm_order(per_clip, baseline=args.baseline)
    gate_table = C.gate(per_clip, runs, baseline=args.baseline,
                        allow_partial=args.allow_partial)

    print("\n-- gate " + "-" * 63)
    for r in gate_table.itertuples():
        marker = {"ok": OK, "warn": WARN, "fail": STOP}[r.status]
        print(f"{marker}  {r.check:<16}{r.detail}")
    print("-" * 71)

    if (gate_table["status"] == "fail").any():
        out.mkdir(parents=True, exist_ok=True)
        gate_table.to_parquet(out / "gate.parquet", index=False)
        print(f"\nRefused. The gate table is at {out / 'gate.parquet'};\n"
              "no comparison was computed, because every check above fails in a\n"
              "way that still produces numbers.")
        return 1

    if args.dry_run:
        print("\n--dry-run: stopping before the bootstrap. Nothing written.")
        return 0

    metrics = [m for m in args.metrics.split(",") if m in per_clip.columns]
    X, index = C.paired_matrix(per_clip, metrics=metrics, arms=arms,
                               strata=tuple(args.strata.split(",")))
    tables = C.build_tables(X, index, arms, metrics, n_boot=args.n_boot,
                            seed=args.seed, min_stratum_clips=args.min_stratum_clips,
                            strata=tuple(args.strata.split(",")))

    out.mkdir(parents=True, exist_ok=True)
    gate_table.to_parquet(out / "gate.parquet", index=False)
    for name in ("metrics", "divergence", "order"):
        if len(tables[name]):
            tables[name].to_parquet(out / f"{name}.parquet", index=False)
    (out / "analysis.json").write_text(json.dumps({
        "arms": arms, "baseline": arms[0], "metrics": metrics,
        "n_clips": int(X.shape[0]), "n_boot": args.n_boot, "seed": args.seed,
        "min_stratum_clips": args.min_stratum_clips,
        "strata": tables["strata"], "dropped_strata": tables["dropped"],
        "runs": runs[runs["has_per_clip"]][["run_dir", "run_id"]].to_dict("records"),
        "schema_version": int(per_clip["schema_version"].iloc[0]),
    }, indent=2, ensure_ascii=False))

    report(tables, gate_table)
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
