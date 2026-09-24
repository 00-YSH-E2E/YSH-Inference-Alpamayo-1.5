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

"""Comparing latency between runs: the stricter gate, and the paired arithmetic.

Runs are written with the real writer, so the timing tables here have the
dtypes a real run's have -- nulls included.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from alpamayo1_5.trace import compare as C
from alpamayo1_5.trace import timing_schema as TS
from alpamayo1_5.trace import writer as W

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    "machine": "thor", "power_mode": "MAXN", "nvidia_driver": "580.00", "l4t_release": "R38.4.0",
    "torch_version": "2.13.0", "transformers_version": "4.57.1", "cuda_runtime": "13.0",
    "attn_impl": "sdpa", "dtype": "bfloat16", "model": "nvidia/Alpamayo-1.5-10B",
    "data_spec": "Cam-4", "num_traj_samples": 6, "inference_step": 10, "max_new_tokens": 256,
    "cuda_graph": False, "cuda_graph_max_graphs": None, "trace_level": "basic",
    "sample_hz": 10.0, "warmup": 2, "timing_schema_version": TS.TIMING_SCHEMA_VERSION,
    "tracer_version": TS.TRACER_VERSION,
}


def segments(k: float, i: int) -> dict:
    """A pass whose every segment is linear in K, plus a per-clip offset."""
    seg = {"t_vision_ms": 400.0 * k + 10 * i, "t_prefill_ms": 800.0 * k + 20 * i,
           "t_decode_ms": 3000.0 + 100.0 * k + 50 * i, "t_postgen_ms": 50.0,
           "t_expert_ms": 2000.0 + 50.0 * k, "t_other_ms": 100.0}
    total = sum(seg.values())
    return {**seg, "t_total_ms": total, "t_wall_ms": total + 300.0,
            "t_first_traj_ms": total + 100.0,
            "t_ttft_ms": seg["t_vision_ms"] + seg["t_prefill_ms"] + 20.0,
            "n_decode_steps": 30 + i, "n_expert_calls": 10}


def write_run(root: Path, name: str, n: int = 6, probe: bool = True, per_clip=None,
              **conditions) -> Path:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    cond = {**CONDITIONS, **conditions}
    k = float(cond["num_traj_samples"])
    rows = []
    for i in range(n):
        base = {**cond, "run_id": name, "git_commit": "a" * 40, "clip_id": f"clip{i:02d}",
                "t0_us": 5_100_000, "clip_index": i, "timing_measured": True}
        rows.append({**base, **segments(k, i), "row_kind": "main", "pass_index": 0,
                     **(per_clip(i) if per_clip else {})})
        if probe and i == 0:
            rows.append({**base, **segments(k, i), "row_kind": "probe", "pass_index": 1,
                         "trace_level": "off"})
    W.write_timing(out, rows)
    (out / "run.json").write_text(json.dumps({"run_id": name, "params": {
        "num_traj_samples": cond["num_traj_samples"],
        "inference_step": cond["inference_step"]}}))
    return out


def k_sweep(root: Path, ks=(1, 3, 6), **conditions) -> None:
    for k in ks:
        write_run(root, f"run_K{k}", num_traj_samples=k, **conditions)


@pytest.fixture(autouse=True)
def _needs_parquet():
    pytest.importorskip("pyarrow", reason="the run directory is parquet")


def load(root: Path, axis: str = "num_traj_samples"):
    return C.load_timing(C.discover_runs(root), axis=axis)


def status(table, check):
    return table.set_index("check").at[check, "status"]


def test_discovery_says_which_runs_have_a_timing_table(tmp_path):
    k_sweep(tmp_path, ks=(1,))
    (tmp_path / "old_run").mkdir()
    runs = C.discover_runs(tmp_path).set_index("run_dir")
    assert bool(runs.at["run_K1", "has_timing"]) and not bool(runs.at["old_run", "has_timing"])


def test_a_clean_k_sweep_passes_and_the_line_is_the_cost_per_sample(tmp_path):
    k_sweep(tmp_path)
    timing = load(tmp_path)
    assert len(timing) == 18 and set(timing["arm"]) == {"K1", "K3", "K6"}
    gate = C.timing_gate(timing, axis="num_traj_samples")
    assert (gate["status"] == "ok").all(), gate.to_string()
    arms = C.arm_order(timing, baseline=1)
    tables = C.timing_tables(timing, arms, n_boot=200)
    fits = tables["fits"].set_index("metric")
    # 400 + 800 + 100 + 50 per sample, and the line is exact.
    assert fits.at["t_total_ms", "slope"] == pytest.approx(1350.0)
    assert fits.at["t_total_ms", "r2"] == pytest.approx(1.0)
    assert fits.at["t_vision_ms", "slope"] == pytest.approx(400.0)
    metrics = tables["metrics"]
    k6 = metrics[(metrics["metric"] == "t_vision_ms") & (metrics["arm"] == "K6")].iloc[0]
    # Every clip is slower at K=6, and the paired interval excludes zero.
    assert k6["n_worse"] == 6 and k6["delta_lo"] > 0.0
    shares = tables["shares"]
    for arm, group in shares.groupby("arm"):
        assert group["share"].sum() == pytest.approx(1.0), arm


def test_ms_per_step_is_the_decode_over_its_steps(tmp_path):
    k_sweep(tmp_path, ks=(1,))
    t = load(tmp_path).set_index("clip_id")
    assert t.at["clip02", "ms_per_step"] == pytest.approx((3000.0 + 100.0 + 100.0) / 32)


@pytest.mark.parametrize("col, value", [
    ("machine", "orin"), ("power_mode", "30W"), ("nvidia_driver", "575.00"),
    ("trace_level", "step"), ("tracer_version", TS.TRACER_VERSION - 1),
    ("torch_version", "2.12.0"),
])
def test_latency_across_conditions_is_refused(tmp_path, col, value):
    """Accuracy survives these; latency does not."""
    write_run(tmp_path, "run_K1", num_traj_samples=1)
    write_run(tmp_path, "run_K6", num_traj_samples=6, **{col: value})
    gate = C.timing_gate(load(tmp_path), axis="num_traj_samples")
    assert status(gate, "conditions") == "fail"
    assert col in gate.set_index("check").at["conditions", "detail"]


def test_sweeping_graphs_exempts_the_graph_count_that_follows_it(tmp_path):
    write_run(tmp_path, "run_cg0", cuda_graph=False, cuda_graph_max_graphs=None)
    write_run(tmp_path, "run_cg1", cuda_graph=True, cuda_graph_max_graphs=4)
    timing = load(tmp_path, axis="cuda_graph")
    assert set(timing["arm"]) == {"cg0", "cg1"}
    assert status(C.timing_gate(timing, axis="cuda_graph"), "conditions") == "ok"


def test_what_blends_two_configurations_is_flagged(tmp_path):
    write_run(tmp_path, "run_K1", num_traj_samples=1, probe=False, warmup=0)
    write_run(tmp_path, "run_K6", num_traj_samples=6,
              per_clip=lambda i: {"graph_fallbacks": 1 if i == 0 else 0,
                                  "throttle_state_max": 1 if i < 3 else 0})
    gate = C.timing_gate(load(tmp_path), axis="num_traj_samples")
    for check in ("graph_fallbacks", "throttling", "overhead_probe", "warmup"):
        assert status(gate, check) == "warn", check
    assert not (gate["status"] == "fail").any()


def test_arms_on_different_clips_are_refused_unless_partial_is_allowed(tmp_path):
    write_run(tmp_path, "run_K1", num_traj_samples=1, n=6)
    write_run(tmp_path, "run_K6", num_traj_samples=6, n=5)
    timing = load(tmp_path)
    assert status(C.timing_gate(timing, axis="num_traj_samples"), "clip_sets") == "fail"
    assert status(C.timing_gate(timing, axis="num_traj_samples", allow_partial=True),
                  "clip_sets") == "warn"


def test_a_default_step_count_is_named_by_the_steps_that_ran(tmp_path):
    write_run(tmp_path, "run_s_default", inference_step=None)
    write_run(tmp_path, "run_s5", inference_step=5,
              per_clip=lambda i: {"n_expert_calls": 5})
    timing = load(tmp_path, axis="inference_step")
    assert set(timing["arm"]) == {"s10", "s5"}


def test_two_runs_of_one_arm_are_refused(tmp_path):
    write_run(tmp_path, "run_a", num_traj_samples=6)
    write_run(tmp_path, "run_b", num_traj_samples=6)
    write_run(tmp_path, "run_c", num_traj_samples=1)
    assert status(C.timing_gate(load(tmp_path), axis="num_traj_samples"), "arms") == "fail"


def test_the_script_compares_and_refuses(tmp_path):
    k_sweep(tmp_path)
    out = tmp_path / "_analysis" / "k"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_sweep.py"),
                           "--runs-root", str(tmp_path), "--timing", "--axis",
                           "num_traj_samples", "--baseline", "1", "--n-boot", "200",
                           "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "where the time goes" in done.stdout
    for name in ("timing_gate", "timing_metrics", "timing_shares", "timing_fits"):
        assert (out / f"{name}.parquet").exists(), name
    write_run(tmp_path, "run_K9", num_traj_samples=9, machine="orin")
    refused = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_sweep.py"),
                              "--runs-root", str(tmp_path), "--timing", "--axis",
                              "num_traj_samples", "--out", str(tmp_path / "_analysis" / "r")],
                             capture_output=True, text=True, cwd=ROOT)
    assert refused.returncode == 1 and "Refused" in refused.stdout
