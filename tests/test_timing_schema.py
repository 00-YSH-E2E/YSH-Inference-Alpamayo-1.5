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

"""The timing registry: what it declares, and what it lets a run report.

Pure Python -- no torch -- so this runs in CI.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from alpamayo1_5.trace import timing_schema as TS

# The characters the tracking helper lets through. Anything else is replaced,
# and two different names could collapse onto one key.
_MLFLOW_KEY = re.compile(r"^[0-9A-Za-z_\-./ ]+$")


def test_column_names_are_unique():
    names = TS.names()
    assert len(names) == len(set(names))


def test_every_column_is_fully_declared():
    for col in TS.COLUMNS:
        assert col.dtype in ("f64", "f32", "i16", "i32", "i64", "b", "s", "lf32"), col
        assert col.direction in ("L", "H", "N"), col
        assert col.description.strip(), col
        assert 1 <= col.since <= TS.TIMING_SCHEMA_VERSION, col


def test_every_version_has_a_changelog_line():
    assert set(TS.CHANGELOG) == set(range(1, TS.TIMING_SCHEMA_VERSION + 1))


def test_legacy_keys_are_columns():
    assert set(TS.LEGACY_KEYS) <= set(TS.names())


def test_aggregate_keys_are_safe_unique_and_bounded():
    keys = TS.AGGREGATE_KEYS
    assert len(keys) == len(set(keys))
    assert all(_MLFLOW_KEY.match(k) for k in keys)
    # One run logs every key through one batch against a single-worker server;
    # a namespace that grows without a bound would be felt there first.
    assert len(keys) <= 250


def test_missing_columns_become_null_not_zero():
    row = TS.coerce_row({"clip_id": "c1"})
    assert list(row) == list(TS.names())
    assert row["t_total_ms"] is None
    assert row["graph_replays"] is None


def test_undeclared_keys_raise_when_strict():
    with pytest.raises(ValueError, match="undeclared"):
        TS.coerce_row({"clip_id": "c1", "t_typo_ms": 3.0}, strict=True)
    assert "t_typo_ms" not in TS.coerce_row({"t_typo_ms": 3.0})


def test_coercion_follows_the_declared_types():
    row = TS.coerce_row({"t_total_ms": np.float32(1.5), "n_decode_steps": np.int64(7),
                         "timing_measured": np.bool_(True), "expert_step_ms": np.arange(3)})
    assert isinstance(row["t_total_ms"], float) and row["t_total_ms"] == 1.5
    assert isinstance(row["n_decode_steps"], int) and row["n_decode_steps"] == 7
    assert row["timing_measured"] is True
    assert row["expert_step_ms"] == [0.0, 1.0, 2.0]


def test_is_number_accepts_numpy_and_refuses_flags_and_nan():
    assert TS.is_number(np.float32(1.0))
    assert TS.is_number(3)
    assert not TS.is_number(True)
    assert not TS.is_number(np.bool_(False))
    assert not TS.is_number(float("nan"))
    assert not TS.is_number("1.0")
    assert not TS.is_number(None)


def test_pa_schema_matches_the_registry():
    pytest.importorskip("pyarrow")
    schema = TS.pa_schema()
    assert schema.names == list(TS.names())


# -- aggregation ------------------------------------------------------------
def row(**kw):
    base = {"row_kind": "main", "timing_measured": True}
    base.update(kw)
    return base


def test_aggregate_ignores_unmeasured_and_extra_passes():
    """An unmeasured pass must not enter a mean as a pass that took no time, and
    a probe or warmup pass must not enter it at all."""
    rows = [row(t_total_ms=100.0),
            row(t_total_ms=0.0, timing_measured=False),
            row(t_total_ms=900.0, row_kind="warmup"),
            row(t_total_ms=300.0)]
    out = TS.aggregate(rows)
    assert out["t_total_ms"] == pytest.approx(200.0)
    assert out["timing.n_main_rows"] == 2.0


def test_aggregate_emits_only_declared_keys():
    rows = [row(t_total_ms=100.0, t_wall_ms=130.0, t_compute_span_ms=90.0,
                expert_step_ms=[30.0, 20.0, 22.0], expert_step_host_ms=[20.0, 19.0, 21.0],
                decode_step_ms=[9.0, 9.5], decode_step_host_ms=[9.0, 9.0],
                n_cuda_allocs=700, n_alloc_retries=0,
                graph_captures=1, graph_replays=3, graph_fallbacks=0,
                graph_capture_ms=40.0, graph_mode="capture", t_expert_ms=72.0)]
    out = TS.aggregate(rows)
    assert set(out) <= set(TS.AGGREGATE_KEYS)
    assert all(isinstance(v, float) for v in out.values())


def test_aggregate_splits_the_first_expert_step_from_the_rest():
    out = TS.aggregate([row(expert_step_ms=[60.0, 20.0, 20.0])])
    assert out["expert_step_first_ms"] == 60.0
    assert out["expert_step_steady_ms"] == 20.0
    assert out["expert_step_spread_ms"] == 40.0


def test_launch_ratio_is_host_over_device():
    out = TS.aggregate([row(expert_step_ms=[10.0, 10.0], expert_step_host_ms=[9.0, 11.0])])
    assert out["expert_launch_ratio"] == pytest.approx(1.0)


def test_latency_p95_is_the_total_p95():
    out = TS.aggregate([row(t_total_ms=float(v)) for v in range(1, 101)])
    assert out["latency_p95"] == out["t_total_ms_p95"]
    assert out["t_total_ms_p50"] == pytest.approx(50.5)


def test_graph_keys_appear_only_with_a_runner():
    assert not any(k.startswith("graph.") for k in TS.aggregate([row(t_total_ms=1.0)]))
    out = TS.aggregate([
        row(graph_captures=0, graph_replays=10, graph_fallbacks=0, graph_mode="replay",
            t_expert_ms=50.0),
        row(graph_captures=0, graph_replays=4, graph_fallbacks=6, graph_mode="mixed",
            t_expert_ms=90.0),
    ])
    assert out["graph.fallbacks_sum"] == 6.0
    assert out["graph.fallback_clip_frac"] == 0.5
    # Only the clean replay clip speaks for replayed latency.
    assert out["graph.t_expert_ms_replay_only"] == 50.0


def test_aggregate_of_nothing_is_a_count_of_zero():
    assert TS.aggregate([]) == {"timing.n_main_rows": 0.0}


def test_the_tracer_cost_is_aggregated():
    out = TS.aggregate([row(t_postgen_model_ms=4.0, t_trace_consume_ms=6.0,
                            trace_hook_host_ms=0.5, trace_n_marks=22)])
    assert out["t_postgen_model_ms"] == 4.0
    assert out["trace.consume_ms"] == 6.0
    assert out["trace.hook_host_ms"] == 0.5
    assert out["trace.n_marks"] == 22.0


def test_columns_added_later_say_when():
    assert TS.column("t_trace_consume_ms").since == 2
    assert TS.column("t_total_ms").since == 1


# -- the measurement protocol (table 4) -----------------------------------------
def probe(clip, level, wall):
    return {"row_kind": "probe", "clip_id": clip, "trace_level": level, "t_wall_ms": wall,
            "timing_measured": True}


def test_overhead_is_the_paired_median_over_clips():
    rows = [probe("a", "off", 1000.0), probe("a", "basic", 1020.0),
            probe("b", "basic", 2060.0), probe("b", "off", 2000.0),
            probe("c", "off", 500.0), probe("c", "basic", 510.0)]
    out = TS.overhead(rows)
    assert out["trace.overhead_pct"] == pytest.approx(2.0)   # median of 2, 3, 2
    assert out["trace.overhead_n"] == 3.0
    assert out["trace.overhead_lo"] <= out["trace.overhead_pct"] <= out["trace.overhead_hi"]


def test_overhead_ignores_unpaired_clips_and_other_rows():
    rows = [probe("a", "off", 1000.0), row(clip_id="a", t_wall_ms=5.0),
            probe("b", "basic", 1000.0)]
    assert TS.overhead(rows) is None


def test_the_interval_covers_a_known_overhead():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        base = float(rng.uniform(8000, 20000))
        rows += [probe(str(i), "off", base),
                 probe(str(i), "basic", base * (1.03 + float(rng.normal(0, 0.005))))]
    out = TS.overhead(rows)
    assert out["trace.overhead_lo"] <= 3.0 <= out["trace.overhead_hi"]


def test_protocol_numbers_come_from_extra_passes():
    rows = [
        {"row_kind": "warmup", "clip_id": "a", "t_wall_ms": 30000.0, "t_start_host_s": 1.0,
         "timing_measured": True},
        row(clip_id="a", t_wall_ms=20000.0, t_start_host_s=2.0),
        {"row_kind": "repeat", "clip_id": "a", "t_wall_ms": 22000.0, "t_start_host_s": 3.0,
         "timing_measured": True, "pass_output_match": True},
        {"row_kind": "repeat", "clip_id": "a", "t_wall_ms": 99999.0, "t_start_host_s": 4.0,
         "timing_measured": True, "pass_output_match": False},
    ]
    out = TS.aggregate(rows)
    assert out["cold_start_ms"] == 30000.0
    assert out["cold_start_excess_ms"] == 10000.0
    # The repeat that did different work is out of the noise band.
    assert out["latency_cv"] == pytest.approx(np.std([20000.0, 22000.0], ddof=1) / 21000.0)
    assert out["pass.output_mismatch_sum"] == 1.0
    assert out["timing.n_extra_rows"] == 3.0
    # And none of the extra passes enters the latency means.
    assert out["t_wall_ms"] == 20000.0


def test_host_stages_and_the_paper_boundary_are_aggregated():
    out = TS.aggregate([row(data_load_ms=2200.0, msg_build_ms=5.0, preprocess_ms=200.0,
                            h2d_ms=15.0, t_fuse_traj_host_ms=3.0, clip_wall_ms=16000.0,
                            t_first_traj_ms=14000.0, rss_bytes=30e9, ctx_invol=7)])
    assert out["host.data_load_ms"] == 2200.0
    assert out["host.paper_preprocess_ms"] == pytest.approx(223.0)
    assert out["t_first_traj_ms"] == 14000.0
    assert out["host.rss_max_gb"] == pytest.approx(30.0)
    assert set(out) <= set(TS.AGGREGATE_KEYS)
