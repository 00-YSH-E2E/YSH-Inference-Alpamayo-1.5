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

"""The work model, without torch: a small model and one pass, worked by hand."""

from __future__ import annotations

import pytest

from alpamayo1_5.trace import profile_parse as PP
from alpamayo1_5.trace import roofline as RL
from alpamayo1_5.trace import timing_schema as TS

MODEL = {
    "vision": {"dense_params": 100, "merger_params": 10, "layers": 2, "hidden": 8,
               "weight_bytes": 1000},
    "lm": {"dense_params": 1000, "layers": 3, "heads": 4, "head_dim": 2, "weight_bytes": 5000},
    "lm_head": {"dense_params": 50, "weight_bytes": 700},
    "expert": {"dense_params": 400, "layers": 3, "heads": 2, "head_dim": 2,
               "weight_bytes": 2000},
}


def one_pass(**extra):
    """K=2 rows of 13 tokens (10 prompt, 3 decoded), 10 bytes per token-row;
    40 patches in 4 images, merged to 10 tokens; 5 Euler steps of 6 tokens."""
    row = {"row_kind": "main", "timing_measured": True, "kv_rows": 2, "kv_final_tokens": 13,
           "kv_bytes": 260, "n_decode_steps": 3, "n_vision_patches": 40, "n_vision_tokens": 10,
           "n_images": 4, "n_expert_calls": 5, "expert_tokens": 6,
           "t_vision_ms": 1.0, "t_prefill_ms": 2.0, "t_decode_ms": 3.0, "t_lm_head_ms": 0.5,
           "expert_step_ms": [1.0, 1.0, 1.0, 1.0, 1.0]}
    row.update(extra)
    return row


def test_the_formulas_on_a_pass_worked_by_hand():
    w = RL.work(one_pass(), MODEL)
    # Patches through the blocks, merged tokens through the merger, and one
    # attention per image of 10 patches: 4 * 2 layers * 8 * 40^2 / 4.
    assert w["vision"] == {"flops": 8000 + 200 + 25600, "bytes": 1000}
    # 20 prompt tokens through 1000 weights; attention 4*3*4*2 = 96 per q*kv, 2 rows of 10x10.
    assert w["prefill"] == {"flops": 40000 + 96 * 2 * 100, "bytes": 5000 + 2 * 10 * 10}
    # Steps attend over 11, 12, 13 tokens: 36 in all. Weights every step; the
    # cache read, then read and written by its concatenation.
    assert w["decode"] == {"flops": 12000 + 96 * 2 * 36, "bytes": 15000 + 3 * 2 * 10 * 36}
    assert w["lm_head"] == {"flops": 2 * 2 * 4 * 50, "bytes": 4 * 700}
    # Six action tokens over 13 + 6.
    assert w["expert"] == {"flops": 48000 + 48 * 2 * 6 * 5 * 19,
                           "bytes": 10000 + 3 * 2 * 10 * 5 * 19}


def test_a_segment_without_its_shapes_is_absent_not_zero():
    w = RL.work(one_pass(n_vision_patches=None, kv_bytes=None), MODEL)
    assert set(w) == {"lm_head"}
    assert RL.work(one_pass(), {}) == {}


def test_efficiency_against_the_roof_that_applies():
    peaks = {"gemm_tflops": 100.0, "read_gbps": 200.0}
    e = RL.efficiency(one_pass(), MODEL, peaks)
    decode = RL.work(one_pass(), MODEL)["decode"]
    assert e["decode"]["ai"] == pytest.approx(decode["flops"] / decode["bytes"])
    assert e["decode"]["gbps"] == pytest.approx(decode["bytes"] / 0.003 / 1e9)
    # Memory-bound here: the roof is intensity times bandwidth.
    roof = e["decode"]["ai"] * 200.0 / 1000.0
    assert e["decode"]["roof_frac"] == pytest.approx(e["decode"]["tflops"] / roof)
    # The head is timed by its Euler steps.
    assert e["expert"]["tflops"] == pytest.approx(
        RL.work(one_pass(), MODEL)["expert"]["flops"] / 0.005 / 1e12)
    assert "roof_frac" not in RL.efficiency(one_pass(), MODEL)["decode"]


def test_a_counted_pass_calibrates_the_formulas():
    w = RL.work(one_pass(), MODEL)
    counted = one_pass(row_kind="flops", fc_flops_decode=w["decode"]["flops"] * 1.05,
                       fc_flops_expert=w["expert"]["flops"] * 1.5,
                       fc_bytes_decode=w["decode"]["bytes"] * 2.0)
    ratios = RL.calibration(counted, MODEL)
    assert ratios["decode"] == pytest.approx({"flops": 1.05, "bytes": 2.0})
    # Bytes are not held to the band: operand bytes include activations.
    assert RL.outside_band(ratios) == pytest.approx({"expert": 1.5})


def test_the_aggregate_reaches_mlflow_through_the_registry():
    counted = one_pass(row_kind="flops", fc_flops_decode=19000.0, fc_bytes_decode=2e9)
    peaks = {"gemm_tflops": 100.0, "read_gbps": 200.0, "copy_gbps": 150.0}
    out = TS.aggregate([one_pass(), counted], model=MODEL, peaks=peaks)
    assert out["roofline.read_gbps"] == 200.0 and "roofline.gemv_gbps" not in out
    assert out["ai.decode"] == pytest.approx(RL.efficiency(one_pass(), MODEL)["decode"]["ai"])
    assert out["flops.decode_gb"] == pytest.approx(2.0)
    assert out["flops.ratio_decode"] == pytest.approx(19000.0 / (12000 + 96 * 2 * 36))
    assert set(out) <= set(TS.AGGREGATE_KEYS)
    # Without the work model there is nothing to rate; the peaks still stand.
    bare = TS.aggregate([one_pass()], peaks=peaks)
    assert not any(k.startswith(("eff.", "ai.", "flops.")) for k in bare)


def test_the_counted_columns_are_declared():
    names = {c.name for c in TS.COLUMNS if c.group == "flops"}
    assert names == {f"fc_{kind}_{s}" for kind in ("flops", "bytes") for s in PP.SEGMENTS} | {
        "fc_n_ops"}


#: A pass's marks in order; the segment the host is in after each one.
MARKS = (
    ("call", "start", "pre"), ("generate", "start", "gen_other"),
    ("vlm", "start", "gen_other"), ("vision", "start", "vision"),
    ("vision", "end", "gen_other"), ("lm", "start", "prefill"), ("lm", "end", "gen_other"),
    ("lm_head", "start", "lm_head"), ("lm_head", "end", "gen_other"),
    ("vlm", "end", "gen_other"), ("lm", "start", "decode"), ("lm", "end", "gen_other"),
    ("generate", "end", "postgen"), ("consume", "start", "trace"),
    ("consume", "end", "postgen"), ("diffusion", "start", "head_other"),
    ("expert", "start", "expert"), ("expert", "end", "head_other"),
    ("diffusion", "end", "tail"), ("call", "end", "tail"),
)


def test_a_counted_pass_follows_the_same_segments_as_a_profiled_one():
    """Mark by mark, the counted pass's segment is the one a profile pass's
    ranges give the same instant."""
    state = PP.SegmentState()
    opened: dict[str, list[float]] = {}
    ranges: dict[str, list[tuple[float, float]]] = {}
    for t, (bucket, kind, expected) in enumerate(MARKS):
        state.mark(bucket, kind)
        assert state.segment == expected, (t, bucket, kind)
        if kind == "start":
            opened.setdefault(bucket, []).append(float(t))
        else:
            ranges.setdefault(bucket, []).append((opened[bucket].pop(), float(t)))
    calls = {b: PP._Calls(sorted(ranges.get(b, []))) for b in PP._BUCKETS}
    for t, (_, _, expected) in enumerate(MARKS[:-1]):
        assert PP._label(t + 0.5, calls) == expected, t
