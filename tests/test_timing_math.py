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

"""Attributing a pass's marks to spans.

Synthetic marks with a known answer. This is where a timing split goes wrong
-- a start paired with the wrong end, a remainder that absorbs a span it
should not -- and none of it needs a GPU to check.
"""

from __future__ import annotations

import pytest

from alpamayo1_5.trace import timing_math as TM
from alpamayo1_5.trace import timing_schema as TS


def marks(*spans):
    """``(bucket, device_start_ms, device_end_ms)`` to records, host = device / 1000."""
    out = []
    for bucket, start, end in spans:
        out.append((bucket, "start", float(start), start / 1000.0))
        out.append((bucket, "end", float(end), end / 1000.0))
    # Host order is time order.
    return sorted(out, key=lambda r: (r[2], r[1] == "start"))


def one_pass():
    """generate 0-100 containing vision 5-25, prefill 30-50, two decode steps;
    diffusion 110-170 containing three Euler steps."""
    return marks(("generate", 0, 100), ("vision", 5, 25), ("lm", 30, 50),
                 ("lm", 55, 65), ("lm", 70, 78), ("diffusion", 110, 170),
                 ("expert", 112, 130), ("expert", 132, 150), ("expert", 152, 168))


def test_the_legacy_split_keeps_its_definitions():
    t = TM.resolve(one_pass(), wall_start_s=0.0, wall_end_s=0.2)
    assert t.vision_ms == pytest.approx(20.0)
    assert t.prefill_ms == pytest.approx(20.0)
    assert t.decode_ms == pytest.approx(18.0)
    assert t.n_decode_steps == 2
    assert t.postgen_ms == pytest.approx(10.0)        # generate end 100 -> diffusion 110
    assert t.expert_ms == pytest.approx(60.0)         # the diffusion span, not the step sum
    assert t.total_ms == pytest.approx(100.0 + 10.0 + 60.0)
    named = t.vision_ms + t.prefill_ms + t.decode_ms + t.postgen_ms + t.expert_ms
    assert t.other_ms == pytest.approx(t.total_ms - named)
    assert t.other_ms >= 0.0


def test_per_call_arrays_keep_the_order_and_the_two_clocks():
    t = TM.resolve(one_pass())
    assert t.decode_step_ms == pytest.approx([10.0, 8.0])
    assert t.expert_step_ms == pytest.approx([18.0, 18.0, 16.0])
    # Host stamps are device / 1000 seconds here, so host ms equals device ms.
    assert t.expert_step_host_ms == pytest.approx(t.expert_step_ms)
    assert t.vision_call_ms == pytest.approx([20.0])


def test_expert_falls_back_to_the_step_sum_without_a_diffusion_span():
    records = marks(("generate", 0, 10), ("expert", 20, 30), ("expert", 30, 45))
    assert TM.resolve(records).expert_ms == pytest.approx(25.0)


def test_wall_and_compute_span():
    t = TM.resolve(one_pass(), wall_start_s=0.0, wall_end_s=0.25)
    assert t.wall_ms == pytest.approx(250.0)
    assert t.compute_span_ms == pytest.approx(20.0 + 20.0 + 18.0 + 60.0)


def test_nothing_recorded_is_unmeasured_and_absent():
    t = TM.resolve([])
    assert t.measured is False
    legacy = t.legacy()
    assert legacy["timing_measured"] is False
    assert legacy["t_total_ms"] is None and legacy["t_vision_ms"] is None
    assert t.wall_ms is None


def test_an_unclosed_span_is_dropped():
    records = [("lm", "start", 0.0, 0.0), ("lm", "end", 5.0, 0.005), ("lm", "start", 6.0, 0.006)]
    t = TM.resolve(records)
    assert t.prefill_ms == pytest.approx(5.0)
    assert t.n_decode_steps == 0


def test_legacy_keys_match_the_registry():
    assert tuple(TM.TimingResult().legacy()) == TS.LEGACY_KEYS


def test_row_keys_are_all_declared():
    assert not TS.unknown_keys(TM.resolve(one_pass()).row())


def test_row_lists_are_copies():
    t = TM.resolve(one_pass())
    row = t.row()
    row["expert_step_ms"].clear()
    assert len(t.expert_step_ms) == 3


@pytest.mark.parametrize("captures, replays, fallbacks, mode", [
    (None, None, None, None),
    (0, 0, 0, "none"),
    (0, 10, 0, "replay"),
    (1, 10, 0, "capture"),
    (0, 0, 10, "fallback"),
    (0, 4, 6, "mixed"),
    (1, 1, 9, "mixed"),
])
def test_graph_mode(captures, replays, fallbacks, mode):
    assert TM.graph_mode(captures, replays, fallbacks) == mode


def test_graph_counters_are_deltas_and_absent_without_a_runner():
    t = TM.resolve(one_pass())
    assert t.graph_captures is None and t.graph_mode is None
    t = TM.resolve(one_pass(),
                   graph_before={"captures": 2, "replays": 20, "eager_fallbacks": 1, "graphs": 2},
                   graph_after={"captures": 3, "replays": 30, "eager_fallbacks": 1, "graphs": 3},
                   capture_ms=41.5)
    assert (t.graph_captures, t.graph_replays, t.graph_fallbacks) == (1, 10, 0)
    assert t.graph_n_graphs == 3
    assert t.graph_capture_ms == 41.5
    assert t.graph_mode == "capture"


def test_allocator_counters_are_deltas_and_absent_when_unreadable():
    assert TM.resolve([]).n_cuda_allocs is None
    t = TM.resolve([], alloc_before=(100, 1), alloc_after=(844, 1))
    assert (t.n_cuda_allocs, t.n_alloc_retries) == (744, 0)


# -- the tracer's own cost (tracer 2) -------------------------------------------
def test_the_tracer_logits_pass_is_taken_out_of_postgen_but_left_in_it():
    """t_postgen_ms keeps its schema-3 meaning; t_postgen_model_ms is the model's part."""
    records = marks(("generate", 0, 100), ("consume", 101, 107), ("diffusion", 110, 170))
    t = TM.resolve(records, hook_ms=0.4)
    assert t.postgen_ms == pytest.approx(10.0)
    assert t.trace_consume_ms == pytest.approx(6.0)
    assert t.postgen_model_ms == pytest.approx(4.0)
    assert t.trace_n_marks == len(records)
    assert t.trace_hook_host_ms == 0.4


def test_without_a_consume_mark_the_model_share_is_absent_not_postgen():
    t = TM.resolve(one_pass())
    assert t.trace_consume_ms is None
    assert t.postgen_model_ms is None


# -- generate's remainder, split (tracer 3) -------------------------------------
def generate_pass():
    """generate 0-100; three forwards 10-40 (with vision 12-20), 50-60, 70-80;
    the language model and lm_head nested inside each forward."""
    return marks(("generate", 0, 100),
                 ("vlm", 10, 40), ("vision", 12, 20), ("lm", 21, 35), ("lm_head", 35, 38),
                 ("vlm", 50, 60), ("lm", 51, 57), ("lm_head", 57, 59),
                 ("vlm", 70, 80), ("lm", 71, 77), ("lm_head", 77, 79))


def test_the_split_adds_back_up_to_other():
    t = TM.resolve(generate_pass(), wall_start_s=-0.005, wall_end_s=0.2)
    assert t.gen_preamble_ms == pytest.approx(10.0)
    assert t.lm_head_ms == pytest.approx(3.0 + 2.0 + 2.0)
    # forwards 50 ms - vision 8 - lm 26 - lm_head 7
    assert t.vlm_glue_ms == pytest.approx(9.0)
    # generate 100 - preamble 10 - forwards 50
    assert t.gen_loop_ms == pytest.approx(40.0)
    parts = t.gen_preamble_ms + t.lm_head_ms + t.vlm_glue_ms + t.gen_loop_ms
    assert parts == pytest.approx(t.other_ms)
    assert t.span_violations == 0


def test_decode_gaps_and_the_first_token():
    t = TM.resolve(generate_pass(), wall_start_s=-0.005)
    assert t.decode_gap_ms == pytest.approx([10.0, 10.0])
    assert t.lm_head_step_ms == pytest.approx([3.0, 2.0, 2.0])
    assert t.n_vlm_forwards == 3
    # Host stamp of forward #2 (50 ms) from the model call (-5 ms).
    assert t.ttft_ms == pytest.approx(55.0)


def test_a_forward_outside_generate_is_counted_not_hidden():
    records = marks(("generate", 10, 20), ("vlm", 0, 5))
    t = TM.resolve(records)
    assert t.span_violations >= 1
    assert t.gen_preamble_ms == 0.0


def test_spans_that_were_never_marked_are_absent_not_zero():
    """Level off marks only the call. A row of 0.0 ms spans would read as a
    pass that took no time in any of them."""
    t = TM.resolve([("call", "start", 0.0, 0.0), ("call", "end", 50.0, 0.05)],
                   wall_start_s=0.0, wall_end_s=0.06)
    assert t.measured is True and t.wall_ms == pytest.approx(60.0)
    assert t.vision_ms is None and t.prefill_ms is None and t.decode_ms is None
    assert t.expert_ms is None and t.postgen_ms is None and t.total_ms is None
    assert t.compute_span_ms is None


# -- the call's own span, and the CPU (table 5) ----------------------------------
def host_pass():
    """call 0-200 with cpu stamps; generate 20-100; diffusion 110-170; a2t 175-185."""
    spans = (("call", 0, 200), ("generate", 20, 100), ("lm", 30, 50), ("lm", 60, 70),
             ("diffusion", 110, 170), ("a2t", 175, 185))
    out = []
    for bucket, start, end in spans:
        # CPU runs at half the host clock here, so every CPU figure is half.
        out.append((bucket, "start", float(start), start / 1000.0, start / 2000.0))
        out.append((bucket, "end", float(end), end / 1000.0, end / 2000.0))
    return sorted(out, key=lambda r: (r[2], r[1] == "start"))


def test_the_call_brackets_what_generate_and_the_head_leave_out():
    t = TM.resolve(host_pass())
    assert t.pre_generate_ms == pytest.approx(20.0)
    assert t.pre_generate_host_ms == pytest.approx(20.0)
    assert t.tail_ms == pytest.approx(30.0)
    assert t.action_to_traj_ms == pytest.approx(10.0)
    assert t.first_traj_ms == pytest.approx(185.0)


def test_cpu_time_is_split_by_segment():
    cpu = TM.resolve(host_pass()).cpu_ms
    assert cpu["pass"] == pytest.approx(100.0)
    assert cpu["pre_generate"] == pytest.approx(10.0)
    assert cpu["prefill"] == pytest.approx(10.0)
    assert cpu["decode"] == pytest.approx(5.0)
    assert cpu["postgen"] == pytest.approx(5.0)
    assert cpu["expert"] == pytest.approx(30.0)
    assert cpu["tail"] == pytest.approx(15.0)


def test_four_field_records_still_resolve_without_cpu():
    t = TM.resolve(one_pass())
    assert t.cpu_ms == {} and t.total_ms is not None


def test_host_and_process_figures_pass_through():
    t = TM.resolve(host_pass(), host_ms={"fuse_traj": 1.5, "expand_inputs": 4.0},
                   process={"proc_cpu_ms": 90.0, "ctx_vol": 3, "ctx_invol": 1,
                            "rss_bytes": 2 ** 30})
    row = t.row()
    assert row["t_fuse_traj_host_ms"] == 1.5 and row["t_expand_inputs_host_ms"] == 4.0
    assert row["t_rope_index_host_ms"] is None
    assert (row["ctx_vol"], row["ctx_invol"], row["rss_bytes"]) == (3, 1, 2 ** 30)
    assert row["cpu_pass_ms"] == pytest.approx(100.0)
    assert not TS.unknown_keys(row)


# -- memory per segment (table 6) -------------------------------------------------
def boundary(label, current, peak, reserved=0, ooms=0):
    return {"label": label, "current": current, "peak": peak, "reserved_peak": reserved,
            "ooms": ooms}


def test_each_boundary_closes_its_own_segment():
    mem = TM.memory_segments([
        boundary("start", 100, 999, ooms=2),       # before the pass: not the pass's peak
        boundary("vision_start", 110, 120),
        boundary("vision_end", 130, 400, reserved=500),
        boundary("prefill_end", 150, 700),
        boundary("generate_end", 160, 650),
        boundary("diffusion_start", 160, 300),
        boundary("diffusion_end", 170, 800, reserved=900),
        boundary("end", 105, 200, ooms=3),
    ])
    assert mem["mem_start_bytes"] == 100 and mem["mem_end_bytes"] == 105
    assert (mem["mem_peak_pre_bytes"], mem["mem_peak_vision_bytes"],
            mem["mem_peak_prefill_bytes"], mem["mem_peak_decode_bytes"],
            mem["mem_peak_postgen_bytes"], mem["mem_peak_expert_bytes"],
            mem["mem_peak_tail_bytes"]) == (120, 400, 700, 650, 300, 800, 200)
    assert mem["mem_peak_clip_bytes"] == 800
    assert mem["mem_reserved_peak_bytes"] == 900
    assert mem["n_ooms"] == 1
    assert not TS.unknown_keys(mem)


def test_level_off_gets_only_the_pass_peak():
    mem = TM.memory_segments([boundary("start", 100, 50), boundary("end", 100, 640)])
    assert mem["mem_peak_clip_bytes"] == 640
    assert "mem_peak_tail_bytes" not in mem and "mem_peak_vision_bytes" not in mem


def test_memory_and_shapes_reach_the_row():
    t = TM.resolve(one_pass(), memory=[boundary("start", 1, 1), boundary("end", 2, 3)],
                   shapes={"kv_bytes": 10, "n_images": 96},
                   process={"host_mem_avail_min_bytes": 5})
    row = t.row()
    assert row["mem_peak_clip_bytes"] == 3 and row["kv_bytes"] == 10
    assert row["n_images"] == 96 and row["host_mem_avail_min_bytes"] == 5
    assert not TS.unknown_keys(row)
