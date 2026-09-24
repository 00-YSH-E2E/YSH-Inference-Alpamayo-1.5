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
