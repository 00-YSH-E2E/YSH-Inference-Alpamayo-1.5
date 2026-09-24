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

"""A profile pass's trace, read without torch.

One hand-built pass, in microseconds, with every number below worked out
from it by hand: where each kernel was launched, what it was, and where the
host was while the GPU ran nothing.
"""

from __future__ import annotations

import gzip
import json

import pytest

from alpamayo1_5.trace import profile_parse as PP
from alpamayo1_5.trace import timing_schema as TS


def _x(cat, name, ts, dur, pid=1, tid=1, **args):
    return {"ph": "X", "cat": cat, "name": name, "ts": float(ts), "dur": float(dur),
            "pid": pid, "tid": tid, "args": args}


class Trace:
    """Events the way kineto exports them: host ranges and ops on one thread,
    launch calls linked to device events by correlation id."""

    def __init__(self) -> None:
        self.events: list[dict] = [{"ph": "M", "name": "process_name", "args": {}}]
        self.corr = 0
        self.ext = 100

    def range(self, bucket, start, end):
        self.events.append(_x("user_annotation", f"trace::{bucket}", start, end - start))

    def op(self, name, start, end):
        self.ext += 1
        self.events.append(_x("cpu_op", name, start, end - start, **{"External id": self.ext}))
        return self.ext

    def device(self, name, launch_at, start, dur, ext=None, kind="kernel", api="cudaLaunchKernel",
               launch_dur=2.0):
        self.corr += 1
        self.events.append(_x("cuda_runtime", api, launch_at, launch_dur,
                              correlation=self.corr, **{"External id": ext}))
        cat = {"kernel": "kernel", "memcpy": "gpu_memcpy", "memset": "gpu_memset"}[kind]
        self.events.append(_x(cat, name, start, dur, pid=0, tid=7, correlation=self.corr,
                              stream=7, grid=[4, 2, 1], block=[128, 1, 1],
                              **{"External id": ext}))

    def unlinked(self, name, start, dur):
        self.events.append(_x("kernel", name, start, dur, pid=0, tid=7, correlation=999_999))

    def api(self, name, at, dur):
        self.events.append(_x("cuda_runtime", name, at, dur))


def one_pass() -> Trace:
    """call 0-1000: pre 0-10, generate 10-600 (vision 20-100, prefill 110-200,
    decode 300-350 and 400-450, lm_head 352-358), postgen 600-650 (the
    tracer's logits pass 615-630), diffusion 650-950 (Euler steps 660-760
    and 800-900), tail after. Device intervals, in order: 5-8 pre, 50-70
    vision, 92-93 pageable copy, 140-180 prefill, 312-330 and 334-344 decode, 358-362 lm_head,
    364-368 gen_other, 412-420 decode, 460-470 unlinked, 606-608 memcpy,
    675-690 expert, 776-790 head_other, 815-820 expert, 990-1010 tail."""
    t = Trace()
    for bucket, start, end in (("call", 0, 1000), ("generate", 10, 600), ("vision", 20, 100),
                               ("lm", 110, 200), ("lm", 300, 350), ("lm_head", 352, 358),
                               ("lm", 400, 450), ("consume", 615, 630),
                               ("diffusion", 650, 950), ("expert", 660, 760),
                               ("expert", 800, 900)):
        t.range(bucket, start, end)
    t.device("fill_kernel", 3, 5, 3, ext=t.op("aten::fill_", 2, 6))
    t.device("nvjet_sm110_tst_128x128", 32, 50, 20, ext=t.op("aten::mm", 30, 40))
    # A copy to pageable memory, the way .tolist() does it: a sync, not a launch.
    t.device("Memcpy DtoH (Device -> Pageable)", 90, 92, 1, kind="memcpy",
             api="cudaMemcpyAsync", launch_dur=5.0, ext=t.op("aten::copy_", 89, 97))
    t.op("aten::scaled_dot_product_attention", 120, 160)
    t.device("pytorch_flash::flash_fwd_kernel", 130, 140, 40,
             ext=t.op("aten::_scaled_dot_product_flash_attention", 121, 159))
    t.op("aten::scaled_dot_product_attention", 305, 321)
    t.device("fmha_cutlassF_bf16", 310, 312, 18,
             ext=t.op("aten::_scaled_dot_product_efficient_attention", 306, 320))
    t.device("CatArrayBatchedCopy", 332, 334, 10, ext=t.op("aten::cat", 330, 336))
    t.device("nvjet_lm_head", 354, 358, 4, ext=t.op("aten::linear", 352.5, 357))
    t.device("reduce_kernel_argmax", 362, 364, 4, ext=t.op("aten::argmax", 360, 370))
    t.device("nvjet_sm110_tst_64x64", 410, 412, 8, ext=t.op("aten::mm", 405, 415))
    t.unlinked("mystery", 460, 10)
    t.device("Memcpy DtoH (Device -> Pinned)", 605, 606, 2, kind="memcpy",
             api="cudaMemcpyAsync", launch_dur=3.0, ext=t.op("aten::_local_scalar_dense", 604, 645))
    t.api("cudaStreamSynchronize", 610, 30)
    t.op("aten::scaled_dot_product_attention", 665, 700)
    t.op("aten::_scaled_dot_product_attention_math", 666, 699)
    t.device("gemm_bmm", 672, 675, 15, ext=t.op("aten::bmm", 670, 680))
    t.device("vectorized_elementwise_kernel", 772, 776, 14, ext=t.op("aten::mul", 770, 775))
    t.device("elementwise_kernel", 812, 815, 5, ext=t.op("aten::add", 810, 815))
    t.device("direct_copy_kernel", 962, 990, 20, ext=t.op("aten::copy_", 960, 965))
    return t


def test_each_kernel_counts_where_it_was_launched():
    row, kernels = PP.analyze(one_pass().events)
    assert row["prof_n_kernels"] == 13 and row["prof_n_memory_ops"] == 2
    expected = {"pre": 1, "vision": 1, "prefill": 1, "decode": 3, "lm_head": 1, "gen_other": 2,
                "postgen": 0, "trace": 0, "expert": 2, "head_other": 1, "tail": 1}
    assert {s: row[f"prof_kernels_{s}"] for s in PP.SEGMENTS} == expected
    assert row["prof_kernel_ms_decode"] == pytest.approx(0.036)
    assert row["prof_kernels_per_decode_step"] == pytest.approx(1.5)
    assert row["prof_kernels_per_expert_step"] == pytest.approx(1.0)
    # The unlinked kernel is placed by its own start, inside generate.
    assert row["prof_n_unlinked"] == 1
    steps = {k["name"]: k["step_index"] for k in kernels}
    assert steps["fmha_cutlassF_bf16"] == 1 and steps["nvjet_sm110_tst_64x64"] == 2
    assert steps["gemm_bmm"] == 0 and steps["elementwise_kernel"] == 1
    assert steps["fill_kernel"] is None


def test_idle_is_cut_by_where_the_host_was():
    row, _ = PP.analyze(one_pass().events)
    # The window runs to the tail kernel's end, past the call's.
    assert row["prof_window_ms"] == pytest.approx(1.010)
    assert row["prof_gpu_busy_ms"] == pytest.approx(0.174)
    assert row["prof_gpu_idle_ms"] == pytest.approx(0.836)
    idle_us = {"pre": 7, "vision": 59, "prefill": 50, "decode": 64, "lm_head": 6,
               "gen_other": 296, "postgen": 33, "trace": 15, "expert": 180, "head_other": 86,
               "tail": 40}
    for seg, us in idle_us.items():
        assert row[f"prof_idle_ms_{seg}"] == pytest.approx(us / 1000.0), seg
    assert sum(row[f"prof_idle_ms_{s}"] for s in PP.SEGMENTS) == pytest.approx(
        row["prof_gpu_idle_ms"])


def test_categories_come_from_the_launching_op_and_sdpa_wins():
    row, kernels = PP.analyze(one_pass().events)
    cat = {k["name"]: k["category"] for k in kernels}
    # The math backend's bmm is attention, not a GEMM.
    assert cat["gemm_bmm"] == "attention"
    assert cat["nvjet_sm110_tst_128x128"] == "gemm" and cat["CatArrayBatchedCopy"] == "cat"
    assert cat["reduce_kernel_argmax"] == "reduction" and cat["fill_kernel"] == "elementwise"
    assert cat["direct_copy_kernel"] == "copy" and cat["mystery"] == "other"
    assert cat["Memcpy DtoH (Device -> Pinned)"] == "memory"
    expected_us = {"attention": 73, "gemm": 32, "cat": 10, "reduction": 4, "elementwise": 22,
                   "copy": 20, "memory": 3, "other": 10, "conv": 0, "norm": 0, "softmax": 0}
    for c, us in expected_us.items():
        assert row[f"prof_cat_ms_{c}"] == pytest.approx(us / 1000.0), c
    assert sum(row[f"prof_cat_ms_{c}"] for c in PP.CATEGORIES) == pytest.approx(
        row["prof_gpu_busy_ms"])


def test_each_phase_names_the_attention_backend_it_ran():
    row, _ = PP.analyze(one_pass().events)
    assert row["prof_sdpa_prefill"] == "flash"
    assert row["prof_sdpa_decode"] == "efficient"
    assert row["prof_sdpa_expert"] == "math"
    assert row["prof_sdpa_vision"] is None
    assert row["prof_n_sdpa_calls"] == 3


def test_launch_and_sync_calls_are_timed_where_they_happened():
    row, kernels = PP.analyze(one_pass().events)
    # Twelve kernel launches of 2 us and the memcpy's 3 us.
    assert row["prof_n_launches"] == 13
    assert row["prof_launch_api_ms"] == pytest.approx(0.027)
    assert row["prof_launch_us_p50"] == pytest.approx(2.0)
    assert row["prof_launch_api_ms_decode"] == pytest.approx(0.006)
    assert row["prof_launch_api_ms_postgen"] == pytest.approx(0.003)
    # The stream synchronize, and the pageable copy's 5 us in vision.
    assert row["prof_n_syncs"] == 2
    assert row["prof_sync_api_ms_postgen"] == pytest.approx(0.030)
    assert row["prof_sync_api_ms_vision"] == pytest.approx(0.005)
    assert row["prof_sync_api_ms"] == pytest.approx(0.035)
    # Lead, launch return to kernel start: 0,0,0,0,0,1,1,2,2,8,16,26 us.
    assert row["prof_lead_ms_p50"] == pytest.approx(0.001)
    assert row["prof_lead_ms_p50_decode"] == 0.0 and row["prof_lead_ms_p50_vision"] == 0.016
    assert row["prof_lead_ms_p50_expert"] == pytest.approx(0.001)
    assert row["prof_lead_ms_p50_postgen"] is None
    first = next(k for k in kernels if k["name"] == "nvjet_sm110_tst_128x128")
    assert first["lead_us"] == pytest.approx(16.0) and first["launch_us"] == 2.0
    assert first["grid_size"] == 8 and first["block_size"] == 128
    assert first["start_us"] == pytest.approx(50.0) and first["op"] == "aten::mm"


def test_work_outside_the_call_is_left_out():
    t = one_pass()
    t.device("before", -50, -45, 3, ext=t.op("aten::add", -52, -40))
    t.api("cudaDeviceSynchronize", 1200, 50)
    row, kernels = PP.analyze(t.events)
    assert "before" not in {k["name"] for k in kernels}
    assert row["prof_n_kernels"] == 13 and row["prof_n_syncs"] == 2


def test_a_trace_the_tracer_did_not_bracket_has_no_row():
    events = [e for e in one_pass().events if e.get("name") != "trace::call"]
    assert PP.analyze(events) == ({}, [])


def test_the_row_is_all_declared_columns():
    row, _ = PP.analyze(one_pass().events)
    assert not TS.unknown_keys(row)
    declared = {c.name for c in TS.COLUMNS if c.group == "profile"}
    assert set(row) | {"prof_export_ms", "prof_parse_ms"} == declared


def test_the_host_segment_follows_the_innermost_open_span():
    def at(open_, lm=None, started=(), ended=()):
        return PP.segment_of(lambda b: b in open_, lm, lambda b: b in started,
                             lambda b: b in ended)

    assert at({"generate", "vision"}) == "vision"
    assert at({"generate"}, lm=0) == "prefill" and at({"generate"}, lm=3) == "decode"
    assert at({"generate", "lm_head"}) == "lm_head" and at({"generate"}) == "gen_other"
    assert at({"consume"}, started={"generate"}, ended={"generate"}) == "trace"
    assert at(set()) == "pre"
    assert at(set(), started={"generate"}, ended={"generate"}) == "postgen"
    assert at({"diffusion", "expert"}, started={"generate", "diffusion"}) == "expert"
    assert at(set(), started={"generate", "diffusion"},
              ended={"generate", "diffusion"}) == "tail"


def test_names_decide_only_when_no_op_launched_the_kernel():
    assert PP.category("kernel", None, False, "nvjet_sm110_tst_128x128_64x6") == "gemm"
    assert PP.category("kernel", None, False,
                       "void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<cutlass::"
                       "bfloat16_t>>") == "attention"
    assert PP.category("kernel", None, False, "CatArrayBatchedCopy<OpaqueType<2>>") == "cat"
    assert PP.category("kernel", "aten::linear", False, "anything") == "gemm"
    assert PP.category("kernel", "aten::silu", False, "gemm_lookalike") == "elementwise"
    assert PP.category("memset", "aten::zero_", False, "Memset (Device)") == "memory"


def test_traces_load_plain_or_gzipped(tmp_path):
    events = one_pass().events
    plain, packed = tmp_path / "t.json", tmp_path / "t.json.gz"
    plain.write_text(json.dumps({"traceEvents": events}))
    with gzip.open(packed, "wt") as f:
        json.dump({"traceEvents": events}, f)
    assert PP.load_events(plain) == PP.load_events(packed) == events
