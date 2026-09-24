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

"""The tracer itself: what it installs, what it reads, and that it leaves no trace.

The arithmetic is tested without torch in test_timing_math.py. What is pinned
here needs torch: the counters the tracer reads off the process, and -- on a
GPU -- a full install, pass and removal on a stand-in model, because a mismatch
between what a hook records and what finalize expects only shows up when events
are actually resolved.
"""

from __future__ import annotations

import types

import pytest

# importorskip, not a bare import: a plain `import torch` raises during
# collection on a machine without it (CI), which fails the whole suite instead
# of skipping this module.
torch = pytest.importorskip("torch", reason="the tracer needs torch")

from alpamayo1_5.trace import token_trace as TT  # noqa: E402
from alpamayo1_5.trace.token_trace import InferenceTracer, trace_inference  # noqa: E402


def test_the_old_record_name_still_resolves():
    assert TT.SegmentTiming is TT.TimingResult


def _tracer(model) -> InferenceTracer:
    return InferenceTracer(model)


def test_graph_counters_are_absent_when_no_runner_is_installed():
    """None, not zeros: zeros would read as a graph that captured nothing."""
    assert _tracer(types.SimpleNamespace(expert=None))._graph_counters() is None


def test_graph_counters_come_from_the_model_when_it_holds_the_runner():
    runner = types.SimpleNamespace(stats={"captures": 1, "replays": 9, "eager_fallbacks": 0})
    model = types.SimpleNamespace(_diffusion_expert_cuda_graph=runner, expert=None)
    assert _tracer(model)._graph_counters()["replays"] == 9


def test_graph_counters_fall_back_to_the_expert_that_holds_the_runner():
    """``enable_diffusion_expert_cuda_graph`` sets the attribute on the expert."""
    runner = types.SimpleNamespace(stats={"captures": 2, "replays": 8, "eager_fallbacks": 1})
    expert = types.SimpleNamespace(_diffusion_expert_cuda_graph=runner)
    model = types.SimpleNamespace(expert=expert)
    assert _tracer(model)._graph_counters()["eager_fallbacks"] == 1


def test_a_runner_that_cannot_report_does_not_break_the_run():
    """Instrumentation must never be the reason an inference fails."""

    class Broken:
        @property
        def stats(self):
            raise RuntimeError("no")

    model = types.SimpleNamespace(_diffusion_expert_cuda_graph=Broken(), expert=None)
    assert _tracer(model)._graph_counters() is None


def test_allocator_counters_are_absent_without_cuda_and_integers_with_it():
    counters = InferenceTracer._alloc_counters()
    if not torch.cuda.is_available():
        assert counters is None
    else:
        allocs, retries = counters
        assert isinstance(allocs, int) and isinstance(retries, int)


def test_a_pass_with_nothing_recorded_reports_unmeasured_rather_than_zero():
    timing = _tracer(types.SimpleNamespace(expert=None)).finalize()
    assert timing.measured is False
    assert timing.expert_step_ms == []
    assert timing.legacy()["t_total_ms"] is None


# -- a full pass on a stand-in model -----------------------------------------
class _Lin(torch.nn.Module):
    """A module whose call enqueues one matmul, so its span has device time."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(256, 256))

    def forward(self, x):
        return x @ self.w


class _Vlm(torch.nn.Module):
    """Like the real VLM, generate calls the module itself once per step, and each
    forward runs vision (prefill only), the language model, then lm_head."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = _Lin()
        self.model.language_model = _Lin()
        self.lm_head = _Lin()

    def forward(self, x, prefill=False):
        if prefill:
            self.model.visual(x)
        return self.lm_head(self.model.language_model(x))

    def generate(self, input_ids=None, **kwargs):
        x = torch.randn(4, 256, device=input_ids.device)
        for step in range(3):  # one prefill, two decode steps
            self(x, prefill=step == 0)
        generated = torch.full((1, 3), 7, dtype=input_ids.dtype, device=input_ids.device)
        return types.SimpleNamespace(sequences=torch.cat([input_ids, generated], dim=1))


class _Model:
    """Just the attributes the tracer hooks, arranged the way the real one is."""

    def __init__(self) -> None:
        self.vlm = _Vlm().cuda()
        self.expert = _Lin().cuda()
        self.action_in_proj = _Lin().cuda()
        # An instance attribute, like a sampler set on an object: removal must
        # restore it rather than delete it.
        self.diffusion = types.SimpleNamespace(sample=self._sample)

    def _sample(self, **kwargs):
        x = torch.randn(4, 256, device="cuda")
        for _ in range(4):
            self.action_in_proj(x)
            x = self.expert(x)
        return x

    def run(self) -> None:
        ids = torch.ones(1, 5, dtype=torch.long, device="cuda")
        self.vlm.generate(input_ids=ids)
        self.diffusion.sample()


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@needs_gpu
def test_a_traced_pass_fills_both_clocks_and_every_array():
    model = _Model()
    with trace_inference(model) as tracer:
        model.run()
    t = tracer.timing
    assert t.measured is True
    assert (t.n_vision_calls, t.n_decode_steps, t.n_expert_calls) == (1, 2, 4)
    assert len(t.expert_step_ms) == len(t.expert_step_host_ms) == 4
    assert len(t.decode_step_ms) == len(t.decode_step_host_ms) == 2
    assert t.total_ms > 0.0 and t.wall_ms > 0.0
    # Every device span lies between install and the final synchronize.
    assert t.wall_ms >= t.total_ms - 1e-3
    assert t.n_cuda_allocs is not None
    assert t.graph_mode is None  # no runner installed
    # The x0 capture still rides on the trace that the generate wrapper built.
    assert tracer.trace is not None and tracer.trace.x0 is not None
    assert tracer.trace.x0.shape == (4, 256)
    # Tracer 2 accounts for itself: its logits pass, its marks, their host cost.
    # vision 2 + lm 6 + vlm 6 + lm_head 6 + generate 2 + consume 2 + diffusion 2
    # + expert 8.
    assert t.trace_n_marks == 34
    assert t.trace_consume_ms is not None and t.trace_consume_ms >= 0.0
    assert t.postgen_model_ms <= t.postgen_ms
    assert t.trace_hook_host_ms > 0.0
    # Tracer 3 splits generate's remainder, and the split adds back up.
    assert t.n_vlm_forwards == 3
    assert len(t.lm_head_step_ms) == 3 and len(t.decode_gap_ms) == 2
    assert t.span_violations == 0
    parts = t.gen_preamble_ms + t.lm_head_ms + t.vlm_glue_ms + t.gen_loop_ms
    assert parts == pytest.approx(t.other_ms, abs=1e-3)
    assert 0.0 < t.ttft_ms <= t.wall_ms


@needs_gpu
def test_removal_leaves_the_model_exactly_as_it_was():
    model = _Model()
    sample = model.diffusion.sample
    with trace_inference(model):
        model.run()
    assert "generate" not in vars(model.vlm)       # a class method: the shadow is deleted
    assert model.diffusion.sample == sample        # an instance attribute: restored
    for module in (model.vlm, model.vlm.lm_head, model.vlm.model.visual, model.expert,
                   model.action_in_proj):
        assert not module._forward_hooks and not module._forward_pre_hooks


@needs_gpu
def test_events_are_reused_across_passes_not_reallocated():
    model = _Model()
    with trace_inference(model):
        model.run()
    size = len(TT._POOL._events)
    with trace_inference(model):
        model.run()
    assert len(TT._POOL._events) == size
    assert TT._POOL.owner is None
