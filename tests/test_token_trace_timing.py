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
        self.action_space = types.SimpleNamespace(action_to_traj=lambda x: x * 2.0)

    def _sample(self, **kwargs):
        x = torch.randn(4, 256, device="cuda")
        for _ in range(4):
            self.action_in_proj(x)
            x = self.expert(x)
        return x

    def run(self) -> None:
        ids = torch.ones(1, 5, dtype=torch.long, device="cuda")
        self.vlm.generate(input_ids=ids)
        self.action_space.action_to_traj(self.diffusion.sample())


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
    # + expert 8 + call 2 + a2t 2.
    assert t.trace_n_marks == 38
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
    # Tracer 4 brackets the call and stamps the thread's CPU time.
    assert t.pre_generate_ms >= 0.0 and t.tail_ms >= 0.0
    assert 0.0 < t.first_traj_ms <= t.wall_ms
    assert t.action_to_traj_ms >= 0.0
    assert t.cpu_ms["pass"] > 0.0
    assert t.rss_bytes > 0 and t.ctx_vol is not None
    # Tracer 5 reads the allocator at every segment boundary.
    for seg in ("pre", "vision", "prefill", "decode", "postgen", "expert", "tail"):
        assert t.memory[f"mem_peak_{seg}_bytes"] > 0, seg
    assert t.memory["mem_peak_clip_bytes"] <= TT.run_peak_bytes()
    assert t.shapes["n_vision_patches"] == 4


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


@needs_gpu
def test_level_off_installs_nothing_and_keeps_the_wall_clock():
    """The baseline the tracer's cost is measured against: no hook of any kind."""
    model = _Model()
    with trace_inference(model, level="off") as tracer:
        for module in (model.vlm, model.expert, model.vlm.model.language_model):
            assert not module._forward_hooks and not module._forward_pre_hooks
        assert "generate" not in vars(model.vlm)
        model.run()
    t = tracer.timing
    assert t.measured is True and t.wall_ms > 0.0
    assert t.trace_n_marks == 2
    assert t.total_ms is None and t.decode_ms is None
    assert tracer.trace is None


def test_an_unknown_level_is_refused():
    with pytest.raises(ValueError, match="trace level"):
        InferenceTracer(types.SimpleNamespace(), level="deep")


def test_the_inventory_weighs_each_part_it_finds():
    from alpamayo1_5.trace import metrics as M

    model = torch.nn.Module()
    model.expert = torch.nn.Linear(10, 10, bias=False)             # 400 bytes in fp32
    model.action_out_proj = torch.nn.Linear(10, 2, bias=False)      # 80 bytes
    model.register_buffer("scale", torch.ones(5, dtype=torch.float16))  # 10 bytes, "other"
    inv = M.module_inventory(model)
    assert inv["weights.expert_gb"] == pytest.approx(400e-9)
    assert inv["weights.action_out_proj_gb"] == pytest.approx(80e-9)
    assert inv["weights.total_gb"] == pytest.approx(490e-9)
    assert inv["weights.other_gb"] == pytest.approx(10e-9)
    assert inv["weights.dtype.float16_gb"] == pytest.approx(10e-9)
    assert "weights.visual_gb" not in inv
    assert inv["weights.n_quant_modules"] == 0.0


class _Cache:
    """Like DynamicCache: ``update`` concatenates a layer's keys and values onto
    what it holds, by layer index, and returns the result."""

    def __init__(self) -> None:
        self.held: dict[int, tuple] = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.held:
            key = torch.cat([self.held[layer_idx][0], key])
            value = torch.cat([self.held[layer_idx][1], value])
        self.held[layer_idx] = (key, value)
        return key, value


class _VisionAttn(torch.nn.Module):
    """The vision tower's attention: one fused projection in, ``proj`` out."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = _Lin()
        self.proj = _Lin()

    def forward(self, x, **kwargs):
        return self.proj(self.qkv(x))


class _TextAttn(torch.nn.Module):
    """The language model's and the head's: four projections, and the cache
    updated between them with the layer's index, as Qwen3-VL does."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = _Lin(), _Lin(), _Lin(), _Lin()

    def forward(self, x, past_key_values=None, **kwargs):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx, None)
        return self.o_proj(q + v[-q.shape[0]:])


class _Layer(torch.nn.Module):
    """A layer whose parts are named the way the real stacks name them."""

    def __init__(self, attn_name: str, attn: torch.nn.Module) -> None:
        super().__init__()
        self.attn_name = attn_name
        setattr(self, attn_name, attn)
        self.mlp = _Lin()

    def forward(self, x, **kwargs):
        return self.mlp(getattr(self, self.attn_name)(x, **kwargs))


class _Stack(torch.nn.Module):
    def __init__(self, name: str, layers: list) -> None:
        super().__init__()
        self.stack_name = name
        setattr(self, name, torch.nn.ModuleList(layers))

    def forward(self, x, **kwargs):
        for layer in getattr(self, self.stack_name):
            x = layer(x, **kwargs)
        return x


class _LayeredVlm(_Vlm):
    """Hands the language model the cache, as generate does."""

    def forward(self, x, prefill=False):
        if prefill:
            self.model.visual(x)
        return self.lm_head(self.model.language_model(x, past_key_values=self.cache))


class _LayeredModel(_Model):
    """The stand-in with stacks where the real model has them: vision blocks
    with ``attn``, language-model and head layers with ``self_attn``, and one
    cache the language model fills and the head attends to."""

    def __init__(self) -> None:
        super().__init__()
        self.cache = _Cache()
        self.vlm = _LayeredVlm().cuda()
        self.vlm.cache = self.cache
        self.vlm.model.visual = _Stack(
            "blocks", [_Layer("attn", _VisionAttn()) for _ in range(2)]).cuda()
        self.vlm.model.language_model = _Stack(
            "layers", [_Layer("self_attn", _TextAttn(i)) for i in range(3)]).cuda()
        self.expert = _Stack("layers", [_Layer("self_attn", _TextAttn(i)) for i in range(2)]).cuda()

    def _sample(self, **kwargs):
        x = torch.randn(4, 256, device="cuda")
        for _ in range(4):
            self.action_in_proj(x)
            x = self.expert(x, past_key_values=self.cache)
        return x


@needs_gpu
def test_level_layer_times_every_layer_of_every_stack():
    model = _LayeredModel()
    with trace_inference(model, level="layer") as tracer:
        model.run()
    t = tracer.timing
    # Vision: 2 blocks x 1 call x (block, attn, mlp, qkv, o_proj). The language
    # model, 3 layers x 3 calls (prefill and two decode steps), and the head, 2 x
    # 4 Euler steps, x (block, attn, mlp, q, k, v, o, kv_cat).
    assert len(t.layers) == 2 * 5 + 3 * 3 * 8 + 2 * 4 * 8
    assert {(s["stack"], s["phase"]) for s in t.layers} == {
        ("vision", "vision"), ("lm", "prefill"), ("lm", "decode"), ("expert", "expert")}
    assert {s["part"] for s in t.layers if s["stack"] == "vision"} == {
        "block", "attn", "mlp", "qkv", "o_proj"}
    s = t.layer_summary
    assert s["n_layer_spans"] == len(t.layers)
    assert "layer_kv_cat_ms_vision" not in s
    for phase in ("vision", "prefill", "decode", "expert"):
        # The parts are inside the layer, and the projections and the cache
        # update inside the attention, on the device clock.
        assert 0.0 < s[f"layer_attn_ms_{phase}"] + s[f"layer_mlp_ms_{phase}"] \
            <= s[f"layer_block_ms_{phase}"] + 1e-3
        inside = s[f"layer_qkv_ms_{phase}"] + s[f"layer_o_proj_ms_{phase}"] \
            + s.get(f"layer_kv_cat_ms_{phase}", 0.0)
        assert 0.0 < inside <= s[f"layer_attn_ms_{phase}"] + 1e-3, phase
    # Each layer's update is inside the step level's span of the same update.
    for phase in ("decode", "expert"):
        assert 0.0 < s[f"layer_kv_cat_ms_{phase}"] <= t.step[f"kv_cat_ms_{phase}"] + 1e-3
    # Layer level includes step level, and leaves no hook and no wrapper behind.
    assert t.step["n_syncs_total"] is not None
    assert "update" not in vars(model.cache)
    for module in list(model.vlm.modules()) + list(model.expert.modules()):
        assert not module._forward_hooks and not module._forward_pre_hooks


@needs_gpu
def test_level_layer_marks_nothing_inside_a_capture_it_did_not_start(monkeypatch):
    tracer = InferenceTracer(types.SimpleNamespace(expert=None), level="layer")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    tracer._mark("L:expert:0:attn", "start")
    assert tracer._marks == []


@needs_gpu
def test_a_profiled_pass_attributes_its_kernels_to_the_spans(tmp_path):
    """The tracer's spans as profiler ranges, read back by the parser: the whole
    path a profile pass takes, on a real trace."""
    from torch.profiler import ProfilerActivity, profile

    from alpamayo1_5.trace import profile_parse as PP

    model = _Model()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with trace_inference(model, ranges=True) as tracer:
            model.run()
    path = tmp_path / "trace.json"
    prof.export_chrome_trace(str(path))
    row, kernels = PP.analyze(PP.load_events(path))
    # Every range the pass opened was closed.
    assert tracer._ranges and all(not opened for opened in tracer._ranges.values())
    # One matmul per module call: vision 1, prefill 1, two decode steps, four Euler steps.
    assert row["prof_kernels_vision"] >= 1 and row["prof_kernels_prefill"] >= 1
    assert row["prof_kernels_decode"] >= 2 and row["prof_kernels_expert"] >= 4
    assert row["prof_kernels_per_expert_step"] >= 1.0
    gemms = [k for k in kernels if k["segment"] == "expert" and k["category"] == "gemm"]
    assert gemms and all(k["op"] is not None for k in gemms)
    assert 0.0 <= row["prof_gpu_idle_ms"] <= row["prof_window_ms"]
    assert row["prof_n_launches"] >= row["prof_n_kernels"] - row["prof_n_unlinked"] - 1


@needs_gpu
def test_level_step_audits_syncs_and_restores_what_it_changed():
    """The tracer's own logits pass brings logits to the host: at least one sync
    lands in the consume phase. Afterwards the sync debug mode and the warning
    hook are exactly as they were."""
    import warnings

    model = _Model()
    shown = warnings.showwarning
    mode = torch.cuda.get_sync_debug_mode()
    with trace_inference(model, level="step") as tracer:
        model.run()
    t = tracer.timing
    assert t.step["n_syncs_consume"] >= 1
    assert t.sync_sites
    assert len(t.step["expert_in_proj_ms"]) == 4
    assert torch.cuda.get_sync_debug_mode() == mode
    assert warnings.showwarning is shown

