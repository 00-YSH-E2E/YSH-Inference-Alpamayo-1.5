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

"""The torch half of the work model: structure read off a model, a counted pass,
and the probe."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch", reason="the counters need torch")

from alpamayo1_5.trace import flop_count as FC  # noqa: E402
from alpamayo1_5.trace.token_trace import trace_inference  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _config(**kw):
    return types.SimpleNamespace(**kw)


class _Layer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = torch.nn.Linear(width, 3 * width, bias=False)
        self.mlp = torch.nn.Linear(width, 2 * width, bias=False)


def test_the_structure_is_read_off_the_modules_as_loaded():
    model = torch.nn.Module()
    model.vlm = torch.nn.Module()
    model.vlm.model = torch.nn.Module()
    visual = torch.nn.Module()
    visual.blocks = torch.nn.ModuleList([_Layer(4), _Layer(4)])
    visual.patch_embed = torch.nn.Conv3d(3, 4, kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                         bias=False)
    visual.merger = torch.nn.Linear(16, 8, bias=False)
    visual.config = _config(hidden_size=4)
    model.vlm.model.visual = visual
    language = torch.nn.Module()
    language.layers = torch.nn.ModuleList([_Layer(8) for _ in range(3)])
    language.config = _config(num_attention_heads=2, head_dim=4, hidden_size=8)
    model.vlm.model.language_model = language
    model.vlm.lm_head = torch.nn.Linear(8, 100, bias=False).to(torch.bfloat16)
    model.expert = torch.nn.Module()
    model.expert.layers = torch.nn.ModuleList([_Layer(8)])
    model.expert.config = _config(num_attention_heads=2, hidden_size=8)
    w = FC.work_model(model)
    # Per block 4*12 + 4*8 = 80; the patch kernel 4*3*2*2*2 = 96.
    assert w["vision"]["dense_params"] == 2 * 80 + 96
    assert w["vision"]["merger_params"] == 128 and w["vision"]["layers"] == 2
    assert w["lm"]["dense_params"] == 3 * (8 * 24 + 8 * 16)
    assert w["lm"]["weight_bytes"] == 3 * (8 * 24 + 8 * 16) * 4
    assert w["lm_head"] == {"dense_params": 800, "weight_bytes": 1600}
    assert w["expert"]["head_dim"] == 4 and w["expert"]["layers"] == 1


# -- a counted pass on the tracer tests' stand-in model -------------------------
class _Lin(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(256, 256))

    def forward(self, x):
        return x @ self.w


class _Vlm(torch.nn.Module):
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
        for step in range(3):
            self(x, prefill=step == 0)
        generated = torch.full((1, 3), 7, dtype=input_ids.dtype, device=input_ids.device)
        return types.SimpleNamespace(sequences=torch.cat([input_ids, generated], dim=1))


class _Model:
    def __init__(self) -> None:
        self.vlm = _Vlm().cuda()
        self.expert = _Lin().cuda()
        self.action_in_proj = _Lin().cuda()
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


@needs_gpu
def test_a_counted_pass_credits_each_matmul_to_its_segment():
    model = _Model()

    def run(listener):
        with trace_inference(model, listener=listener) as tracer:
            model.run()
        return tracer

    tracer, row = FC.counted(run)
    mm = 2 * 4 * 256 * 256
    assert row["fc_flops_vision"] == mm and row["fc_flops_prefill"] == mm
    assert row["fc_flops_decode"] == 2 * mm and row["fc_flops_lm_head"] == 3 * mm
    assert row["fc_flops_expert"] == 4 * mm
    # The input projection runs inside the sampler, outside the expert.
    assert row["fc_flops_head_other"] == 4 * mm
    # A matmul reads x and w and writes its output: at least that much.
    assert row["fc_bytes_expert"] >= 4 * (2 * 4 * 256 * 4 + 256 * 256 * 4)
    assert row["fc_n_ops"] > 0
    assert tracer.timing.measured


@needs_gpu
def test_the_probe_measures_every_peak():
    peaks = FC.probe(reps=1)
    for key in ("gemm_tflops", "gemv_gbps", "read_gbps", "copy_gbps", "kvcat_gbps"):
        assert peaks[key] > 0.0, key
