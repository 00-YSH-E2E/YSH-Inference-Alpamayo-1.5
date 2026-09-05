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

"""The sampler that passes a target time, and accepts a given initial noise.

Two changes to the upstream Euler loop, both additive.

**Target time.** Each step hands the head ``[t_target, t_start]`` as a
``(B, 1, 2)`` tensor instead of ``(B, 1, 1)``. The upstream head reads only
the last element, so a checkpoint with the upstream head is unaffected; the
shortcut head reads both. With ``inference_step=1`` the single call is
``F(x_0, t_start=0, t_target=1)`` -- the one-step map the student is trained
to make. ``pass_target_time=False`` restores the upstream tensor shape exactly.

**Given x0.** The upstream loop draws its noise inside the loop and returns
only the endpoint. The paired evaluation needs the student to start from the
*same* noise the teacher started from, so ``x0`` can be supplied. Nothing
else changes: the same noise through ten steps of the same head gives the
upstream trajectory to the bit.

The override is on :meth:`sample`, not only :meth:`_euler`, because the
upstream sample accepts **kwargs and drops them -- a diffusion_kwargs
entry the model passes through would never reach the loop otherwise.

Time runs from 0 (noise) to 1 (data): ``x_{t+dt} = x_t + dt * v``. The test
suite pins that direction against a hand-rolled two-step loop, because the
shortcut target is built from it and a reversed sign trains the wrong map
without any error being raised.
"""

from __future__ import annotations

from typing import Literal

import torch

from alpamayo1_5.diffusion.base import StepFn
from alpamayo1_5.diffusion.flow_matching import FlowMatching


class ShortcutFlowMatching(FlowMatching):
    """Upstream Euler sampling, plus a target time per step and an injectable x0."""

    @torch.no_grad()
    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        step_fn: StepFn,
        unguided_step_fn: StepFn | None = None,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        int_method: Literal["euler"] | None = None,
        use_classifier_free_guidance: bool | None = None,
        inference_guidance_weight: float | None = None,
        temperature: float = 1.0,
        x0: torch.Tensor | None = None,
        pass_target_time: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        int_method = int_method or self.int_method
        inference_step = inference_step or self.num_inference_steps
        if use_classifier_free_guidance is None:
            use_classifier_free_guidance = self.use_classifier_free_guidance
        if inference_guidance_weight is None:
            inference_guidance_weight = self.inference_guidance_weight
        if use_classifier_free_guidance and unguided_step_fn is None:
            raise ValueError("unguided_step_fn is required when using classifier free guidance")
        if int_method != "euler":
            raise ValueError(f"Invalid integration method: {int_method}")
        return self._euler(
            batch_size=batch_size, step_fn=step_fn, unguided_step_fn=unguided_step_fn,
            device=device, return_all_steps=return_all_steps, inference_step=inference_step,
            inference_guidance_weight=inference_guidance_weight,
            use_classifier_free_guidance=use_classifier_free_guidance,
            temperature=temperature, x0=x0, pass_target_time=pass_target_time,
        )

    def _euler(  # type: ignore[override]
        self,
        batch_size: int,
        step_fn: StepFn,
        unguided_step_fn: StepFn | None = None,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        inference_guidance_weight: float | None = None,
        use_classifier_free_guidance: bool | None = None,
        temperature: float = 1.0,
        x0: torch.Tensor | None = None,
        pass_target_time: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        inference_step = inference_step or self.num_inference_steps
        if x0 is None:
            x = torch.randn(batch_size, *self.x_dims, device=device) * temperature
        else:
            x = torch.as_tensor(x0, dtype=torch.float32, device=device)
            want = (batch_size, *self.x_dims)
            if tuple(x.shape) != want:
                raise ValueError(f"x0 has shape {tuple(x.shape)}, expected {want}")
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(self.x_dims)
        ones = [1] * n_dim
        all_steps = [x] if return_all_steps else None

        for i in range(inference_step):
            dt = (time_steps[i + 1] - time_steps[i]).view(1, *ones).expand(batch_size, *ones)
            t_start = time_steps[i].view(1, *ones).expand(batch_size, *ones)
            if pass_target_time:
                t_target = time_steps[i + 1].view(1, *ones).expand(batch_size, *ones)
                t = torch.cat([t_target, t_start], dim=-1)            # (B, 1, 2)
            else:
                t = t_start                                            # (B, 1, 1)
            if use_classifier_free_guidance:
                v = self._guided_v(step_fn=step_fn, x=x, t=t, unguided_step_fn=unguided_step_fn,
                                   inference_guidance_weight=inference_guidance_weight)
            else:
                v = step_fn(x=x, t=t)
            x = x + dt * v
            if return_all_steps:
                all_steps.append(x)
        if return_all_steps:
            return torch.stack(all_steps, dim=1), time_steps
        return x
