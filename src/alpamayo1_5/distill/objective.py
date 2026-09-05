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

"""The SnapFlow objective: keep the flow, and teach it to take one step.

Two terms on one velocity field ``F(x, t_start, t_target | c)``:

* **Flow matching**, ``t ~ U(0,1)``, ``x_t = (1 - t) x_noise + t x_1``,
  target ``x_1 - x_noise``, asked at ``(t, t)``. This is the pretraining
  objective and it is what keeps the multi-step field intact while the
  shortcut is learned; the paper's ablation loses both with it removed
  (``alpha = 0``) and with it alone (``alpha = 1``).
* **Shortcut**, asked at ``(0, 1)`` from fresh noise ``x_0``: match the
  average velocity of the model's own two-step Euler path from ``x_0``,
  ``v = ½[F(x_0, 0, 0) + F(x_half, ½, ½)]`` with ``x_half = x_0 + ½ F(x_0, 0, 0)``,
  both evaluated without gradient. No external teacher, no EMA: the target
  is the current model, two steps deep.

The sign inside ``x_half`` is the one thing here that a test has to hold.
This codebase integrates ``x += dt * v`` from ``t = 0`` (noise) to ``1``
(data); the paper's notation runs the other way, and a copied minus sign
trains the field to land two steps *backwards* while every loss still
decreases.

The velocity field is injected as a callable so this module knows nothing
about the model, the cache, or the pool -- and so the sampler's own two-step
loop can be used as the reference in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""``(x, t_start, t_target) -> v``; ``x``/``v`` are ``(B, 64, 2)``, times are ``(B,)``."""


@dataclass(frozen=True)
class SnapFlowConfig:
    alpha: float = 0.5      # weight on flow matching; (1 - alpha) on the shortcut
    lam: float = 0.1        # extra scale on the shortcut term
    t_mid: float = 0.5      # where the second stop-gradient evaluation sits


@torch.no_grad()
def trapezoid_target(velocity: VelocityFn, x0: torch.Tensor, t_mid: float = 0.5) -> torch.Tensor:
    """Average velocity of the model's own two-step Euler path from ``x0``.

    Two forwards, neither carrying gradient. One step of the returned
    velocity lands exactly where the two Euler steps land; that identity is
    what the tests check.
    """
    B = x0.shape[0]
    zeros = x0.new_zeros(B)
    mids = x0.new_full((B,), t_mid)
    v0 = velocity(x0, zeros, zeros)
    x_mid = x0 + t_mid * v0
    v_mid = velocity(x_mid, mids, mids)
    return 0.5 * (v0 + v_mid)


def snapflow_loss(
    velocity: VelocityFn,
    x1: torch.Tensor,
    cfg: SnapFlowConfig = SnapFlowConfig(),
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """One training step's losses for a batch of GT actions ``x1`` ``(B, 64, 2)``.

    Returns ``loss`` (what to back-propagate) alongside its two parts and a
    few scalars worth logging. The flow-matching and shortcut questions are
    asked in **one** call to ``velocity`` on a batch of ``2B``, so the caller
    pays one expert forward with gradient rather than two.

    Contract for ``velocity``: it is called with batch ``B`` (the two
    stop-gradient target evaluations) and with batch ``2B`` (the gradient
    call), and **row j uses conditioning j mod B**. The caller binds it to a
    batch of conditionings and is responsible for tiling them; this module
    never sees a conditioning.
    """
    B = x1.shape[0]
    dev, dt = x1.device, x1.dtype
    x_noise = torch.randn(x1.shape, device=dev, dtype=dt, generator=generator)
    t = torch.rand(B, device=dev, dtype=dt, generator=generator)
    x_t = (1.0 - t.view(B, 1, 1)) * x_noise + t.view(B, 1, 1) * x1
    v_fm_target = x1 - x_noise

    x0 = torch.randn(x1.shape, device=dev, dtype=dt, generator=generator)
    v_sc_target = trapezoid_target(velocity, x0, cfg.t_mid)

    # Both questions in one forward: rows [0, B) are flow matching at (t, t),
    # rows [B, 2B) are the shortcut at (0, 1).
    x_all = torch.cat([x_t, x0], dim=0)
    t_start = torch.cat([t, x1.new_zeros(B)], dim=0)
    t_target = torch.cat([t, x1.new_ones(B)], dim=0)
    v_all = velocity(x_all, t_start, t_target)

    loss_fm = torch.nn.functional.mse_loss(v_all[:B], v_fm_target)
    loss_sc = torch.nn.functional.mse_loss(v_all[B:], v_sc_target)
    loss = cfg.alpha * loss_fm + (1.0 - cfg.alpha) * cfg.lam * loss_sc
    return {
        "loss": loss,
        "loss_fm": loss_fm.detach(),
        "loss_shortcut": loss_sc.detach(),
        "v_target_norm": v_sc_target.norm(dim=(1, 2)).mean().detach(),
        "fm_t_mean": t.mean().detach(),
    }
