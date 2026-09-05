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

"""The action projection with a second time input.

The pretrained head maps ``(x_t, t)`` to the expert's input embeddings. A
shortcut objective asks the velocity field a second question -- "from t_start,
what average velocity lands at t_target?" -- so the head needs to see both
times. This subclass adds ``t_target`` as an extra Fourier-encoded term on the
first hidden layer, behind a projection initialised to **zero**.

Zero is load-bearing. With the projection at zero the forward is
arithmetically the pretrained one (``h + 0``), so a base checkpoint loaded
into this class starts training exactly where it left off, and the shortcut
head cannot be worse than the original before the first gradient step.
``timesteps`` keeps its upstream shape convention: the last element is the
start time the base class already reads, and an optional leading element is
the target. A ``(B, 1, 1)`` input therefore reproduces the base class
bit-for-bit whether or not the projection has been trained.

One trap the tests pin: ``from_pretrained`` fills a *missing* key with
``N(0, 0.02)``, not zeros. Loading base weights into this class therefore
leaves ``target_proj`` random unless :meth:`reset_target_time` is called, and
the model would not equal the pretrained one at step zero.
"""

from __future__ import annotations

import torch
from torch import nn

from alpamayo1_5.models.action_in_proj import FourierEncoderV2, PerWaypointActionInProjV2


class PerWaypointActionInProjV2Shortcut(PerWaypointActionInProjV2):
    """``PerWaypointActionInProjV2`` plus a zero-initialised target-time term."""

    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ) -> None:
        super().__init__(
            in_dims=in_dims, out_dim=out_dim, num_enc_layers=num_enc_layers,
            hidden_size=hidden_size, max_freq=max_freq, num_fourier_feats=num_fourier_feats,
        )
        self.target_fourier = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
        self.target_proj = nn.Linear(num_fourier_feats, hidden_size)
        self.reset_target_time()

    def reset_target_time(self) -> None:
        """Zero the target-time projection, making the forward the pretrained one."""
        nn.init.zeros_(self.target_proj.weight)
        nn.init.zeros_(self.target_proj.bias)

    def target_time_is_zero(self) -> bool:
        return not bool(self.target_proj.weight.any()) and not bool(self.target_proj.bias.any())

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(B, T, action_dim)``; ``timesteps`` is ``(B, 1, 1)`` holding
        ``t_start``, or ``(B, 1, 2)`` holding ``[t_target, t_start]``."""
        B, T, _ = x.shape
        action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1]).repeat(1, T, 1)
        feats = torch.cat((action_feats, timestep_feats), dim=-1).flatten(0, 1)

        trunk = self.encoder.trunk
        h = trunk[0](feats)                                   # Linear(in -> hidden)
        if timesteps.shape[-1] >= 2:
            # Injected before the first nonlinearity so the MLP can mix the
            # two times with the action; a post-norm bias would shift every
            # waypoint identically and bypass the normalisation.
            target = self.target_proj(self.target_fourier(timesteps[..., 0]))
            h = h + target.repeat(1, T, 1).flatten(0, 1)
        h = trunk[1:](h)                                      # the rest, unchanged
        return self.norm(h.reshape(B, T, -1))
