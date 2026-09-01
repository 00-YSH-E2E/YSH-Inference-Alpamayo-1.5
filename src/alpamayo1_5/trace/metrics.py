# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Metrics over a finished inference run.

Nothing here is stored in the run's parquet. These are derived quantities, and
derived quantities change: which horizons matter, whether ADE follows the best
sample or the mean, where the jerk percentile sits. Freezing them next to the
raw predictions would mean re-running the model every time a definition moves,
and worse, would leave a stale column that still looks authoritative.

So the run stores predictions, history and token traces; this module turns them
into numbers, every time, from scratch.

All metric keys use only ``[0-9A-Za-z_\\-./ ]`` -- MLflow silently rejects the
rest, so horizons read ``ade_3.0s`` rather than ``ade@3.0s``.
"""

from __future__ import annotations

import numpy as np
import torch

from alpamayo1_5.action_space.action_space import ActionSpace
from alpamayo1_5.geometry.rotation import so3_to_yaw_torch

# Horizons to report, in seconds. Indices are derived from the dataset's own
# timing rather than hardcoded: the logged future starts one step after t0, so
# element i is at (i + 1) * dt. Hardcoding 9/19/29/... silently shifts by a
# step the moment dt or the start offset changes.
HORIZONS_SEC = (1.0, 2.0, 3.0, 4.0, 5.0, 6.4)

# Comfort threshold for lateral acceleration. Not a physical limit -- it is the
# point where a passenger notices being thrown sideways.
LATERAL_ACCEL_COMFORT_MS2 = 4.0


def horizon_indices(dt: float = 0.1, start_offset_steps: int = 1) -> dict[str, int]:
    """Map each horizon to its 0-based index in the future trajectory.

    Args:
        dt: Seconds between waypoints.
        start_offset_steps: How many steps after t0 the first waypoint sits.
            The dataset loader emits ``[t0+dt, ..., t0+64*dt]``, so this is 1.

    Returns:
        ``{"3.0s": 29, ...}`` -- horizons that do not land on a waypoint are
        dropped rather than rounded, so a mislabelled number never appears.
    """
    indices = {}
    for horizon in HORIZONS_SEC:
        exact = horizon / dt - start_offset_steps
        index = int(round(exact))
        if abs(exact - index) < 1e-6 and index >= 0:
            indices[f"{horizon}s"] = index
    return indices


def displacement(pred_xy: np.ndarray, gt_xy: np.ndarray, dt: float = 0.1) -> dict[str, float]:
    """Displacement error of K sampled trajectories against the logged future.

    Args:
        pred_xy: Predicted trajectories, shape ``[K, T, 2]``.
        gt_xy: Logged future, shape ``[T, 2]``.
        dt: Waypoint spacing, used to place the horizon breakdown.

    Returns:
        Best-sample and mean-sample ADE/FDE, plus ADE and endpoint error per
        horizon. The horizon breakdown is what exposes a model that degrades
        with distance -- an aggregate ADE averages exactly that away.
    """
    error = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=-1)  # [K, T]
    ade_per_sample = error.mean(axis=-1)
    fde_per_sample = error[:, -1]
    best = int(np.argmin(ade_per_sample))

    out = {
        "min_ade": float(ade_per_sample.min()),
        "min_fde": float(fde_per_sample.min()),
        "mean_ade": float(ade_per_sample.mean()),
        "mean_fde": float(fde_per_sample.mean()),
    }
    # The breakdown follows the best sample so it stays consistent with min_ade.
    for name, index in horizon_indices(dt).items():
        if index < error.shape[1]:
            out[f"ade_{name}"] = float(error[best, : index + 1].mean())
            out[f"de_{name}"] = float(error[best, index])
    return out


def _denormalize(action: torch.Tensor, space: ActionSpace) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo the action-space normalization.

    The two channels are normalized with very different scales, so comparing a
    threshold in normalized space is meaningless. Every kinematic number below
    is computed after this.
    """
    accel = action[..., 0] * space.accel_std.to(action) + space.accel_mean.to(action)
    kappa = action[..., 1] * space.curvature_std.to(action) + space.curvature_mean.to(action)
    return accel, kappa


def kinematics(
    space: ActionSpace,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
) -> dict[str, float]:
    """Kinematic feasibility of predicted trajectories, in physical units.

    A trajectory can track the logged future closely and still be undrivable --
    tracking it by yanking the wheel. That difference does not show up in ADE,
    which is why this exists.

    ``within_bounds_ratio`` is reported alongside as a hard-failure canary only.
    Its gates sit around 10 sigma and it folds every waypoint into one boolean,
    so it answers "did anything explode", never "is this comfortable to ride in".
    Note that only its acceleration half can ever fire: ``traj_to_action``
    clamps curvature to ``curvature_bounds`` before returning, so the curvature
    half is satisfied by construction. A ``curvature_violation_rate`` used to be
    reported here and was necessarily 0.0 on every run -- a metric that cannot
    take a second value looks like a passing check.

    A caveat that applies to everything below. These read the action recovered
    by inverting the trajectory through ``traj_to_action``, which is a
    regularized least-squares fit, not the action the model emitted. The fit is
    biased toward smoothness, so jerk and the violation gates are, if anything,
    flattering. The emitted action is available -- ``token_trace`` already
    intercepts the diffusion head's return value -- and routing it here would
    make these numbers exact.

    Args:
        space: The model's action space, holding this checkpoint's normalization
            constants and dt.
        history_xyz: Ego history, ``[K, N_hist, 3]``.
        history_rot: Ego history rotations, ``[K, N_hist, 3, 3]``.
        pred_xyz: Predicted trajectories, ``[K, T, 3]``.
        pred_rot: Predicted rotations, ``[K, T, 3, 3]``.
    """
    action, states = space.traj_to_action(
        traj_history_xyz=history_xyz,
        traj_history_rot=history_rot,
        traj_future_xyz=pred_xyz,
        traj_future_rot=pred_rot,
        output_all_states=True,
    )
    accel, kappa = _denormalize(action, space)
    speed = states[..., 0]  # v is physical; states[..., 1] is the *normalized* accel

    jerk = torch.diff(accel, dim=-1) / space.dt
    lateral = speed**2 * kappa

    accel_lo, accel_hi = space.accel_bounds

    def stat(tensor: torch.Tensor) -> tuple[float, float]:
        flat = tensor.abs().flatten().float()
        return float(flat.mean()), float(torch.quantile(flat, 0.95))

    jerk_mean, jerk_p95 = stat(jerk)
    lat_mean, lat_p95 = stat(lateral)
    return {
        "jerk_mean": jerk_mean,
        "jerk_p95": jerk_p95,
        "lat_accel_mean": lat_mean,
        "lat_accel_p95": lat_p95,
        "lat_accel_over_4_ratio": float(
            (lateral.abs() > LATERAL_ACCEL_COMFORT_MS2).float().mean()
        ),
        # +-9.8 m/s^2 is a 1 g gate: this fires when the fit produced something
        # physically absurd, not when the ride is uncomfortable. Use
        # lat_accel_over_4_ratio for that.
        "accel_violation_rate": float(((accel < accel_lo) | (accel > accel_hi)).float().mean()),
        "within_bounds_ratio": float(space.is_within_bounds(action).float().mean()),
        "speed_mean": float(speed.abs().mean()),
    }


def heading(pred_xyz: torch.Tensor, pred_rot: torch.Tensor) -> dict[str, float]:
    """Net heading change and final lateral offset, averaged over samples.

    These two together separate a straight run from a curve and, less obviously,
    a lane change from either: a lane change barely changes heading but moves
    the vehicle sideways. A heading-only rule misses it entirely.
    """
    yaw = so3_to_yaw_torch(pred_rot)  # [K, T]
    delta = yaw[..., -1] - yaw[..., 0]
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))  # wrap to (-pi, pi]
    lateral_offset = pred_xyz[..., -1, 1]
    return {
        "net_heading_deg": float(torch.rad2deg(delta).mean()),
        "net_heading_abs_deg": float(torch.rad2deg(delta).abs().mean()),
        "lateral_offset_m": float(lateral_offset.mean()),
        "lateral_offset_abs_m": float(lateral_offset.abs().mean()),
    }


def classify_scene(net_heading_abs_deg: float, lateral_offset_abs_m: float) -> str:
    """Label a maneuver from egomotion alone.

    Order matters: a lane change is checked before "straight" because it has a
    small heading change and would otherwise be swallowed by it.
    """
    if net_heading_abs_deg > 20.0:
        return "curve"
    if lateral_offset_abs_m > 2.0:
        return "lane_change"
    if net_heading_abs_deg < 5.0 and lateral_offset_abs_m < 1.0:
        return "straight"
    return "other"


def token_quality(
    logprob: np.ndarray, entropy: np.ndarray, n_valid: int, low_conf_threshold: float = -3.0
) -> dict[str, float]:
    """Reasoning quality from the per-token trace.

    A compressed model tends to lose confidence before its text visibly
    degrades: it still emits fluent driving language while the distribution
    behind each token flattens out. Reading twenty samples will not catch that;
    these numbers will, and unlike a hallucination rate they are continuous, so
    they go straight onto a comparison curve.

    Args:
        logprob: Per-step log probability of the chosen token, ``[T]``.
        entropy: Per-step entropy of the distribution, ``[T]``.
        n_valid: Real generated length. Steps past this are post-EOS padding and
            would drag every average toward whatever the model emits after it
            has stopped saying anything.
        low_conf_threshold: log-probability below which a token counts as
            low-confidence.
    """
    if n_valid <= 0:
        return {}
    lp = np.asarray(logprob, dtype=np.float64)[:n_valid]
    ent = np.asarray(entropy, dtype=np.float64)[:n_valid]
    lp = lp[np.isfinite(lp)]
    ent = ent[np.isfinite(ent)]
    if lp.size == 0 or ent.size == 0:
        return {}
    return {
        "logprob_mean": float(lp.mean()),
        "perplexity": float(np.exp(-lp.mean())),
        "entropy_mean": float(ent.mean()),
        "entropy_p95": float(np.percentile(ent, 95)),
        "low_confidence_ratio": float((lp < low_conf_threshold).mean()),
    }


def model_size(model: torch.nn.Module) -> dict[str, float]:
    """Parameter count and peak inference memory -- the "does it fit" column."""
    params = list(model.parameters())
    out = {
        "params_billions": sum(p.numel() for p in params) / 1e9,
        "param_bytes_gb": sum(p.numel() * p.element_size() for p in params) / 1e9,
    }
    if torch.cuda.is_available():
        out["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out
