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

"""The displacement arithmetic, on hand-computed cases.

Every number a run is ranked by comes out of this file, and all of it is plain
numpy. ``metrics.py`` imports torch at module level for the kinematics half,
which needs a real checkpoint's action space and a GPU -- so the module is
loaded here against stand-ins for the three things the pure functions never
touch. That keeps the ADE math testable on a machine with no torch at all,
which is where CI runs.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def M():
    """metrics.py with torch and the action space stubbed out."""
    saved = {k: sys.modules.get(k) for k in (
        "torch", "alpamayo1_5.action_space", "alpamayo1_5.action_space.action_space",
        "alpamayo1_5.geometry", "alpamayo1_5.geometry.rotation", "_metrics_pure",
    )}
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = torch
    for name in ("alpamayo1_5.action_space", "alpamayo1_5.geometry"):
        sys.modules.setdefault(name, types.ModuleType(name))
    space = types.ModuleType("alpamayo1_5.action_space.action_space")
    space.ActionSpace = object
    sys.modules["alpamayo1_5.action_space.action_space"] = space
    rotation = types.ModuleType("alpamayo1_5.geometry.rotation")
    rotation.so3_to_yaw_torch = lambda x: x
    sys.modules["alpamayo1_5.geometry.rotation"] = rotation

    module = types.ModuleType("_metrics_pure")
    sys.modules["_metrics_pure"] = module
    source = (ROOT / "src/alpamayo1_5/trace/metrics.py").read_text()
    exec(compile(source, "metrics.py", "exec"), module.__dict__)  # noqa: S102
    try:
        yield module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


# -- horizons --------------------------------------------------------------
def test_horizon_indices_track_dt(M):
    """The labels are derived from the checkpoint's step, not hardcoded.

    A checkpoint with a different dt would otherwise mislabel every horizon --
    ade_1.0s would silently mean something else.
    """
    assert M.horizon_indices(0.1)["1.0s"] == 9      # t0+0.1s is index 0
    assert M.horizon_indices(0.1)["6.4s"] == 63
    assert M.horizon_indices(0.05)["1.0s"] == 19    # half the step, twice the index


def test_horizon_indices_refuse_to_round(M):
    """A horizon that does not land on a step is dropped, not approximated."""
    assert "1.0s" not in M.horizon_indices(0.3)


# -- displacement ----------------------------------------------------------
def test_min_ade_picks_the_best_sample(M):
    """Two samples, constant offsets of 1 m and 3 m."""
    gt = np.zeros((64, 2))
    pred = np.stack([np.full((64, 2), (1.0, 0.0)), np.full((64, 2), (3.0, 0.0))])
    out = M.displacement(pred, gt)
    assert out["min_ade"] == pytest.approx(1.0)
    assert out["mean_ade"] == pytest.approx(2.0)   # over both samples
    assert out["min_fde"] == pytest.approx(1.0)


def test_min_fde_is_min_over_samples_not_the_best_ade_sample(M):
    """Standard definition: the two can come from different samples.

    Sample A drifts late (small ADE, large FDE); sample B is offset evenly.
    """
    gt = np.zeros((4, 2))
    a = np.array([[0.0, 0], [0, 0], [0, 0], [8.0, 0]])   # ade 2.0, fde 8.0
    b = np.full((4, 2), (3.0, 0.0))                      # ade 3.0, fde 3.0
    out = M.displacement(np.stack([a, b]), gt, dt=0.1)
    assert out["min_ade"] == pytest.approx(2.0)          # from a
    assert out["min_fde"] == pytest.approx(3.0)          # from b


def test_ade_at_full_horizon_equals_min_ade(M):
    """A free consistency check: if these disagree the indexing is wrong."""
    rng = np.random.RandomState(0)
    out = M.displacement(rng.randn(6, 64, 2), np.zeros((64, 2)))
    assert out["ade_6.4s"] == pytest.approx(out["min_ade"])


def test_perfect_prediction_scores_zero(M):
    gt = np.arange(128, dtype=float).reshape(64, 2)
    out = M.displacement(gt[None].repeat(3, axis=0), gt)
    assert out["min_ade"] == pytest.approx(0.0)
    assert out["min_fde"] == pytest.approx(0.0)


def test_displacement_is_euclidean_not_per_axis(M):
    """3-4-5: an offset of (3, 4) is 5 m, not 3.5."""
    gt = np.zeros((4, 2))
    pred = np.full((1, 4, 2), (3.0, 4.0))
    assert M.displacement(pred, gt, dt=0.1)["min_ade"] == pytest.approx(5.0)


# -- scenes ----------------------------------------------------------------
@pytest.mark.parametrize(
    "heading_deg, offset_m, expected",
    [(25.0, 0.1, "curve"), (1.0, 3.0, "lane_change"), (0.5, 0.1, "straight")],
)
def test_classify_scene(M, heading_deg, offset_m, expected):
    assert M.classify_scene(heading_deg, offset_m) == expected


# -- the situation the data presented ---------------------------------------
def _traj(v0, accel, curve_deg=0.0, lateral_m=0.0, steps=64, dt=0.1):
    """A logged future in the ego frame at t0."""
    t = np.arange(steps) * dt
    distance = v0 * t + 0.5 * accel * t**2
    heading = np.deg2rad(curve_deg) * (t / t[-1])
    x = distance * np.cos(heading)
    y = distance * np.sin(heading) + lateral_m * (t / t[-1])
    return np.stack([x, y], axis=-1)


@pytest.mark.parametrize(
    "traj_kwargs, lateral, longitudinal",
    [
        (dict(v0=15, accel=0.0), "straight", "cruise"),
        (dict(v0=15, accel=-1.6), "straight", "decel"),
        (dict(v0=10, accel=+1.6), "straight", "accel"),
        (dict(v0=15, accel=0.0, curve_deg=30), "curve", "cruise"),
        (dict(v0=15, accel=-1.6, curve_deg=30), "curve", "decel"),
        (dict(v0=15, accel=0.0, lateral_m=3.5), "lane_change", "cruise"),
        (dict(v0=15, accel=-1.6, lateral_m=3.5), "lane_change", "decel"),
    ],
)
def test_classify_maneuver_reads_both_axes(M, traj_kwargs, lateral, longitudinal):
    """Cornering and braking happen at once; one label cannot hold both."""
    out = M.classify_maneuver(_traj(**traj_kwargs))
    assert out == {"lateral": lateral, "longitudinal": longitudinal}


def test_classify_maneuver_without_ground_truth_is_unknown(M):
    """Absent, not guessed.

    The label describes what the vehicle did. With no logged future there is
    nothing to describe, and inventing "straight" would put the clip in a
    bucket it was never observed to belong to.
    """
    assert M.classify_maneuver(np.zeros((0, 2))) == {
        "lateral": "unknown", "longitudinal": "unknown"}
    assert M.classify_maneuver(np.zeros((2, 2)))["lateral"] == "unknown"


def test_gentle_speed_drift_is_still_cruise(M):
    """The threshold separates driver intent from noise on a steady speed."""
    assert M.classify_maneuver(_traj(v0=15, accel=-0.2))["longitudinal"] == "cruise"


def test_classification_does_not_depend_on_the_prediction(M):
    """The axis the whole breakdown rests on.

    Two variants must see the same clip in the same bucket, or a per-situation
    comparison is comparing different clip sets. Nothing but the logged future
    reaches this function -- there is no prediction argument to pass.
    """
    import inspect

    params = list(inspect.signature(M.classify_maneuver).parameters)
    assert params == ["gt_xy", "dt"]


# -- token quality ---------------------------------------------------------
def test_token_quality_perplexity_is_exp_of_negative_mean_logprob(M):
    logprob = np.full(10, -np.log(4.0))
    out = M.token_quality(logprob, np.zeros(10), 10)
    assert out["perplexity"] == pytest.approx(4.0)


def test_token_quality_honours_the_valid_length(M):
    """Rows run past their own end: the batch decodes until the last row stops.

    Averaging over the padding would drag every statistic toward whatever the
    padded slots happen to hold.
    """
    logprob = np.concatenate([np.full(5, -1.0), np.full(5, -100.0)])
    out = M.token_quality(logprob, np.zeros(10), 5)
    assert out["logprob_mean"] == pytest.approx(-1.0)


def test_token_quality_on_an_empty_row_is_empty_not_zero(M):
    assert M.token_quality(np.array([]), np.array([]), 0) == {}
