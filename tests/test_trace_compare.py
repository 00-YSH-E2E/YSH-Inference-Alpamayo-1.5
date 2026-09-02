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

"""Comparing runs: the failures that produce numbers instead of errors.

Almost nothing here guards against a crash. The failures worth a test in this
module are the ones that finish, write a full set of tables, and are wrong --
an arm silently dropped by a groupby, a percent change dominated by one easy
clip, a divergence whose operands are the right magnitude in the wrong order.

What these tests cannot see: whether the model ran correctly. Everything is
synthetic or read from a fixture directory. A green run here says the
comparison machinery is sound, not that the comparison is interesting.

compare.py imports no torch, so all of this runs on the CI runner.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from alpamayo1_5.trace import compare as C
from alpamayo1_5.trace import writer as W


# -- fixtures --------------------------------------------------------------

def per_clip_frame(step, n=8, rng_seed=0, scene=None, speed=None, **override):
    """A per_clip frame obeying the identities the gate checks.

    ``sample_gain`` and ``ade_6.4s`` are not free columns -- they are defined
    in terms of the others -- so a fixture that sets them independently would
    fail checks that have nothing to do with what the test is about.
    """
    rng = np.random.default_rng(rng_seed)
    min_ade = np.abs(rng.normal(0.8, 0.3, n)) + 0.05
    mean_ade = min_ade + np.abs(rng.normal(1.0, 0.3, n))
    frame = pd.DataFrame({
        "schema_version": W.SCHEMA_VERSION,
        "run_id": f"run{step:02d}" + "0" * 26,
        "variant": "Vanilla",
        "git_commit": "abc1234",
        "clip_id": [f"c{i:03d}" for i in range(n)],
        "t0_us": 5_100_000,
        "min_ade": min_ade,
        "mean_ade": mean_ade,
        "min_fde": min_ade * 2,
        "mean_fde": mean_ade * 2,
        "ade_6.4s": min_ade,
        "sample_gain": mean_ade - min_ade,
        "diversity_mean_m": np.abs(rng.normal(3.0, 0.5, n)),
        "diversity_final_m": np.abs(rng.normal(4.0, 0.5, n)),
        "diversity_max_m": np.abs(rng.normal(6.0, 0.5, n)),
        "scene": scene if scene is not None else ["straight", "curve"] * (n // 2),
        "speed_profile": speed if speed is not None else ["cruise", "decel"] * (n // 2),
        "model": "nvidia/Alpamayo-1.5-10B",
        "data_spec": "Cam-4",
        "attn_impl": "sdpa",
        "dtype": "bfloat16",
        "inference_step": step,
        "max_new_tokens": 256,
        "temperature": 0.6,
        "top_p": 0.98,
        "seed": 42,
        "num_traj_samples": 6,
        "conditioning_source": "generated",
    })
    for key, value in override.items():
        frame[key] = value
    return frame


def write_run(root, step, *, calls=None, n=8, rng_seed=0, name=None,
              per_clip=True, predictions=True, run_json=True, **override):
    """A run directory on disk. Any of the three files can be withheld."""
    calls = step if calls is None else calls
    out = root / (name or f"Alpamayo-1.5_Cam-4_Vanilla_{n}clip_s{step}_box_26.09.02_{step:08d}")
    out.mkdir(parents=True, exist_ok=True)
    frame = per_clip_frame(step, n=n, rng_seed=rng_seed, **override)
    if per_clip:
        frame.to_parquet(out / "per_clip.parquet", index=False)
    if predictions:
        pd.DataFrame({
            "schema_version": W.SCHEMA_VERSION,
            "clip_id": np.repeat(frame["clip_id"].to_numpy(), 2),
            "sample_k": np.tile([0, 1], n),
            "n_expert_calls": calls,
            "token_ids": [[1, 2, 3]] * (2 * n),
        }).to_parquet(out / "predictions.parquet", index=False)
    if run_json:
        (out / "run.json").write_text(json.dumps({
            "run_id": f"run{step:02d}" + "0" * 26, "variant": "Vanilla",
            "machine": "box", "date": "26.09.02",
            "schema_version": W.SCHEMA_VERSION,
            "clips": list(frame["clip_id"]),
            "params": {"inference_step": step, "seed": 42, "num_traj_samples": 6,
                       "temperature": 0.6, "n_clips": n},
        }))
    return out


@pytest.fixture(autouse=True)
def _needs_parquet():
    pytest.importorskip("pyarrow", reason="the run directory is parquet")


# -- discovery -------------------------------------------------------------

def test_an_interrupted_run_is_a_row_not_an_absence(tmp_path):
    """The failure this module exists to prevent, in its cheapest form.

    ``writer.load_runs`` globs for ``*/predictions.parquet``. A run killed
    before writing one is not an error there, not a warning, and not a row --
    it simply is not in the result, and the arm it was going to be drops out
    of the comparison in silence. A sweep that loses its baseline that way
    still produces a full set of tables, every one of them measured against
    whichever arm happened to sort first.
    """
    write_run(tmp_path, 10)
    write_run(tmp_path, 4, per_clip=False, predictions=False, run_json=False)

    runs = C.discover_runs(tmp_path)

    assert len(runs) == 2, "the dead run must still be visible"
    dead = runs[~runs["has_per_clip"]]
    assert len(dead) == 1
    assert "s4" in dead["run_dir"].iloc[0]


def test_the_analysis_output_is_not_discovered_as_an_arm(tmp_path):
    """The tool writes into ``_analysis/`` under the same root it reads.

    Without the leading-underscore skip, running the analysis twice makes the
    second run discover the first run's output directory and try to load it as
    an arm. The underscore in the output path is load-bearing, not cosmetic.
    """
    write_run(tmp_path, 10)
    (tmp_path / "_analysis").mkdir()
    (tmp_path / "_analysis" / "per_clip.parquet").write_bytes(b"not parquet")

    runs = C.discover_runs(tmp_path)

    assert list(runs["run_dir"]) == [d for d in runs["run_dir"] if not d.startswith("_")]
    assert len(runs) == 1


def test_a_truncated_run_json_reads_as_missing_not_as_a_traceback(tmp_path):
    """A run killed mid-write leaves half a file. That is one bad arm, not a
    dead analysis -- the other eleven have to stay visible."""
    out = write_run(tmp_path, 10)
    (out / "run.json").write_text('{"run_id": "trunc')

    runs = C.discover_runs(tmp_path)

    assert len(runs) == 1
    assert runs["run_id"].isna().all()
    assert bool(runs["has_per_clip"].iloc[0])


# -- loading ---------------------------------------------------------------

def test_a_null_axis_arm_survives_instead_of_being_dropped(tmp_path):
    """Observed on disk: ``inference_step`` written as ``None``.

    Two things break at once, and only one of them is loud.
    ``sorted(set(steps))`` raises comparing int to NoneType -- annoying but
    visible. ``groupby("step")`` drops the arm entirely, because pandas
    excludes NaN keys by default, and that one has no symptom at all.

    The recovery is ``n_expert_calls``, which records what the model actually
    did rather than what the shell meant to ask for.
    """
    write_run(tmp_path, 10)
    write_run(tmp_path, 4, inference_step=None, name="null_axis_run")

    per_clip = C.load_per_clip(C.discover_runs(tmp_path))

    assert set(per_clip["arm"]) == {"s10", "s4"}
    recovered = per_clip[per_clip["run_dir"] == "null_axis_run"]
    assert set(recovered["step"]) == {4}
    assert set(recovered["step_source"]) == {"executed"}
    assert per_clip.groupby("arm").size().to_dict() == {"s10": 8, "s4": 8}


def test_an_unrecoverable_axis_refuses_rather_than_guessing(tmp_path):
    """With no predictions to fall back on there is nothing to recover from.

    Picking a plausible value here would invent the label the entire
    comparison is grouped by, which is worse than stopping.
    """
    write_run(tmp_path, 10)
    write_run(tmp_path, 4, inference_step=None, predictions=False, name="no_evidence")

    with pytest.raises(ValueError, match="could not be recovered"):
        C.load_per_clip(C.discover_runs(tmp_path))


def test_no_usable_run_names_what_it_looked_at(tmp_path):
    """An empty frame would send the caller looking for a bug in the stats."""
    write_run(tmp_path, 10, per_clip=False)

    with pytest.raises(ValueError, match="nothing to compare"):
        C.load_per_clip(C.discover_runs(tmp_path))


def test_arm_order_puts_the_baseline_first(tmp_path):
    write_run(tmp_path, 10)
    write_run(tmp_path, 4)
    write_run(tmp_path, 1)
    per_clip = C.load_per_clip(C.discover_runs(tmp_path))

    assert C.arm_order(per_clip) == ["s10", "s4", "s1"]
    assert C.arm_order(per_clip, baseline=4) == ["s4", "s10", "s1"]
    with pytest.raises(ValueError, match="not among the arms"):
        C.arm_order(per_clip, baseline=7)


# -- the gate --------------------------------------------------------------

def statuses(table):
    return dict(zip(table["check"], table["status"]))


def test_a_clean_sweep_passes_every_check(tmp_path):
    """The gate has to be quiet on good data or nobody will read it."""
    for step in (10, 4, 1):
        write_run(tmp_path, step)
    runs = C.discover_runs(tmp_path)
    per_clip = C.load_per_clip(runs)

    table = C.gate(per_clip, runs, baseline=10)

    assert "fail" not in set(table["status"]), table.to_string()
    assert len(table) >= 11, "every check reports, including the ones that pass"


def test_an_arm_that_ran_the_default_step_count_is_caught(tmp_path):
    """The failure that produces a beautiful wrong answer.

    ``flow_matching`` resolves ``inference_step or self.num_inference_steps``,
    so a falsy value runs the default ten and reports success. An arm labelled
    s1 in that state is identical to the baseline, the table reads "reducing
    steps costs nothing", and no other part of the pipeline disagrees.
    """
    write_run(tmp_path, 10, calls=10)
    write_run(tmp_path, 1, calls=10)          # labelled s1, actually ran ten
    runs = C.discover_runs(tmp_path)

    table = C.gate(C.load_per_clip(runs), runs)

    assert statuses(table)["executed_steps"] == "fail"
    assert "did not run the step count it claims" in \
        table.set_index("check").at["executed_steps", "detail"]


def test_a_constant_calls_per_step_factor_is_fine(tmp_path):
    """Classifier-free guidance makes it two expert calls per step.

    The check is a ratio for this reason. An equality would reject a correct
    sweep the day guidance is switched on.
    """
    write_run(tmp_path, 10, calls=20)
    write_run(tmp_path, 4, calls=8)
    runs = C.discover_runs(tmp_path)

    table = C.gate(C.load_per_clip(runs), runs)

    assert statuses(table)["executed_steps"] == "ok"


def test_arms_run_at_different_settings_are_refused(tmp_path):
    """Then a difference between them is not attributable to the axis."""
    write_run(tmp_path, 10)
    write_run(tmp_path, 4, seed=7)          # the column, not the fixture rng
    runs = C.discover_runs(tmp_path)

    table = C.gate(C.load_per_clip(runs), runs)

    assert statuses(table)["config"] == "fail"


def test_arms_covering_different_clips_are_refused(tmp_path):
    """An arm that died on the hard clips scores best on what it finished."""
    write_run(tmp_path, 10, n=8)
    write_run(tmp_path, 4, n=4)
    runs = C.discover_runs(tmp_path)

    table = C.gate(C.load_per_clip(runs), runs)

    assert statuses(table)["clip_sets"] == "fail"
    assert statuses(C.gate(C.load_per_clip(runs), runs, allow_partial=True))["clip_sets"] \
        == "warn"


def test_strata_that_disagree_between_arms_are_refused(tmp_path):
    """``scene`` comes from the logged future, so it cannot depend on the arm.

    If it does, a threshold moved mid-sweep and the strata are two different
    partitions wearing the same names.
    """
    write_run(tmp_path, 10, scene=["straight"] * 8)
    write_run(tmp_path, 4, scene=["curve"] * 8)
    runs = C.discover_runs(tmp_path)

    table = C.gate(C.load_per_clip(runs), runs)

    assert statuses(table)["strata"] == "fail"


def test_absent_diversity_warns_and_is_never_zero_filled(tmp_path):
    """K=1 records no spread, so the column is absent -- not zero.

    Filling it with zero would report a total mode collapse that was never
    measured, in exactly the experiment whose headline is about collapse.
    """
    write_run(tmp_path, 10)
    write_run(tmp_path, 4, diversity_final_m=np.nan)
    runs = C.discover_runs(tmp_path)
    per_clip = C.load_per_clip(runs)

    table = C.gate(per_clip, runs)

    assert statuses(table)["diversity"] == "warn"
    absent = per_clip[per_clip["arm"] == "s4"]["diversity_final_m"]
    assert absent.isna().all(), "absent must stay absent"
    assert not (absent.fillna(-1) == 0).any()


def test_a_broken_internal_identity_fails(tmp_path):
    """``sample_gain`` is defined as ``mean_ade - min_ade``.

    If it is not, the file was written by code that disagrees with this code
    about what the columns mean, and no amount of careful statistics downstream
    repairs that.
    """
    write_run(tmp_path, 10)
    write_run(tmp_path, 4, sample_gain=99.0)
    runs = C.discover_runs(tmp_path)

    table = C.gate(C.load_per_clip(runs), runs)

    assert statuses(table)["sample_gain"] == "fail"


def test_a_different_cot_between_arms_is_reported(tmp_path):
    """The pairing rests on the arms sharing their initial noise.

    ``_euler`` draws noise once before the loop, so identical seed plus
    identical token stream implies identical noise. A systematic mismatch means
    the per-clip differences are not attributable to the step count.
    """
    write_run(tmp_path, 10)
    out = write_run(tmp_path, 4)
    pred = pd.read_parquet(out / "predictions.parquet")
    pred["token_ids"] = [[9, 9, 9]] * len(pred)
    pred.to_parquet(out / "predictions.parquet", index=False)
    runs = C.discover_runs(tmp_path)

    table = C.gate(C.load_per_clip(runs), runs)

    assert statuses(table)["token_ids"] == "fail"


def test_one_arm_is_not_a_comparison(tmp_path):
    write_run(tmp_path, 10)
    runs = C.discover_runs(tmp_path)

    table = C.gate(C.load_per_clip(runs), runs)

    assert statuses(table)["arms"] == "fail"


# -- percent change --------------------------------------------------------

def test_percent_change_is_the_ratio_of_means_not_the_mean_of_ratios(tmp_path):
    """One easy clip must not decide the summary.

    Every clip here is 10% worse except one, whose baseline minADE is 0.02 m
    and which moves to 0.42 m -- a +2000% per-clip ratio. Averaging the ratios
    turns a 10% effect into a 30% one on the strength of that single row. The
    ratio of the means is unmoved, because 0.02 m is a small number in a sum
    of large ones, which is the correct weight for it to carry.
    """
    X = np.zeros((100, 2, 1))
    X[:, 0, 0] = 1.0
    X[:, 1, 0] = 1.1                      # every clip is 10% worse
    X[0, 0, 0] = 0.02                     # ...except one that starts tiny
    X[0, 1, 0] = 0.42

    delta = C.ratio_deltas(X.mean(axis=0))[1, 0]
    mean_of_ratios = float(np.mean(100.0 * (X[:, 1, 0] / X[:, 0, 0] - 1.0)))

    assert delta == pytest.approx(10.40, abs=0.05), "close to the true 10%"
    assert mean_of_ratios > 2.5 * delta, "one clip in a hundred trebled the headline"


def test_percent_change_broadcasts_so_the_interval_uses_the_same_arithmetic():
    """The CI and the estimate must not be computed by two expressions."""
    means = np.array([[2.0, 4.0], [3.0, 2.0]])
    block = np.broadcast_to(means, (5, 3, 2, 2))

    point = C.ratio_deltas(means)
    over_block = C.ratio_deltas(block)

    assert point.shape == (2, 2)
    assert over_block.shape == (5, 3, 2, 2)
    assert np.allclose(over_block, point)


# -- divergence: the sign ---------------------------------------------------

def test_divergence_is_positive_when_min_degrades_while_mean_improves():
    """The one error in this module that cannot be caught by eye.

    Reverse the operands and every number keeps its magnitude, every stratum
    keeps its rank, and the conclusion reverses. So the two components are
    asserted separately before their difference -- a test on the difference
    alone passes for the wrong reason if both signs are flipped.
    """
    means = np.array([[1.0, 2.0],     # baseline: minADE 1.0, meanADE 2.0
                      [1.25, 1.5]])   # arm:      minADE worse, meanADE better

    deltas = C.ratio_deltas(means)

    assert deltas[1, 0] == pytest.approx(+25.0), "minADE got worse"
    assert deltas[1, 1] == pytest.approx(-25.0), "meanADE got better"
    assert C.divergence(deltas, i_min=0, i_mean=1)[1] == pytest.approx(+50.0)


def test_divergence_flips_sign_when_the_two_metrics_swap():
    """Independent of which column happens to be minADE.

    This survives an edit to the fixture above, which the first test does not.
    """
    means = np.array([[1.0, 2.0], [1.25, 1.5]])
    deltas = C.ratio_deltas(means)

    forward = C.divergence(deltas, i_min=0, i_mean=1)
    backward = C.divergence(deltas, i_min=1, i_mean=0)

    assert np.allclose(backward, -forward)


def test_a_collapsing_sampler_reads_as_positive_divergence():
    """Jensen's inequality, so no tolerance needs tuning.

    Contract the samples toward their own mean and the mean distance to the
    target can only fall while the minimum distance can only rise. That is the
    whole claim the sweep is testing, stated as arithmetic: if the sign
    convention here were backwards, this would come out negative.
    """
    rng = np.random.default_rng(0)
    target = np.zeros(2)
    centres = rng.normal(0.0, 2.0, size=(300, 2))
    cloud = rng.normal(0.0, 2.0, size=(300, 6, 2))

    rows = []
    for shrink in (1.0, 0.5, 0.15):
        samples = centres[:, None, :] + cloud * shrink
        dist = np.linalg.norm(samples - target, axis=2)
        rows.append(np.stack([dist.min(axis=1), dist.mean(axis=1)], axis=1))
    X = np.stack(rows, axis=1)                      # [clip, arm, metric]

    deltas = C.ratio_deltas(X.mean(axis=0))
    div = C.divergence(deltas, i_min=0, i_mean=1)

    assert deltas[2, 0] > 0.0, "minADE must degrade as coverage is lost"
    assert deltas[2, 1] < 0.0, "meanADE must improve as samples concentrate"
    assert div[0] == pytest.approx(0.0)
    assert div[1] < div[2], "divergence must grow with the collapse"


# -- strata ----------------------------------------------------------------

def test_scene_and_speed_profile_are_kept_as_separate_axes():
    """They overlap. A clip turning while braking is in ``curve`` and in
    ``decel`` both -- the runner says so where it assigns them.

    Laid along one axis of a chart, ``straight``/``decel``/``lane_change``/
    ``curve`` double-counts the overlap and reads as a partition that does not
    exist.
    """
    index = pd.DataFrame({
        "scene": ["curve"] * 30 + ["straight"] * 70,
        "speed_profile": ["decel"] * 50 + ["cruise"] * 50,
    })

    labels, masks, _ = C.stratum_masks(index, min_clips=10)

    axes = {(lab["axis"], lab["stratum"]): m for lab, m in zip(labels, masks)}
    assert ("scene", "curve") in axes and ("speed_profile", "decel") in axes
    overlap = axes[("scene", "curve")] & axes[("speed_profile", "decel")]
    assert overlap.sum() > 0, "the two axes are meant to overlap"
    assert sum(lab["n"] for lab in labels if lab["axis"] == "scene") == len(index)
    assert sum(lab["n"] for lab in labels if lab["axis"] == "speed_profile") == len(index)


def test_a_stratum_below_the_floor_is_reported_not_silently_dropped():
    """A table that omits a stratum without saying so reads as one that does
    not exist."""
    index = pd.DataFrame({"scene": ["curve"] * 3 + ["straight"] * 97})

    labels, _, dropped = C.stratum_masks(index, columns=("scene",), min_clips=20)

    assert [lab["stratum"] for lab in labels] == ["all", "straight"]
    assert dropped == [{"axis": "scene", "stratum": "curve", "n": 3, "min_clips": 20}]


# -- bootstrap -------------------------------------------------------------

def test_every_replicate_resamples_the_same_clips_for_all_arms():
    """This is the pairing, expressed as a property of the resampler.

    Make arm 1 exactly arm 0 plus a constant and the difference must be that
    constant in *every* replicate, with no scatter. Resampling the arms
    independently would leave a spread here, and that spread is exactly the
    paired precision the experiment was designed to buy.
    """
    rng = np.random.default_rng(0)
    X = np.zeros((200, 2, 1))
    X[:, 0, 0] = rng.normal(10.0, 3.0, 200)
    X[:, 1, 0] = X[:, 0, 0] + 5.0

    block = C.bootstrap_group_means(X, n_boot=500, seed=0)
    difference = block[:, 0, 1, 0] - block[:, 0, 0, 0]

    assert np.allclose(difference, 5.0)
    assert block[:, 0, 0, 0].std() > 0.1, "the resampling did happen"


def test_the_bootstrap_returns_replicates_not_intervals():
    """Because a derived quantity has to be computed inside the replicate.

    Two intervals derived independently and then subtracted ignore that the
    quantities move together, and the result is wide enough to turn a real
    effect into "not significant" with no symptom.
    """
    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(2.0, 0.5, size=(300, 2, 2))) + 0.2
    X[:, 1, 0] *= 1.3       # arm worse on metric 0
    X[:, 1, 1] *= 0.8       # arm better on metric 1

    block = C.bootstrap_group_means(X, n_boot=2000, seed=0)
    inside = C.divergence(C.ratio_deltas(block), 0, 1)[:, 0, 1]
    lo_in, hi_in = C.percentile_interval(inside)

    per_metric = C.ratio_deltas(block)[:, 0, 1, :]
    lo0, hi0 = C.percentile_interval(per_metric[:, 0])
    lo1, hi1 = C.percentile_interval(per_metric[:, 1])
    naive_width = (hi0 - lo1) - (lo0 - hi1)

    assert (hi_in - lo_in) < naive_width
    assert block.ndim == 4


def test_the_bootstrap_does_not_materialise_the_resampled_data():
    """The naive ``X[idx]`` is 2.5 GB at the size this analysis actually runs.

    A bootstrap mean is a multinomially weighted mean exactly, so the whole
    thing is one matmul per group. This asserts the output size, which is what
    regresses if somebody rewrites it the obvious way.
    """
    X = np.ones((1181, 4, 7))
    n_boot = 200

    block = C.bootstrap_group_means(X, n_boot=n_boot, seed=0)

    assert block.shape == (n_boot, 1, 4, 7)
    assert block.nbytes == n_boot * 1 * 4 * 7 * 8


def test_group_means_use_only_the_group():
    rng = np.random.default_rng(2)
    X = np.zeros((400, 1, 1))
    X[:200, 0, 0] = rng.normal(1.0, 0.01, 200)
    X[200:, 0, 0] = rng.normal(9.0, 0.01, 200)
    masks = np.zeros((2, 400), dtype=bool)
    masks[0, :200] = True
    masks[1, 200:] = True

    block = C.bootstrap_group_means(X, masks, n_boot=200, seed=0)

    assert block[:, 0, 0, 0].mean() == pytest.approx(1.0, abs=0.05)
    assert block[:, 1, 0, 0].mean() == pytest.approx(9.0, abs=0.05)


# -- sign test -------------------------------------------------------------

def test_the_sign_test_excludes_ties_rather_than_splitting_them():
    """Ties are real here -- a collapsed sampler in bfloat16 produces
    bit-identical samples and an exactly zero difference. Splitting them
    halfway manufactures evidence out of the one outcome carrying none.
    """
    baseline = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    arm = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    result = C.sign_test(baseline, arm)

    assert (result["n_better"], result["n_worse"], result["n_tie"]) == (3, 0, 3)
    assert result["p_value"] == pytest.approx(0.25)   # 2 * (1/2)^3, not (1/2)^6


def test_the_sign_test_matches_the_exact_binomial():
    baseline = np.zeros(10)
    arm = np.array([-1.0] * 8 + [1.0] * 2)

    result = C.sign_test(baseline, arm)

    expected = 2 * sum(math.comb(10, k) for k in range(3)) / 2 ** 10
    assert result["p_value"] == pytest.approx(expected)


def test_an_underflowing_p_value_stays_readable_in_logs():
    """Below about 1e-308 the float is 0.0, and printing ``p = 0`` is a claim
    nobody should make. The log stays finite."""
    baseline = np.zeros(4000)
    arm = -np.ones(4000)

    result = C.sign_test(baseline, arm)

    assert result["p_value"] == 0.0
    assert result["p_log10"] < -1000
    assert np.isfinite(result["p_log10"])


# -- paired matrix ---------------------------------------------------------

def test_the_matrix_puts_clips_on_axis_zero_so_arms_cannot_misalign(tmp_path):
    write_run(tmp_path, 10)
    write_run(tmp_path, 4)
    per_clip = C.load_per_clip(C.discover_runs(tmp_path))

    X, index = C.paired_matrix(per_clip, metrics=["min_ade", "mean_ade"],
                               arms=["s10", "s4"])

    assert X.shape == (8, 2, 2)
    assert list(index["clip_id"]) == sorted(index["clip_id"])
    baseline = per_clip[per_clip["arm"] == "s10"].set_index("clip_id")
    assert X[0, 0, 0] == pytest.approx(baseline.at[index["clip_id"].iloc[0], "min_ade"])


def test_a_clip_missing_a_metric_is_dropped_whole(tmp_path):
    """Otherwise each metric gets its own denominator and a table of seven
    metrics silently describes seven different clip sets."""
    write_run(tmp_path, 10)
    out = write_run(tmp_path, 4)
    frame = pd.read_parquet(out / "per_clip.parquet")
    frame.loc[0, "mean_ade"] = np.nan
    frame.to_parquet(out / "per_clip.parquet", index=False)
    per_clip = C.load_per_clip(C.discover_runs(tmp_path))

    X, index = C.paired_matrix(per_clip, metrics=["min_ade", "mean_ade"],
                               arms=["s10", "s4"])

    assert X.shape == (7, 2, 2)
    assert len(index) == 7
    assert not np.isnan(X).any()


def test_duplicate_clips_within_an_arm_are_refused(tmp_path):
    """One row per clip is what makes the pairing a pairing."""
    out = write_run(tmp_path, 10)
    frame = pd.read_parquet(out / "per_clip.parquet")
    pd.concat([frame, frame.head(1)]).to_parquet(out / "per_clip.parquet", index=False)
    write_run(tmp_path, 4)
    per_clip = C.load_per_clip(C.discover_runs(tmp_path))

    with pytest.raises(ValueError, match="duplicate"):
        C.paired_matrix(per_clip, metrics=["min_ade"], arms=["s10", "s4"])
