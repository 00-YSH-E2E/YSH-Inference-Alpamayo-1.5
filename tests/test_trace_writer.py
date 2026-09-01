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

"""The run directory: naming, columns, and what leaves the machine.

writer.py imports no torch, so all of this runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpamayo1_5.trace import writer as W


def sample(k: int = 0, **extra):
    row = {
        "clip_id": "c1",
        "t0_us": 5_100_000,
        "sample_k": k,
        "pred_xy": np.zeros((4, 2)),
        "hist_xy": np.zeros((2, 2)),
    }
    row.update(extra)
    return row


# -- naming ----------------------------------------------------------------
def test_run_dir_name_ends_with_the_run_id():
    """This is the whole bidirectional link.

    From a directory you find the MLflow run; from a run's output_uri you find
    the directory. Two runs of the same variant on the same day are routine
    while something is being debugged, so the date alone collides.
    """
    name = W.run_dir_name("Pruned-24L", "26.09.01", "deadbeefcafe1234")
    assert name == "Alpamayo-1.5_Cam-4_Pruned-24L_26.09.01_deadbeef"


def test_run_dir_name_is_filesystem_safe():
    name = W.run_dir_name("INT8/w4 a16", "26.09.01", "abcdef0123")
    assert "/" not in name and " " not in name


# -- columns ---------------------------------------------------------------
def test_prompt_len_reaches_the_row():
    """The regression this file exists for.

    TokenTrace carried prompt_len but sample() did not return it, so the
    writer's .get("prompt_len", 0) took the default on every run ever
    uploaded. Zero is a plausible-looking integer, and writer's own docstring
    calls the field necessary for slicing generated tokens off the padding --
    so an offline reader silently sliced from index 0 into it.
    """
    frame = W.build_rows([sample(prompt_len=1234)], {"run_id": "r"})
    assert frame["prompt_len"].iloc[0] == 1234


def test_timing_columns_are_present_and_sum_to_the_total():
    row = sample(t_vision_ms=10.0, t_prefill_ms=20.0, t_decode_ms=600.0,
                 t_postgen_ms=30.0, t_expert_ms=40.0, t_other_ms=50.0,
                 t_total_ms=750.0, n_decode_steps=120, timing_measured=True)
    frame = W.build_rows([row], {"run_id": "r"})
    named = sum(
        float(frame[k].iloc[0])
        for k in ("t_vision_ms", "t_prefill_ms", "t_decode_ms",
                  "t_postgen_ms", "t_expert_ms", "t_other_ms")
    )
    assert named == pytest.approx(float(frame["t_total_ms"].iloc[0]))


def test_absent_optional_values_become_null_not_zero():
    """A run that was never timed must not claim it took no time."""
    frame = W.build_rows([sample()], {"run_id": "r"})
    assert frame["t_total_ms"].isna().all()
    assert frame["prompt_len"].isna().all()


def test_config_columns_repeat_on_every_row():
    """Repetition is what makes cross-run concatenation work."""
    rows = [sample(k) for k in range(3)]
    frame = W.build_rows(rows, {"run_id": "r", "variant": "Vanilla",
                                "git_commit": "a" * 40})
    assert set(frame["variant"]) == {"Vanilla"}
    assert set(frame["git_commit"]) == {"a" * 40}
    assert list(frame["sample_k"]) == [0, 1, 2]


def test_schema_version_is_stamped_on_every_row():
    frame = W.build_rows([sample(), sample(1)], {"run_id": "r"})
    assert set(frame["schema_version"]) == {W.SCHEMA_VERSION}


def test_array_columns_are_flattened_to_float32_lists():
    frame = W.build_rows([sample(pred_xy=np.ones((4, 2)))], {"run_id": "r"})
    assert frame["pred_xy"].iloc[0] == [1.0] * 8


# -- what leaves the machine -----------------------------------------------
def test_upload_paths_excludes_the_logged_future(tmp_path):
    """gt.parquet stays local: it is recoverable from (clip_id, t0_us).

    The exclusion is structural -- gt is simply not in the list -- rather than
    a filter someone can bypass by passing a different glob.
    """
    (tmp_path / "predictions.parquet").touch()
    (tmp_path / "run.json").touch()
    (tmp_path / "gt.parquet").touch()
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "c1.png").touch()

    names = {p.name for p in W.upload_paths(tmp_path)}
    assert "gt.parquet" not in names
    assert {"predictions.parquet", "run.json", "c1.png"} <= names


def test_upload_paths_on_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert W.upload_paths(tmp_path / "nope") == []


# -- round trip ------------------------------------------------------------
def test_write_and_load_round_trip(tmp_path):
    pytest.importorskip("pyarrow", reason="parquet I/O needs pyarrow")
    out = tmp_path / "Alpamayo-1.5_Cam-4_Vanilla_26.09.01_abcdef01"
    W.write_run(out, [sample(k, prompt_len=7) for k in range(2)],
                {"run_id": "r", "variant": "Vanilla"}, {"n_clips": 1})
    assert (out / "predictions.parquet").is_file()
    assert (out / "run.json").is_file()

    frame = W.load_runs(tmp_path)
    assert len(frame) == 2
    assert set(frame["prompt_len"]) == {7}


def test_load_runs_refuses_to_mix_schema_versions(tmp_path, monkeypatch):
    """A silently misaligned comparison is the failure the version guards."""
    pytest.importorskip("pyarrow", reason="parquet I/O needs pyarrow")
    for version, name in ((W.SCHEMA_VERSION, "a"), (W.SCHEMA_VERSION + 1, "b")):
        monkeypatch.setattr(W, "SCHEMA_VERSION", version)
        W.write_run(tmp_path / name, [sample()], {"run_id": name}, {})
    with pytest.raises(ValueError, match="schema versions differ"):
        W.load_runs(tmp_path)


def test_load_runs_with_no_runs_is_empty_not_an_error(tmp_path):
    assert W.load_runs(tmp_path).empty
