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
import pandas as pd
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
    name = W.run_dir_name("Pruned-24L", "26.09.01", "deadbeefcafe1234", machine="thor")
    assert name == "Alpamayo-1.5_Cam-4_Pruned-24L_thor_26.09.01_deadbeef"


def test_run_dir_name_carries_the_machine():
    """Two runs of the same checkpoint on different boards are not comparable.

    Without the machine in the name they produce directories identical in every
    visible field, and the latency inside them means different things. It also
    matters off the tailnet, where there is no MLflow record yet and the
    directory name is the only thing saying where the work happened.
    """
    thor = W.run_dir_name("Vanilla", "26.09.01", "abcdef0123", machine="thor")
    pro = W.run_dir_name("Vanilla", "26.09.01", "abcdef0123", machine="pro6000")
    assert thor != pro
    assert "_thor_" in thor and "_pro6000_" in pro


def test_machine_defaults_to_the_short_hostname():
    """The same word env.host records, so the directory and the tag agree."""
    import socket

    assert W.machine_name() == socket.gethostname().split(".")[0]
    assert W.machine_name("rented-a100") == "rented-a100"


def test_label_distinguishes_runs_that_share_a_variant():
    """A 1-clip smoke and a 100-clip evaluation are not the same run.

    Without the label they read identically except for the run id -- unique,
    but you cannot pick the one you want by eye, and mistaking a smoke test for
    a real measurement is easy and quiet.
    """
    smoke = W.run_dir_name("Vanilla", "26.09.02", "aaaaaaaa11",
                           machine="thor", label="1clip_k6-temp0.6")
    real = W.run_dir_name("Vanilla", "26.09.02", "bbbbbbbb22",
                          machine="thor", label="100clip_k6-temp0.6")
    assert "_1clip_k6-temp0.6_" in smoke
    assert "_100clip_k6-temp0.6_" in real


def test_names_stay_inside_a_workable_width():
    """Long enough to identify, short enough to read in a listing."""
    longest = W.run_dir_name("Pruned-24L", "26.09.02", "aaaaaaaa11", machine="pro6000",
                             label="100clip_k1-temp0.9-s20")
    assert len(longest) < 100, longest


def test_no_label_leaves_the_name_unchanged():
    """A sweep over one axis should not pay for a segment it does not use."""
    plain = W.run_dir_name("Vanilla", "26.09.02", "aaaaaaaa11", machine="thor")
    assert W.run_dir_name("Vanilla", "26.09.02", "aaaaaaaa11",
                          machine="thor", label=None) == plain
    assert W.run_dir_name("Vanilla", "26.09.02", "aaaaaaaa11",
                          machine="thor", label="") == plain


def test_run_dir_name_is_filesystem_safe():
    name = W.run_dir_name("INT8/w4 a16", "26.09.01", "abcdef0123", machine="box one")
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


def test_call_counts_reach_the_parquet():
    """n_expert_calls is the executed diffusion step count.

    --inference-step records None when unset and the sampler then uses the
    checkpoint's own default, so this column is the only place the number that
    actually ran is visible. All three were being produced by the tracer and
    dropped for want of a column.
    """
    row = sample(n_decode_steps=120, n_vision_calls=1, n_expert_calls=10)
    frame = W.build_rows([row], {"run_id": "r"})
    assert frame["n_decode_steps"].iloc[0] == 120
    assert frame["n_vision_calls"].iloc[0] == 1
    assert frame["n_expert_calls"].iloc[0] == 10


def test_x0_is_stored_flat_beside_the_prediction_it_produced():
    """The flow head's initial noise is the other irrecoverable input.

    A one-step student is scored on reproducing the teacher's map from this
    tensor to the trajectory, point by point; without it the only pairing key
    is the clip, and six samples of one clip are then indistinguishable.
    """
    x0 = np.arange(8, dtype=np.float32).reshape(4, 2)
    frame = W.build_rows([sample(x0=x0)], {})

    assert frame["x0"].iloc[0] == x0.reshape(-1).tolist()
    assert W._ARRAY_SHAPES["x0"] == ("T", 2), "a reader reshapes it like pred_xy"


def test_a_row_without_x0_is_null_not_zero():
    """A run made without the tracer lacks the column; it did not draw zeros."""
    frame = W.build_rows([sample()], {})

    assert frame["x0"].iloc[0] is None


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


# -- per_clip identity ------------------------------------------------------
def clip(clip_id: str = "c1", **extra):
    row = {"clip_id": clip_id, "t0_us": 5_100_000, "min_ade": 0.5, "scene": "curve"}
    row.update(extra)
    return row


def test_per_clip_carries_run_identity(tmp_path):
    """Without it a sweep's per_clip files concatenate into an unusable pile.

    This file exists to be pivoted, and the pivot anyone actually wants is
    across runs. That needs the run to be a column.
    """
    pytest.importorskip("pyarrow", reason="parquet I/O needs pyarrow")
    path = W.write_per_clip(
        tmp_path, [clip("c1"), clip("c2")],
        {"run_id": "r1", "variant": "Vanilla", "git_commit": "a" * 40,
         "columns": {"inference_step": 2, "seed": 42}},
    )
    frame = pd.read_parquet(path)
    assert set(frame["run_id"]) == {"r1"}
    assert set(frame["variant"]) == {"Vanilla"}
    assert set(frame["git_commit"]) == {"a" * 40}
    assert set(frame["inference_step"]) == {2}
    assert set(frame["schema_version"]) == {W.SCHEMA_VERSION}
    assert list(frame["clip_id"]) == ["c1", "c2"]


def test_per_clip_t0_comes_from_the_clip_not_the_config(tmp_path):
    """The pairing key is (clip_id, t0_us), so t0 has to survive per row.

    It is constant today, which is exactly why a config-level t0 would look
    correct right up until someone sweeps the sample timestamp.
    """
    pytest.importorskip("pyarrow", reason="parquet I/O needs pyarrow")
    path = W.write_per_clip(
        tmp_path,
        [clip("c1", t0_us=3_100_000), clip("c2", t0_us=7_100_000)],
        {"run_id": "r1", "columns": {"inference_step": 2}},
    )
    assert list(pd.read_parquet(path)["t0_us"]) == [3_100_000, 7_100_000]


def test_per_clip_without_a_config_still_writes(tmp_path):
    """config is optional, so an older caller keeps working."""
    pytest.importorskip("pyarrow", reason="parquet I/O needs pyarrow")
    frame = pd.read_parquet(W.write_per_clip(tmp_path, [clip()]))
    assert set(frame["schema_version"]) == {W.SCHEMA_VERSION}
    assert "run_id" not in frame.columns


def test_per_clip_two_runs_concatenate_with_an_axis_to_group_on(tmp_path):
    """The thing the missing identity made impossible."""
    pytest.importorskip("pyarrow", reason="parquet I/O needs pyarrow")
    frames = []
    for step in (10, 2):
        out = tmp_path / f"run_s{step}"
        out.mkdir()
        frames.append(pd.read_parquet(W.write_per_clip(
            out, [clip("c1"), clip("c2")],
            {"run_id": f"r{step}", "variant": "Vanilla",
             "columns": {"inference_step": step}},
        )))
    merged = pd.concat(frames, ignore_index=True)
    assert set(merged["inference_step"]) == {10, 2}
    assert len(merged.groupby(["clip_id", "inference_step"])) == 4


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


# -- the predictions table is frozen; timing lives beside it ---------------
#: Every column build_rows produced at schema 3, in order. New measurement
#: goes to timing.parquet; a change here cuts new runs off from the whole
#: schema-3 corpus they are compared against, so it has to be a decision.
_SCHEMA_3_COLUMNS = [
    "schema_version", "run_id", "variant", "git_commit", "clip_id", "t0_us", "sample_k",
    "pred_xy", "x0", "pred_rot", "hist_xy", "hist_rot", "cot", "meta_action",
    "token_ids", "token_logprob", "token_entropy", "n_generated_tokens", "n_cot_tokens",
    "eos_missing", "prompt_len", "t_vision_ms", "t_prefill_ms", "t_decode_ms",
    "t_postgen_ms", "t_expert_ms", "t_other_ms", "t_total_ms", "n_decode_steps",
    "n_vision_calls", "n_expert_calls", "timing_measured",
]


def test_predictions_schema_is_frozen_at_v3():
    frame = W.build_rows([sample()], {"run_id": "r"})
    assert W.SCHEMA_VERSION == 3
    assert list(frame.columns) == _SCHEMA_3_COLUMNS


def timing_row(clip: str = "c1", **extra):
    from alpamayo1_5.trace import timing_schema as TS

    row = {"timing_schema_version": TS.TIMING_SCHEMA_VERSION,
           "tracer_version": TS.TRACER_VERSION, "run_id": "r", "clip_id": clip,
           "row_kind": "main", "pass_index": 0, "timing_measured": True,
           "t_total_ms": 100.0, "t_wall_ms": 120.0,
           "expert_step_ms": [30.0, 20.0, 20.0]}
    row.update(extra)
    return row


def test_timing_round_trip_keeps_list_columns(tmp_path):
    path = W.write_timing(tmp_path, [timing_row(), timing_row("c2")])
    frame = pd.read_parquet(path)
    assert list(frame["clip_id"]) == ["c1", "c2"]
    assert list(frame["expert_step_ms"].iloc[0]) == [30.0, 20.0, 20.0]


def test_a_column_null_in_every_row_keeps_its_type(tmp_path):
    """An instrument that was off in one run and on in the next must still
    concatenate. pandas would have inferred the all-null column as null-typed."""
    import pyarrow.parquet as pq

    path = W.write_timing(tmp_path, [timing_row(graph_replays=None)])
    schema = pq.read_schema(path)
    assert str(schema.field("graph_replays").type) == "int32"
    assert str(schema.field("expert_step_host_ms").type) == "list<element: float>"


def test_undeclared_timing_keys_are_dropped_not_fatal(tmp_path):
    frame = pd.read_parquet(W.write_timing(tmp_path, [timing_row(not_a_column=1.0)]))
    assert "not_a_column" not in frame.columns


def test_write_run_writes_timing_and_names_its_version(tmp_path):
    import json

    from alpamayo1_5.trace import timing_schema as TS

    W.write_run(tmp_path, [sample()], {"run_id": "r"}, {}, timing=[timing_row()])
    assert (tmp_path / "timing.parquet").exists()
    meta = json.loads((tmp_path / "run.json").read_text())
    assert meta["timing"]["schema_version"] == TS.TIMING_SCHEMA_VERSION
    assert meta["timing"]["tracer_version"] == TS.TRACER_VERSION
    assert meta["schema_version"] == 3


def test_two_runs_of_timing_concatenate(tmp_path):
    for name in ("run_a", "run_b"):
        (tmp_path / name).mkdir()
        W.write_timing(tmp_path / name, [timing_row(run_id=name)])
    frame = W.load_timing(tmp_path)
    assert sorted(frame["run_id"].unique()) == ["run_a", "run_b"]


def test_load_timing_refuses_to_mix_versions(tmp_path):
    for name, version in (("old", 1), ("new", 2)):
        (tmp_path / name).mkdir()
        W.write_timing(tmp_path / name, [timing_row(timing_schema_version=version)])
    with pytest.raises(ValueError, match="timing schema versions differ"):
        W.load_timing(tmp_path)


def test_upload_paths_include_timing(tmp_path):
    (tmp_path / "timing.parquet").touch()
    (tmp_path / "run.json").touch()
    (tmp_path / "gt.parquet").touch()
    names = {p.name for p in W.upload_paths(tmp_path)}
    assert "timing.parquet" in names
    assert "gt.parquet" not in names


def test_thermal_series_round_trips(tmp_path):
    from alpamayo1_5.trace import thermal as TH

    log = TH.ThermalLog(mode="unknown")
    log._record({"power.VIN": 150.0, "freq.gpu": 1575.0}, 10.0, 0.3)
    path = W.write_thermal(tmp_path, log, run_id="r")
    frame = pd.read_parquet(path)
    assert sorted(frame["sensor"]) == ["freq.gpu", "power.VIN"]
    assert set(frame["unit"]) == {"MHz", "W"}
    assert (frame["thermal_schema_version"] == TH.THERMAL_SCHEMA_VERSION).all()
    assert (frame["run_id"] == "r").all()


def test_an_empty_thermal_log_writes_nothing(tmp_path):
    from alpamayo1_5.trace import thermal as TH

    assert W.write_thermal(tmp_path, TH.ThermalLog(mode="unknown")) is None


def test_upload_paths_include_thermal(tmp_path):
    (tmp_path / "thermal.parquet").touch()
    assert "thermal.parquet" in {p.name for p in W.upload_paths(tmp_path)}
