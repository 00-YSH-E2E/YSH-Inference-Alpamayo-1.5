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

"""The recording helper's pure parts.

These need no GPU, no MLflow and no network, which is the point: the rule
engine and the coordinate parsing decide whether a run is findable later, and
they are the cheapest things in the repo to get wrong silently.
"""

from __future__ import annotations

import ml_platform_track as mlp
import pytest


# -- coordinates -----------------------------------------------------------
@pytest.mark.parametrize(
    "coord, expected",
    [
        ("owner/name", ("owner/name", "main", "")),
        ("hf:owner/name", ("owner/name", "main", "")),
        ("owner/name@abc123", ("owner/name", "abc123", "")),
        ("owner/name#sub/dir", ("owner/name", "main", "sub/dir")),
        ("hf:owner/name@abc#sub/dir", ("owner/name", "abc", "sub/dir")),
        ("owner/name:model@abc", ("owner/name:model", "abc", "")),
    ],
)
def test_parse_coord(coord, expected):
    assert mlp.parse_coord(coord) == expected


def test_sha_regex_rejects_short_and_branch_names():
    assert mlp._SHA_RE.match("a" * 40)
    assert not mlp._SHA_RE.match("a" * 12)   # a truncated sha is not a coordinate
    assert not mlp._SHA_RE.match("main")


def test_metric_key_sanitising():
    # MLflow rejects anything outside this alphabet, and an exception raised
    # mid-run loses everything computed so far.
    assert mlp._METRIC_KEY_RE.sub("_", "ade@3.0s") == "ade_3.0s"
    assert mlp._METRIC_KEY_RE.sub("_", "scenario.curve.score") == "scenario.curve.score"


# -- credentials -----------------------------------------------------------
@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://x-access-token:ghp_SECRET@github.com/o/r.git",
         "https://github.com/o/r.git"),
        ("https://user:pw@gitlab.com/o/r.git", "https://gitlab.com/o/r.git"),
        ("https://github.com/o/r.git", "https://github.com/o/r.git"),
        ("git@github.com:o/r.git", "git@github.com:o/r.git"),  # ssh, no userinfo
        ("", ""),
    ],
)
def test_git_remote_never_carries_a_credential(url, expected):
    """The tracking server has no authentication and tags cannot be edited.

    A token written into git_remote is published to everyone who can reach it,
    permanently.
    """
    out = mlp._strip_credentials(url)
    assert out == expected
    assert "ghp_" not in out and ":pw@" not in out


# -- params ----------------------------------------------------------------
def test_flatten_produces_dotted_keys():
    flat = mlp.flatten({"opt": {"lr": 1e-4, "sched": {"warmup": 100}}, "bs": 8})
    assert flat == {"opt.lr": 1e-4, "opt.sched.warmup": 100, "bs": 8}


def test_flatten_leaves_lists_alone():
    # A list is a value, not a level: opt.betas.0 would be worse than useless.
    assert mlp.flatten({"opt": {"betas": [0.9, 0.99]}}) == {"opt.betas": [0.9, 0.99]}


# -- the rule engine -------------------------------------------------------
def _run(run_type: str, **tags):
    """A Run with its state set directly; check() touches no MLflow."""
    run = object.__new__(mlp.Run)
    run.run_type = run_type
    run._metrics = set(tags.pop("_metrics", ()))
    run.unresolved = list(tags.pop("_unresolved", ()))
    run._tags = {
        "git_commit": "a" * 40,
        "hf_models": '["o/n@' + "b" * 40 + '"]',
        "model_source": "hf:o/n:model@" + "b" * 40,
        **tags,
    }
    return run


DURABLE = "hf:owner/name-evals@" + "c" * 40 + "#runs/x/"


def test_complete_infer_run_passes():
    assert _run("infer", output_uri=DURABLE).check() == []


def test_complete_eval_run_passes():
    assert _run("eval", output_uri=DURABLE, _metrics={"score"}).check() == []


def test_train_run_needs_no_conditioning_source():
    """It is not one of the recording rules' coordinates.

    A helper every project copies cannot fail a run over a concept that
    project does not have.
    """
    assert _run("train", _metrics={"val_loss"}).check() == []


@pytest.mark.parametrize("run_type", ["eval", "infer"])
def test_local_path_is_not_an_archive(run_type):
    """The failure this check exists for.

    upload_run_dir swallows its exceptions and returns None, and the caller
    falls back to path:<dir>. Accepting that hands back a PASS for a run whose
    outputs exist on exactly one machine's disk.
    """
    problems = _run(run_type, output_uri="path:/home/thor/out/x",
                    _metrics={"score"}).check()
    assert any("survive this machine" in p for p in problems)


def test_infer_without_output_uri_fails():
    problems = _run("infer").check()
    assert any("output_uri" in p for p in problems)


def test_eval_without_score_fails():
    problems = _run("eval", output_uri=DURABLE).check()
    assert any("score" in p for p in problems)


def test_missing_git_commit_fails():
    problems = _run("train", git_commit="", _metrics={"val_loss"}).check()
    assert any("git_commit" in p for p in problems)


def test_missing_data_coordinate_fails():
    problems = _run("train", hf_models="", _metrics={"val_loss"}).check()
    assert any("data coordinate" in p for p in problems)


def test_unresolved_revision_is_a_failure_not_a_shrug():
    """An unpinned coordinate that looks pinned is worse than an absent one."""
    problems = _run("train", _metrics={"val_loss"}, _unresolved=["o/n@main"]).check()
    assert any("unresolved" in p for p in problems)


# -- environment -----------------------------------------------------------
def test_env_tags_never_raises_and_reports_the_machine():
    tags = mlp.env_tags(seed=42, notes="hello")
    assert tags["seed"] == "42"
    assert tags["notes"] == "hello"
    for key in ("env.host", "env.python", "env.platform", "cmd"):
        assert tags.get(key), f"{key} missing -- it cannot be backfilled later"


def test_env_tags_survives_a_broken_collector(monkeypatch):
    """One failing item must not take the others with it.

    A single try around the torch block used to lose the device count and the
    compute capability because the device *name* raised.
    """
    monkeypatch.setattr(mlp.socket, "gethostname",
                        lambda: (_ for _ in ()).throw(OSError("no host")))
    tags = mlp.env_tags()
    assert "env.host" not in tags
    assert tags.get("env.python")


def test_checkpoint_refuses_rather_than_uploading():
    """Checkpoints go to Hugging Face regardless of size. This is not advice."""
    run = object.__new__(mlp.Run)
    with pytest.raises(RuntimeError, match="never go through MLflow"):
        run.checkpoint("ckpt/best.pth")
