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

"""Which checkpoint revision a run loads, and the coordinate it records for it.

The runner imports torch and the dataset package at module level, so this is
skipped where they are absent (CI) and runs in the project venv.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="the runner imports torch")
pytest.importorskip("physical_ai_av", reason="the runner imports the dataset package")

import run_inference_tracked as R  # noqa: E402


def test_the_default_model_loads_its_pinned_snapshot():
    assert R.resolve_model_revision(R.MODEL_REPO, None) == R.MODEL_REVISION
    assert len(R.MODEL_REVISION) == 40


def test_an_explicit_revision_wins():
    sha = "a" * 40
    assert R.resolve_model_revision(R.MODEL_REPO, sha) == sha


def test_another_hub_model_does_not_inherit_nvidias_sha():
    """NVIDIA's sha names nothing in someone else's repository."""
    assert R.resolve_model_revision("someone/pruned-24l", None) is None


def test_a_local_directory_has_no_revision(tmp_path):
    assert R.resolve_model_revision(str(tmp_path), "b" * 40) is None
    assert R.model_coordinate(str(tmp_path), None) == f"path:{tmp_path}"


def test_coordinates_name_a_sha_when_one_is_known():
    assert R.model_coordinate(R.MODEL_REPO, R.MODEL_REVISION) == \
        f"hf:{R.MODEL_REPO}@{R.MODEL_REVISION}"
    # Unpinned hub models still resolve main at record time, as before.
    assert R.model_coordinate("someone/pruned-24l", None) == "hf:someone/pruned-24l@main"
