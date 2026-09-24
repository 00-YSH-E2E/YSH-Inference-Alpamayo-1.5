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

"""Which extra passes each clip gets, and in what order.

The runner imports torch and the dataset package, so this runs in the project
venv and is skipped in CI.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("torch", reason="the runner imports torch")
pytest.importorskip("physical_ai_av", reason="the runner imports the dataset package")

import run_inference_tracked as R  # noqa: E402


def args(**kw):
    base = dict(overhead_probe=0, timing_repeats=0, repeat_clips=5, trace_level="basic")
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_no_extra_passes_by_default():
    assert R.extra_passes_for(0, args()) == []


def test_the_probe_pair_alternates_its_order():
    """A slow drift must not load onto one side of the off/on comparison."""
    a = args(overhead_probe=2)
    assert R.extra_passes_for(0, a) == [("probe", "off"), ("probe", "basic")]
    assert R.extra_passes_for(1, a) == [("probe", "basic"), ("probe", "off")]
    assert R.extra_passes_for(2, a) == []


def test_repeats_follow_the_probe_on_the_first_clips():
    a = args(overhead_probe=1, timing_repeats=2, repeat_clips=1)
    assert R.extra_passes_for(0, a) == [("probe", "off"), ("probe", "basic"),
                                        ("repeat", "basic"), ("repeat", "basic")]
    assert R.extra_passes_for(1, a) == []


def test_a_run_at_level_off_has_nothing_to_probe():
    assert R.extra_passes_for(0, args(overhead_probe=4, trace_level="off")) == []
