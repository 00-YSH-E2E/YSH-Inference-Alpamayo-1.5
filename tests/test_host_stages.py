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

"""Host stages around the model call, with stand-ins for the dataset and processor.

Pure Python; runs in CI.
"""

from __future__ import annotations

import time
import types

import pytest

from alpamayo1_5.trace import host_stages as HS
from alpamayo1_5.trace import timing_schema as TS


@pytest.fixture(autouse=True)
def fresh_clock():
    HS.CURRENT.clock = HS.StageClock()
    yield
    HS.CURRENT.clock = None


def test_spans_accumulate_and_the_remainder_is_what_is_left():
    clock = HS.CURRENT.clock
    clock.add("data_load_ms", 100.0)
    clock.add("preprocess_ms", 30.0)
    clock.add("preprocess_image_ms", 20.0)
    clock.add("model_call_ms", 500.0)
    clock.add("data_load_ms", 50.0)          # a second visit adds up
    row = clock.row(clip_wall_ms=700.0)
    assert row["data_load_ms"] == 150.0
    assert row["preprocess_text_ms"] == 10.0
    # Nested stages (the image half) are not subtracted twice.
    assert row["clip_other_ms"] == pytest.approx(700.0 - 150.0 - 30.0 - 500.0)


def test_stages_that_did_not_run_are_absent():
    row = HS.StageClock().row()
    assert row["render_ms"] is None and row["preprocess_text_ms"] is None
    assert row["clip_other_ms"] is None


def test_every_stage_is_a_declared_column():
    assert not TS.unknown_keys(HS.StageClock().row(clip_wall_ms=1.0))


class _Camera:
    def decode_images_from_timestamps(self, ts):
        time.sleep(0.002)
        return "frames", ts


class _Dataset:
    features = types.SimpleNamespace(LABELS=types.SimpleNamespace(EGOMOTION="ego"))

    def get_clip_feature(self, clip_id, feature, maybe_stream=False):
        time.sleep(0.001)
        return "interpolator" if feature == "ego" else _Camera()


def test_the_dataset_split_separates_ego_fetch_and_decode():
    avdi = HS.wrap_dataset(_Dataset())
    avdi.get_clip_feature("c1", "ego")
    for cam in ("front", "left"):
        avdi.get_clip_feature("c1", cam).decode_images_from_timestamps([1, 2])
    ms = HS.CURRENT.clock.ms
    assert ms["data_ego_ms"] > 0.0
    assert ms["data_cam_fetch_ms"] > 0.0
    assert ms["data_cam_decode_ms"] >= 2 * 2.0 * 0.9


def test_a_reader_that_refuses_attributes_is_left_untimed_not_broken():
    class Slotted:
        __slots__ = ()

        def decode_images_from_timestamps(self, ts):
            return ts

    class Dataset(_Dataset):
        def get_clip_feature(self, clip_id, feature, maybe_stream=False):
            return Slotted()

    avdi = HS.wrap_dataset(Dataset())
    assert avdi.get_clip_feature("c1", "front").decode_images_from_timestamps([3]) == [3]


def test_the_processor_split_times_the_image_half():
    image = types.SimpleNamespace(preprocess=lambda *a, **k: time.sleep(0.002) or "pixels")
    processor = HS.wrap_processor(types.SimpleNamespace(image_processor=image))
    assert processor.image_processor.preprocess() == "pixels"
    assert HS.CURRENT.clock.ms["preprocess_image_ms"] >= 1.5


def test_nothing_is_written_without_a_current_clock():
    HS.CURRENT.clock = None
    avdi = HS.wrap_dataset(_Dataset())
    avdi.get_clip_feature("c1", "ego")   # must not raise
