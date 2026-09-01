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

"""Thermal logging: aggregation, thresholds, and the non-Jetson path.

The sysfs globs resolve at import, so on any machine that is not a Jetson they
are empty and every reader returns nothing. That path matters more than it
looks: it is where CI runs, and a logger that raised there would take the run
down with it.

Note what this cannot check. The zone names (``tj-thermal``) and rail names
(``VDD_IN``) differ between Orin and Thor, so passing here says nothing about
what the numbers mean on the board the runs happen on.
"""

from __future__ import annotations

import time

import pytest

from alpamayo1_5.trace import thermal as TH


# -- degradation -----------------------------------------------------------
def test_readers_are_empty_not_broken_without_sensors(monkeypatch):
    monkeypatch.setattr(TH, "_ZONE_TYPES", [])
    monkeypatch.setattr(TH, "_RAILS", [])
    assert TH.read_temps() == {}
    assert TH.read_power_w() == {}


def test_summary_of_nothing_is_empty(monkeypatch):
    monkeypatch.setattr(TH, "_ZONE_TYPES", [])
    monkeypatch.setattr(TH, "_RAILS", [])
    log = TH.ThermalLog(mode="unknown")
    log.sample()
    assert log.summary() == {}


def test_peak_of_nothing_is_nan_not_zero(monkeypatch):
    """Zero degrees is a reading. "No sensor" is not."""
    monkeypatch.setattr(TH, "_ZONE_TYPES", [])
    monkeypatch.setattr(TH, "_RAILS", [])
    peak = TH.ThermalLog(mode="unknown").peak_c
    assert peak != peak


def test_verdict_says_so_when_there_is_nothing_to_say(monkeypatch):
    monkeypatch.setattr(TH, "_ZONE_TYPES", [])
    monkeypatch.setattr(TH, "_RAILS", [])
    assert "no readings" in TH.ThermalLog(mode="unknown").verdict()


# -- aggregation -----------------------------------------------------------
def _log(*readings):
    log = TH.ThermalLog(mode="25W")
    log.samples.extend(readings)
    return log


def test_summary_reports_max_and_mean_per_zone():
    log = _log({"tj-thermal": 50.0}, {"tj-thermal": 70.0})
    out = log.summary()
    assert out["temp.tj_max_c"] == 70.0
    assert out["temp.tj_mean_c"] == 60.0
    assert out["temp.peak_c"] == 70.0


def test_power_is_averaged_across_the_run():
    out = _log({"power.VDD_IN": 4.0}, {"power.VDD_IN": 8.0}).summary()
    assert out["power.VDD_IN_mean_w"] == 6.0


def test_a_missing_zone_in_one_sample_does_not_break_the_mean():
    """Sensors come and go; a partial reading is still worth keeping."""
    out = _log({"tj-thermal": 50.0}, {}, {"tj-thermal": 70.0}).summary()
    assert out["temp.tj_mean_c"] == 60.0


# -- thresholds ------------------------------------------------------------
@pytest.mark.parametrize(
    "peak, fan, throttle",
    [(40.0, 0.0, 0.0), (85.0, 1.0, 0.0), (105.0, 1.0, 1.0)],
)
def test_threshold_flags_are_numbers_so_they_can_be_filtered(peak, fan, throttle):
    out = _log({"tj-thermal": peak}).summary()
    assert out["temp.fan_engaged"] == fan
    assert out["temp.throttle_risk"] == throttle


def test_verdict_escalates_with_temperature():
    assert "cool throughout" in _log({"tj-thermal": 40.0}).verdict()
    assert "fan engaged" in _log({"tj-thermal": 85.0}).verdict()
    assert "suspect" in _log({"tj-thermal": 105.0}).verdict()
    assert "not comparable" in _log({"tj-thermal": 112.0}).verdict()


# -- background sampling ---------------------------------------------------
def test_sampling_collects_while_the_block_runs_and_stops_after():
    """Power has to be read while the GPU is busy.

    Sampling between clips put every reading after inference had finished and
    after the figure was drawn, so the power averages described an idle board.
    """
    log = TH.ThermalLog(mode="unknown")
    with log.sampling(period_s=0.02):
        time.sleep(0.25)
    during = len(log.samples)
    assert during >= 2, "the sampler never ran"
    assert log._thread is None, "the thread outlived its block"

    time.sleep(0.1)
    assert len(log.samples) == during, "still sampling after the block exited"


def test_sampling_survives_a_failing_reader(monkeypatch):
    """A sampler must never be the reason a run dies."""
    def boom():
        raise OSError("sysfs went away")

    monkeypatch.setattr(TH, "read_temps", boom)
    log = TH.ThermalLog(mode="unknown")
    with log.sampling(period_s=0.02):
        time.sleep(0.1)
    assert log.samples == []


def test_power_mode_is_read_once():
    """It cannot change mid-run, and it costs a subprocess."""
    TH.power_mode.cache_clear()
    first = TH.power_mode()
    assert TH.power_mode() is first
    assert TH.power_mode.cache_info().hits >= 1
