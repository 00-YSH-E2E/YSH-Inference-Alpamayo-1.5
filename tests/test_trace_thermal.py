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

import pathlib
import time

import pytest

from alpamayo1_5.trace import thermal as TH


@pytest.fixture
def no_sensors(monkeypatch, tmp_path):
    """A board with nothing to read -- every sensor list empty, every path absent."""
    for name in ("_ZONE_TYPES", "_RAILS", "_VIN_POWER", "_CPU_POLICIES", "_COOLING",
                 "_OC_DIRS", "_FAN_DIRS", "_TACH_DIRS"):
        monkeypatch.setattr(TH, name, [])
    for name in ("_GPU_DEVFREQ", "_EMC_DEVFREQ", "_EMC_CAP", "_MEMINFO"):
        monkeypatch.setattr(TH, name, str(tmp_path / "absent"))
    monkeypatch.setattr(TH, "_gpu_busy_pct", lambda: None)


# -- degradation -----------------------------------------------------------
def test_readers_are_empty_not_broken_without_sensors(no_sensors):
    assert TH.read_temps() == {}
    assert TH.read_power_w() == {}
    assert TH.read_fast() == {}
    assert TH.read_slow() == {}


def test_summary_of_nothing_is_empty(no_sensors):
    log = TH.ThermalLog(mode="unknown")
    log.sample()
    # Nothing about the board -- only how long the empty read took.
    assert set(log.summary()) == {"sampler.read_ms_mean"}


def test_peak_of_nothing_is_nan_not_zero(no_sensors):
    """Zero degrees is a reading. "No sensor" is not."""
    peak = TH.ThermalLog(mode="unknown").peak_c
    assert peak != peak


def test_verdict_says_so_when_there_is_nothing_to_say(no_sensors):
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
    """A sampler must never be the reason a run dies -- and a failing sensor costs
    that sensor, not the reading. The temperatures failing used to drop every
    sample whole, power included; now the rest of the reading is kept."""
    def boom():
        raise OSError("sysfs went away")

    monkeypatch.setattr(TH, "read_temps", boom)
    log = TH.ThermalLog(mode="unknown")
    with log.sampling(period_s=0.02):
        time.sleep(0.15)
    assert log.samples, "the sampler stopped"
    assert all(not any(k.endswith("-thermal") for k in s) for s in log.samples)


def test_power_mode_is_read_once():
    """It cannot change mid-run, and it costs a subprocess."""
    TH.power_mode.cache_clear()
    first = TH.power_mode()
    assert TH.power_mode() is first
    assert TH.power_mode.cache_info().hits >= 1


def test_a_zone_that_fails_with_typeerror_is_skipped_not_fatal(monkeypatch, tmp_path):
    """Thor's GPU zone raises TypeError, not OSError, while the GPU is powered off.

    Before this was caught, one such zone discarded the whole reading, and in
    the sampler thread every sample was lost without a word.
    """
    for name, temp in (("cpu-thermal", "41000"), ("gpu-thermal", None)):
        zone = tmp_path / name
        zone.mkdir()
        (zone / "type").write_text(name + "\n")
        (zone / "temp").write_text(temp or "0")
    monkeypatch.setattr(TH, "_ZONE_TYPES", sorted(str(p / "type") for p in tmp_path.iterdir()))

    real = pathlib.Path.read_text

    def read_text(self, *args, **kwargs):
        if self.parent.name == "gpu-thermal" and self.name == "temp":
            raise TypeError("can't concat NoneType to bytes")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)
    assert TH.read_temps() == {"cpu-thermal": 41.0}



# -- the two tiers, on a fake board ------------------------------------------------
@pytest.fixture
def board(no_sensors, monkeypatch, tmp_path):
    """A small fake sysfs tree: one INA3221 rail, the INA238 input, the GPU clock,
    the over-current counters, a cooling device, a CPU policy, memory."""
    def write(rel, text):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return str(path)

    rail = write("ina3221/in1_label", "VDD_GPU\n")
    write("ina3221/in1_input", "19700\n")      # mV
    write("ina3221/curr1_input", "5060\n")     # mA: the current limit
    vin = write("ina238/power1_input", "150000000\n")   # uW
    write("ina238/in1_label", "VIN\n")
    gpu = str(tmp_path / "gpu-gpc-0")
    write("gpu-gpc-0/cur_freq", "1575000000\n")
    for index, count in ((1, 0), (2, 8), (3, 813)):
        write(f"oc/oc{index}_event_cnt", f"{count}\n")
    cool = write("cooling_device0/type", "gpu-throttle-alert\n")
    write("cooling_device0/cur_state", "1\n")
    write("policy0/scaling_cur_freq", "2601000\n")
    meminfo = write("meminfo", "MemTotal: 128790092 kB\nMemAvailable: 104857600 kB\n")
    zone = write("zone0/type", "tj-thermal\n")
    write("zone0/temp", "64500\n")

    monkeypatch.setattr(TH, "_RAILS", [rail])
    monkeypatch.setattr(TH, "_VIN_POWER", [vin])
    monkeypatch.setattr(TH, "_GPU_DEVFREQ", gpu)
    monkeypatch.setattr(TH, "_OC_DIRS", [str(tmp_path / "oc")])
    monkeypatch.setattr(TH, "_COOLING", [cool])
    monkeypatch.setattr(TH, "_CPU_POLICIES", [str(tmp_path / "policy0")])
    monkeypatch.setattr(TH, "_MEMINFO", meminfo)
    monkeypatch.setattr(TH, "_ZONE_TYPES", [zone])
    return tmp_path


def test_the_fast_tier_reads_power_clock_and_counters(board):
    fast = TH.read_fast()
    assert fast["power.VDD_GPU"] == pytest.approx(19.7 * 5.06)
    assert fast["curr.VDD_GPU"] == pytest.approx(5.06)
    assert fast["power.VIN"] == pytest.approx(150.0)
    assert fast["freq.gpu"] == pytest.approx(1575.0)
    assert fast["oc.oc3"] == 813.0


def test_the_slow_tier_reads_the_rest(board):
    slow = TH.read_slow()
    assert slow["tj-thermal"] == 64.5
    assert slow["cool.gpu-throttle-alert"] == 1.0
    assert slow["freq.cpu_policy0"] == pytest.approx(2601.0)
    assert slow["mem.available"] == pytest.approx(104857600 * 1024 / 1e9)


def test_a_powered_off_gpu_is_absent_not_zero(board):
    (board / "gpu-gpc-0" / "cur_freq").write_text("0\n")
    assert "freq.gpu" not in TH.read_fast()


def test_one_unreadable_sensor_costs_only_itself(board):
    (board / "ina238" / "power1_input").unlink()
    fast = TH.read_fast()
    assert "power.VIN" not in fast and "power.VDD_GPU" in fast


def test_units_follow_the_sensor_family():
    assert TH.unit_of("power.VIN") == "W" and TH.unit_of("curr.VDD_GPU") == "A"
    assert TH.unit_of("freq.gpu") == "MHz" and TH.unit_of("tj-thermal") == "C"
    assert TH.unit_of("oc.oc3") == "count" and TH.unit_of("fan.rpm") == "rpm"


def test_the_series_is_stamped_labelled_and_tabled(board):
    log = TH.ThermalLog(mode="unknown")
    log.mark("infer", 3)
    log.sample()
    log.mark("post")
    log.sample()
    stamps, values = log.series("power.VIN")
    assert len(stamps) == 2 and values == [150.0, 150.0]
    table = log.table()
    assert set(table["phase"]) == {"infer", "post"}
    assert set(table["clip_index"]) == {3}
    rows = [i for i, s in enumerate(table["sensor"]) if s == "freq.gpu"]
    assert all(table["unit"][i] == "MHz" for i in rows)


def test_the_summary_keeps_its_old_keys_and_adds_the_new_ones(board):
    log = TH.ThermalLog(mode="unknown")
    log.sample()
    (board / "oc" / "oc3_event_cnt").write_text("820\n")
    log.sample()
    out = log.summary()
    assert out["temp.tj_max_c"] == 64.5
    assert out["power.VIN_mean_w"] == pytest.approx(150.0)
    assert out["clock.gpu_mhz_mean"] == pytest.approx(1575.0)
    assert out["oc.oc3_events"] == 7.0
    assert out["temp.throttled_samples"] == 2.0


def test_the_sampler_runs_both_tiers_at_their_rates(board):
    log = TH.ThermalLog(mode="unknown", hz=50.0)
    with log.sampling():
        time.sleep(0.3)
    stamps, _ = log.series("power.VIN")
    slow, _ = log.series("tj-thermal")
    assert len(stamps) >= 8
    # Slow every five fast periods (0.1 s): the first tick and a couple more.
    assert 2 <= len(slow) < len(stamps)
    assert log.effective_hz > 20.0


def test_phases_can_be_marked_without_a_handle(board):
    log = TH.ThermalLog(mode="unknown", hz=50.0)
    with log.sampling():
        TH.mark("infer", 7)
        time.sleep(0.1)
    assert TH.CURRENT.log is None
    table = log.table()
    assert "infer" in table["phase"] and 7 in table["clip_index"]


def test_the_l4t_release_keeps_its_revision():
    text = "# R38 (release), REVISION: 4.0, GCID: 43443517, BOARD: generic, EABI: aarch64"
    assert TH.parse_l4t_release(text) == "R38.4.0"


def test_a_driver_without_a_number_keeps_its_build_date():
    line = ("NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  TempVersion  Release "
            "Build  (bugfix_main)  (buildbrain@x)  Tue Dec 30 15:54:35 PST 2025\nGCC version: 13")
    assert TH.parse_nvrm_version(line) == "TempVersion Tue Dec 30 15:54:35 PST 2025"
    assert TH.parse_nvrm_version("NVRM version: NVIDIA Module for x86_64  580.00  Release") \
        == "580.00"
    assert TH.parse_nvrm_version("") is None
