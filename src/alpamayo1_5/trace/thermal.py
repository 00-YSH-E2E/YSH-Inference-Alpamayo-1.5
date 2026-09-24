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

"""The board, sampled alongside a run: temperature, power, clocks, throttling.

Latency is the number this project is judged on, and the board moves it
without moving anything else. A long run that starts cool and ends hot
produces trajectories that look identical and timings that do not; a clock
that the governor has not yet ramped makes the first passes of a run slower
than the rest; and on the Thor the GPU runs into a hardware current limit
under load. After the fact none of that can be told apart from a real
regression -- the state of the board at the time is simply gone.

So it is recorded, in two tiers:

* **fast** (``hz``, 10 by default): the power rails, the GPU clock and the
  over-current counters. Fast enough that a clip's energy can be integrated
  from them, and a segment's too when it lasts a couple of seconds.
* **slow** (five fast periods, at most one second -- 2 Hz at the default,
  the old cadence): temperatures, cooling-device states, CPU and memory
  clocks, the fan, free memory and the GPU's busy share from NVML.

Every reading keeps its ``perf_counter`` stamp, the same clock the tracer
uses, and the whole series is written to ``thermal.parquet`` -- so a reading
can be attributed to the clip, and the phase of the clip, it was taken in.

Readings come straight from sysfs, about a quarter of a millisecond each (the
INA sensors sit on I2C). Every sensor fails on its own: a GPU powered off with
no CUDA context makes its zone unreadable, and that must cost the zone, not
the reading.

The over-current counters matter most. On the Thor the GPU sits at the
INA3221's current limit (about 5 A on VDD_GPU, near 100 W) under a heavy
load, and each time it is hit the SoC pulls clocks back and ``oc3`` counts
it. Throttle reasons are not readable without root, and temperature alone
reads "fine" through all of it; these counters are the evidence.

Thresholds on Jetson AGX Thor (from the thermal zone trip points): the fan
steps up at 80/86/91/100C, the GPU throttles at 109C, and 114.5C is the
critical shutdown. Nothing here protects the board -- the hardware does that
on its own. The point is to know, later, whether a timing can be trusted.
"""

from __future__ import annotations

import array
import contextlib
import functools
import glob
import os
import pathlib
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

# The first fan step. Below this, nothing has begun to intervene.
FAN_STEP_C = 80.0
# Past this the SoC is actively pulling clocks back; a latency measured here is
# not comparable to one measured cool.
THROTTLE_WARN_C = 100.0
GPU_THROTTLE_C = 109.0

#: Version of thermal.parquet's columns.
THERMAL_SCHEMA_VERSION = 1

# Sensors, found once by path. Module-level so tests can point them at a fake
# tree. hwmon numbers change between boots; driver paths do not.
_ZONE_TYPES = sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*/type"))
_RAILS = sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"))
_VIN_POWER = sorted(glob.glob("/sys/bus/i2c/drivers/ina238/*/hwmon/hwmon*/power1_input"))
_GPU_DEVFREQ = "/sys/class/devfreq/gpu-gpc-0"
_EMC_DEVFREQ = "/sys/class/devfreq/bwmgr"
_EMC_CAP = "/sys/kernel/nvpmodel_clk_cap/emc"
_CPU_POLICIES = sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*"))
_COOLING = sorted(glob.glob("/sys/class/thermal/cooling_device*/type"))
_MEMINFO = "/proc/meminfo"


def _hwmon_named(fragment: str) -> list[str]:
    """hwmon directories whose ``name`` contains ``fragment``."""
    out = []
    for name_path in sorted(glob.glob("/sys/class/hwmon/hwmon*/name")):
        try:
            if fragment in pathlib.Path(name_path).read_text().strip():
                out.append(os.path.dirname(name_path))
        except (OSError, TypeError, ValueError):
            continue
    return out


_OC_DIRS = _hwmon_named("soctherm_oc")
_FAN_DIRS = [d for d in _hwmon_named("fan") if os.path.exists(os.path.join(d, "pwm1"))]
_TACH_DIRS = [d for d in sorted(glob.glob("/sys/class/hwmon/hwmon*"))
              if os.path.exists(os.path.join(d, "rpm"))]

#: Unit of each sensor family, by the prefix of its name.
UNITS = {"power": "W", "curr": "A", "freq": "MHz", "oc": "count", "cool": "state",
         "mem": "GB", "util": "%"}


def unit_of(sensor: str) -> str:
    if sensor.endswith("-thermal"):
        return "C"
    if sensor.startswith("fan."):
        return "rpm" if sensor.endswith("rpm") else "pwm"
    return UNITS.get(sensor.split(".", 1)[0], "")


def read_temps() -> dict[str, float]:
    """Every thermal zone, in Celsius. Missing sensors are skipped, never fatal."""
    out: dict[str, float] = {}
    for type_path in _ZONE_TYPES:
        try:
            # Read through pathlib so the handles close deterministically. Now
            # that a background thread calls this several times a second, an
            # open handle per zone per sample adds up before the refcount
            # collector gets to it.
            name = pathlib.Path(type_path).read_text().strip()
            raw = pathlib.Path(type_path.replace("/type", "/temp")).read_text()
            out[name] = int(raw) / 1000.0
        except (OSError, ValueError, TypeError):
            # TypeError is not hypothetical. On Thor the GPU powers off when no
            # CUDA context exists, and reading gpu-thermal then fails inside
            # CPython's buffered reader with "can't concat NoneType to bytes"
            # rather than an OSError. Uncaught, it discarded every zone of the
            # sample -- and, in the sampler thread, the whole sample silently.
            continue
    return out


def read_power_w() -> dict[str, float]:
    """Per-rail power in watts, where the board exposes it."""
    out: dict[str, float] = {}
    for label_path in _RAILS:
        path = pathlib.Path(label_path)
        # Only the filename may be rewritten -- the directory contains "ina3221",
        # and a blind replace would rewrite that too.
        channel = path.name.replace("in", "").replace("_label", "")
        try:
            name = path.read_text().strip()
            volts = int((path.parent / f"in{channel}_input").read_text()) / 1000.0
            amps = int((path.parent / f"curr{channel}_input").read_text()) / 1000.0
            out[name] = volts * amps
        except (OSError, ValueError, TypeError):
            # Same failure as read_temps: a rail read while its domain is off.
            continue
    return out


@functools.lru_cache(maxsize=1)
def power_mode() -> str:
    """The nvpmodel profile, which caps clocks and therefore latency.

    Cached: it does not change mid-run, and it was being read by a subprocess
    twice per run -- once for the params and once per ThermalLog.
    """
    try:
        done = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5, check=False
        )
        for line in done.stdout.splitlines():
            if "Power Mode" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return os.environ.get("NV_POWER_MODE", "unknown")


def _read(path: str) -> str | None:
    """One sysfs attribute, or None. Every failure mode of a powered-off domain
    -- OSError, EAGAIN, TypeError from the buffered reader -- lands here."""
    try:
        with open(path, "rb", buffering=0) as handle:
            return handle.read(256).decode().strip()
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None


def _number(path: str) -> float | None:
    text = _read(path)
    try:
        return float(text.split()[0]) if text else None
    except (ValueError, IndexError):
        return None


def read_fast() -> dict[str, float]:
    """The fast tier: rails, board input, GPU clock, over-current counters."""
    out: dict[str, float] = {}
    for label_path in _RAILS:
        directory = os.path.dirname(label_path)
        channel = os.path.basename(label_path).replace("in", "").replace("_label", "")
        name = _read(label_path)
        millivolts = _number(os.path.join(directory, f"in{channel}_input"))
        milliamps = _number(os.path.join(directory, f"curr{channel}_input"))
        if name and millivolts is not None and milliamps is not None:
            out[f"power.{name}"] = millivolts * milliamps / 1e6
            out[f"curr.{name}"] = milliamps / 1000.0
    for power_path in _VIN_POWER:
        # INA238: the whole module's input. Label from in1_label when present.
        label = _read(os.path.join(os.path.dirname(power_path), "in1_label")) or "VIN"
        microwatts = _number(power_path)
        if microwatts is not None:
            out[f"power.{label}"] = microwatts / 1e6
    hertz = _number(os.path.join(_GPU_DEVFREQ, "cur_freq"))
    # Zero means the GPU is powered off, which is an absence, not a clock.
    if hertz:
        out["freq.gpu"] = hertz / 1e6
    for directory in _OC_DIRS:
        for index in (1, 2, 3):
            count = _number(os.path.join(directory, f"oc{index}_event_cnt"))
            if count is not None:
                out[f"oc.oc{index}"] = count
    return out


def read_slow() -> dict[str, float]:
    """The slow tier: temperatures, cooling states, CPU and memory clocks, fan,
    free memory, the GPU's busy share."""
    out: dict[str, float] = {}
    try:
        out.update(read_temps())
    except Exception:
        # The temperatures are the first thing asked about and the last thing
        # allowed to take the rest of the reading down with them.
        pass
    for type_path in _COOLING:
        kind = _read(type_path)
        state = _number(type_path.replace("/type", "/cur_state"))
        if kind and state is not None:
            out[f"cool.{kind}"] = state
    for policy in _CPU_POLICIES:
        khz = _number(os.path.join(policy, "scaling_cur_freq"))
        if khz is not None:
            out[f"freq.cpu_{os.path.basename(policy)}"] = khz / 1000.0
    hertz = _number(os.path.join(_EMC_DEVFREQ, "cur_freq"))
    if hertz:
        out["freq.emc"] = hertz / 1e6
    for directory in _FAN_DIRS[:1]:
        pwm = _number(os.path.join(directory, "pwm1"))
        if pwm is not None:
            out["fan.pwm"] = pwm
    for directory in _TACH_DIRS[:1]:
        rpm = _number(os.path.join(directory, "rpm"))
        if rpm is not None:
            out["fan.rpm"] = rpm
    try:
        with open(_MEMINFO) as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    out["mem.available"] = int(line.split()[1]) * 1024 / 1e9
                    break
    except (OSError, ValueError, IndexError):
        pass
    busy = _gpu_busy_pct()
    if busy is not None:
        out["util.gpu"] = busy
    return out


@functools.lru_cache(maxsize=1)
def _nvml_handle() -> Any:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def _gpu_busy_pct() -> float | None:
    """NVML's GPU busy share. The only source on the Thor -- its GPU is a PCI
    device under nvidia.ko, with no load file in sysfs. About 2 ms a call,
    which is why it lives in the slow tier."""
    handle = _nvml_handle()
    if handle is None:
        return None
    try:
        import pynvml

        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        return None


def hw_inventory() -> dict[str, Any]:
    """What the board is, once per run: release, driver, clock ranges, governors.

    A latency number belongs to a board configuration, and most of it cannot
    be recovered afterwards -- a driver update or a JetPack upgrade leaves no
    trace in the run otherwise.
    """
    out: dict[str, Any] = {}
    release = _read("/etc/nv_tegra_release")
    if release:
        out["hw.l4t_release"] = parse_l4t_release(release)
    driver = _nvml_driver() or parse_nvrm_version(_read("/proc/driver/nvidia/version") or "")
    if driver:
        out["hw.nvidia_driver"] = driver
    for key, path in (
        ("hw.gpu_freq_min_mhz", os.path.join(_GPU_DEVFREQ, "min_freq")),
        ("hw.gpu_freq_max_mhz", os.path.join(_GPU_DEVFREQ, "max_freq")),
        ("hw.emc_freq_max_mhz", os.path.join(_EMC_DEVFREQ, "max_freq")),
        ("hw.emc_cap_mhz", _EMC_CAP),
    ):
        hertz = _number(path)
        if hertz is not None:
            out[key] = hertz / 1e6
    governor = _read(os.path.join(_GPU_DEVFREQ, "governor"))
    if governor:
        out["hw.gpu_governor"] = governor
    if _CPU_POLICIES:
        governor = _read(os.path.join(_CPU_POLICIES[0], "scaling_governor"))
        if governor:
            out["hw.cpu_governor"] = governor
        out["hw.n_cpu_policies"] = len(_CPU_POLICIES)
    return out


def parse_l4t_release(text: str) -> str:
    """``# R38 (release), REVISION: 4.0, ...`` -> ``R38.4.0``."""
    first = text.splitlines()[0] if text else ""
    match = re.search(r"R(\d+).*?REVISION:\s*([\d.]+)", first)
    return f"R{match.group(1)}.{match.group(2)}" if match else first.lstrip("# ").strip()[:60]


def parse_nvrm_version(text: str) -> str | None:
    """The kernel module's version token, with the build date when it is not a
    number -- on this board it reads ``TempVersion``, which alone would make two
    different drivers look the same."""
    first = text.splitlines()[0] if text else ""
    if not first.startswith("NVRM version:"):
        return None
    tokens = first.split()
    for arch in ("aarch64", "x86_64"):
        if arch in tokens and tokens.index(arch) + 1 < len(tokens):
            version = tokens[tokens.index(arch) + 1]
            if not version[:1].isdigit():
                version = f"{version} {' '.join(tokens[-6:])}"
            return version
    return first[len("NVRM version:"):].strip()[:80]


def _nvml_driver() -> str | None:
    """The driver version NVML reports (580.00 on this board)."""
    if _nvml_handle() is None:
        return None
    try:
        import pynvml

        version = pynvml.nvmlSystemGetDriverVersion()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:
        return None


class _Current:
    """The run's sampler, so pipeline code can mark phases without a handle."""

    log: "ThermalLog | None" = None


CURRENT = _Current()


def mark(phase: str, clip_index: int | None = None) -> None:
    """Label readings from now on with ``phase`` (and ``clip_index``, if given)."""
    if CURRENT.log is not None:
        CURRENT.log.mark(phase, clip_index)


@dataclass
class ThermalLog:
    """The board over a run, and what it says about whether a timing can be trusted.

    ``samples`` holds the slow tier's full readings as dicts -- what the
    summary and the verdict are computed from. Every reading of both tiers
    also goes into a compact series, stamped and labelled, which is what
    ``thermal.parquet`` is written from.
    """

    samples: list[dict[str, float]] = field(default_factory=list)
    mode: str = field(default_factory=power_mode)
    hz: float = 10.0

    #: Set while a background sampler is running; cleared to stop it.
    _stop: threading.Event | None = None
    _thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._t_host = array.array("d")
        self._t_unix = array.array("d")
        self._read_ms = array.array("f")
        self._clip = array.array("i")
        self._phase = array.array("H")
        self._phases: list[str] = ["idle"]
        self._series: dict[str, tuple[array.array, array.array]] = {}
        self._cur_phase = 0
        self._cur_clip = -1
        self.effective_hz = float("nan")

    # -- labelling ------------------------------------------------------------
    def mark(self, phase: str, clip_index: int | None = None) -> None:
        if phase not in self._phases:
            self._phases.append(phase)
        self._cur_phase = self._phases.index(phase)
        if clip_index is not None:
            self._cur_clip = int(clip_index)

    # -- reading ----------------------------------------------------------------
    def _record(self, values: dict[str, float], stamp: float, read_ms: float) -> None:
        with self._lock:
            tick = len(self._t_host)
            self._t_host.append(stamp)
            self._t_unix.append(time.time())
            self._read_ms.append(read_ms)
            self._clip.append(self._cur_clip)
            self._phase.append(self._cur_phase)
            for sensor, value in values.items():
                series = self._series.get(sensor)
                if series is None:
                    series = self._series[sensor] = (array.array("i"), array.array("d"))
                series[0].append(tick)
                series[1].append(float(value))

    def _read(self, slow: bool) -> dict[str, float]:
        values: dict[str, float] = {}
        for reader in (read_fast, read_slow) if slow else (read_fast,):
            try:
                values.update(reader())
            except Exception:
                # One tier failing costs its sensors, never the other tier's.
                continue
        return values

    def sample(self) -> dict[str, float]:
        """Take one full reading now. Cheap enough to call between clips."""
        stamp = time.perf_counter()
        reading = self._read(slow=True)
        self._record(reading, stamp, (time.perf_counter() - stamp) * 1000.0)
        self.samples.append(reading)
        return reading

    def latest(self) -> dict[str, float]:
        """The most recent slow reading, without taking another."""
        return self.samples[-1] if self.samples else {}

    @contextlib.contextmanager
    def sampling(self, period_s: float | None = None) -> Iterator["ThermalLog"]:
        """Sample in the background for the duration of the block.

        Between-clip sampling is fine for temperature, which moves on a
        timescale of seconds, but it is the wrong instrument for power: every
        reading lands after inference finished and after the sample figure was
        drawn, so a power average describes an idle GPU. That is not an
        underestimate, it is a different quantity.

        The cadence is drift-free -- tick k is due at start + k * period, not
        a period after the last one finished -- so a slow read does not
        stretch the series.
        """
        period = float(period_s) if period_s else 1.0 / float(self.hz)
        slow_period = min(1.0, 5.0 * period)
        stop = threading.Event()

        def loop() -> None:
            start = time.perf_counter()
            last_slow = -float("inf")
            tick = 0
            while not stop.is_set():
                stamp = time.perf_counter()
                slow = stamp - last_slow >= slow_period
                try:
                    reading = self._read(slow)
                    self._record(reading, stamp, (time.perf_counter() - stamp) * 1000.0)
                    if slow:
                        last_slow = stamp
                        self.samples.append(reading)
                except Exception:
                    # A sampler must never be the reason a run dies.
                    pass
                tick += 1
                stop.wait(max(0.0, start + tick * period - time.perf_counter()))
            elapsed = time.perf_counter() - start
            if elapsed > 0:
                self.effective_hz = tick / elapsed

        thread = threading.Thread(target=loop, name="thermal-sampler", daemon=True)
        self._stop, self._thread = stop, thread
        CURRENT.log = self
        thread.start()
        try:
            yield self
        finally:
            stop.set()
            thread.join(timeout=max(period * 4, 2.0))
            self._stop = self._thread = None
            if CURRENT.log is self:
                CURRENT.log = None

    # -- the series ---------------------------------------------------------------
    @property
    def n_ticks(self) -> int:
        return len(self._t_host)

    def series(self, sensor: str) -> tuple[list[float], list[float]]:
        """``(host times, values)`` of one sensor, in order."""
        with self._lock:
            entry = self._series.get(sensor)
            if entry is None:
                return [], []
            stamps = [self._t_host[t] for t in entry[0]]
            values = list(entry[1])
        return stamps, values

    def sensors(self) -> list[str]:
        with self._lock:
            return sorted(self._series)

    def table(self) -> dict[str, list[Any]]:
        """Long format for thermal.parquet: one row per sensor reading."""
        with self._lock:
            phases = list(self._phases)
            cols: dict[str, list[Any]] = {k: [] for k in (
                "tick", "t_host_s", "t_unix_s", "read_ms", "clip_index", "phase",
                "sensor", "value", "unit")}
            for sensor in sorted(self._series):
                ticks, values = self._series[sensor]
                unit = unit_of(sensor)
                for tick, value in zip(ticks, values):
                    cols["tick"].append(tick)
                    cols["t_host_s"].append(self._t_host[tick])
                    cols["t_unix_s"].append(self._t_unix[tick])
                    cols["read_ms"].append(float(self._read_ms[tick]))
                    cols["clip_index"].append(self._clip[tick])
                    cols["phase"].append(phases[self._phase[tick]])
                    cols["sensor"].append(sensor)
                    cols["value"].append(value)
                    cols["unit"].append(unit)
        return cols

    # -- the summary ------------------------------------------------------------
    def _series_from_samples(self, key: str) -> list[float]:
        return [s[key] for s in self.samples if key in s]

    @property
    def peak_c(self) -> float:
        """Hottest junction reading seen. This is the number that decides trust."""
        tj = self._series_from_samples("tj-thermal") or [
            v for s in self.samples for k, v in s.items() if k.endswith("-thermal")
        ]
        return max(tj) if tj else float("nan")

    def summary(self) -> dict[str, float]:
        """Aggregates for the run record. Every key the 2 Hz sampler reported keeps
        its name; power means now come from the fast tier when it ran."""
        if not self.samples and not self._series:
            return {}
        out: dict[str, float] = {}
        for key in ("tj-thermal", "gpu-thermal", "cpu-thermal", "soc012-thermal",
                    "soc345-thermal"):
            series = self._series_from_samples(key)
            if series:
                short = key.replace("-thermal", "")
                out[f"temp.{short}_max_c"] = max(series)
                out[f"temp.{short}_mean_c"] = sum(series) / len(series)
        power_keys = {k for s in self.samples for k in s if k.startswith("power.")}
        power_keys |= {k for k in self.sensors() if k.startswith("power.")}
        for key in sorted(power_keys):
            _, fast = self.series(key)
            series = fast or self._series_from_samples(key)
            if series:
                out[f"{key}_mean_w"] = sum(series) / len(series)
        peak = self.peak_c
        if peak == peak:  # not NaN
            out["temp.peak_c"] = peak
            # Recorded as numbers so a query can filter on them later.
            out["temp.fan_engaged"] = float(peak >= FAN_STEP_C)
            out["temp.throttle_risk"] = float(peak >= THROTTLE_WARN_C)
        # A cooling device other than the fan in a non-zero state is the SoC
        # capping itself -- evidence of throttling that needs no threshold.
        caps = sorted({k for s in self.samples for k in s
                       if k.startswith("cool.") and "fan" not in k})
        if caps:
            out["temp.throttled_samples"] = float(sum(
                1 for s in self.samples if any(s.get(k, 0) > 0 for k in caps)))
        _, gpu = self.series("freq.gpu")
        if gpu:
            out["clock.gpu_mhz_mean"] = sum(gpu) / len(gpu)
            out["clock.gpu_mhz_min"] = min(gpu)
        for index in (1, 2, 3):
            _, counts = self.series(f"oc.oc{index}")
            if counts:
                out[f"oc.oc{index}_events"] = counts[-1] - counts[0]
        if len(self._read_ms):
            out["sampler.read_ms_mean"] = float(sum(self._read_ms) / len(self._read_ms))
        if self.effective_hz == self.effective_hz:
            out["sampler.hz"] = float(self.effective_hz)
        return out

    def verdict(self) -> str:
        """One line for the console: is the latency from this run usable?"""
        peak = self.peak_c
        if peak != peak:
            return "thermal: no readings"
        if peak >= GPU_THROTTLE_C:
            return f"thermal: peak {peak:.1f}C -- GPU THROTTLED, latency is not comparable"
        if peak >= THROTTLE_WARN_C:
            return f"thermal: peak {peak:.1f}C -- clocks likely pulled back, treat latency as suspect"
        if peak >= FAN_STEP_C:
            return f"thermal: peak {peak:.1f}C -- fan engaged, latency still fine"
        return f"thermal: peak {peak:.1f}C -- cool throughout ({self.mode})"
