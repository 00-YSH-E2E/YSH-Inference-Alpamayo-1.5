# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Board temperature and power, sampled alongside a run.

Latency is the number this project is judged on, and thermal throttling moves
it without moving anything else. A long run that starts cool and ends hot
produces trajectories that look identical and timings that do not, and after
the fact there is no way to tell that apart from a real regression -- the
temperature at the time is simply gone.

So it gets recorded. Readings come straight from sysfs rather than by shelling
out to tegrastats: about 0.6ms per sample, which against a twenty-second clip
is small enough not to disturb what it is measuring.

Thresholds on Jetson AGX Thor (from the thermal zone trip points): the fan
steps up at 80/86/91/100C, the GPU throttles at 109C, and 114.5C is the
critical shutdown. Nothing here protects the board -- the hardware does that
on its own. The point is to know, later, whether a timing can be trusted.
"""

from __future__ import annotations

import glob
import os
import pathlib
import subprocess
from dataclasses import dataclass, field

# The first fan step. Below this, nothing has begun to intervene.
FAN_STEP_C = 80.0
# Past this the SoC is actively pulling clocks back; a latency measured here is
# not comparable to one measured cool.
THROTTLE_WARN_C = 100.0
GPU_THROTTLE_C = 109.0

_ZONE_TYPES = sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*/type"))
_RAILS = sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"))


def read_temps() -> dict[str, float]:
    """Every thermal zone, in Celsius. Missing sensors are skipped, never fatal."""
    out: dict[str, float] = {}
    for type_path in _ZONE_TYPES:
        try:
            name = open(type_path).read().strip()
            out[name] = int(open(type_path.replace("/type", "/temp")).read()) / 1000.0
        except (OSError, ValueError):
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
        except (OSError, ValueError):
            continue
    return out


def power_mode() -> str:
    """The nvpmodel profile, which caps clocks and therefore latency."""
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


@dataclass
class ThermalLog:
    """Samples taken over a run, reduced to what makes a timing trustworthy or not."""

    samples: list[dict[str, float]] = field(default_factory=list)
    mode: str = field(default_factory=power_mode)

    def sample(self) -> dict[str, float]:
        """Take one reading. Cheap enough to call between clips."""
        reading = read_temps()
        reading.update({f"power.{k}": v for k, v in read_power_w().items()})
        self.samples.append(reading)
        return reading

    def _series(self, key: str) -> list[float]:
        return [s[key] for s in self.samples if key in s]

    @property
    def peak_c(self) -> float:
        """Hottest junction reading seen. This is the number that decides trust."""
        tj = self._series("tj-thermal") or [
            v for s in self.samples for k, v in s.items() if k.endswith("-thermal")
        ]
        return max(tj) if tj else float("nan")

    def summary(self) -> dict[str, float]:
        """Aggregates for the run record."""
        if not self.samples:
            return {}
        out: dict[str, float] = {}
        for key in ("tj-thermal", "gpu-thermal", "cpu-thermal"):
            series = self._series(key)
            if series:
                short = key.replace("-thermal", "")
                out[f"temp.{short}_max_c"] = max(series)
                out[f"temp.{short}_mean_c"] = sum(series) / len(series)
        for key in {k for s in self.samples for k in s if k.startswith("power.")}:
            series = self._series(key)
            if series:
                out[f"{key}_mean_w"] = sum(series) / len(series)
        peak = self.peak_c
        if peak == peak:  # not NaN
            out["temp.peak_c"] = peak
            # Recorded as numbers so a query can filter on them later.
            out["temp.fan_engaged"] = float(peak >= FAN_STEP_C)
            out["temp.throttle_risk"] = float(peak >= THROTTLE_WARN_C)
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
