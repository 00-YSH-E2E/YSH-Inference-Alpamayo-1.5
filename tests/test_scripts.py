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

"""The launch scripts: syntax, sweep axes, machine files and the single-run lock.

Nothing here starts inference. A dry sweep exits before any pre-flight check,
and the lock test is refused before the first combination runs.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("run.sh", "run_sweep.sh", "run_queue.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "ALPAMAYO_LOCK_HELD"}
    env.update(extra)
    return env


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_parse(name):
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)


def test_a_dry_sweep_names_the_graph_arms(tmp_path):
    """Two arms that differ only in CUDA_GRAPH must not share a directory name."""
    config = tmp_path / "sweep.sh"
    config.write_text('SWEEP_NAME="cg-test"\nSWEEP_INFERENCE_STEP=(10)\nSWEEP_CUDA_GRAPH=(0 1)\n')
    done = subprocess.run(["bash", str(ROOT / "scripts" / "run_sweep.sh")],
                          env=_env(DRY_RUN="1", SWEEP_CONFIG=str(config)),
                          capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    assert "cg0" in done.stdout and "cg1" in done.stdout
    assert "2개 조합" in done.stdout


def _copy_scripts(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in SCRIPTS:
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    return scripts


def test_env_file_overrides_the_tracked_settings(tmp_path):
    """Machine facts live in the untracked .env; the tracked block is the default."""
    scripts = _copy_scripts(tmp_path)
    (tmp_path / ".env").write_text('VARIANT="FromEnv"\nNETWORK_PROXY=""\n')
    out = subprocess.run(
        ["bash", "-c", f'source "{scripts / "run.sh"}"; echo "$VARIANT|$ENV_LOADED|$NETWORK_PROXY|"'],
        env=_env(), capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "FromEnv|1||"


def test_without_an_env_file_the_tracked_settings_stand(tmp_path):
    scripts = _copy_scripts(tmp_path)
    out = subprocess.run(
        ["bash", "-c", f'source "{scripts / "run.sh"}"; echo "$ENV_LOADED"'],
        env=_env(), capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "0"


def test_a_held_lock_refuses_a_second_sweep(tmp_path):
    """Two runs on one GPU inflate each other's latency; the second must not start."""
    scripts = _copy_scripts(tmp_path)
    lock = tmp_path / "test.lock"
    (tmp_path / ".env").write_text(f'LOCK_FILE="{lock}"\n')
    config = tmp_path / "sweep.sh"
    config.write_text('SWEEP_NAME="lock-test"\nSWEEP_INFERENCE_STEP=(10)\n')
    with open(lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        done = subprocess.run(["bash", str(scripts / "run_sweep.sh")],
                              env=_env(DRY_RUN="0", SWEEP_CONFIG=str(config)),
                              capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert done.returncode == 1
    assert "막힘" in done.stderr
    # Refused before the first combination: nothing was launched.
    assert "[1/" not in done.stdout
