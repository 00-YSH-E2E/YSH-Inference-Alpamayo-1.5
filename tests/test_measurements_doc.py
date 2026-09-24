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

"""docs/MEASUREMENTS.md is rendered from the code, and must be current."""

from __future__ import annotations

import re
from pathlib import Path

from alpamayo1_5.trace import measurements_doc as MD
from alpamayo1_5.trace import timing_schema as TS

ROOT = Path(__file__).resolve().parents[1]


def test_the_page_matches_the_code():
    on_disk = (ROOT / "docs" / "MEASUREMENTS.md").read_text()
    assert on_disk == MD.render(), (
        "docs/MEASUREMENTS.md is stale; regenerate it with\n"
        "    python -m alpamayo1_5.trace.measurements_doc > docs/MEASUREMENTS.md")


def test_every_column_group_is_described():
    assert {c.group for c in TS.COLUMNS} <= set(MD.GROUP_TEXT)


def test_every_setting_the_page_names_exists():
    """The settings table is written by hand; each row must still be real."""
    run_sh = (ROOT / "scripts" / "run.sh").read_text()
    runner = (ROOT / "scripts" / "run_inference_tracked.py").read_text()
    for variable, flag, _ in MD.FLAGS:
        assert re.search(rf"^{variable}=", run_sh, re.M), variable
        assert f'"{flag}"' in runner, flag


def test_every_row_kind_the_runner_makes_is_listed():
    runner = (ROOT / "scripts" / "run_inference_tracked.py").read_text()
    made = set(re.findall(r'\("(probe|repeat|memsnap|flops|profile)", ', runner))
    assert made <= {kind for kind, _, _ in MD.ROW_KINDS}
