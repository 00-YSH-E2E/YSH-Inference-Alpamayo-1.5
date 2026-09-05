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

"""Which clips, which reasoning, and how a clip becomes model inputs.

Three sources, none of them derived here:

* the clip list with its ``split`` column -- training reads only ``train``,
  and refuses a list without the column rather than assume;
* the reasoning bank written by ``scripts/prepare_cot.py`` -- one recorded
  reasoning sequence per (clip, t0, sample), with the hash that pairs a
  student sample to the teacher sample it must reproduce;
* the dataset itself, through the same loader and processor the inference
  harness uses, so the prompt the student is conditioned on is byte-for-byte
  the prompt the teacher saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset


def load_clips(path: str, split: str = "train") -> list[str]:
    """Clip ids of one split, in file order. Refuses a list with no split."""
    table = pd.read_parquet(path)
    if "split" not in table.columns:
        raise ValueError(f"{path} has no `split` column; training must not guess which "
                         "clips are held out")
    chosen = table[table["split"] == split]["clip_id"].tolist()
    if not chosen:
        raise ValueError(f"no clips with split={split!r} in {path}")
    return chosen


@dataclass(frozen=True)
class CotSample:
    k: int
    ids: np.ndarray           # int64, the generated sequence ending with the markers
    n_cot: int
    cot_hash: str


def load_cot_bank(path: str, split: str | None = "train") -> dict[tuple[str, int], list[CotSample]]:
    """``(clip_id, t0_us) -> [CotSample, ...]`` sorted by sample index."""
    table = pd.read_parquet(path)
    if split is not None:
        if "split" not in table.columns:
            raise ValueError(f"{path} has no `split` column")
        table = table[table["split"] == split]
    bank: dict[tuple[str, int], list[CotSample]] = {}
    for row in table.sort_values(["clip_id", "t0_us", "sample_k"]).itertuples():
        bank.setdefault((row.clip_id, int(row.t0_us)), []).append(CotSample(
            k=int(row.sample_k), ids=np.asarray(row.token_ids, dtype=np.int64),
            n_cot=int(row.n_cot_tokens), cot_hash=str(row.cot_hash),
        ))
    if not bank:
        raise ValueError(f"reasoning bank {path} is empty for split={split!r}")
    return bank


def load_clip(avdi, processor, clip_id: str, t0_us: int, device: str | torch.device = "cuda",
              with_future: bool = True) -> dict:
    """The model inputs for one (clip, t0), exactly as the inference harness builds them.

    Returns the dict :func:`conditioning.build_prompt_kv` consumes:
    ``tokenized_data``, ``ego_history_xyz``/``_rot`` and, when ``with_future``,
    ``ego_future_xyz``/``_rot`` for the GT action.
    """
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi, maybe_stream=False)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"])
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    payload = {"tokenized_data": inputs,
               "ego_history_xyz": data["ego_history_xyz"],
               "ego_history_rot": data["ego_history_rot"]}
    if with_future:
        payload["ego_future_xyz"] = data["ego_future_xyz"]
        payload["ego_future_rot"] = data["ego_future_rot"]
    return helper.to_device(payload, str(device))
