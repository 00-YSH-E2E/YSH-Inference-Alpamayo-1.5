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

"""A student checkpoint that ``from_pretrained`` loads, without 22 GB per step.

The base checkpoint is five safetensors shards. The first three hold only
the VLM; the fourth holds most of the expert plus 43 VLM tensors; the fifth
holds the rest of the expert and the two projections. The VLM never trains,
so a student differs from the base in the expert and the projections only
-- 4.6 GB of the 22.

A checkpoint directory here is therefore: the trained head written once as
its own shard, the VLM-only shards **symlinked** to the base blobs, the 43
VLM tensors from shard four rewritten once into a shard of their own (so
shard four's stale expert copy is never read), and an index that points every
key at the right file. ``from_pretrained`` follows the index and nothing else,
so the directory loads exactly like the base, with the head replaced.

The config is the base config with three fields changed: the two
``_target_`` strings that name the shortcut head and sampler -- hydra
instantiates from them, and a checkpoint that names the upstream classes
still loads as the upstream model -- and ``attn_implementation`` set to
``sdpa`` because flash-attn is not installed anywhere this runs.

Upload pushes the directory as-is. The hub deduplicates by content hash, so
after the first push the unchanged VLM shards cost a hash and no transfer;
each checkpoint adds the 4.6 GB head.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HEAD_TARGET = "alpamayo1_5.distill.head.PerWaypointActionInProjV2Shortcut"
SAMPLER_TARGET = "alpamayo1_5.distill.flow_matching_shortcut.ShortcutFlowMatching"
HEAD_PREFIXES = ("expert.", "action_in_proj.", "action_out_proj.")
INDEX = "model.safetensors.index.json"
HEAD_SHARD = "head-00001-of-00001.safetensors"


def is_head_key(key: str) -> bool:
    return key.startswith(HEAD_PREFIXES)


def prepare_base_shards(base_snapshot: Path, out: Path) -> Path:
    """One-time: symlink the VLM-only shards and split the VLM tensors out of
    the mixed one. Idempotent; returns ``out``."""
    base_snapshot, out = Path(base_snapshot), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((base_snapshot / INDEX).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    plan: dict[str, str] = {}                       # key -> file under `out`
    for shard in shards:
        keys = [k for k, s in weight_map.items() if s == shard]
        vlm = [k for k in keys if not is_head_key(k)]
        if len(vlm) == len(keys):
            # pure VLM shard: link it
            target = out / shard
            if not target.exists():
                os.symlink(os.path.realpath(base_snapshot / shard), target)
            plan.update({k: shard for k in keys})
        elif vlm:
            # mixed shard: rewrite only its VLM tensors
            name = shard.replace("model-", "vlm-")
            target = out / name
            if not target.exists():
                with safe_open(str(base_snapshot / shard), framework="pt") as fh:
                    tensors = {k: fh.get_tensor(k) for k in vlm}
                save_file(tensors, str(target), metadata={"format": "pt"})
            plan.update({k: name for k in vlm})
        # head-only shard: nothing to keep
    (out / "vlm_plan.json").write_text(json.dumps(plan, indent=0))
    return out


def head_state(model) -> dict[str, torch.Tensor]:
    """The trainable half of the model as ``bf16`` CPU tensors, base-key names."""
    out: dict[str, torch.Tensor] = {}
    for prefix, module in (("expert.", model.expert), ("action_in_proj.", model.action_in_proj),
                           ("action_out_proj.", model.action_out_proj)):
        for k, v in module.state_dict().items():
            out[prefix + k] = v.detach().to(torch.bfloat16).cpu().contiguous()
    return out


def write_student_checkpoint(model, step: int, run_dir: Path,
                             base_snapshot: Path | None = None) -> Path:
    """``run_dir/ckpt/step_XXXXXX/`` loadable by ``Alpamayo1_5.from_pretrained``."""
    run_dir = Path(run_dir)
    if base_snapshot is None:
        base_snapshot = Path(model.config._name_or_path if os.path.isdir(
            getattr(model.config, "_name_or_path", "")) else _resolve_snapshot(model))
    base = prepare_base_shards(base_snapshot, run_dir / "_base")
    plan = json.loads((base / "vlm_plan.json").read_text())

    ckpt = run_dir / "ckpt" / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    tensors = head_state(model)
    save_file(tensors, str(ckpt / HEAD_SHARD), metadata={"format": "pt"})

    weight_map = {k: HEAD_SHARD for k in tensors}
    for k, f in plan.items():
        weight_map[k] = f
        link = ckpt / f
        if not link.exists():
            os.symlink(os.path.realpath(base / f), link)
    total = sum(v.numel() * v.element_size() for v in tensors.values())
    for f in set(plan.values()):
        total += os.path.getsize(os.path.realpath(base / f))
    (ckpt / INDEX).write_text(json.dumps(
        {"metadata": {"total_size": total, "step": step}, "weight_map": weight_map}, indent=1))

    cfg = json.loads((Path(base_snapshot) / "config.json").read_text())
    cfg["action_in_proj_cfg"]["_target_"] = HEAD_TARGET
    cfg["diffusion_cfg"]["_target_"] = SAMPLER_TARGET
    cfg["attn_implementation"] = "sdpa"
    cfg["distill"] = {"step": step, "base": str(base_snapshot)}
    (ckpt / "config.json").write_text(json.dumps(cfg, indent=2))
    return ckpt


def load_head_into(model, ckpt: Path) -> None:
    """Resume: put a checkpoint's head tensors back into a live model."""
    tensors = load_file(str(Path(ckpt) / HEAD_SHARD))
    state = model.state_dict()
    for k, v in tensors.items():
        state[k].copy_(v.to(state[k].dtype))


def upload_checkpoint(ckpt: Path, repo_id: str, step: int) -> str | None:
    """Push the directory to a private model repo under ``ckpt/step_XXXXXX``."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(ckpt), repo_id=repo_id, repo_type="model",
        path_in_repo=f"ckpt/step_{step:06d}",
        commit_message=f"SnapFlow student, step {step}",
    )
    api.upload_file(path_or_fileobj=json.dumps({"latest_step": step}).encode(),
                    path_in_repo="latest.json", repo_id=repo_id, repo_type="model")
    return getattr(info, "commit_url", None) or str(info)


def _resolve_snapshot(model) -> str:
    """The cached snapshot directory the model was loaded from."""
    from huggingface_hub import snapshot_download
    name = getattr(model.config, "_name_or_path", "nvidia/Alpamayo-1.5-10B")
    return snapshot_download(name, local_files_only=True)
