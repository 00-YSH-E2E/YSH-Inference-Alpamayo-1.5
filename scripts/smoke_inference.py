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

"""Does the model run here at all? One clip, open loop, no instrumentation.

Same pipeline as ``src/alpamayo1_5/test_inference.py`` with one change: the
attention backend is passed explicitly. The default is ``flash_attention_2``,
which has no aarch64 wheel and needs nvcc to build, so on this board the stock
script fails at model construction before anything interesting happens.

Dropping to SDPA costs less than it sounds: the diffusion expert is forced to
SDPA regardless (``models/alpamayo1_5.py:106-109`` -- its 4D float mask and
``is_causal=False`` are not something the FA2 kernel accepts), so only the VLM
backbone is affected. It does change latency, which is why the backend is
printed here and recorded as a run parameter by the tracked runner.

Run it before the instrumented pipeline. If this prints a plausible minADE, the
environment is good and everything after it is bookkeeping.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    return parser.parse_args()


def describe_device() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable -- this will not work")
        return
    free, total = torch.cuda.mem_get_info()
    print(f"device     : {torch.cuda.get_device_name(0)}")
    print("capability : sm_%d%d" % torch.cuda.get_device_capability(0))
    print(f"torch      : {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"GPU memory : {free / 1e9:.1f} / {total / 1e9:.1f} GB free")


def main() -> None:
    args = parse_args()
    describe_device()
    print(f"attention  : {args.attn}")

    print(f"\nLoading clip {args.clip_id} ...")
    t0 = time.perf_counter()
    data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    print(f"  dataset loaded in {time.perf_counter() - t0:.1f}s")

    print(f"\nLoading {args.model} (~22GB on first run) ...")
    t0 = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to("cuda")
    model.eval()
    print(f"  model loaded in {time.perf_counter() - t0:.1f}s")
    params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {params / 1e9:.2f}B")

    processor = helper.get_processor(model.tokenizer)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(f"  prompt tokens: {inputs['input_ids'].shape[1]}")
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )

    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.synchronize()  # otherwise the timer measures kernel queueing
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cot = str(np.asarray(extra["cot"]).reshape(-1)[0])
    print("\nChain-of-Causation:\n ", cot)

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    min_ade = float(np.linalg.norm(pred_xy - gt_xy[None], axis=-1).mean(-1).min())

    print(f"\nminADE     : {min_ade:.3f} m")
    print(f"latency    : {elapsed:.1f}s for {args.num_traj_samples} sample(s)")
    print(f"VRAM peak  : {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    if min_ade >= 1.0:
        print("NOTE: minADE above 1.0m. Sampling is stochastic; rerun before worrying.")


if __name__ == "__main__":
    main()
