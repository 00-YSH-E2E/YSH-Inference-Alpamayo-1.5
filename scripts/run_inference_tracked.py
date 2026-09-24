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

"""Open-loop inference over a clip list, recorded so it can be compared later.

    clips -> traced inference -> metrics -> run directory -> Hugging Face + MLflow

Numbers land on MLflow, files on Hugging Face, and the run carries a coordinate
tying them together. What gets stored is only what re-running the model is the
only way to recover; metrics are recomputed from that every time, because their
definitions are still moving.

Two flags exist because getting them wrong fails quietly rather than loudly:

``--data-cache`` must point at the downloaded chunks. Without it the dataset
interface falls back to streaming and a run that should take minutes takes
hours, with nothing in the output to say why. ``--allow-stream`` opts back in
deliberately.

``TORCH_DISABLE_NATIVE_JIT=1`` is set here before torch loads. Otherwise torch
routes part of the rotary embedding through triton, which compiles a C
extension at runtime and needs Python headers this system does not have. It is
recorded as a parameter: installing those headers changes which kernels run,
and therefore the latency, so runs on either side of that are not comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Must precede the torch import.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The repo holding the code being run, found from this file rather than from the
# working directory. Launched from one directory up -- which is not a git repo --
# every git coordinate would come back empty and the run would carry no record of
# what produced it. Launched from a *different* repo, it would record that repo's
# commit as though it described this code.
REPO_ROOT = Path(__file__).resolve().parent.parent

import ml_platform_track as mlp  # noqa: E402

import physical_ai_av  # noqa: E402
from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from alpamayo1_5.trace import flop_count as FC  # noqa: E402
from alpamayo1_5.trace import host_stages as HS  # noqa: E402
from alpamayo1_5.trace import metrics as M  # noqa: E402
from alpamayo1_5.trace import profile_parse as PP  # noqa: E402
from alpamayo1_5.trace import roofline as RL  # noqa: E402
from alpamayo1_5.trace import thermal as TH  # noqa: E402
from alpamayo1_5.trace import timing_schema as TS  # noqa: E402
from alpamayo1_5.trace import writer as W  # noqa: E402
from alpamayo1_5.trace.token_trace import (  # noqa: E402
    DEFAULT_SPECIAL_IDS,
    TRACE_LEVELS,
    run_peak_bytes,
    trace_inference,
)

MODEL_REPO = "nvidia/Alpamayo-1.5-10B"
DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
EVALS_REPO = "YSHRobotics/Alpamayo-Evals"
DATA_CACHE = "/home/thor/Documents/Alpamayo/Data/Alpamayo-1.5_Cam-4_Vanilla"
# Outputs live beside the data rather than inside the code repo, and share its
# naming: {model}_{data}_{variant}_{date}_{run_id}.
OUT_ROOT = "/home/thor/Documents/Alpamayo/Inference"
DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"
MAX_SAMPLE_IMAGES = 20  # representative figures per run, per the recording rules

# Allocator history snapshots written this run (local files; see --memory-snapshot).
_MEMSNAPS: list[str] = []

# Where host-device synchronizations happened, over the run's main passes (trace
# level step). run.json keeps the busiest sites: the code a sync-free decode
# loop would have to change.
_SYNC_SITES: dict[str, int] = {}

# The key a main pass's layer spans travel under (trace level layer), popped the
# same way.
LAYERS = "__layers__"

# The key a profile pass's kernels travel under, popped the same way.
KERNELS = "__kernels__"

# Chrome traces kept this run (local files; see --profile-trace).
_PROFILE_TRACES: list[str] = []

# The key a pass's host-clock windows travel under from run_clip to the loop.
# Popped before the row is stored: windows are how energy is attributed, not a
# column.
WINDOWS = "__windows__"

# Per-clip values that become run-level means. Spelled out rather than swept
# off whatever per_clip happens to contain: with a sweep, the metric namespace
# of every future run is decided by whatever metrics.py last returned, and a
# column can appear or vanish between runs without anyone choosing that.
# Anything computed but absent here is printed at the end of the run.
#
# ade_6.4s is min_ade by construction (the full horizon of the best sample).
# Kept as a free consistency check; if the two ever disagree, the horizon
# indexing is wrong.
_CLIP_METRICS = (
    "min_ade", "min_fde", "mean_ade", "mean_fde",
    "ade_1.0s", "ade_2.0s", "ade_3.0s", "ade_4.0s", "ade_5.0s", "ade_6.4s",
    "de_1.0s", "de_2.0s", "de_3.0s", "de_4.0s", "de_5.0s", "de_6.4s",
    "jerk_mean", "jerk_p95",
    "lat_accel_mean", "lat_accel_p95", "lat_accel_over_4_ratio",
    "accel_violation_rate", "within_bounds_ratio", "speed_mean",
    "net_heading_deg", "net_heading_abs_deg",
    "lateral_offset_m", "lateral_offset_abs_m",
    "diversity_mean_m", "diversity_final_m", "diversity_max_m", "sample_gain",
)

# Numeric per-clip values that locate a row rather than measure it. Averaging one
# says nothing -- mean(t0_us) is t0_us -- so they are excluded from the warning
# below instead of being added to _CLIP_METRICS, which would log them to MLflow
# as though they were results and put a constant on every comparison chart.
_CLIP_COORDS = frozenset({"t0_us"})

# What each situation bucket reports, beyond its clip count. Short on purpose:
# every entry here is multiplied by the number of buckets, and the per-clip
# table in the run directory already carries everything for offline pivots.
#
# min_ade is renamed to `score` in the output so the bucket's headline matches
# the run's. The rest are there to answer "why": mean_ade against min_ade shows
# how much the K samples disagreed, sample_gain is that difference directly, and
# diversity says whether the model was hedging or committing.
_BUCKET_METRICS = ("min_ade", "mean_ade", "min_fde", "sample_gain", "diversity_final_m")


# The snapshot every cached chunk was downloaded from. Passed explicitly
# because `PhysicalAIAVDatasetInterface(revision=None)` resolves `main` over
# the network at construction time: the day NVIDIA pushes a commit, every clip
# misses the cache and raises -- or, with --allow-stream, silently re-downloads
# 79 GB. For a run that has to survive a week unattended, the revision is part
# of the experiment definition, not an environment detail.
DATASET_REVISION = "b719eea7f0a63619ef51ec7f54178af0937ef050"

# The same for the checkpoint: the snapshot this machine's cache holds. The
# recording rules forbid `@main` as a coordinate -- tomorrow it names another
# commit -- and until now both coordinates were recorded as `@main` and resolved
# to whatever main was at record time, which is not necessarily what was read.
# Applied only to MODEL_REPO: another hub checkpoint has its own history, and
# NVIDIA's sha would not name anything in it.
MODEL_REVISION = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def resolve_model_revision(model: str, revision: str | None) -> str | None:
    """The revision to load and to record. None: a local directory, or a hub
    checkpoint whose revision nobody pinned (the tracker resolves `@main` then)."""
    if Path(model).is_dir():
        return None
    if revision:
        return revision
    return MODEL_REVISION if model == MODEL_REPO else None


def model_coordinate(model: str, revision: str | None) -> str:
    """``model_source`` as the recording rules want it: a path, or a hub id at a sha."""
    if Path(model).is_dir():
        return f"path:{model}"
    return f"hf:{model}@{revision or 'main'}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip-id", action="append", help="Repeat for several clips.")
    p.add_argument("--clip-list", help="Parquet with a clip_id column.")
    p.add_argument("--limit", type=int, help="Use only the first N clips of the list.")
    p.add_argument("--t0-us", type=int, default=5_100_000)
    p.add_argument("--num-traj-samples", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--inference-step", type=int, default=None,
                   help="Diffusion Euler steps. 0 is rejected: the sampler reads it as "
                        "'unset' and silently runs its default, corrupting any latency number.")
    p.add_argument("--seed", type=int, default=42,
                   help="XORed with a hash of each clip id before seeding, so every clip "
                        "starts the CUDA generator at its own point. Re-seeding every clip "
                        "with the same value made sample k always draw the same slice of the "
                        "stream, and sample 0 came out 0.5 m worse than the others.")
    p.add_argument("--diffusion-temperature", type=float, default=1.0,
                   help="Scale on the flow head's initial noise. 1.0 is the training prior. "
                        "Distinct from --temperature, which is the CoT text temperature.")
    p.add_argument("--dataset-revision", default=DATASET_REVISION,
                   help="Dataset snapshot to read from the cache. Pinned; see DATASET_REVISION.")
    p.add_argument("--flush-every", type=int, default=100,
                   help="Rewrite the run directory every N clips, so a crash at clip "
                        "1299 of 1300 loses N clips rather than the run. 0 disables.")
    p.add_argument("--x0-from", default=None,
                   help="A teacher run's predictions.parquet (schema 3). Its recorded x0 for "
                        "each (clip, sample_k) is injected into the sampler, so this run's "
                        "samples start from the teacher's noise and can be paired on it.")
    p.add_argument("--split", default=None,
                   help="Comma-separated subset of the clip list's `split` column, e.g. "
                        "val,test. Keeps evaluation clips out of any training run.")
    p.add_argument("--model", default=MODEL_REPO)
    p.add_argument("--model-revision", default=None,
                   help="Hub revision to load and record. Defaults to MODEL_REVISION for the "
                        "default model; other hub ids fall back to resolving main.")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    # The graph runner has been in the tree, tested, and documented in the
    # README since it was written, and nothing here ever turned it on -- so
    # every recorded run measures the eager path and the replayed path has
    # never been measured at all. A flag is what makes it an axis.
    p.add_argument("--cuda-graph", action="store_true",
                   help="Replay the diffusion expert with exact-shape CUDA graphs.")
    p.add_argument("--cuda-graph-max-graphs", type=int, default=4,
                   help="Distinct input signatures to keep captured (default 4).")
    p.add_argument("--trace-level", default="basic", choices=TRACE_LEVELS,
                   help="How deep the instrumentation goes. 'off' installs no hook and keeps "
                        "only the wall clock -- the baseline the tracer's cost is measured "
                        "against. 'step' goes inside the decode and Euler steps; 'layer' "
                        "adds every layer's attention and MLP (layers.parquet). Never "
                        "compare latency across levels.")
    p.add_argument("--warmup", type=int, default=2,
                   help="Untimed passes on the first clip before the run. The first pass of a "
                        "process carries autotuning and allocator growth; on the Thor clip 1 "
                        "came out 31%% slower than the median. Recorded as warmup rows.")
    p.add_argument("--overhead-probe", type=int, default=0,
                   help="Give the first N clips an extra off/on pass pair, re-seeded, to "
                        "measure the tracer's own cost (trace.overhead_pct).")
    p.add_argument("--timing-repeats", type=int, default=0,
                   help="Extra passes per repeated clip, for the latency noise band.")
    p.add_argument("--repeat-clips", type=int, default=5,
                   help="How many clips, from the start, get --timing-repeats.")
    p.add_argument("--sample-hz", type=float, default=10.0,
                   help="Board sampler rate: power, GPU clock and over-current counters. "
                        "Temperatures and the slower sensors run at a fifth of it, at most "
                        "1 Hz slower. 10 Hz resolves a clip's energy, and a segment's once "
                        "it lasts a couple of seconds.")
    p.add_argument("--memory-snapshot", type=int, default=0,
                   help="Give the first N clips an extra pass under the allocator's history "
                        "recorder, and write each snapshot next to the run (local only; open "
                        "it at pytorch.org/memory_viz). Shows every allocation the KV "
                        "concatenation makes.")
    p.add_argument("--profile-clips", type=int, default=0,
                   help="Give the first N clips an extra pass under torch.profiler, at trace "
                        "level basic: every kernel to kernels.parquet, and the pass's true GPU "
                        "idle, kernels per step, launch and sync time and SDPA backends to its "
                        "timing row. The profiler slows the pass; its latency is never used.")
    p.add_argument("--flop-count", type=int, default=0,
                   help="Give the first N clips an extra pass with every op's FLOPs and "
                        "operand bytes counted by segment (FlopCounterMode and a dispatch "
                        "mode; the expert's CUDA graph is set aside for it). It checks the "
                        "analytic work model the efficiency numbers use. Never timed.")
    p.add_argument("--roofline-probe", action="store_true",
                   help="Before the first clip, measure this board's bf16 GEMM throughput and "
                        "the bandwidth of a GEMV, a read, a copy and the KV concatenation "
                        "(~10 s). The efficiency numbers then say how close each segment "
                        "came to its roof.")
    p.add_argument("--deadline-ms", type=float, default=None,
                   help="A latency budget to report against: the share of main passes over "
                        "it (device total, host wall, time to the first trajectory) and the "
                        "p95 margin under it. Reporting only; nothing is dropped.")
    p.add_argument("--steady-skip", type=int, default=0,
                   help="Leave the first N clips out of the steady-state numbers, where "
                        "clocks ramp and the allocator grows. The headline means keep them.")
    p.add_argument("--profile-trace", action="store_true",
                   help="Also keep each profile pass's Chrome trace, gzipped, next to the run "
                        "(local only; open it in Perfetto).")
    p.add_argument("--variant", default="Vanilla", help="Vanilla, Pruned-24L, INT8 ...")
    p.add_argument("--data-spec", default="Cam-4")
    p.add_argument("--machine", default=None,
                   help="Where this ran, for the run directory and run name. Defaults to "
                        "the short hostname, which is also what env.host records.")
    p.add_argument("--label", default=None,
                   help="Extra name segment, appended after the standard <n>clip_k<K>-temp<T>. "
                        "Sweeps use it for axes beyond those, so two runs of one batch stay "
                        "distinguishable by eye rather than only by run id.")
    p.add_argument("--sweep", default=None,
                   help="Tag every run of one sweep with this name so the batch can be "
                        "filtered as a unit. A sweep is N runs, not one run with N results "
                        "-- the recording rules forbid folding several evaluations into one.")
    p.add_argument("--experiment", default="alpamayo-1.5")
    p.add_argument("--notes", help="One or two human sentences: why this run exists.")
    p.add_argument("--data-cache", default=DATA_CACHE)
    p.add_argument("--allow-stream", action="store_true",
                   help="Permit streaming when a clip is not cached. Off by default so a "
                        "missing cache fails loudly instead of running 10x slower.")
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--evals-repo", default=EVALS_REPO)
    p.add_argument("--include-gt", action="store_true",
                   help="Also keep the logged future locally. It is recoverable from "
                        "(clip_id, t0_us), so it is never uploaded either way.")
    p.add_argument("--no-samples", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--no-track", action="store_true")
    return p.parse_args()


def resolve_clips(args: argparse.Namespace) -> list[str]:
    if args.clip_id:
        return args.clip_id
    if args.clip_list:
        import pandas as pd

        table = pd.read_parquet(args.clip_list)
        if args.split:
            wanted = [x.strip() for x in args.split.split(",") if x.strip()]
            if "split" not in table.columns:
                raise SystemExit(
                    f"--split {args.split} asked for, but {args.clip_list} has no `split` "
                    "column. Use notebooks/clip_ids_cached1300.parquet, which carries one."
                )
            table = table[table["split"].isin(wanted)]
            if not len(table):
                raise SystemExit(f"no clips in {args.clip_list} with split in {wanted}")
        # The split rides on the clip so per_clip.parquet can be cut by it
        # offline: a headline computed over training-split clips is a leak, and
        # nothing downstream can tell unless the row says which split it was.
        if "split" in table.columns:
            args.clip_split = dict(zip(table["clip_id"], table["split"]))
        clips = table["clip_id"].tolist()
        return clips[: args.limit] if args.limit else clips
    return [DEFAULT_CLIP]


def render_sample(result: dict, data: dict, path: Path) -> bool:
    """One BEV + camera figure. A run archived without viewable samples shows
    nothing on the hub, which defeats keeping the outputs at all."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from alpamayo1_5 import viz_utils

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1.3]})
        try:
            axes[0].imshow(
                viz_utils.make_camera_grid(data["image_frames"], data["camera_indices"])
            )
            axes[0].axis("off")
            viz_utils.plot_condition(
                axes[1], result["pred_xy"], color="tab:blue", label="prediction"
            )
            if result.get("gt_xy") is not None:
                gt = result["gt_xy"]
                axes[1].plot(gt[:, 0], gt[:, 1], "k--", linewidth=2, label="logged future")
            axes[1].set_aspect("equal", adjustable="datalim")
            axes[1].set_xlabel("x [m]")
            axes[1].set_ylabel("y [m]")
            axes[1].legend(loc="best", fontsize=8)
            title = result["clip_id"]
            if result.get("min_ade") is not None:
                title += f"  |  minADE {result['min_ade']:.2f} m"
            axes[1].set_title(title, fontsize=10)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(path, dpi=110)
        finally:
            # pyplot keeps a global reference to every open figure. Closing only
            # on success leaks one decoded camera grid per failure, and the
            # failures come in runs -- a full disk fails all of them.
            plt.close(fig)
        return True
    except Exception as exc:  # a figure is never worth failing a run over
        print(f"[samples] {result['clip_id']}: {exc}", file=sys.stderr)
        return False


def clip_seed(seed: int, clip_id: str) -> int:
    """The seed for one clip: the run seed XORed with a hash of the clip id.

    Seeding every clip with the same value is what made the K samples
    non-exchangeable. Each clip then started the CUDA Philox generator from
    the same state, and because the CoT decode consumes only a small,
    similar-sized slice before the trajectory noise is drawn, sample k landed
    on nearly the same counter range every time. Over 1181 clips the per-index
    means never averaged out: sample 0 sat 1.32 m from the sample centroid
    against 0.78-1.07 m for the others and scored 0.51 m worse in ADE.

    XOR with a clip hash keeps what mattered about per-clip seeding -- the same
    clip gives the same trajectories in every arm, so arms stay paired -- and
    makes the result independent of which other clips are in the list and in
    what order. Seeding once at the start of the run would not: a different
    --clip-list would then change what every clip after the first sees.
    """
    digest = hashlib.blake2b(clip_id.encode(), digest_size=4).hexdigest()
    return seed ^ int(digest, 16)


def prepare_clip(processor, avdi, clip_id: str, args) -> tuple[dict, dict, dict]:
    """Load one clip and build the model's inputs.

    Kept apart from inference so that extra passes over a clip -- the overhead
    probe, the repeats -- reuse these inputs rather than decoding the video
    again. The model deep-copies what it is given, so the inputs survive a pass.
    """
    # Each stage adds to the clip's clock (host_stages); outside a clip -- the
    # warmup -- there is no clock and nothing is recorded.
    started = time.perf_counter()
    data = load_physical_aiavdataset(
        clip_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.allow_stream
    )
    HS.add_since("data_load_ms", started)
    started = time.perf_counter()
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    HS.add_since("msg_build_ms", started)
    started = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    HS.add_since("preprocess_ms", started)
    started = time.perf_counter()
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )
    HS.add_since("h2d_ms", started)
    diffusion_kwargs = {"temperature": args.diffusion_temperature}
    if args.inference_step is not None:
        diffusion_kwargs["inference_step"] = args.inference_step
    if getattr(args, "x0_table", None) is not None:
        x0 = args.x0_table.get(clip_id)
        if x0 is None or x0.shape[0] != args.num_traj_samples:
            raise SystemExit(f"--x0-from has no {args.num_traj_samples} rows for clip {clip_id}")
        diffusion_kwargs["x0"] = torch.as_tensor(x0, dtype=torch.float32, device="cuda")
    return data, model_inputs, diffusion_kwargs


def infer(model, clip_id: str, model_inputs: dict, diffusion_kwargs: dict, args, level: str,
          ranges: bool = False, listener=None):
    """One traced inference pass, re-seeded for the clip.

    Every pass over a clip starts the generator at the same point, so extra
    passes are replicates of the main one: same tokens, same noise, same
    trajectories. ``pass_output_match`` checks that it held -- a timing
    difference between passes that did different work is not noise.
    """
    torch.cuda.manual_seed_all(clip_seed(args.seed, clip_id))
    with trace_inference(model, level=level, ranges=ranges, listener=listener) as tracer:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
                diffusion_kwargs=diffusion_kwargs,
                return_extra=True,
            )
    return pred_xyz, pred_rot, extra, tracer


def extra_passes_for(index: int, args) -> list[tuple[str, str]]:
    """The extra passes clip ``index`` gets after its main pass: ``(row_kind, level)``.

    The probe pair alternates its order by clip, so a slow drift over the run --
    the board warming -- does not load onto one side of the comparison.
    """
    passes: list[tuple[str, str]] = []
    if index < args.overhead_probe and args.trace_level != "off":
        pair = [("probe", "off"), ("probe", args.trace_level)]
        passes += pair if index % 2 == 0 else pair[::-1]
    if index < args.repeat_clips:
        passes += [("repeat", args.trace_level)] * args.timing_repeats
    if index < getattr(args, "memory_snapshot", 0):
        passes.append(("memsnap", args.trace_level))
    if index < getattr(args, "flop_count", 0):
        passes.append(("flops", "basic"))
    # Last: the profiler's buffers are the largest thing any pass leaves behind.
    if index < getattr(args, "profile_clips", 0):
        passes.append(("profile", "basic"))
    return passes


def counted_infer(model, clip_id: str, model_inputs: dict, diffusion_kwargs: dict, args):
    """A --flop-count pass: the inference with its FLOPs and operand bytes counted
    by segment. The expert's CUDA graph is set aside for it -- a replay runs no
    op the counters could see -- and put back however the pass ends."""
    runner = getattr(model.expert, "_diffusion_expert_cuda_graph", None)
    if runner is not None:
        model.expert.forward = runner._original_forward
    try:
        out, row = FC.counted(lambda listener: infer(
            model, clip_id, model_inputs, diffusion_kwargs, args, "basic", listener=listener))
    finally:
        if runner is not None:
            model.expert.forward = runner.forward
    return (*out, row)


def profiled_infer(model, clip_id: str, model_inputs: dict, diffusion_kwargs: dict, args,
                   out_dir: Path):
    """A profile pass: the inference under torch.profiler, the tracer's spans as ranges.

    Returns the pass as ``infer`` does, then the profile's timing-row columns
    and its device events. Reading the trace is instrumentation: if it fails
    the pass keeps its tracer row and loses only the profile columns.
    """
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = infer(model, clip_id, model_inputs, diffusion_kwargs, args, "basic", ranges=True)
    raw = out_dir / f".profile_{clip_id[:8]}.json"
    row: dict = {}
    kernels: list[dict] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        prof.export_chrome_trace(str(raw))
        export_ms = (time.perf_counter() - started) * 1000.0
        del prof
        started = time.perf_counter()
        row, kernels = PP.analyze(PP.load_events(raw))
        row["prof_export_ms"] = export_ms
        row["prof_parse_ms"] = (time.perf_counter() - started) * 1000.0
        if args.profile_trace:
            import gzip
            import shutil

            kept = out_dir / f"profile_{clip_id[:8]}.json.gz"
            with open(raw, "rb") as src, gzip.open(kept, "wb") as dst:
                shutil.copyfileobj(src, dst)
            _PROFILE_TRACES.append(str(kept))
    except Exception as exc:
        print(f"[profile] {clip_id[:8]}: {exc}", file=sys.stderr)
    finally:
        raw.unlink(missing_ok=True)
    return (*out, row, kernels)


def _dump_memory_snapshot(path: Path) -> None:
    """Write the allocator's recorded history and stop recording. A snapshot that
    cannot be written costs the file, never the run."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))
        _MEMSNAPS.append(str(path))
    except Exception as exc:
        print(f"[memsnap] {path.name}: {exc}", file=sys.stderr)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def run_clip(
    model, processor, avdi, clip_id: str, args, out_dir: Path,
    extra_passes: list[tuple[str, str]] = (),
) -> tuple[list[dict], dict, list[dict]]:
    """Inference for one clip.

    Returns one row per sample, the per-clip extras, and a timing row per pass:
    the main one, then any extra passes. The sample rows carry only the legacy
    timing keys -- the predictions table is frozen at schema 3 -- and every
    other measurement goes to the timing table.
    """
    data, model_inputs, diffusion_kwargs = prepare_clip(processor, avdi, clip_id, args)
    TH.mark("infer")
    started = time.perf_counter()
    pred_xyz, pred_rot, extra, tracer = infer(
        model, clip_id, model_inputs, diffusion_kwargs, args, args.trace_level
    )
    HS.add_since("model_call_ms", started)
    TH.mark("post")
    timing = tracer.timing
    trace = tracer.trace
    for site, count in timing.sync_sites.items():
        _SYNC_SITES[site] = _SYNC_SITES.get(site, 0) + count

    started = time.perf_counter()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]  # [K, T, 2]
    gt = data.get("ego_future_xyz")
    gt_xy = gt.cpu()[0, 0, :, :2].numpy() if gt is not None else None
    cot_texts = [str(c) for c in np.asarray(extra["cot"]).reshape(-1)]
    # extract_text_tokens already decodes this next to the reasoning trace; the
    # parquet has had a meta_action column since the first run and nothing was
    # ever putting anything in it.
    meta_actions = [str(m) for m in np.asarray(extra.get("meta_action", [])).reshape(-1)]

    hist_xy = data["ego_history_xyz"][0, 0, :, :2].cpu().numpy()
    hist_rot = data["ego_history_rot"][0, 0].cpu().numpy()
    rows = []
    for k in range(pred_xy.shape[0]):
        row = {
            "clip_id": clip_id,
            "t0_us": args.t0_us,
            "sample_k": k,
            "pred_xy": pred_xy[k],
            "pred_rot": pred_rot.cpu().numpy()[0, 0, k],
            "hist_xy": hist_xy,
            "hist_rot": hist_rot,
            "cot": cot_texts[k] if k < len(cot_texts) else "",
            "meta_action": meta_actions[k] if k < len(meta_actions) else None,
            **timing.legacy(),
        }
        if trace is not None and k < trace.token_ids.shape[0]:
            row.update(trace.sample(k))
        rows.append(row)
    HS.add_since("result_cpu_ms", started)
    started = time.perf_counter()

    # t0_us belongs to the clip, not to the run: the pairing key across runs is
    # (clip_id, t0_us), and it is degenerate only for as long as nobody sweeps
    # the sample timestamp. Carry it here rather than in the config columns so
    # it stays a per-row value when that day comes.
    extras = {"clip_id": clip_id, "t0_us": args.t0_us, "gt_xy": gt_xy, "data": data,
              "split": getattr(args, "clip_split", {}).get(clip_id)}
    if gt_xy is not None:
        # Pass the checkpoint's own dt rather than letting the 0.1 default
        # stand. The horizon labels (ade_1.0s and friends) are derived from it,
        # so a checkpoint with a different step would mislabel every one of
        # them -- which is the failure displacement()'s docstring claims to
        # have avoided by not hardcoding the indices.
        extras.update(M.displacement(pred_xy, gt_xy, dt=float(model.action_space.dt)))
    extras.update(
        M.kinematics(
            model.action_space,
            data["ego_history_xyz"][:, -1].repeat(pred_xy.shape[0], 1, 1).cuda(),
            data["ego_history_rot"][:, -1].repeat(pred_xy.shape[0], 1, 1, 1).cuda(),
            pred_xyz[0, 0],
            pred_rot[0, 0],
        )
    )
    extras.update(M.heading(pred_xyz[0, 0], pred_rot[0, 0]))
    # How far apart the K samples are. minADE improves either because the model
    # got better or because it spread wider and one sample landed; without this
    # the two are indistinguishable.
    extras.update(M.diversity(pred_xy))
    if extras.get("mean_ade") is not None and extras.get("min_ade") is not None:
        # What drawing K samples actually bought. Large next to a wide spread
        # means the model is hedging rather than committing.
        extras["sample_gain"] = float(extras["mean_ade"] - extras["min_ade"])
    # The situation the clip presented, read from the logged future. Not from
    # the prediction: a variant that predicts straighter would otherwise get
    # more clips labelled "straight", and the per-situation comparison would be
    # over different clip sets for each variant.
    maneuver = M.classify_maneuver(
        gt_xy, dt=float(model.action_space.dt)
    ) if gt_xy is not None else {"lateral": "unknown", "longitudinal": "unknown"}
    extras["scene"] = maneuver["lateral"]
    extras["speed_profile"] = maneuver["longitudinal"]
    # What the model *thought* the situation was, from its own trajectory. Kept
    # alongside because a disagreement with `scene` is itself a finding.
    extras["scene_predicted"] = M.classify_scene(
        extras.get("net_heading_abs_deg", 0.0), extras.get("lateral_offset_abs_m", 0.0)
    )
    extras["pred_xy"] = pred_xy
    HS.add_since("metrics_ms", started)

    timing_rows = [{**timing.row(), "row_kind": "main", "pass_index": 0,
                    "trace_level": args.trace_level, WINDOWS: timing.windows,
                    LAYERS: timing.layers}]
    started = time.perf_counter()
    if extra_passes:
        TH.mark("extra")
    for index, (kind, level) in enumerate(extra_passes, start=1):
        if kind == "memsnap":
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        profiled: dict = {}
        try:
            if kind == "profile":
                xyz, _, _, other, prof_row, kernels = profiled_infer(
                    model, clip_id, model_inputs, diffusion_kwargs, args, out_dir)
                profiled = {**prof_row, KERNELS: kernels}
            elif kind == "flops":
                xyz, _, _, other, profiled = counted_infer(
                    model, clip_id, model_inputs, diffusion_kwargs, args)
            else:
                xyz, _, _, other = infer(model, clip_id, model_inputs, diffusion_kwargs, args,
                                         level)
        finally:
            if kind == "memsnap":
                _dump_memory_snapshot(out_dir / f"memsnap_{clip_id[:8]}.pickle")
        timing_rows.append({
            **other.timing.row(), "row_kind": kind, "pass_index": index, "trace_level": level,
            "pass_output_match": bool(torch.equal(xyz, pred_xyz)), WINDOWS: other.timing.windows,
            **profiled,
        })
    if extra_passes:
        HS.add_since("extra_passes_ms", started)
        TH.mark("post")
    return rows, extras, timing_rows


def main() -> None:
    args = parse_args()
    if args.inference_step is not None and args.inference_step < 1:
        raise SystemExit(
            "--inference-step must be >= 1. The sampler reads 0 as 'unset' and silently "
            "runs its default, which quietly corrupts any latency measurement."
        )
    clips = resolve_clips(args)
    date = datetime.now(timezone.utc).astimezone().strftime("%y.%m.%d")
    args.model_revision = resolve_model_revision(args.model, args.model_revision)

    params = {
        "model": args.model,
        "model_revision": args.model_revision,
        "variant": args.variant,
        "data_spec": args.data_spec,
        "n_clips": len(clips),
        "num_traj_samples": args.num_traj_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # Never passed, so generate runs without a top-k warper. Recorded because
        # "unset" and "unrecorded" read the same, and a later run that sets it
        # would otherwise look comparable.
        "top_k": None,
        "max_generation_length": args.max_generation_length,
        "inference_step": args.inference_step,
        "seed": args.seed,
        # How the seed reaches the generator. Runs before this scheme existed
        # re-seeded every clip with the bare value; they must not share an
        # MLflow axis with these, because their samples are not exchangeable.
        "seed_scheme": "clip-hash",
        "diffusion_temperature": args.diffusion_temperature,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "x0_from": args.x0_from,
        "attn_impl": args.attn,
        "cuda_graph": args.cuda_graph,
        "cuda_graph_max_graphs": args.cuda_graph_max_graphs if args.cuda_graph else None,
        "dtype": "bfloat16",
        "t0_us": args.t0_us,
        # Latency is only comparable within one table version and one tracer.
        "timing_schema_version": TS.TIMING_SCHEMA_VERSION,
        "tracer_version": TS.TRACER_VERSION,
        "trace_level": args.trace_level,
        "warmup": args.warmup,
        "overhead_probe": args.overhead_probe,
        "timing_repeats": args.timing_repeats,
        "repeat_clips": args.repeat_clips,
        "memory_snapshot": args.memory_snapshot,
        "profile_clips": args.profile_clips,
        "profile_trace": args.profile_trace,
        "flop_count": args.flop_count,
        "roofline_probe": args.roofline_probe,
        "deadline_ms": args.deadline_ms,
        "steady_skip": args.steady_skip,
        "sample_hz": args.sample_hz,
        "torch_disable_native_jit": os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
        "torch_version": torch.__version__,
        "data_cache": args.data_cache,
        "power_mode": TH.power_mode(),
        # Latency is meaningless without it, and a run made off the tailnet
        # has no env.host tag until it is imported.
        "machine": W.machine_name(args.machine),
        # Switches that change what was read or produced. They were reachable
        # only through the `cmd` tag -- an opaque string that records what was
        # typed and says nothing about the defaults, and cannot be filtered on.
        #
        # allow_stream is the one that matters: it decides whether a clip came
        # from the local cache or from the hub. A half-populated cache under it
        # produces a run that read a mix of cached and live data, and nothing
        # else in the record distinguishes that from a fully cached run.
        "allow_stream": bool(args.allow_stream),
        "include_gt": bool(args.include_gt),
        "save_samples": not args.no_samples,
        "upload": not args.no_upload,
    }

    def execute(run: mlp.Run | None) -> None:
        # Wrapped so the data load splits into the ego fetch, the camera fetches
        # and the CPU frame decode -- the last of which is most of it.
        avdi = HS.wrap_dataset(physical_ai_av.PhysicalAIAVDatasetInterface(
            cache_dir=args.data_cache, revision=args.dataset_revision
        ))
        load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": args.attn}
        if args.model_revision:
            # Load the snapshot the coordinate names, not whatever main is today.
            load_kwargs["revision"] = args.model_revision
        model = Alpamayo1_5.from_pretrained(args.model, **load_kwargs).to("cuda").eval()
        if args.cuda_graph:
            # After .to("cuda").eval(): the runner refuses a CPU or training
            # model, and refusing is the right behaviour -- a graph captured
            # in train mode would replay dropout.
            model.enable_diffusion_expert_cuda_graph(
                max_batch_size=args.num_traj_samples,
                max_graphs=args.cuda_graph_max_graphs,
            )
        # What this board delivers, measured before the first clip -- and before
        # the peak-memory reset below, which the probe's buffers would set.
        peaks = None
        if args.roofline_probe:
            peaks = FC.probe()
            peaks["gpu_mhz_after"] = TH.read_fast().get("freq.gpu")
            print("[roofline] " + "  ".join(f"{k} {v:.1f}" for k, v in peaks.items()
                                             if isinstance(v, float)))
        work = FC.work_model(model)
        args.x0_table = None
        if args.x0_from:
            import inspect
            if "x0" not in inspect.signature(model.diffusion.sample).parameters:
                raise SystemExit(
                    "--x0-from was given, but this model's sampler does not accept x0 -- the "
                    "upstream sample() drops unknown kwargs, so the run would look paired "
                    "without being paired. Load a student checkpoint (its config names "
                    "ShortcutFlowMatching), or drop --x0-from."
                )
            import pandas as pd
            t = pd.read_parquet(args.x0_from, columns=["clip_id", "sample_k", "x0"])
            if t["x0"].isna().any():
                raise SystemExit(f"{args.x0_from} has rows without x0 (schema < 3?)")
            args.x0_table = {
                cid: np.stack([np.asarray(v, np.float32).reshape(-1, 2)
                               for v in g.sort_values("sample_k")["x0"]])
                for cid, g in t.groupby("clip_id")
            }
            print(f"[x0] {len(args.x0_table)} clips of teacher noise from {args.x0_from}")
        # Wrapped so tokenization splits into its image half and its text half.
        processor = HS.wrap_processor(helper.get_processor(model.tokenizer))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        run_id = run.run_id if run is not None else f"local{int(time.time()):x}"
        machine = W.machine_name(args.machine)
        # 이름에 늘 들어가는 꼬리. 클립 수와 샘플링은 결과를 직접 바꾸는데,
        # 없으면 3클립 스모크와 100클립 본 run 이 폴더 이름으로 구분이 안 된다 —
        # 그 둘을 헷갈리는 건 실제로 일어나고, 헷갈린 채 비교하면 조용히 틀린다.
        # temperature 는 temp 로 풀어 쓴다: 이 코드베이스엔 t0_us 가 있어서 t 가 모호하다.
        label = (f"{len(clips)}clip_k{args.num_traj_samples}"
                 f"-temp{args.temperature:g}")
        if args.label:              # sweep 이 붙이는 추가 축 (예: -s20)
            label = f"{label}-{args.label}"
        out_dir = Path(args.out_root) / W.run_dir_name(
            args.variant, date, run_id, data=args.data_spec, machine=machine,
            label=label,
        )
        config = {
            "run_id": run_id,
            "variant": args.variant,
            "git_commit": mlp.git_tags(REPO_ROOT).get("git_commit"),
            "columns": {
                "model": args.model,
                "data_spec": args.data_spec,
                "attn_impl": args.attn,
                "dtype": "bfloat16",
                "inference_step": args.inference_step,
                "max_new_tokens": args.max_generation_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "seed_scheme": "clip-hash",
                "diffusion_temperature": args.diffusion_temperature,
                "x0_source": "teacher" if args.x0_from else "sampled",
                "num_traj_samples": args.num_traj_samples,
                "conditioning_source": "generated",
            },
        }
        # Written on every flush and at the end. Everything here is known before
        # the first clip, so there is no reason for it to wait for the last.
        identity = {
            "run_id": run_id, "variant": args.variant, "date": date,
            "machine": machine, "clips": clips, "params": params,
        }
        rows, per_clip, gt_rows, timing_rows, layer_rows, kernel_rows = [], [], [], [], [], []
        # What the board is: release, driver, clock ranges. Into every timing
        # row (the two that decide comparability), the params and run.json.
        hw = TH.hw_inventory()
        # Repeated on every timing row, for the same reason the config columns
        # repeat on every prediction row: runs concatenate, and a latency number
        # is only comparable with one taken under the same conditions -- which
        # is a stricter set than for accuracy (machine and power mode matter).
        import transformers

        timing_base = {
            "timing_schema_version": TS.TIMING_SCHEMA_VERSION,
            "tracer_version": TS.TRACER_VERSION,
            "run_id": run_id,
            "variant": args.variant,
            "git_commit": config["git_commit"],
            "machine": machine,
            "power_mode": TH.power_mode(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "attn_impl": args.attn,
            "dtype": "bfloat16",
            "model": args.model,
            "data_spec": args.data_spec,
            "num_traj_samples": args.num_traj_samples,
            "inference_step": args.inference_step,
            "max_new_tokens": args.max_generation_length,
            "cuda_graph": args.cuda_graph,
            "cuda_graph_max_graphs": args.cuda_graph_max_graphs if args.cuda_graph else None,
            "trace_level": args.trace_level,
            "sample_hz": args.sample_hz,
            "l4t_release": hw.get("hw.l4t_release"),
            "nvidia_driver": hw.get("hw.nvidia_driver"),
            "warmup": args.warmup,
            "overhead_probe": args.overhead_probe,
            "timing_repeats": args.timing_repeats,
            "repeat_clips": args.repeat_clips,
            "memory_snapshot": args.memory_snapshot,
        }
        # Throttling moves latency without moving anything else, and after the
        # run there is no way to tell that from a regression.
        #
        # Sampled in the background rather than between clips. Between-clip
        # readings are fine for temperature, which moves over seconds, but every
        # one of them lands after inference finished and after the figure was
        # drawn -- so the power averages described an idle GPU, which is a
        # different quantity rather than a low estimate.
        thermal = TH.ThermalLog(hz=args.sample_hz)
        thermal.sample()
        board = TH.PendingAttribution(thermal)
        if args.trace_level == "off":
            print("[trace] level off: no token statistics, no x0 and no spans are recorded -- "
                  "only the wall clock. predictions.parquet will lack token_ids.")
        if args.trace_level == "layer" and args.cuda_graph:
            print("[trace] level layer with --cuda-graph: layer hooks do not fire on a replayed "
                  "step, so replayed Euler steps have no layer rows.")
        with thermal.sampling():
            if args.warmup and clips:
                TH.mark("warmup", 0)
                # On the first clip, inside the sampler so the board's state during
                # warmup is on record too. Nothing from these passes reaches the
                # predictions: they exist to absorb the process's first-pass costs.
                started = time.perf_counter()
                _, warm_inputs, warm_kwargs = prepare_clip(processor, avdi, clips[0], args)
                for n in range(args.warmup):
                    _, _, _, warm = infer(model, clips[0], warm_inputs, warm_kwargs, args,
                                          args.trace_level)
                    timing_rows.append({
                        **timing_base, **warm.timing.row(), "clip_id": clips[0],
                        "t0_us": args.t0_us, "clip_index": 0, "row_kind": "warmup",
                        "pass_index": n + 1, "trace_level": args.trace_level,
                    })
                    board.add(timing_rows[-1], warm.timing.windows)
                del warm_inputs, warm_kwargs
                print(f"[warmup] {args.warmup} pass(es) on {clips[0][:8]} "
                      f"{time.perf_counter() - started:.1f}s")
            for i, clip_id in enumerate(clips):
                started = time.perf_counter()
                clock = HS.StageClock()
                HS.CURRENT.clock = clock
                # Every board reading from here on belongs to clip i, and to the
                # phase of it the pipeline is in -- run_clip moves it along.
                TH.mark("load", i)
                clip_rows, extras, clip_timing = run_clip(
                    model, processor, avdi, clip_id, args, out_dir,
                    extra_passes=extra_passes_for(i, args),
                )
                rows.extend(clip_rows)
                per_clip.append(extras)
                for timing_row in clip_timing:
                    windows = timing_row.pop(WINDOWS, None)
                    join = {"run_id": run_id, "clip_id": clip_id, "clip_index": i,
                            "row_kind": timing_row.get("row_kind"),
                            "pass_index": timing_row.get("pass_index"),
                            "timing_schema_version": TS.TIMING_SCHEMA_VERSION,
                            "tracer_version": TS.TRACER_VERSION}
                    for span in timing_row.pop(LAYERS, None) or []:
                        layer_rows.append({**span, **join})
                    for kernel in timing_row.pop(KERNELS, None) or []:
                        kernel_rows.append({**kernel, **join})
                    timing_rows.append({**timing_base, **timing_row, "clip_id": clip_id,
                                        "t0_us": args.t0_us, "clip_index": i})
                    board.add(timing_rows[-1], windows)
                # The main pass is the clip's first row; its host stages are
                # complete only once the figure and the flush below are done.
                main_row = timing_rows[-len(clip_timing)]
                if extras.get("gt_xy") is not None:
                    gt_rows.append({"clip_id": clip_id, "t0_us": args.t0_us, "gt_xy": extras["gt_xy"]})
                if not args.no_samples and i < MAX_SAMPLE_IMAGES:
                    with clock.span("render_ms"):
                        render_sample(
                            {**extras, "pred_xy": extras["pred_xy"]},
                            extras["data"],
                            out_dir / "samples" / f"{clip_id}.png",
                        )
                extras.pop("data", None)  # frames are large; do not hold them for the whole run
                # A partial run directory is a valid one. It carries every row
                # finished so far and a run.json that says it is partial, so a
                # crash costs at most flush_every clips and the analysis can
                # still read what completed. The final write below replaces it.
                done = i + 1
                if args.flush_every and done % args.flush_every == 0 and done < len(clips):
                    board.settle()
                    with clock.span("flush_ms"):
                        W.write_run(out_dir, rows, config,
                                    {**identity, "partial": True, "n_clips_done": done},
                                    gt=gt_rows if args.include_gt else None, per_clip=per_clip,
                                    timing=timing_rows, thermal=thermal, layers=layer_rows,
                                    kernels=kernel_rows)
                main_row.update(clock.row(
                    clip_wall_ms=(time.perf_counter() - clock.started) * 1000.0))
                HS.CURRENT.clock = None
                TH.mark("between")
                board.settle()
                reading = thermal.latest()
                print(
                    f"[{i + 1}/{len(clips)}] {clip_id[:8]} "
                    f"minADE {extras.get('min_ade', float('nan')):.3f} "
                    f"scene {extras['scene']} {time.perf_counter() - started:.1f}s "
                    f"tj {reading.get('tj-thermal', float('nan')):.0f}C"
                )

        # The sampler has stopped: every reading is in, so every pass still
        # waiting gets its energy from the complete series.
        board.settle(final=True)
        space = model.action_space
        meta = {
            **identity,
            # Explicit, so a reader that finds a run.json can tell a finished
            # run from a flush that happened to be the last thing written
            # before the process died.
            "partial": False,
            "n_clips_done": len(clips),
            # Normalization differs per checkpoint; without it, kinematics
            # recomputed offline are quietly wrong.
            "accel_mean": float(space.accel_mean),
            "accel_std": float(space.accel_std),
            "curvature_mean": float(space.curvature_mean),
            "curvature_std": float(space.curvature_std),
            "dt": float(space.dt),
            "n_waypoints": int(space.n_waypoints),
            # None, never 0, when the tracer did not run. Zero is a plausible
            # length and a reader slicing on it walks into the padding -- which
            # is how this field came to be wrong in every run before schema 2.
            # Fixing sample() closed the common path; this closes the one where
            # the tracer itself failed and the rows never carried the key.
            "prompt_len": next(
                (int(r["prompt_len"]) for r in rows if r.get("prompt_len") is not None), None
            ),
            # How many rows the token tracer actually reached. It swallows its
            # own failures so a bad hook cannot kill a run, which means a silent
            # zero here would otherwise be the only sign it never ran.
            "n_traced_rows": sum(1 for r in rows if r.get("token_ids") is not None),
            "n_rows_total": len(rows),
            # Archived so a reader can find the reasoning and meta-action spans
            # inside token_ids without knowing this checkpoint. Several of these
            # ids are used nowhere in this repo, which is the point -- the
            # parquet is meant to outlive the code that wrote it.
            "special_token_ids": DEFAULT_SPECIAL_IDS,
            "params_billions": sum(p.numel() for p in model.parameters()) / 1e9,
        }
        meta.update(M.model_size(model))
        # The tracer reads and resets the allocator's peak at every segment
        # boundary, folding each read into a run-level maximum first; without
        # that, max_memory_allocated would describe only the last segment.
        meta["vram_peak_gb"] = run_peak_bytes() / 1e9
        # What each part of the model weighs as loaded -- quantized checkpoints
        # restore into different tensors than they were saved as.
        inventory = M.module_inventory(model)
        meta["modules"] = inventory
        if _MEMSNAPS:
            meta["memory_snapshots"] = list(_MEMSNAPS)
        if _SYNC_SITES:
            meta["sync_sites"] = sorted(_SYNC_SITES.items(), key=lambda kv: -kv[1])[:30]
        # What the efficiency numbers are computed from, so they can be again.
        meta["work_model"] = work
        if peaks:
            meta["roofline"] = peaks
        counted = [r for r in timing_rows if r.get("row_kind") == "flops"]
        if counted:
            meta["work_calibration"] = {}
            for r in counted:
                ratios = RL.calibration(r, work)
                meta["work_calibration"][r["clip_id"]] = ratios
                off = RL.outside_band(ratios)
                if off:
                    print(f"[flops] {r['clip_id'][:8]}: counted/analytic FLOPs outside "
                          f"{RL.CALIBRATION_BAND}: {off} -- a formula in roofline.work is "
                          "wrong, and so are the efficiency numbers that use it.")
        profiled = [r for r in timing_rows if r.get("row_kind") == "profile"]
        if profiled:
            meta["profile"] = {
                "passes": len(profiled), "traces": list(_PROFILE_TRACES),
                "kernels_schema_version": TS.KERNELS_SCHEMA_VERSION,
                # The backend each phase dispatched to, over every profile pass:
                # an attention implementation can be demoted inside one submodule
                # and not another, and only the profile sees which.
                "sdpa": {p: sorted({b for r in profiled if r.get(f"prof_sdpa_{p}")
                                    for b in str(r[f"prof_sdpa_{p}"]).split("+")})
                         for p in PP.SDPA_PHASES},
            }
        # The recording rules want the tracer's cost in run.json as well as in
        # MLflow: it is what a reader needs to decide how far to trust any
        # latency here, and run.json is what survives without the tracker.
        probe = TS.overhead(timing_rows) or {}
        meta["trace"] = {
            "level": args.trace_level,
            "warmup": args.warmup,
            "overhead_probe": args.overhead_probe,
            "timing_repeats": args.timing_repeats,
            "repeat_clips": args.repeat_clips,
            **{key.split(".", 1)[1]: value for key, value in probe.items()},
        }
        meta["thermal"] = thermal.summary()
        meta["power_mode"] = thermal.mode
        meta["hw"] = hw
        print(f"\n{thermal.verdict()}")
        W.write_run(out_dir, rows, config, meta,
                    gt=gt_rows if args.include_gt else None, per_clip=per_clip,
                    timing=timing_rows, thermal=thermal, layers=layer_rows,
                    kernels=kernel_rows)
        print(f"\nrun directory: {out_dir}")

        # Archive before the tracking short-circuit below. --no-track means "do
        # not record this run", not "do not keep its outputs" -- and --no-upload
        # already exists for the latter. Coupling them left every --no-track run
        # as a single copy on this machine's disk with nothing pointing at it.
        sha = None
        if not args.no_upload:
            sha = mlp.upload_run_dir(
                W.upload_paths(out_dir), args.evals_repo, f"runs/{out_dir.name}"
            )

        if run is None:
            return

        scored = [c["min_ade"] for c in per_clip if c.get("min_ade") is not None]
        run.metric("n_clips", len(clips))
        if scored:
            values = np.asarray(scored, dtype=float)
            run.score(float(values.mean()))
            # The denominator. score is a mean over clips that had ground
            # truth, which is not the same as n_clips, and reading the two side
            # by side without this makes it look like it was.
            run.metric("n_scored", float(values.size))
            # The shape of the distribution, not just its middle. A mean that
            # holds while the tail doubles is the regression this misses.
            for pct in (50, 90, 95):
                run.metric(f"min_ade_p{pct}", float(np.percentile(values, pct)))
            run.metric("min_ade_max", float(values.max()))
            run.metric("frac_over_2m", float((values > 2.0).mean()))

        # Averages over per-clip values. Per-sample rows stay in the parquet.
        #
        # Declared, not swept. Collecting whatever keys per_clip happens to hold
        # means the run's metric namespace is decided by whatever metrics.py
        # last returned: add a key there and every future run silently grows a
        # column, drop one and old runs have a column new ones lack.
        #
        # TS.is_number rather than isinstance(v, (int, float)): the latter skips
        # np.float32 silently -- a metric computed with numpy never reached MLflow
        # and was not even named in the warning below -- and accepts True as 1.
        for key in _CLIP_METRICS:
            values = [float(c[key]) for c in per_clip if TS.is_number(c.get(key))]
            if values:
                run.metric(key, float(np.mean(values)))
        unrecorded = sorted(
            {k for c in per_clip for k, v in c.items() if TS.is_number(v)}
            - set(_CLIP_METRICS)
            - _CLIP_COORDS
        )
        if unrecorded:
            print(f"[trace] not recorded (add to _CLIP_METRICS): {', '.join(unrecorded)}")
        for key in ("logprob_mean", "perplexity", "entropy_mean", "entropy_p95",
                    "low_confidence_ratio"):
            vals = []
            for r in rows:
                q = M.token_quality(
                    r.get("token_logprob", []), r.get("token_entropy", []),
                    int(r.get("n_generated_tokens", 0)),
                )
                if key in q:
                    vals.append(q[key])
            if vals:
                run.metric(key, float(np.mean(vals)))
        def mean_present(key: str) -> float | None:
            """Mean over rows that actually carry the key.

            Defaulting a missing measurement to 0.0 records "took no time" as
            though it were an observation. A metric that was never taken must
            be absent, not zero.
            """
            vals = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        for key in ("n_generated_tokens", "n_cot_tokens"):
            value = mean_present(key)
            if value is not None:
                run.metric(f"{key}_mean", value)
        # The share of samples that ran to max_new_tokens without the end
        # marker. The recording rules require it: a compressed model that stops
        # closing its reasoning gets slower and worse at once, and a mean token
        # count hides it among the samples that did close.
        missing = [bool(r["eos_missing"]) for r in rows if r.get("eos_missing") is not None]
        if missing:
            run.metric("eos_missing_rate", float(np.mean(missing)))

        # Latency. One decode span covers a whole clip's K samples, so the rows
        # of one clip share it -- averaging it against a single row's token
        # count divides a batch total by a per-row denominator and lands ~K
        # times off. Group by clip and use denominators that match the span.
        by_clip: dict[str, list[dict]] = {}
        for r in rows:
            by_clip.setdefault(r.get("clip_id"), []).append(r)
        per_step, per_token = [], []
        for group in by_clip.values():
            head = group[0]
            if not head.get("timing_measured") or head.get("t_decode_ms") is None:
                continue
            decode_ms = float(head["t_decode_ms"])
            steps = head.get("n_decode_steps") or 0
            tokens = sum(int(r.get("n_generated_tokens") or 0) for r in group)
            if steps:
                per_step.append(decode_ms / steps)
            if tokens:
                per_token.append(decode_ms / tokens)
        # Deliberately not named ms_per_token: runs before schema 2 recorded a
        # key by that name whose value was off by roughly K, and two identical
        # runs disagreed by 46%. Reusing the name would mix the two silently on
        # one axis. A series that visibly stops is easier to trust.
        if per_step:
            run.metric("ms_per_decode_step", float(np.mean(per_step)))
        if per_token:
            run.metric("ms_per_generated_token", float(np.mean(per_token)))

        # How many clips the timings above actually rest on. Without it a mean
        # over 3 measured clips out of 100 looks like a mean over 100.
        run.metric("n_timed_clips", float(len(per_step)))
        # Everything else about latency, from the timing table: the legacy
        # spans, tails, the host clock, per-step shapes, allocator and graph
        # counters. One batched request -- the tracking server is a
        # single-worker box and each metric call is a round trip -- and the key
        # set is declared in timing_schema.AGGREGATE_KEYS rather than swept off
        # whatever a run happened to compute. Only measured main passes count:
        # the per-row averaging this replaces took 0.0 from any pass whose
        # timing failed, and reported it as a pass that took no time.
        try:
            run.metrics(TS.aggregate(timing_rows, model=work, peaks=peaks,
                                     deadline_ms=args.deadline_ms,
                                     steady_skip=args.steady_skip))
        except Exception as exc:  # a lost metric batch must not lose the run's link
            print(f"[timing] aggregate metrics not recorded: {exc}", file=sys.stderr)
        # Measured during the loop, so it could not be among the parameters
        # logged when the run opened. New keys may be added; logged ones may not
        # change, so this is logged once, here.
        # The model's weights by part are known only once it has loaded, which
        # is after the run opened: they join the same late batch.
        late = {key: round(value, 4) for key, value in inventory.items()}
        late.update(hw)
        if probe:
            late["trace.overhead_pct"] = round(probe["trace.overhead_pct"], 4)
            late["trace.overhead_n"] = int(probe["trace.overhead_n"])
        for phase, backends in (meta.get("profile") or {}).get("sdpa", {}).items():
            late[f"prof.sdpa_{phase}"] = "+".join(backends) or "none"
        try:
            run.params(late)
        except Exception as exc:
            print(f"[timing] late params not recorded: {exc}", file=sys.stderr)
        # Breakdown by scene: fixed cardinality, and it answers the question
        # actually being asked -- does the model give up on curves? The previous
        # code wrote a metric key per clip UUID, which put 100 keys in a 148-key
        # run. A UUID is a primary key, not an axis: it cannot be sorted,
        # grouped or plotted, the key set differs between runs so the
        # experiment's union grows without bound, and each one cost its own HTTP
        # round trip to a single-worker server. Per-clip values are already in
        # predictions.parquet, where metrics.py argues they belong.
        # Two axes, each a partition of the scored clips: every clip lands in
        # exactly one lateral bucket and one longitudinal bucket, so each axis
        # sums back to n_scored. A clip that is both cornering and braking is
        # counted in "curve" and in "decel" -- the axes overlap by design, which
        # is what lets you read them as two tables rather than nine thin cells.
        buckets: dict[str, list[dict]] = {}
        for c in per_clip:
            if c.get("min_ade") is None:
                continue
            for key in ("scene", "speed_profile"):
                label = c.get(key)
                if label:
                    buckets.setdefault(label, []).append(c)

        def bucket_stats(clips: list[dict]) -> dict[str, float]:
            """The few numbers that say *how* a bucket went, not just how badly.

            score alone cannot separate "the model is worse here" from "the
            model hedged and one sample landed here" -- mean_ade against
            min_ade, and the spread between samples, are what tell them apart.
            Deliberately short: this set multiplies by the number of buckets.
            """
            out: dict[str, float] = {"count": len(clips)}
            for name in _BUCKET_METRICS:
                values = [float(c[name]) for c in clips if TS.is_number(c.get(name))]
                if values:
                    out[name] = float(np.mean(values))
            out["score"] = out.pop("min_ade", float("nan"))
            return {k: v for k, v in out.items() if v == v}   # drop NaN

        run.by_scenario({s: bucket_stats(v) for s, v in buckets.items()})
        # How often the model read the situation differently from the data. Not
        # an accuracy number in itself, but a variant whose agreement drops is
        # misjudging what it is looking at before it misplaces a trajectory.
        judged = [c for c in per_clip
                  if c.get("scene") not in (None, "unknown") and c.get("scene_predicted")]
        if judged:
            agree = sum(1 for c in judged if c["scene"] == c["scene_predicted"])
            run.metric("scene_agreement", agree / len(judged))
        # The clips a person would actually open. Bounded, and readable in the
        # UI as one tag instead of hunting through a metric list.
        worst = sorted(
            ((c["clip_id"], float(c["min_ade"])) for c in per_clip
             if c.get("min_ade") is not None),
            key=lambda kv: -kv[1],
        )[:10]
        if worst:
            run.tag("worst_clips", json.dumps([[k, round(v, 4)] for k, v in worst]))
        run.metrics(thermal.summary())
        run.artifact(out_dir / "run.json", name="eval")

        if sha:
            run.result_path(f"hf:{args.evals_repo}@{sha}#runs/{out_dir.name}/")
        else:
            # Either the upload was skipped or it failed. Both leave the files
            # on this machine only, so record the path honestly -- and when it
            # was a failure, say so, because a local path standing in for an
            # archive is exactly what makes a lost run look like a good one.
            run.result_path(f"path:{out_dir}")
            if not args.no_upload:
                run.tag("upload_failed", "true")

    if args.no_track:
        execute(None)
        return
    with mlp.evaluate(
        args.experiment,
        # 폴더 이름과 같은 표기를 쓴다. 표에서 본 run 을 디스크에서 찾을 때
        # 머릿속에서 변환하지 않아도 되도록.
        run_name=f"{args.variant}@{W.machine_name(args.machine)}"
                 f"-{len(clips)}clip-k{args.num_traj_samples}"
                 f"-temp{args.temperature:g}"
                 f"{'-' + args.label if args.label else ''}",
        model=model_coordinate(args.model, args.model_revision),
        # The snapshot the data was actually read from, not main at record time.
        hf_datasets=[f"{DATASET_REPO}@{args.dataset_revision}"],
        params=params,
        variant=args.variant,
        split=f"clips:{len(clips)}",
        conditioning_source="generated",
        root=REPO_ROOT,
        seed=args.seed,
        notes=args.notes,
    ) as run:
        if args.sweep:
            run.tag("sweep", args.sweep)
        if args.label:
            run.tag("label", args.label)
        execute(run)


if __name__ == "__main__":
    main()
