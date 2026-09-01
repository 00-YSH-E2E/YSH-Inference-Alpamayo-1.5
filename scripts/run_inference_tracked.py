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

import ml_platform_track as mlp  # noqa: E402

import physical_ai_av  # noqa: E402
from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from alpamayo1_5.trace import metrics as M  # noqa: E402
from alpamayo1_5.trace import thermal as TH  # noqa: E402
from alpamayo1_5.trace import writer as W  # noqa: E402
from alpamayo1_5.trace.token_trace import DEFAULT_SPECIAL_IDS, trace_inference  # noqa: E402

MODEL_REPO = "nvidia/Alpamayo-1.5-10B"
DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
EVALS_REPO = "YSHRobotics/Alpamayo-Evals"
DATA_CACHE = "/home/thor/Documents/Alpamayo/Data/Alpamayo-1.5_Cam-4_Vanilla"
DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"
MAX_SAMPLE_IMAGES = 20  # representative figures per run, per the recording rules

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
)


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=MODEL_REPO)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--variant", default="Vanilla", help="Vanilla, Pruned-24L, INT8 ...")
    p.add_argument("--data-spec", default="Cam-4")
    p.add_argument("--experiment", default="alpamayo-1.5")
    p.add_argument("--notes", help="One or two human sentences: why this run exists.")
    p.add_argument("--data-cache", default=DATA_CACHE)
    p.add_argument("--allow-stream", action="store_true",
                   help="Permit streaming when a clip is not cached. Off by default so a "
                        "missing cache fails loudly instead of running 10x slower.")
    p.add_argument("--out-root", default="out")
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

        clips = pd.read_parquet(args.clip_list)["clip_id"].tolist()
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


def run_clip(model, processor, avdi, clip_id: str, args, out_dir: Path) -> tuple[list[dict], dict]:
    """Inference for one clip. Returns one row per sample, plus per-clip extras."""
    data = load_physical_aiavdataset(
        clip_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.allow_stream
    )
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )
    diffusion_kwargs = {}
    if args.inference_step is not None:
        diffusion_kwargs["inference_step"] = args.inference_step

    torch.cuda.manual_seed_all(args.seed)
    with trace_inference(model) as tracer:
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
    timing = tracer.timing.as_dict()
    trace = tracer.trace

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
            **timing,
        }
        if trace is not None and k < trace.token_ids.shape[0]:
            row.update(trace.sample(k))
        rows.append(row)

    extras = {"clip_id": clip_id, "gt_xy": gt_xy, "data": data}
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
    extras["scene"] = M.classify_scene(
        extras.get("net_heading_abs_deg", 0.0), extras.get("lateral_offset_abs_m", 0.0)
    )
    extras["pred_xy"] = pred_xy
    return rows, extras


def main() -> None:
    args = parse_args()
    if args.inference_step is not None and args.inference_step < 1:
        raise SystemExit(
            "--inference-step must be >= 1. The sampler reads 0 as 'unset' and silently "
            "runs its default, which quietly corrupts any latency measurement."
        )
    clips = resolve_clips(args)
    date = datetime.now(timezone.utc).astimezone().strftime("%y.%m.%d")

    params = {
        "model": args.model,
        "variant": args.variant,
        "data_spec": args.data_spec,
        "n_clips": len(clips),
        "num_traj_samples": args.num_traj_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_generation_length": args.max_generation_length,
        "inference_step": args.inference_step,
        "seed": args.seed,
        "attn_impl": args.attn,
        "dtype": "bfloat16",
        "t0_us": args.t0_us,
        "torch_disable_native_jit": os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
        "torch_version": torch.__version__,
        "data_cache": args.data_cache,
        "power_mode": TH.power_mode(),
    }

    def execute(run: mlp.Run | None) -> None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.data_cache)
        model = Alpamayo1_5.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=args.attn
        ).to("cuda").eval()
        processor = helper.get_processor(model.tokenizer)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        run_id = run.run_id if run is not None else f"local{int(time.time()):x}"
        out_dir = Path(args.out_root) / W.run_dir_name(
            args.variant, date, run_id, data=args.data_spec
        )
        rows, per_clip, gt_rows = [], [], []
        # Throttling moves latency without moving anything else, and after the
        # run there is no way to tell that from a regression.
        #
        # Sampled in the background rather than between clips. Between-clip
        # readings are fine for temperature, which moves over seconds, but every
        # one of them lands after inference finished and after the figure was
        # drawn -- so the power averages described an idle GPU, which is a
        # different quantity rather than a low estimate.
        thermal = TH.ThermalLog()
        thermal.sample()
        with thermal.sampling():
            for i, clip_id in enumerate(clips):
                started = time.perf_counter()
                clip_rows, extras = run_clip(model, processor, avdi, clip_id, args, out_dir)
                rows.extend(clip_rows)
                per_clip.append(extras)
                if extras.get("gt_xy") is not None:
                    gt_rows.append({"clip_id": clip_id, "t0_us": args.t0_us, "gt_xy": extras["gt_xy"]})
                if not args.no_samples and i < MAX_SAMPLE_IMAGES:
                    render_sample(
                        {**extras, "pred_xy": extras["pred_xy"]},
                        extras["data"],
                        out_dir / "samples" / f"{clip_id}.png",
                    )
                extras.pop("data", None)  # frames are large; do not hold them for the whole run
                reading = thermal.sample()
                print(
                    f"[{i + 1}/{len(clips)}] {clip_id[:8]} "
                    f"minADE {extras.get('min_ade', float('nan')):.3f} "
                    f"scene {extras['scene']} {time.perf_counter() - started:.1f}s "
                    f"tj {reading.get('tj-thermal', float('nan')):.0f}C"
                )

        config = {
            "run_id": run_id,
            "variant": args.variant,
            "git_commit": mlp.git_tags().get("git_commit"),
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
                "num_traj_samples": args.num_traj_samples,
                "conditioning_source": "generated",
            },
        }
        space = model.action_space
        meta = {
            "run_id": run_id,
            "variant": args.variant,
            "date": date,
            "clips": clips,
            "params": params,
            # Normalization differs per checkpoint; without it, kinematics
            # recomputed offline are quietly wrong.
            "accel_mean": float(space.accel_mean),
            "accel_std": float(space.accel_std),
            "curvature_mean": float(space.curvature_mean),
            "curvature_std": float(space.curvature_std),
            "dt": float(space.dt),
            "n_waypoints": int(space.n_waypoints),
            "prompt_len": int(rows[0].get("prompt_len", 0)) if rows else 0,
            # Archived so a reader can find the reasoning and meta-action spans
            # inside token_ids without knowing this checkpoint. Several of these
            # ids are used nowhere in this repo, which is the point -- the
            # parquet is meant to outlive the code that wrote it.
            "special_token_ids": DEFAULT_SPECIAL_IDS,
            "params_billions": sum(p.numel() for p in model.parameters()) / 1e9,
        }
        meta.update(M.model_size(model))
        meta["thermal"] = thermal.summary()
        meta["power_mode"] = thermal.mode
        print(f"\n{thermal.verdict()}")
        W.write_run(out_dir, rows, config, meta, gt=gt_rows if args.include_gt else None)
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
        for key in _CLIP_METRICS:
            values = [c[key] for c in per_clip if isinstance(c.get(key), (int, float))]
            if values:
                run.metric(key, float(np.mean(values)))
        unrecorded = sorted(
            {k for c in per_clip for k, v in c.items() if isinstance(v, (int, float))}
            - set(_CLIP_METRICS)
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

        for key in ("t_vision_ms", "t_prefill_ms", "t_decode_ms", "t_postgen_ms",
                    "t_expert_ms", "t_other_ms", "t_total_ms"):
            value = mean_present(key)
            if value is not None:
                run.metric(key, value)
        # How many clips the timings above actually rest on. Without it a mean
        # over 3 measured clips out of 100 looks like a mean over 100.
        run.metric("n_timed_clips", float(len(per_step)))
        # Breakdown by scene: fixed cardinality, and it answers the question
        # actually being asked -- does the model give up on curves? The previous
        # code wrote a metric key per clip UUID, which put 100 keys in a 148-key
        # run. A UUID is a primary key, not an axis: it cannot be sorted,
        # grouped or plotted, the key set differs between runs so the
        # experiment's union grows without bound, and each one cost its own HTTP
        # round trip to a single-worker server. Per-clip values are already in
        # predictions.parquet, where metrics.py argues they belong.
        by_scene: dict[str, list[float]] = {}
        for c in per_clip:
            if c.get("scene") and c.get("min_ade") is not None:
                by_scene.setdefault(c["scene"], []).append(float(c["min_ade"]))
        run.by_scenario({s: (float(np.mean(v)), len(v)) for s, v in by_scene.items()})
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
        run_name=f"{args.variant}-{len(clips)}clip-n{args.num_traj_samples}",
        model=f"hf:{args.model}@main",
        hf_datasets=[f"{DATASET_REPO}@main"],
        params=params,
        variant=args.variant,
        split=f"clips:{len(clips)}",
        conditioning_source="generated",
        seed=args.seed,
        notes=args.notes,
    ) as run:
        execute(run)


if __name__ == "__main__":
    main()
