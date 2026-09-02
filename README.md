<div align="center">

# 🏔️ Alpamayo 1.5

### Supercharging Autonomous Driving with Interactive, Steerable Reasoning

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Alpamayo--1.5--10B-blue)](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

> **This is a fork of [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5)**
> that adds instrumented, recorded inference on Jetson AGX Thor. Everything below
> is upstream's documentation; what this fork adds — and where to start on a
> Jetson, because the first command upstream suggests does not work there — is in
> [Running tracked inference](#running-tracked-inference) at the end.

## Updates

- [May 2026] SFT and RL post-training scripts are available in [Alpamayo Recipes](https://github.com/NVlabs/alpamayo-recipes): [Alpamayo 1.5 SFT](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_sft) and [Alpamayo 1.x RL post-training](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl).

**📖 Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) first!**
The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations. This GitHub README focuses on setup, usage, and frequently asked questions.

## Support

📣 **Usage questions and discussion about Alpamayo 1.5**: please join us on the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Code-level bugs, documentation issues, and feature requests**: file a [GitHub issue](../../issues/new/choose) using the appropriate template (Bug report, Documentation request, or Feature request). The relevant NVIDIA responder is auto-assigned via the `assignees:` field on the template.

🚨 **Security vulnerabilities**: please use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). Do not file security issues publicly here.

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x** with `nvcc` (required to compile `flash-attn` from source). If you don't have it, see [Troubleshooting](#flash-attention-issues) for a fallback using PyTorch's built-in SDPA.
- **Python 3.12**

### Hardware requirements

| Configuration                                           | VRAM   |
| ------------------------------------------------------- | ------ |
| Single-sample inference (`num_traj_samples=1`)          | ~24 GB |
| Multi-sample inference (`num_traj_samples=16`)          | ~40 GB |
| Multi-sample inference with CFG (`num_traj_samples=16`) | ~60 GB |

Measured on an NVIDIA H100 80GB GPU.

## Getting Started

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Set up the environment

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
```

> **Note:** If `uv sync` fails on `flash-attn`, see [Troubleshooting](#flash-attention-issues) below.

### 3. Authenticate with HuggingFace

The model and dataset require access to gated resources. Request access here:

- 🤗 [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo-1.5-10B Model](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Then authenticate:

```bash
hf auth login
```

Get your token at: https://huggingface.co/settings/tokens

> **Note:** The `physical_ai_av` package (auto-installed via dependencies) streams data from the HuggingFace dataset. You must have accepted the dataset access request above before running inference.

## Running Inference

### Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo1_5/test_inference.py
```

> **On Jetson (Thor, Orin), run `python scripts/smoke_inference.py` instead.**
> `test_inference.py` loads the model without naming an attention
> implementation, which defaults to flash-attn — and flash-attn has no aarch64
> wheel, so the command above fails on the hardware this fork targets.
> `scripts/smoke_inference.py` is the same pipeline with SDPA and a device
> report. See [Running tracked inference](#running-tracked-inference).

In case you would like to obtain more trajectories and reasoning traces, please feel free to increase
the `num_traj_samples` argument in the script.

### Interactive notebooks

We provide notebooks that demonstrate the different capabilities of Alpamayo 1.5 under `notebooks/`, including standard model inference, incorporating navigation guidance, modifying the number of cameras, and visual question answering.

### Inference methods

Alpamayo 1.5 provides two inference methods:

- **`sample_trajectories_from_data_with_vlm_rollout`** -- Full pipeline: the VLM generates chain-of-causation reasoning, then a diffusion expert produces trajectory predictions conditioned on the VLM's hidden states. This is the primary inference method used by the test script and most notebooks.

- **`generate_text`** -- Text-only generation for visual question answering (VQA). Returns extracted text fields.

### Optional CUDA graph acceleration

Repeated trajectory inference can replay the diffusion expert with exact-shape CUDA graphs. Enable
this after moving the model to CUDA and calling `eval()`:

```python
model.eval()
model.enable_diffusion_expert_cuda_graph(
    max_batch_size=16,
    max_graphs=4,
)
```

Set `max_batch_size` to at least `batch_size * num_traj_samples * num_traj_sets`. The first
supported input shape is captured lazily; up to `max_graphs` exact shape signatures are retained,
and additional signatures fall back to eager execution. Captured graphs keep static CUDA buffers,
so this option trades additional GPU memory for lower diffusion-expert launch overhead. Inspect
`model.diffusion_expert_cuda_graph_stats` for capture, replay, and fallback counts.


## Fine-tuning and Post-training Recipes

SFT and RL post-training scripts are maintained in [Alpamayo Recipes](https://github.com/NVlabs/alpamayo-recipes):

- [Alpamayo 1.5 SFT](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_sft)
- [Alpamayo 1.x RL post-training](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl), including Alpamayo 1.5

## Project Structure

```
alpamayo_1.5_release/
├── notebooks/
│   ├── inference.ipynb                  # Standard model inference
│   ├── inference_cam_num.ipynb          # Inference with different camera counts
│   ├── inference_nav.ipynb              # Inference with navigation guidance
│   └── inference_vqa.ipynb              # Visual question answering
├── src/
│   └── alpamayo1_5/
│       ├── action_space/
│       │   └── ...                      # Action space definitions
│       ├── diffusion/
│       │   └── ...                      # Diffusion model components
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   ├── ...                      # Model components and utils functions
│       ├── __init__.py                  # Package marker
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       ├── test_inference.py            # Inference test script
├── pyproject.toml                       # Project dependencies
└── uv.lock                              # Locked dependency versions
```

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. `flash-attn` requires CUDA Toolkit (specifically `nvcc`) at build time. If you see build errors during `uv sync`:

**Option A: Install without flash-attn and use SDPA fallback**

```bash
uv sync --active --no-install-package flash-attn
```

Then load the model with PyTorch's built-in scaled dot-product attention:

```python
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
```

**Option B: Install CUDA Toolkit, then retry**

Install CUDA Toolkit 12.x (e.g., via your package manager or [NVIDIA's install guide](https://developer.nvidia.com/cuda-downloads)), ensure `nvcc` is on your PATH, then re-run:

```bash
uv sync --active
```

## Frequently Asked Questions (FAQ)

<details>
<summary><strong>How does Alpamayo 1.5 relate to Alpamayo 1?</strong></summary>

Alpamayo 1.5 expands upon the architecture released in Alpamayo 1 and fully realizes what is described in our paper [*"Alpamayo 1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
"*](https://arxiv.org/abs/2511.00088). Specifically:

| Feature                                 | Description                                                      | Alpamayo 1             | Alpamayo 1.5       |
| --------------------------------------- | ---------------------------------------------------------------- | ---------------------- | ------------------ |
| **Chain-of-Causation (CoC) reasoning**  | Hybrid auto-labeling with human in the loop for reasoning traces | ✅ Included            | ✅ Included        |
| **Vision-Language-Action architecture** | Cosmos-Reason backbone + action expert                           | ✅ Included            | ✅ Included        |
| **Trajectory prediction**               | 6.4s horizon, 64 waypoints at 10 Hz                              | ✅ Supported           | ✅ Supported       |
| **RL post-training**                    | Reinforcement learning for reasoning/action consistency          | ❌ Not RL post-trained | ✅ RL post-trained |
| **Navigation conditioning**             | Explicit navigation inputs                                       | ❌ Not supported       | ✅ Supported       |
| **General VQA**                         | Supports visual question answering                               | ❌ Not supported       | ✅ Supported       |
| **Flexible multi-camera support**       | Supports a variable number of input cameras                      | ❌ Not supported       | ✅ Supported       |

</details>

<details>
<summary><strong>Does Alpamayo 1.5 accept navigation inputs?</strong></summary>

Yes! Please see `notebooks/inference_nav.ipynb` for examples.

</details>

<details>
<summary><strong>Does Alpamayo 1.5 support general VQA?</strong></summary>

Yes! Please see `notebooks/inference_vqa.ipynb` for examples.

</details>

<details>
<summary><strong>Was Alpamayo 1.5 post-trained with Reinforcement Learning (RL)?</strong></summary>

Yes! Alpamayo 1.5 has undergone RL post-training, achieving improvements in reasoning quality and reasoning-trajectory alignment as a result.

</details>

<details>
<summary><strong>Does Alpamayo 1.5 accept different numbers of cameras?</strong></summary>

Yes! Please see `notebooks/inference_cam_num.ipynb` for examples. Note that model accuracy may degrade with fewer cameras, the magnitude of which will depend on the specific scenario. For instance, it is expected that Alpamayo 1.5 would struggle to see cross-traffic in a right turn if only provided a front-facing camera.

</details>

<details>
<summary><strong>What are the minimum GPU requirements?</strong></summary>

You need an NVIDIA GPU with at least **24 GB VRAM** for inference. Tested configurations include RTX 3090, A100, H100, and B200. Running on GPUs with less memory (e.g., 16 GB) will likely result in CUDA out-of-memory errors. Please refer to our [hardware requirements](#hardware-requirements) for more information.

</details>

<details>
<summary><strong>Can I use this model in production / commercial applications?</strong></summary>

Yes. See the [License](#license) section and the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for details.

</details>

## License

- **Inference code**: Apache License 2.0 - see [LICENSE](./LICENSE) for details.
- **Model weights**: OpenMDW-1.1 - see the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for details.

## Disclaimer

Alpamayo 1.5 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1.5 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1.5 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.

## Citation

If you use Alpamayo 1.5 in your research, please cite:

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
---

# Running tracked inference

*Everything from here down is this fork, not upstream NVIDIA.*

Upstream gives you a model and a script that prints trajectories. This fork adds
the part that makes a run mean something a month later: it records what code,
what data and what machine produced each number, writes the predictions to a
file, and pushes both to somewhere they survive the machine that made them.

The motivation is narrow. Comparing a pruned or quantized variant against the
baseline on a Jetson means comparing latency, and latency on a Jetson depends on
how hot the board was. After the fact there is no way to tell a real regression
from a warm heatsink — unless the temperature was recorded alongside the number.

## Where things go

```
MLflow  ──  numbers you sort and graph: minADE, latency, temperature, power
HF      ──  the files behind them: predictions.parquet, run.json, samples/*.png
            a coordinate on the run points from one to the other
```

Neither half stands alone. Metrics with no artifacts cannot be inspected;
artifacts with no metrics cannot be ranked.

## Start here

The short way: **edit the settings block at the top of `scripts/run.sh` and run
it.** Every flag has a named line there, and it checks what it can before
spending any GPU time — the variant name, whether the working tree is committed,
whether the data cache exists, whether MLflow is reachable. It reports every
problem at once rather than one per attempt, and it takes three seconds.

```bash
$EDITOR scripts/run.sh     # VARIANT, MACHINE, LIMIT, NOTES ...
./scripts/run.sh
```

To run several configurations, edit the axes at the top of `scripts/sweep.sh`
and run that instead. It is hydra's `--multirun` idea — list values on an axis
and the combinations all run:

```bash
SWEEP_VARIANT=("Vanilla" "Pruned-24L" "INT8")
SWEEP_MODEL=("nvidia/Alpamayo-1.5-10B" "hf:me/pruned-24l" "hf:me/int8")
SWEEP_NUM_TRAJ_SAMPLES=(1 6)          # 3 x 2 = 6 runs
```

**`--variant` is a label; `--model` decides what actually runs.** Listing three
variants against one checkpoint runs the same weights three times under three
names, and with a fixed seed the numbers come out identical — which reads as
"the variant made no difference". The sweep refuses that combination rather
than producing it.

Directory names gain a segment naming only the axes that varied, so a batch is
identifiable by eye rather than only by run id:

```
Alpamayo-1.5_Cam-4_Vanilla_k1-t0.6_thor_26.09.02_39581f9b
Alpamayo-1.5_Cam-4_Vanilla_k6-t0.9_thor_26.09.02_a1b2c3d4
```

A sweep that only varies the variant gets no such segment — the variant is
already in the name.

What it does *not* do is fold those into one run. The recording rules forbid
several evaluations in one run — the earlier result gets overwritten — so a
sweep is N separate runs carrying a shared `sweep` tag. They line up one per row
in the comparison table, and the tag selects the batch. Defaults come from
`run.sh`, which is sourced rather than duplicated, so settings live in one file.
A failed combination does not stop the rest, and the summary at the end says
which ones landed.

That MLflow check matters more than it looks. The tracking server is bound to
the tailnet address only, and the mlflow client retries seven times at a
120-second timeout — so from outside the tailnet a run hangs for minutes before
dying, having never started inference. The script gives up after three seconds
and tells you the three ways out.

The long way, if you would rather pass flags:

```bash
# 1. does the model load and run at all on this board?
python scripts/smoke_inference.py

# 2. a real, recorded run
export MLFLOW_TRACKING_URI=http://<your-hub>:5000     # or ML_PLATFORM_HOST
export HF_TOKEN=hf_...                                # to pin revisions to shas
git commit -am "..."                                  # BEFORE the run, see below
python scripts/run_inference_tracked.py --limit 3 --variant Vanilla
```

The run prints `PASS` or `FAIL` against the recording rules when it finishes. A
`FAIL` is a bug in the script, not in the run — the next run trips on the same
thing, so fix it there.

Useful flags: `--clip-list clips.parquet` / `--limit N` to choose clips,
`--variant` to name what makes this run different (it becomes a comparison
axis), `--num-traj-samples`, `--temperature`, `--seed`, `--no-upload` to keep
outputs local, `--no-track` to skip MLflow entirely.

## What a run leaves behind

```
out/Alpamayo-1.5_Cam-4_Vanilla_thor_26.09.01_39581f9b/
├── predictions.parquet   one row per (clip, t0, trajectory sample) — raw output
├── per_clip.parquet      one row per clip — situation label and its metrics
├── run.json              params, normalization constants, parquet schema
├── gt.parquet            logged future — local only, never uploaded
└── samples/<clip_id>.png camera grid + predicted vs logged trajectory
```

The name is the coordinate, and it reads in the order the questions get asked:

```
{model}_{data}_{variant}_{machine}_{YY.MM.DD}_{run_id[:8]}
```

**Machine is in the name because latency is meaningless without it.** The same
checkpoint on a Thor and on a Pro 6000 otherwise produces two directories
identical in every visible field, holding numbers that are not comparable. It
defaults to the short hostname — the same word `env.host` records — and
`--machine` overrides it for a rented box whose hostname means nothing.

The last eight characters are the MLflow run id, which makes the link
bidirectional: a directory names its run, and a run's `output_uri` names its
directory. Re-running the same configuration on the same day is routine, so the
date alone would collide.

`per_clip.parquet` is what makes any breakdown possible. `predictions.parquet`
holds raw model output by design, so minADE and the situation label are in
neither it nor MLflow — without this file, asking "how did the braking clips
do" means re-fetching the gated dataset and recomputing.

Everything except `gt.parquet` is pushed to the evals repo
(`--evals-repo`, default `YSHRobotics/Alpamayo-Evals`). **That repo has to stay
private:** the sample images embed frames from
`nvidia/PhysicalAI-Autonomous-Vehicles`, which is gated. Making it public is one
click and cannot be undone for anything already cloned.

## What gets recorded

| | |
|---|---|
| Code | `git_commit`, `git_branch`, `git_dirty` |
| Data & model | `hf_datasets`, `hf_models`, `model_source` — revisions pinned to 40-character shas, not branch names |
| Machine | `env.host`, `env.gpu`, `env.compute_capability`, `env.torch`, `env.cuda`, `power_mode` |
| Result | `score` (mean minADE, metres, lower is better), percentiles, per-situation breakdown |
| Situation | `scenario.<bucket>.<metric>` on two axes — `straight`/`curve`/`lane_change`/`other` and `cruise`/`accel`/`decel` — classified from the **logged future**, so every variant is compared over the same clips |
| Latency | `t_vision_ms`, `t_prefill_ms`, `t_decode_ms`, `t_postgen_ms`, `t_expert_ms`, `t_other_ms` — the six sum to `t_total_ms` |
| Thermal | `temp.tj_max_c`, `temp.peak_c`, `power.<rail>_mean_w`, `temp.throttle_risk` |
| Where the outputs went | `output_uri` — must be durable; a local path fails the check |

Per-clip values are **not** metrics. They live in `predictions.parquet`. A clip
id is a primary key, not an axis: keyed as metrics they cannot be grouped or
sorted, and the key set differs between runs.

## Two things the code cannot do for you

**Commit before running.** `git status --porcelain` is recorded as `git_dirty`,
and a dirty run is not reproducible. This is not hypothetical — half the runs
recorded so far are dirty, because the instrumentation was being written while
it was being used.

**Run from inside this repository.** MLflow reads the working directory to fill
in `mlflow.source.git.commit`, and that takes precedence over the tag set here.
Launched from a different repo, a run records that repo's commit instead.

## The modules

| | |
|---|---|
| `scripts/run_inference_tracked.py` | The CLI. Loads clips, runs the model, computes metrics, writes the run directory, records it |
| `scripts/run.sh` | The settings block you edit instead of remembering flags. Pre-flight checks before any GPU time is spent |
| `scripts/sweep.sh` | Several configurations in one command. N runs sharing a `sweep` tag, not one run with N results |
| `scripts/smoke_inference.py` | Does the model load and produce a trajectory on this board? No recording |
| `scripts/ml_platform_track.py` | The recording helper. A byte-identical copy of `examples/ml_platform_track.py` in the ML_Platform repo — that one is canonical |
| `src/alpamayo1_5/trace/token_trace.py` | Per-token logprob and entropy, CUDA-event timing split |
| `src/alpamayo1_5/trace/metrics.py` | ADE/FDE, kinematic feasibility, scene classification, token quality |
| `src/alpamayo1_5/trace/thermal.py` | Jetson temperature and rail power, read from sysfs |
| `src/alpamayo1_5/trace/writer.py` | The run directory: schema, versioning, what may be uploaded |

## Tests

```bash
pytest                    # 52 tests, no GPU required
```

They cover the displacement arithmetic, the run-directory schema and the
recording rules — the parts that fail silently rather than loudly. CI runs the
same set. **Green in CI does not mean green on Thor:** nothing there exercises
the model, the CUDA-event timing, or thermal reading on real hardware.
