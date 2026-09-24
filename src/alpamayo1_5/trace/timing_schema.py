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

"""What ``timing.parquet`` holds, and the one place that says so.

Measurement lives in its own table rather than in ``predictions.parquet``, for
two reasons that are about comparison, not tidiness:

* **Latency and accuracy are compared under different rules.** Accuracy parity
  may cross machines -- a Thor run and a Pro 6000 run of the same checkpoint on
  the same clips are supposed to agree, and ``compare.py`` deliberately leaves
  ``machine`` out of its pairing. Latency may not: a number is only comparable
  with one taken on the same board, power mode, driver, torch build and depth
  of instrumentation. One table cannot carry both rules.
* **The measurement schema moves faster.** Every new instrument adds columns.
  Had they gone into the predictions table, each one would have cut every new
  run off from the schema-3 corpus it is meant to be compared with.

So ``predictions.parquet`` stays frozen at schema 3 -- its legacy ``t_*``
columns keep their names and definitions -- and this table carries everything
else, versioned on its own.

The registry below is the single source of truth. The writer builds the
parquet schema from it, the runner derives the MLflow names from it, the
comparison code reads directions from it and the documentation is generated
from it. A column that is not declared here does not exist.

Nothing in this module imports torch: CI installs numpy, pandas and pyarrow
only, and everything here is tested there.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

#: Bump in every change that adds, removes or redefines a column. Tables with
#: different versions refuse to concatenate: a column that exists in one run
#: and not another would come back as a silent NaN in a comparison.
TIMING_SCHEMA_VERSION = 8

#: Bump when the hooks that produce the basic-level numbers change. Instrument
#: cost moves with them, so two runs measured by different tracers are not
#: comparable on latency even when every other condition matches.
TRACER_VERSION = 5

CHANGELOG = {
    1: "Initial table: legacy spans, host wall clock, per-call arrays, allocator and "
    "CUDA-graph counters, run conditions.",
    2: "The tracer's own cost: t_trace_consume_ms and its host twin, t_postgen_model_ms, "
    "trace_n_marks, trace_hook_host_ms. Tracer 2 marks its logits pass.",
    3: "generate split: t_gen_preamble_ms, t_lm_head_ms, t_vlm_glue_ms, t_gen_loop_ms, "
    "t_ttft_ms, n_vlm_forwards, lm_head_step_ms, decode_gap_ms, trace_span_violations. "
    "Tracer 3 hooks the VLM forward and lm_head.",
    4: "Measurement protocol: overhead_probe, timing_repeats and repeat_clips conditions, "
    "pass_output_match. Warmup, probe and repeat passes are rows of their own kind.",
    5: "Host side: the clip's stages outside the call (data, tokenization, copies, metrics, "
    "figures), the spans before generate and after the head, first trajectory, thread CPU "
    "per segment, context switches, RSS. Tracer 4 marks the call and stamps thread CPU.",
    6: "Memory and shapes: allocator peak per segment, clip peak, reserved peak, OOMs, host "
    "memory available; the KV cache the head attends to, the image and patch counts, and the "
    "memory_snapshot condition. Tracer 5 reads the allocator at segment boundaries.",
    7: "Board conditions: l4t_release and nvidia_driver. sample_hz is now the fast tier's "
    "rate (10 by default).",
    8: "Board attribution per pass: energy per rail and per segment, power mean and peaks, "
    "coverage, GPU clock per segment, EMC minimum, over-current events, throttle state, "
    "junction peak, anchor lag.",
}

#: What a row can be. Only ``main`` rows feed predictions and latency
#: aggregates; the others are extra passes over the same clip whose purpose is
#: to measure something else (the tracer's own cost, noise, kernels, memory).
ROW_KINDS = ("main", "warmup", "probe", "repeat", "profile", "memsnap")

_DTYPES = ("f64", "f32", "i16", "i32", "i64", "b", "s", "lf32")
_DIRECTIONS = {"L": "lower", "H": "higher", "N": "neutral"}


@dataclass(frozen=True)
class Col:
    """One column of ``timing.parquet``.

    ``direction`` is what a comparison should call better: L (lower), H
    (higher) or N (neutral -- a count or a condition, where better has no
    meaning and ranking it would be a mistake).
    """

    name: str
    dtype: str
    unit: str
    direction: str
    group: str
    description: str
    since: int = 1


def _cols(group: str, specs: Iterable[tuple[str, str, str, str, str]],
          since: int = 1) -> tuple[Col, ...]:
    return tuple(Col(name, dtype, unit, direction, group, text, since)
                 for name, dtype, unit, direction, text in specs)


IDENTITY = _cols("identity", (
    ("timing_schema_version", "i16", "", "N", "Version of this table's column set."),
    ("tracer_version", "i16", "", "N",
     "Version of the basic-level instrumentation. Latency is compared only within one."),
    ("run_id", "s", "", "N", "MLflow run id, or a local id when the run was not tracked."),
    ("variant", "s", "", "N", "The run's comparison label."),
    ("git_commit", "s", "", "N", "Commit of the code that produced the row."),
    ("clip_id", "s", "", "N", "Clip the pass ran on."),
    ("t0_us", "i64", "us", "N", "Sample timestamp within the clip."),
    ("clip_index", "i32", "", "N", "Position of the clip in the run's clip list."),
    ("row_kind", "s", "", "N",
     "main | warmup | probe | repeat | profile | memsnap. Only main rows feed predictions "
     "and latency aggregates."),
    ("pass_index", "i16", "", "N",
     "0 for a clip's main pass; extra passes over the same clip count up from 1."),
    ("t_start_unix_s", "f64", "s", "N", "Wall-clock time the pass started, Unix seconds."),
    ("t_start_host_s", "f64", "s", "N",
     "perf_counter when the pass started. The board sampler stamps the same clock, which is "
     "what lets its readings be attributed to a pass."),
))

CONDITIONS = _cols("condition", (
    ("machine", "s", "", "N", "Where the run happened. Latency never crosses machines."),
    ("power_mode", "s", "", "N", "nvpmodel mode name."),
    ("torch_version", "s", "", "N", "torch build; its kernels are part of the measurement."),
    ("transformers_version", "s", "", "N", "transformers version; it owns the decode loop."),
    ("cuda_runtime", "s", "", "N", "CUDA runtime torch was built against."),
    ("attn_impl", "s", "", "N", "Attention implementation requested at load."),
    ("dtype", "s", "", "N", "Compute dtype."),
    ("model", "s", "", "N", "Model coordinate as given on the command line."),
    ("data_spec", "s", "", "N", "Camera and sensor configuration."),
    ("num_traj_samples", "i16", "", "N", "K, trajectories per clip."),
    ("inference_step", "i16", "", "N", "Euler steps requested; null means the default."),
    ("max_new_tokens", "i16", "", "N", "Generation cap actually passed to generate."),
    ("cuda_graph", "b", "", "N", "Whether the expert was replayed through CUDA graphs."),
    ("cuda_graph_max_graphs", "i16", "", "N",
     "Captured signatures kept; null when graphs are off."),
    ("trace_level", "s", "", "N",
     "How deep the instrumentation went. Deeper levels cost more; never compare latency "
     "across levels."),
    ("sample_hz", "f32", "Hz", "N", "Board sampler rate while the pass ran."),
    ("warmup", "i16", "", "N", "Untimed inferences run before the first main pass."),
))

#: The spans ``predictions.parquet`` has always carried, under the same names
#: and definitions. ``t_other_ms`` is the remainder of the named spans, so the
#: seven are checkable: vision + prefill + decode + postgen + expert + other
#: equals total.
LEGACY = _cols("segment", (
    ("t_vision_ms", "f64", "ms", "L", "Vision tower span, device clock."),
    ("t_prefill_ms", "f64", "ms", "L", "First language-model call, device clock."),
    ("t_decode_ms", "f64", "ms", "L", "Every later language-model call, device clock."),
    ("t_postgen_ms", "f64", "ms", "L",
     "From generate returning to the trajectory head starting. Host work lives here."),
    ("t_expert_ms", "f64", "ms", "L", "The diffusion sampler, all Euler steps."),
    ("t_other_ms", "f64", "ms", "L",
     "The generate span minus vision, prefill and decode: lm_head, logits processing, "
     "sampling, stopping criteria."),
    ("t_total_ms", "f64", "ms", "L", "Generate span + postgen + expert."),
    ("n_vision_calls", "i16", "", "N", "Vision tower calls in the pass."),
    ("n_decode_steps", "i16", "", "N",
     "Language-model calls after prefill -- decode's denominator, since the batch decodes "
     "until its last row finishes."),
    ("n_expert_calls", "i16", "", "N", "Euler steps that actually executed."),
    ("timing_measured", "b", "", "N", "False: every time in the row is absent, not zero."),
))

CLOCKS = _cols("clock", (
    ("t_wall_ms", "f64", "ms", "L",
     "Host wall clock from tracer install to after the final synchronize: what the caller "
     "waited. The device spans cannot see host gaps between them; this can."),
    ("t_compute_span_ms", "f64", "ms", "N",
     "Device time inside the vision, prefill, decode and expert spans. Idle inside a span "
     "is included, so this is not GPU busy time -- that comes from the kernel profile."),
))

PER_CALL = _cols("per_call", (
    ("vision_call_ms", "lf32", "ms", "L", "Each vision tower call, device clock."),
    ("decode_step_ms", "lf32", "ms", "L", "Each decode step, device clock."),
    ("decode_step_host_ms", "lf32", "ms", "N",
     "Each decode step, host clock: how long Python took to enqueue it."),
    ("expert_step_ms", "lf32", "ms", "L",
     "Each Euler step of the trajectory head, device clock. The first carries warmup and, "
     "with graphs on, the capture."),
    ("expert_step_host_ms", "lf32", "ms", "N", "Each Euler step, host clock."),
))

ALLOC = _cols("alloc", (
    ("n_cuda_allocs", "i64", "", "L",
     "Allocator allocations during the pass. A KV cache rebuilt by concatenation on every "
     "layer and step lands here in the hundreds."),
    ("n_alloc_retries", "i64", "", "L",
     "Allocator retries: each is a cache flush and a stall in whichever span was unlucky."),
))

GRAPH = _cols("graph", (
    ("graph_captures", "i32", "", "N", "Expert CUDA graphs captured; null when graphs are off."),
    ("graph_replays", "i32", "", "N",
     "Expert steps replayed. A capture step also counts as a replay."),
    ("graph_fallbacks", "i32", "", "L",
     "Expert steps that ran eager with a graph runner installed. Never average latency over "
     "rows that fell back together with rows that did not."),
    ("graph_n_graphs", "i32", "", "N", "Graphs held by the runner after the pass."),
    ("graph_capture_ms", "f64", "ms", "L", "Host time spent capturing during the pass."),
    ("graph_mode", "s", "", "N",
     "none | replay | capture | fallback | mixed: which path the pass's expert steps took."),
))

TRACE = _cols("trace", (
    ("t_trace_consume_ms", "f64", "ms", "L",
     "Device time of the tracer's own logits pass. It runs inside the postgen window, so "
     "t_postgen_ms includes it -- kept there so the legacy column keeps its meaning."),
    ("t_trace_consume_host_ms", "f64", "ms", "L", "The same pass, host clock."),
    ("t_postgen_model_ms", "f64", "ms", "L",
     "postgen minus the tracer's logits pass: what the model's own code between generate "
     "and the trajectory head cost."),
    ("trace_n_marks", "i32", "", "N", "Marks recorded in the pass."),
    ("trace_hook_host_ms", "f64", "ms", "L",
     "Host time spent inside the tracer's marks. The instrument's own cost, per pass."),
), since=2)

#: What t_other_ms was made of. By construction
#: t_other_ms = t_gen_preamble_ms + t_lm_head_ms + t_vlm_glue_ms + t_gen_loop_ms,
#: so the remainder that used to be one opaque number is now four checkable ones.
GENERATE = _cols("generate", (
    ("t_gen_preamble_ms", "f64", "ms", "L",
     "generate's start to the first forward: K-fold input expansion, cache and processor "
     "setup."),
    ("t_lm_head_ms", "f64", "ms", "L",
     "The vocabulary projection, every step. It runs after the language model returns, so "
     "the decode span never included it -- 1.27 GB of weights read per step."),
    ("t_vlm_glue_ms", "f64", "ms", "L",
     "Inside each VLM forward but outside vision, the language model and lm_head: "
     "embeddings, image-token scatter, rope index, deepstack."),
    ("t_gen_loop_ms", "f64", "ms", "L",
     "generate outside every forward: logits processors, sampling, stopping criteria and "
     "the host syncs between steps."),
    ("t_ttft_ms", "f64", "ms", "L",
     "Host clock from the model call to the second forward -- the first reasoning token is "
     "on the host by then, since the loop syncs on it."),
    ("n_vlm_forwards", "i16", "", "N", "VLM forward calls: prefill plus decode steps."),
    ("lm_head_step_ms", "lf32", "ms", "L", "Each lm_head call, device clock."),
    ("decode_gap_ms", "lf32", "ms", "L",
     "Device time between consecutive forwards: the host loop each decode step waits on."),
    ("trace_span_violations", "i16", "", "L",
     "Derived spans that came out negative and were floored. Nonzero means a nesting "
     "assumption broke and the split is suspect."),
), since=3)

PROTOCOL = _cols("condition", (
    ("overhead_probe", "i16", "", "N",
     "Clips that got an extra off/on pass pair to measure the tracer's own cost."),
    ("timing_repeats", "i16", "", "N", "Extra passes per repeated clip, for the noise band."),
    ("repeat_clips", "i16", "", "N", "How many clips, from the start, got the repeats."),
    ("pass_output_match", "b", "", "N",
     "Extra passes only: whether the trajectories came out bit-identical to the main pass. "
     "False means the passes were not replicates, and their timing differences are not noise."),
), since=4)

#: The clip outside the model call, host clock. Main passes only: an extra pass
#: reuses the prepared inputs and has no stages of its own.
HOST = _cols("host", (
    ("data_load_ms", "f64", "ms", "L", "Reading the clip from the cache, cameras included."),
    ("data_ego_ms", "f64", "ms", "L", "Of which: fetching the egomotion feature."),
    ("data_cam_fetch_ms", "f64", "ms", "L", "Of which: fetching the four camera features."),
    ("data_cam_decode_ms", "f64", "ms", "L",
     "Of which: decoding the frames on the CPU (PyAV, one thread)."),
    ("msg_build_ms", "f64", "ms", "L", "Building the chat message around the frames."),
    ("preprocess_ms", "f64", "ms", "L",
     "The processor: image resize and patching plus text tokenization. With the copies and "
     "trajectory tokenization this is the preprocessing component of NVIDIA's own analysis."),
    ("preprocess_image_ms", "f64", "ms", "L", "Of which: the image processor."),
    ("preprocess_text_ms", "f64", "ms", "L", "Of which: everything else -- the tokenizer."),
    ("h2d_ms", "f64", "ms", "L", "Copying the inputs to the GPU (pageable, so it blocks)."),
    ("model_call_ms", "f64", "ms", "L", "The main pass, host clock around the call."),
    ("extra_passes_ms", "f64", "ms", "N", "Probe and repeat passes over this clip."),
    ("result_cpu_ms", "f64", "ms", "L", "Bringing trajectories and text back to the host."),
    ("metrics_ms", "f64", "ms", "L", "Displacement, kinematics, diversity, scene labels."),
    ("render_ms", "f64", "ms", "L", "The sample figure, for clips that get one."),
    ("flush_ms", "f64", "ms", "L", "Rewriting the run directory, on clips that flush."),
    ("clip_wall_ms", "f64", "ms", "L", "The clip from its first stage to its last."),
    ("clip_other_ms", "f64", "ms", "L", "The clip's wall clock that no stage accounts for."),
), since=5)

#: Inside the call but outside generate and the head, and what the CPU did.
PASS_HOST = _cols("pass_host", (
    ("t_pre_generate_ms", "f64", "ms", "L",
     "Device clock from the call to generate: input deep copy and trajectory tokenization."),
    ("t_pre_generate_host_ms", "f64", "ms", "L", "The same span, host clock."),
    ("t_tail_ms", "f64", "ms", "L",
     "Device clock from the head's end to the call's end: action-to-trajectory and text "
     "extraction."),
    ("t_action_to_traj_ms", "f64", "ms", "L", "Converting actions to the trajectory."),
    ("t_first_traj_ms", "f64", "ms", "L",
     "Device clock from the call to the trajectory existing -- what a planner waits for."),
    ("t_fuse_traj_host_ms", "f64", "ms", "L", "Trajectory tokenization, host clock."),
    ("t_expand_inputs_host_ms", "f64", "ms", "L",
     "generate's K-fold copy of the inputs, pixels included, host clock."),
    ("t_rope_index_host_ms", "f64", "ms", "L", "Rope index construction, a host loop per row."),
    ("cpu_pass_ms", "f64", "ms", "N", "Main-thread CPU time over the call."),
    ("cpu_pre_generate_ms", "f64", "ms", "N", "Main-thread CPU time before generate."),
    ("cpu_vision_ms", "f64", "ms", "N", "Main-thread CPU time in the vision span."),
    ("cpu_prefill_ms", "f64", "ms", "N", "Main-thread CPU time in the prefill span."),
    ("cpu_decode_ms", "f64", "ms", "N", "Main-thread CPU time in the decode spans."),
    ("cpu_postgen_ms", "f64", "ms", "N", "Main-thread CPU time between generate and the head."),
    ("cpu_expert_ms", "f64", "ms", "N", "Main-thread CPU time in the head."),
    ("cpu_tail_ms", "f64", "ms", "N", "Main-thread CPU time after the head."),
    ("proc_cpu_ms", "f64", "ms", "N", "Process CPU time over the call, every thread."),
    ("ctx_vol", "i32", "", "N", "Voluntary context switches of the main thread in the call."),
    ("ctx_invol", "i32", "", "L",
     "Involuntary ones: the thread was runnable and the scheduler took the core away."),
    ("rss_bytes", "i64", "B", "L", "Process resident set at the end of the call."),
), since=5)

#: Memory is unified on the Thor: what the GPU allocates, the host no longer
#: has. The allocator's peak is read and reset at every segment boundary, so
#: each segment gets its own peak rather than the run sharing one.
MEMORY = _cols("memory", (
    ("mem_start_bytes", "i64", "B", "N", "Allocated when the call began."),
    ("mem_end_bytes", "i64", "B", "N", "Allocated when the call ended."),
    ("mem_peak_pre_bytes", "i64", "B", "L", "Peak before vision: input copies and expansion."),
    ("mem_peak_vision_bytes", "i64", "B", "L", "Peak during the vision tower."),
    ("mem_peak_prefill_bytes", "i64", "B", "L", "Peak from vision's end through prefill."),
    ("mem_peak_decode_bytes", "i64", "B", "L", "Peak over the decode loop."),
    ("mem_peak_postgen_bytes", "i64", "B", "L",
     "Peak between generate and the head -- includes the tracer's logits pass."),
    ("mem_peak_expert_bytes", "i64", "B", "L", "Peak during the head."),
    ("mem_peak_tail_bytes", "i64", "B", "L", "Peak after the head."),
    ("mem_peak_clip_bytes", "i64", "B", "L", "Peak over the whole call."),
    ("mem_reserved_peak_bytes", "i64", "B", "L",
     "Peak held by the caching allocator -- what the process actually took from the board."),
    ("n_ooms", "i32", "", "L", "Out-of-memory events the allocator recovered from."),
    ("host_mem_avail_min_bytes", "i64", "B", "H",
     "MemAvailable at the lower of the call's two ends. The GPU's memory is the host's."),
), since=6)

SHAPES = _cols("shape", (
    ("kv_bytes", "i64", "B", "N", "The KV cache the head attended to, all layers, K and V."),
    ("kv_final_tokens", "i32", "", "N",
     "Its length: prompt plus decoded tokens. Clips differ, which is also why a graph "
     "captured for one clip rarely fits the next."),
    ("kv_rows", "i16", "", "N", "Its batch rows: K."),
    ("kv_layers", "i16", "", "N", "Its layers."),
    ("expert_tokens", "i16", "", "N", "Action tokens the head appends each Euler step."),
    ("n_images", "i16", "", "N", "Images the vision tower encoded: K copies of every frame."),
    ("n_vision_patches", "i32", "", "N", "Patches it encoded."),
    ("n_vision_tokens", "i32", "", "N", "Tokens those became after the spatial merge."),
    ("memory_snapshot", "i16", "", "N",
     "Clips that got an extra pass under the allocator's history recorder."),
), since=6)

BOARD = _cols("condition", (
    ("l4t_release", "s", "", "N",
     "Jetson Linux release. A JetPack upgrade changes kernels, clocks and power policy."),
    ("nvidia_driver", "s", "", "N", "GPU driver version."),
), since=7)

#: The board over one pass, integrated from the sampler's series over the pass's
#: host-clock window. Segment energy is kept only where a segment lasts long
#: enough (1.9 s) to rest on more than a couple of readings.
ENERGY = _cols("energy", (
    ("e_vin_j", "f64", "J", "L", "Whole-module input energy over the pass (INA238 VIN)."),
    ("e_gpu_j", "f64", "J", "L", "GPU rail energy (VDD_GPU)."),
    ("e_cpu_soc_mss_j", "f64", "J", "L", "CPU, SoC and memory rail energy."),
    ("e_sys5v0_j", "f64", "J", "L", "5 V system rail energy."),
    ("e_vin_vision_j", "f64", "J", "L", "Module input energy in the vision span."),
    ("e_vin_prefill_j", "f64", "J", "L", "Module input energy in prefill."),
    ("e_vin_decode_j", "f64", "J", "L", "Module input energy over the decode loop."),
    ("e_vin_expert_j", "f64", "J", "L", "Module input energy in the head."),
    ("e_gpu_vision_j", "f64", "J", "L", "GPU rail energy in the vision span."),
    ("e_gpu_prefill_j", "f64", "J", "L", "GPU rail energy in prefill."),
    ("e_gpu_decode_j", "f64", "J", "L", "GPU rail energy over the decode loop."),
    ("e_gpu_expert_j", "f64", "J", "L", "GPU rail energy in the head."),
    ("p_vin_mean_w", "f64", "W", "L", "Mean module input power over the pass."),
    ("p_vin_peak_w", "f64", "W", "L", "Highest module input reading in the pass."),
    ("p_gpu_peak_w", "f64", "W", "L",
     "Highest GPU rail reading: near 100 W the rail is at its ~5 A current limit."),
    ("e_coverage", "f32", "", "H",
     "Share of the window between the first and last reading. Below 1 the rest is held "
     "flat -- an assumption the energy rests on."),
    ("n_power_samples", "i32", "", "N", "Module input readings inside the window."),
), since=8)

BOARD_STATE = _cols("board", (
    ("gpu_mhz_mean", "f32", "MHz", "H", "Time-weighted GPU clock over the pass."),
    ("gpu_mhz_min", "f32", "MHz", "H", "Lowest GPU clock read in the pass."),
    ("gpu_mhz_vision", "f32", "MHz", "H", "GPU clock in the vision span."),
    ("gpu_mhz_prefill", "f32", "MHz", "H", "GPU clock in prefill."),
    ("gpu_mhz_decode", "f32", "MHz", "H", "GPU clock over the decode loop."),
    ("gpu_mhz_expert", "f32", "MHz", "H", "GPU clock in the head."),
    ("emc_mhz_min", "f32", "MHz", "H", "Lowest memory clock read in the pass."),
    ("oc1_events", "i32", "", "L", "Over-current events on channel 1 during the pass."),
    ("oc2_events", "i32", "", "L", "Over-current events on channel 2 during the pass."),
    ("oc3_events", "i32", "", "L",
     "Over-current events on channel 3 -- the GPU's current limit. Each one pulls clocks "
     "back; nonzero means the pass ran throttled whatever the temperature said."),
    ("throttle_state_max", "i16", "", "L",
     "Highest state of any capping cooling device (not the fan) during the pass."),
    ("temp_tj_max_c", "f32", "C", "L", "Hottest junction reading in the pass."),
    ("anchor_lag_ms", "f32", "ms", "N",
     "How late the pass's first event ran on the device after being enqueued. Large means "
     "the GPU was still busy with earlier work and the window's start is less certain."),
), since=8)

COLUMNS: tuple[Col, ...] = (IDENTITY + CONDITIONS + LEGACY + CLOCKS + PER_CALL + ALLOC + GRAPH
                            + TRACE + GENERATE + PROTOCOL + HOST + PASS_HOST + MEMORY + SHAPES
                            + BOARD + ENERGY + BOARD_STATE)

#: The keys ``predictions.parquet`` reads off a pass, unchanged since schema 3.
LEGACY_KEYS = tuple(c.name for c in LEGACY)

_BY_NAME = {c.name: c for c in COLUMNS}


def column(name: str) -> Col:
    return _BY_NAME[name]


def names() -> tuple[str, ...]:
    return tuple(c.name for c in COLUMNS)


def directions() -> dict[str, str]:
    """Column name to ``lower`` / ``higher`` / ``neutral``, for comparison tables."""
    return {c.name: _DIRECTIONS[c.direction] for c in COLUMNS}


# -- rows -------------------------------------------------------------------
def is_number(value: Any) -> bool:
    """A finite real number, numpy scalars included, bools excluded.

    ``isinstance(v, (int, float))`` is the obvious test and it is wrong twice:
    it rejects ``np.float32``, so a metric computed with numpy is silently
    skipped, and it accepts ``True``, so a flag gets averaged as a measurement.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def _coerce(col: Col, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    kind = col.dtype
    if kind in ("f64", "f32"):
        return float(value)
    if kind in ("i16", "i32", "i64"):
        return int(value)
    if kind == "b":
        return bool(value)
    if kind == "s":
        return str(value)
    if kind == "lf32":
        return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]
    raise ValueError(f"unknown dtype {kind!r} for {col.name}")


def unknown_keys(row: Mapping[str, Any]) -> list[str]:
    """Keys a row carries that the registry does not declare."""
    return sorted(k for k in row if k not in _BY_NAME)


def coerce_row(row: Mapping[str, Any], strict: bool = False) -> dict[str, Any]:
    """Exactly the declared columns, in declared order, with declared types.

    Missing columns become nulls: a row from a pass that did not run an
    instrument says so by absence, never by a plausible zero. Undeclared keys
    raise under ``strict`` (the tests) and are dropped otherwise (a run), since
    a typo in instrumentation must not cost a run.
    """
    if strict:
        extra = unknown_keys(row)
        if extra:
            raise ValueError(f"undeclared timing columns: {extra}")
    return {c.name: _coerce(c, row.get(c.name)) for c in COLUMNS}


def pa_schema() -> Any:
    """The parquet schema. Explicit, so a column that is null in every row of a
    run keeps its type and the run still concatenates with one where it is not."""
    import pyarrow as pa

    types = {
        "f64": pa.float64(), "f32": pa.float32(),
        "i16": pa.int16(), "i32": pa.int32(), "i64": pa.int64(),
        "b": pa.bool_(), "s": pa.string(), "lf32": pa.list_(pa.float32()),
    }
    return pa.schema([pa.field(c.name, types[c.dtype]) for c in COLUMNS])


# -- aggregation ------------------------------------------------------------
#: Every MLflow key :func:`aggregate` can emit. Declared rather than derived
#: from whatever a run happened to compute, for the same reason the runner
#: declares its per-clip metrics: otherwise the experiment's metric namespace is
#: decided by the last code change, and keys appear and vanish between runs
#: without anyone choosing that.
AGGREGATE_KEYS = (
    # The seven legacy spans, under the names every earlier run used.
    "t_vision_ms", "t_prefill_ms", "t_decode_ms", "t_postgen_ms",
    "t_expert_ms", "t_other_ms", "t_total_ms",
    # Tails. latency_p95 is the recording rules' name for t_total_ms_p95.
    "t_total_ms_p50", "t_total_ms_p95", "t_total_ms_p99", "latency_p95",
    "t_wall_ms", "t_wall_ms_p50", "t_wall_ms_p95", "t_wall_ms_p99",
    "compute_span_ratio",
    "expert_step_first_ms", "expert_step_steady_ms", "expert_step_p95_ms",
    "expert_step_spread_ms", "expert_launch_ratio",
    "decode_step_p95_ms", "decode_launch_ratio",
    "alloc.n_cuda_allocs", "alloc.n_alloc_retries_sum",
    "graph.captures_sum", "graph.replays_sum", "graph.fallbacks_sum",
    "graph.fallback_clip_frac", "graph.capture_ms_sum", "graph.t_expert_ms_replay_only",
    "timing.n_main_rows",
    "t_postgen_model_ms", "trace.consume_ms", "trace.hook_host_ms", "trace.n_marks",
    "t_gen_preamble_ms", "t_lm_head_ms", "t_vlm_glue_ms", "t_gen_loop_ms",
    "t_ttft_ms", "t_ttft_ms_p95", "decode_gap_ms", "decode_gap_ms_p95", "lm_head_step_ms",
    "trace.span_violations_sum",
    "trace.overhead_pct", "trace.overhead_lo", "trace.overhead_hi", "trace.overhead_n",
    "cold_start_ms", "cold_start_excess_ms", "latency_cv", "latency_cv_n",
    "timing.n_extra_rows", "pass.output_mismatch_sum",
    "host.data_load_ms", "host.data_ego_ms", "host.data_cam_fetch_ms", "host.data_cam_decode_ms",
    "host.msg_build_ms", "host.preprocess_ms", "host.preprocess_image_ms",
    "host.preprocess_text_ms", "host.h2d_ms", "host.model_call_ms", "host.result_cpu_ms",
    "host.metrics_ms", "host.render_ms", "host.flush_ms", "host.clip_wall_ms",
    "host.clip_wall_ms_p95", "host.clip_other_ms", "host.paper_preprocess_ms",
    "t_pre_generate_ms", "t_tail_ms", "t_action_to_traj_ms", "t_first_traj_ms",
    "t_first_traj_ms_p95", "host.cpu_pass_ms", "host.cpu_decode_ms", "host.cpu_expert_ms",
    "host.ctx_invol", "host.rss_max_gb",
    "mem.peak_clip_gb", "mem.peak_clip_gb_max", "mem.peak_pre_gb", "mem.peak_vision_gb",
    "mem.peak_prefill_gb", "mem.peak_decode_gb", "mem.peak_postgen_gb", "mem.peak_expert_gb",
    "mem.peak_tail_gb", "mem.reserved_peak_gb_max", "mem.n_ooms_sum", "mem.host_avail_min_gb",
    "kv.bytes_mb", "kv.final_tokens", "kv.cat_decode_gb", "kv.cat_expert_gb", "kv.graph_copy_gb",
    "vision.n_images", "vision.n_patches", "vision.n_tokens",
    "energy.clip_j", "energy.per_traj_j", "energy.gpu_clip_j", "energy.cpu_soc_clip_j",
    "energy.decode_j_per_step", "energy.vin_decode_j", "energy.vin_expert_j",
    "energy.gpu_vision_j", "energy.gpu_prefill_j", "energy.gpu_decode_j", "energy.gpu_expert_j",
    "energy.coverage_min", "power.vin_clip_mean_w", "power.vin_clip_peak_w",
    "power.gpu_clip_peak_w", "clock.gpu_mhz_clip_mean", "clock.gpu_mhz_clip_min",
    "clock.gpu_mhz_decode", "clock.gpu_mhz_expert", "clock.emc_mhz_min",
    "throttle.oc3_events_sum", "throttle.oc3_events_per_clip", "throttle.clips_with_oc_frac",
    "throttle.state_max", "throttle.temp_tj_max_c", "throttle.corr_wall_gpu_mhz",
    "throttle.corr_wall_oc3", "trace.anchor_lag_ms",
)


def kv_traffic(row: Mapping[str, Any]) -> dict[str, float] | None:
    """Bytes the KV cache's concatenation moves in one pass, estimated from shapes.

    Derived here rather than stored: the stored facts are the cache's size and
    length and the step counts, and the model of what a concatenation costs
    may change. transformers' DynamicLayer.update rebuilds every layer's cache
    with torch.cat on every call -- reading the old tensor and writing a new
    one one token longer. So a decode step at cache length L moves about
    2 * L token-rows per layer, and an Euler step of the head, which appends
    its action tokens to the full prompt cache and crops them off again, moves
    about 2 * (L_final + tokens). A CUDA graph captures the same concatenation,
    and its first replay per clip also copies the whole prompt cache into the
    graph's static buffers.
    """
    need = ("kv_bytes", "kv_rows", "kv_final_tokens")
    if not all(is_number(row.get(k)) for k in need) or not row["kv_rows"] or \
            not row["kv_final_tokens"]:
        return None
    rows_, final = float(row["kv_rows"]), float(row["kv_final_tokens"])
    per_token_row = float(row["kv_bytes"]) / (rows_ * final)
    n = float(row.get("n_decode_steps") or 0)
    prompt = max(final - n, 0.0)
    steps = float(row.get("n_expert_calls") or 0)
    tokens = float(row.get("expert_tokens") or 0)
    return {
        "cat_decode": 2.0 * per_token_row * rows_ * (n * prompt + n * (n + 1) / 2.0),
        "cat_expert": 2.0 * per_token_row * rows_ * steps * (final + tokens),
        "graph_copy": float(row["kv_bytes"]) if (row.get("graph_replays") or 0) > 0 else 0.0,
    }


def main_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows a latency number may rest on: main passes that were measured."""
    return [r for r in rows
            if r.get("row_kind", "main") == "main" and r.get("timing_measured") is True]


def _values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if is_number(r.get(key))]


def _arrays(rows: Iterable[Mapping[str, Any]], key: str) -> list[list[float]]:
    out = []
    for r in rows:
        value = r.get(key)
        if value is None:
            continue
        array = [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]
        if array:
            out.append(array)
    return out


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _ratio_of_sums(rows: Iterable[Mapping[str, Any]], num: str, den: str) -> list[float]:
    out = []
    for r in rows:
        a, b = r.get(num), r.get(den)
        if a is None or b is None:
            continue
        a_sum = float(np.sum(np.asarray(a, dtype=np.float64)))
        b_sum = float(np.sum(np.asarray(b, dtype=np.float64)))
        if b_sum > 0.0:
            out.append(a_sum / b_sum)
    return out


def overhead(rows: Iterable[Mapping[str, Any]], n_boot: int = 2000,
             seed: int = 0) -> dict[str, float] | None:
    """The tracer's own cost, from probe passes: off and on over the same clip.

    Per clip, ``100 * (wall_on - wall_off) / wall_off``; the run's number is the
    median over clips, with a bootstrap interval over clips. Paired per clip
    because clips differ by seconds and the cost is milliseconds -- an
    unpaired difference of means would be all clip variance. The probe order
    alternates by clip, so a slow drift (the board warming) does not load onto
    one side.
    """
    by_clip: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r.get("row_kind") != "probe" or not is_number(r.get("t_wall_ms")):
            continue
        side = "off" if r.get("trace_level") == "off" else "on"
        by_clip.setdefault(str(r.get("clip_id")), {"off": [], "on": []})[side].append(
            float(r["t_wall_ms"]))
    pcts = [100.0 * (np.mean(v["on"]) - np.mean(v["off"])) / np.mean(v["off"])
            for v in by_clip.values() if v["on"] and v["off"] and np.mean(v["off"]) > 0.0]
    if not pcts:
        return None
    values = np.asarray(pcts, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(values, size=(n_boot, values.size), replace=True), axis=1)
    return {
        "trace.overhead_pct": float(np.median(values)),
        "trace.overhead_lo": float(np.percentile(boot, 2.5)),
        "trace.overhead_hi": float(np.percentile(boot, 97.5)),
        "trace.overhead_n": float(values.size),
    }


def _protocol(rows: list[Mapping[str, Any]]) -> dict[str, float | None]:
    """Numbers that need the extra passes, not only the main ones."""
    out: dict[str, float | None] = {}
    out.update(overhead(rows) or {})
    extra = [r for r in rows if r.get("row_kind", "main") != "main"]
    if extra:
        out["timing.n_extra_rows"] = float(len(extra))
        # Extra passes are re-seeded replicates of the main one. A mismatch
        # means they were not, and everything measured on them is suspect.
        out["pass.output_mismatch_sum"] = float(
            sum(1 for r in extra if r.get("pass_output_match") is False))

    # The first pass the process made, whatever its kind: cuDNN autotuning,
    # allocator growth and lazy initialisation all land on it. It is a real
    # deployment number -- the first plan after boot -- and a contaminant
    # anywhere else, which is why warmup passes exist.
    stamped = [r for r in rows if is_number(r.get("t_wall_ms"))]
    if stamped:
        first = min(stamped, key=lambda r: (float(r.get("t_start_host_s") or 0.0),
                                            r.get("clip_index") or 0, r.get("pass_index") or 0))
        out["cold_start_ms"] = float(first["t_wall_ms"])
        steady = _values(main_rows(rows), "t_wall_ms")
        if steady:
            out["cold_start_excess_ms"] = out["cold_start_ms"] - float(np.median(steady))

    # Run-to-run noise on one clip: the main pass and its repeats. Only passes
    # that reproduced the main pass's output count -- otherwise the spread
    # includes a different amount of work, not just noise.
    groups: dict[str, list[float]] = {}
    for r in rows:
        kind = r.get("row_kind", "main")
        if kind not in ("main", "repeat") or not is_number(r.get("t_wall_ms")):
            continue
        if kind == "repeat" and r.get("pass_output_match") is False:
            continue
        groups.setdefault(str(r.get("clip_id")), []).append(float(r["t_wall_ms"]))
    cvs = [float(np.std(v, ddof=1) / np.mean(v)) for v in groups.values()
           if len(v) >= 2 and np.mean(v) > 0.0]
    if cvs:
        out["latency_cv"] = float(np.mean(cvs))
        out["latency_cv_n"] = float(len(cvs))
    return out


def _corr(rows: list[Mapping[str, Any]], a: str, b: str) -> float | None:
    """Pearson r across clips, or None when it cannot mean anything (fewer
    than three clips, or a series that never moved)."""
    pairs = [(float(r[a]), float(r[b])) for r in rows if is_number(r.get(a))
             and is_number(r.get(b))]
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs).T
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _board(rows: list[Mapping[str, Any]]) -> dict[str, float | None]:
    """Energy, clocks and throttling over the main passes."""
    out: dict[str, float | None] = {}
    out["energy.clip_j"] = _mean(_values(rows, "e_vin_j"))
    per_traj = [float(r["e_vin_j"]) / float(r["num_traj_samples"]) for r in rows
                if is_number(r.get("e_vin_j")) and is_number(r.get("num_traj_samples"))
                and r["num_traj_samples"]]
    out["energy.per_traj_j"] = _mean(per_traj)
    out["energy.gpu_clip_j"] = _mean(_values(rows, "e_gpu_j"))
    out["energy.cpu_soc_clip_j"] = _mean(_values(rows, "e_cpu_soc_mss_j"))
    per_step = [float(r["e_vin_decode_j"]) / float(r["n_decode_steps"]) for r in rows
                if is_number(r.get("e_vin_decode_j")) and r.get("n_decode_steps")]
    out["energy.decode_j_per_step"] = _mean(per_step)
    out["energy.vin_decode_j"] = _mean(_values(rows, "e_vin_decode_j"))
    out["energy.vin_expert_j"] = _mean(_values(rows, "e_vin_expert_j"))
    for seg in ("vision", "prefill", "decode", "expert"):
        out[f"energy.gpu_{seg}_j"] = _mean(_values(rows, f"e_gpu_{seg}_j"))
    coverage = _values(rows, "e_coverage")
    out["energy.coverage_min"] = min(coverage) if coverage else None
    out["power.vin_clip_mean_w"] = _mean(_values(rows, "p_vin_mean_w"))
    peaks = _values(rows, "p_vin_peak_w")
    out["power.vin_clip_peak_w"] = max(peaks) if peaks else None
    peaks = _values(rows, "p_gpu_peak_w")
    out["power.gpu_clip_peak_w"] = max(peaks) if peaks else None
    out["clock.gpu_mhz_clip_mean"] = _mean(_values(rows, "gpu_mhz_mean"))
    lows = _values(rows, "gpu_mhz_min")
    out["clock.gpu_mhz_clip_min"] = min(lows) if lows else None
    out["clock.gpu_mhz_decode"] = _mean(_values(rows, "gpu_mhz_decode"))
    out["clock.gpu_mhz_expert"] = _mean(_values(rows, "gpu_mhz_expert"))
    emc = _values(rows, "emc_mhz_min")
    out["clock.emc_mhz_min"] = min(emc) if emc else None
    oc = _values(rows, "oc3_events")
    if oc:
        out["throttle.oc3_events_sum"] = float(sum(oc))
        out["throttle.oc3_events_per_clip"] = float(np.mean(oc))
        out["throttle.clips_with_oc_frac"] = float(np.mean([v > 0 for v in oc]))
    states = _values(rows, "throttle_state_max")
    out["throttle.state_max"] = max(states) if states else None
    temps = _values(rows, "temp_tj_max_c")
    out["throttle.temp_tj_max_c"] = max(temps) if temps else None
    # Does the latency spread across clips follow the board? A strong negative
    # correlation with the clock, or a positive one with over-current events,
    # says the clips were not slow for their own reasons.
    out["throttle.corr_wall_gpu_mhz"] = _corr(rows, "t_wall_ms", "gpu_mhz_mean")
    out["throttle.corr_wall_oc3"] = _corr(rows, "t_wall_ms", "oc3_events")
    out["trace.anchor_lag_ms"] = _mean(_values(rows, "anchor_lag_ms"))
    return out


def aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Run-level numbers for MLflow, from the main measured rows only.

    Restricting to measured rows is a fix, not a nicety: the averaging this
    replaces took the legacy spans off every row, and an unmeasured row carried
    0.0 -- so a run where timing failed on some clips reported them as having
    taken no time. The protocol numbers (tracer overhead, cold start, noise
    band) are the exception: they exist only in the extra passes.
    """
    rows = list(rows)
    protocol = _protocol(rows)
    rows = main_rows(rows)
    out: dict[str, float | None] = {"timing.n_main_rows": float(len(rows)), **protocol}
    if not rows:
        return {k: v for k, v in out.items() if v is not None}

    for col in LEGACY:
        if col.unit == "ms":
            out[col.name] = _mean(_values(rows, col.name))
    for key in ("t_total_ms", "t_wall_ms"):
        values = _values(rows, key)
        if values:
            for pct in (50, 95, 99):
                out[f"{key}_p{pct}"] = float(np.percentile(values, pct))
    out["latency_p95"] = out.get("t_total_ms_p95")
    out["t_wall_ms"] = _mean(_values(rows, "t_wall_ms"))

    spans = [float(r["t_compute_span_ms"]) / float(r["t_wall_ms"]) for r in rows
             if is_number(r.get("t_compute_span_ms")) and is_number(r.get("t_wall_ms"))
             and float(r["t_wall_ms"]) > 0.0]
    out["compute_span_ratio"] = _mean(spans)

    steps = _arrays(rows, "expert_step_ms")
    # The first Euler step carries warmup and, with graphs on, the capture.
    # Folded into one mean it describes a step that runs once per clip as
    # though it ran every time.
    out["expert_step_first_ms"] = _mean([s[0] for s in steps])
    out["expert_step_steady_ms"] = _mean([float(np.mean(s[1:])) for s in steps if len(s) > 1])
    out["expert_step_p95_ms"] = _mean([float(np.percentile(s, 95)) for s in steps])
    # Steps are the same shapes doing the same work, so a wide spread is the
    # instrumentation, the allocator or a capture -- not the model.
    out["expert_step_spread_ms"] = _mean([max(s) - min(s) for s in steps if len(s) > 1])
    # Host enqueue time over device time. Near 1: the device waited on the
    # host, the step is launch-bound, and removing launches has something to
    # win. Well below 1: the device is the bottleneck. Above 1 only when work
    # queued earlier hid a slow host.
    out["expert_launch_ratio"] = _mean(
        _ratio_of_sums(rows, "expert_step_host_ms", "expert_step_ms"))

    decode = _arrays(rows, "decode_step_ms")
    out["decode_step_p95_ms"] = _mean([float(np.percentile(s, 95)) for s in decode])
    out["decode_launch_ratio"] = _mean(
        _ratio_of_sums(rows, "decode_step_host_ms", "decode_step_ms"))

    out["alloc.n_cuda_allocs"] = _mean(_values(rows, "n_cuda_allocs"))
    retries = _values(rows, "n_alloc_retries")
    out["alloc.n_alloc_retries_sum"] = float(sum(retries)) if retries else None

    graphed = [r for r in rows if r.get("graph_captures") is not None]
    if graphed:
        # Totals, not means: one fallback anywhere makes every latency mean a
        # blend of two configurations, which is a yes-or-no question.
        for name in ("captures", "replays", "fallbacks"):
            out[f"graph.{name}_sum"] = float(sum(_values(graphed, f"graph_{name}")))
        out["graph.fallback_clip_frac"] = float(
            np.mean([float((r.get("graph_fallbacks") or 0) > 0) for r in graphed]))
        out["graph.capture_ms_sum"] = float(sum(_values(graphed, "graph_capture_ms")))
        replay_only = [r for r in graphed if r.get("graph_mode") == "replay"]
        out["graph.t_expert_ms_replay_only"] = _mean(_values(replay_only, "t_expert_ms"))

    out["t_postgen_model_ms"] = _mean(_values(rows, "t_postgen_model_ms"))
    out["trace.consume_ms"] = _mean(_values(rows, "t_trace_consume_ms"))
    out["trace.hook_host_ms"] = _mean(_values(rows, "trace_hook_host_ms"))
    out["trace.n_marks"] = _mean(_values(rows, "trace_n_marks"))

    for key in ("t_gen_preamble_ms", "t_lm_head_ms", "t_vlm_glue_ms", "t_gen_loop_ms",
                "t_ttft_ms"):
        out[key] = _mean(_values(rows, key))
    ttft = _values(rows, "t_ttft_ms")
    out["t_ttft_ms_p95"] = float(np.percentile(ttft, 95)) if ttft else None
    # Per step, pooled over clips: a gap is a property of one step, and pooling
    # weights each step equally however many steps its clip decoded.
    gaps = [g for a in _arrays(rows, "decode_gap_ms") for g in a]
    out["decode_gap_ms"] = _mean(gaps)
    out["decode_gap_ms_p95"] = float(np.percentile(gaps, 95)) if gaps else None
    out["lm_head_step_ms"] = _mean([h for a in _arrays(rows, "lm_head_step_ms") for h in a])
    violations = _values(rows, "trace_span_violations")
    out["trace.span_violations_sum"] = float(sum(violations)) if violations else None

    for col in HOST:
        if col.name != "extra_passes_ms":
            out[f"host.{col.name}"] = _mean(_values(rows, col.name))
    walls = _values(rows, "clip_wall_ms")
    out["host.clip_wall_ms_p95"] = float(np.percentile(walls, 95)) if walls else None
    # NVIDIA's latency analysis of Alpamayo counts preprocessing as one
    # component: tokenizing images and text, and the trajectory history. The
    # same boundary here, so the two can be put side by side.
    paper = [sum(float(r[k]) for k in ("msg_build_ms", "preprocess_ms", "h2d_ms",
                                         "t_fuse_traj_host_ms"))
             for r in rows if all(is_number(r.get(k)) for k in (
                 "msg_build_ms", "preprocess_ms", "h2d_ms", "t_fuse_traj_host_ms"))]
    out["host.paper_preprocess_ms"] = _mean(paper)
    for key in ("t_pre_generate_ms", "t_tail_ms", "t_action_to_traj_ms", "t_first_traj_ms"):
        out[key] = _mean(_values(rows, key))
    first = _values(rows, "t_first_traj_ms")
    out["t_first_traj_ms_p95"] = float(np.percentile(first, 95)) if first else None
    for key in ("cpu_pass_ms", "cpu_decode_ms", "cpu_expert_ms"):
        out[f"host.{key}"] = _mean(_values(rows, key))
    out["host.ctx_invol"] = _mean(_values(rows, "ctx_invol"))
    rss = _values(rows, "rss_bytes")
    out["host.rss_max_gb"] = max(rss) / 1e9 if rss else None

    clip_peak = _values(rows, "mem_peak_clip_bytes")
    out["mem.peak_clip_gb"] = _mean(clip_peak) / 1e9 if clip_peak else None
    out["mem.peak_clip_gb_max"] = max(clip_peak) / 1e9 if clip_peak else None
    for seg in ("pre", "vision", "prefill", "decode", "postgen", "expert", "tail"):
        values = _values(rows, f"mem_peak_{seg}_bytes")
        out[f"mem.peak_{seg}_gb"] = _mean(values) / 1e9 if values else None
    reserved = _values(rows, "mem_reserved_peak_bytes")
    out["mem.reserved_peak_gb_max"] = max(reserved) / 1e9 if reserved else None
    ooms = _values(rows, "n_ooms")
    out["mem.n_ooms_sum"] = float(sum(ooms)) if ooms else None
    avail = _values(rows, "host_mem_avail_min_bytes")
    out["mem.host_avail_min_gb"] = min(avail) / 1e9 if avail else None

    kv = _values(rows, "kv_bytes")
    out["kv.bytes_mb"] = _mean(kv) / 1e6 if kv else None
    out["kv.final_tokens"] = _mean(_values(rows, "kv_final_tokens"))
    estimates = [kv_traffic(r) for r in rows]
    for key in ("cat_decode", "cat_expert", "graph_copy"):
        values = [e[key] for e in estimates if e and e.get(key) is not None]
        out[f"kv.{key}_gb"] = _mean(values) / 1e9 if values else None
    out["vision.n_images"] = _mean(_values(rows, "n_images"))
    out["vision.n_patches"] = _mean(_values(rows, "n_vision_patches"))
    out["vision.n_tokens"] = _mean(_values(rows, "n_vision_tokens"))

    out.update(_board(rows))

    return {k: float(v) for k, v in out.items() if v is not None and math.isfinite(v)}
