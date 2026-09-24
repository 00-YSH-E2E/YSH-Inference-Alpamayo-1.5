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

"""FLOPs and bytes per segment, and where they put a pass on this board's roofline.

A span's milliseconds say how long it took, not whether that was slow. For
that it needs the work it did: the FLOPs, and the bytes it had to move from
memory. Their ratio, the arithmetic intensity, says which roof applies --
the board's matmul throughput or its memory bandwidth -- and the achieved
rate against that roof says how much is left.

The work is analytic, from the pass's own shapes (timing.parquet) and the
model's structure (run.json ``work_model``), so every main pass gets it
without another run:

* FLOPs: 2 per weight per token through every dense layer, and attention's
  two matmuls, ``4 * batch * heads * q_len * kv_len * head_dim`` per layer --
  counted over the full score matrix, causal or not, the way torch's
  FlopCounterMode counts SDPA, so the two can be compared;
* bytes: each stack's weights once per call; the KV cache read once by
  every attention call; and its concatenation, which reads and writes the
  whole cache on every append (``timing_schema.kv_traffic``). Activations
  are left out: that matters only where a segment is compute-bound anyway,
  and then the bytes do not decide anything.

``--flop-count`` checks the formulas: one pass per clip under FlopCounterMode
and an operand-bytes dispatch mode. Measured over analytic FLOPs outside
[0.9, 1.1] means a formula is wrong. Bytes are not held to that band --
operand bytes count activations the formulas leave out.

Pure python, so it is tested without torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: The segments the formulas cover, and the timing column each is timed by.
#: The head's time is its Euler steps' sum: the diffusion span also holds the
#: sampler's own arithmetic, which is not the expert's work.
WORK_SEGMENTS = ("vision", "prefill", "decode", "lm_head", "expert")
_TIME = {"vision": "t_vision_ms", "prefill": "t_prefill_ms", "decode": "t_decode_ms",
         "lm_head": "t_lm_head_ms"}

#: Measured over analytic FLOPs inside this band: the formulas hold.
CALIBRATION_BAND = (0.9, 1.1)

#: The probe's numbers: bf16 matmul throughput, and the bandwidth of a GEMV, a
#: streaming read, a copy and the KV cache's concatenation.
PEAK_KEYS = ("gemm_tflops", "gemv_gbps", "read_gbps", "copy_gbps", "kvcat_gbps")


def _num(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def segment_time_ms(row: Mapping[str, Any], seg: str) -> float | None:
    """How long a segment took in a pass, on the device clock."""
    if seg == "expert":
        steps = row.get("expert_step_ms")
        if steps is None or not len(steps):
            return None
        return float(sum(float(s) for s in steps))
    return _num(row, _TIME[seg])


def work(row: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Analytic FLOPs and bytes of each segment of one pass.

    A segment whose shapes or structure are missing is absent rather than
    zero -- a pass traced below the level that records its shapes did not
    do no work.
    """
    out: dict[str, dict[str, float]] = {}
    k = _num(row, "kv_rows")
    final = _num(row, "kv_final_tokens")
    kv = _num(row, "kv_bytes")
    n = _num(row, "n_decode_steps") or 0.0
    per_tok = kv / (k * final) if k and final and kv else None
    prompt = final - n if final is not None else None

    vis = model.get("vision")
    patches, tokens, images = (_num(row, "n_vision_patches"), _num(row, "n_vision_tokens"),
                               _num(row, "n_images"))
    if vis and patches and images:
        flops = (2.0 * patches * vis["dense_params"]
                 + 2.0 * (tokens or 0.0) * vis.get("merger_params", 0)
                 # One attention per image, over that image's patches.
                 + 4.0 * vis["layers"] * vis["hidden"] * patches * patches / images)
        out["vision"] = {"flops": flops, "bytes": float(vis["weight_bytes"])}

    lm = model.get("lm")
    if lm and k and prompt and per_tok:
        attn = 4.0 * lm["layers"] * lm["heads"] * lm["head_dim"]
        out["prefill"] = {
            "flops": 2.0 * k * prompt * lm["dense_params"] + attn * k * prompt * prompt,
            # The weights once, and the cache written once.
            "bytes": float(lm["weight_bytes"]) + k * prompt * per_tok,
        }
        if n:
            # Step s attends over prompt + s tokens.
            lengths = n * prompt + n * (n + 1) / 2.0
            out["decode"] = {
                "flops": 2.0 * k * n * lm["dense_params"] + attn * k * lengths,
                # Weights every step; the cache read by attention, and read and
                # written again by its concatenation.
                "bytes": n * float(lm["weight_bytes"]) + 3.0 * k * per_tok * lengths,
            }

    head = model.get("lm_head")
    if head and k:
        # One call per forward, on the last position only (generate asks for
        # the last token's logits).
        calls = n + 1.0
        out["lm_head"] = {"flops": 2.0 * k * calls * head["dense_params"],
                          "bytes": calls * float(head["weight_bytes"])}

    ex = model.get("expert")
    steps = _num(row, "n_expert_calls")
    action = _num(row, "expert_tokens")
    if ex and k and steps and action and final and per_tok:
        span = final + action
        out["expert"] = {
            "flops": (2.0 * k * action * steps * ex["dense_params"]
                      + 4.0 * ex["layers"] * ex["heads"] * ex["head_dim"] * k * action * steps
                      * span),
            "bytes": steps * float(ex["weight_bytes"]) + 3.0 * k * per_tok * steps * span,
        }
    return out


def calibration(row: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Measured over analytic, per segment, for a pass that was counted.

    ``row`` is a ``--flop-count`` pass's timing row: its shapes feed the
    formulas and its ``fc_*`` columns are what was counted.
    """
    analytic = work(row, model)
    out: dict[str, dict[str, float]] = {}
    for seg, w in analytic.items():
        flops, nbytes = _num(row, f"fc_flops_{seg}"), _num(row, f"fc_bytes_{seg}")
        ratios = {}
        if flops is not None and w["flops"] > 0.0:
            ratios["flops"] = flops / w["flops"]
        if nbytes is not None and w["bytes"] > 0.0:
            ratios["bytes"] = nbytes / w["bytes"]
        if ratios:
            out[seg] = ratios
    return out


def outside_band(ratios: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """The segments whose FLOP formula the counted pass contradicts."""
    lo, hi = CALIBRATION_BAND
    return {seg: r["flops"] for seg, r in ratios.items()
            if "flops" in r and not lo <= r["flops"] <= hi}


def efficiency(row: Mapping[str, Any], model: Mapping[str, Any],
               peaks: Mapping[str, float] | None = None) -> dict[str, dict[str, float]]:
    """Per segment: arithmetic intensity, achieved TFLOP/s and GB/s, and --
    with the board's measured peaks -- the fraction of the roof achieved.

    The memory roof is the streaming read the probe measured: what the
    board delivers to a kernel that only reads, which is what the decode
    and the head's steps mostly do.
    """
    out: dict[str, dict[str, float]] = {}
    for seg, w in work(row, model).items():
        ms = segment_time_ms(row, seg)
        if not ms or ms <= 0.0 or w["bytes"] <= 0.0:
            continue
        seconds = ms / 1000.0
        e = {"ai": w["flops"] / w["bytes"], "tflops": w["flops"] / seconds / 1e12,
             "gbps": w["bytes"] / seconds / 1e9}
        if peaks and peaks.get("gemm_tflops") and peaks.get("read_gbps"):
            roof = min(float(peaks["gemm_tflops"]), e["ai"] * float(peaks["read_gbps"]) / 1000.0)
            e["roof_frac"] = e["tflops"] / roof
        out[seg] = e
    return out


def aggregate(rows: Iterable[Mapping[str, Any]], model: Mapping[str, Any] | None,
              peaks: Mapping[str, float] | None) -> dict[str, float]:
    """MLflow numbers: efficiency over the main passes, the counted work and
    its calibration over the ``flops`` passes, and the board's peaks."""
    rows = list(rows)
    out: dict[str, float] = {}
    for key in PEAK_KEYS:
        if peaks and peaks.get(key) is not None:
            out[f"roofline.{key}"] = float(peaks[key])
    if not model:
        return out
    main = [r for r in rows if r.get("row_kind") == "main" and r.get("timing_measured")]
    per: dict[str, dict[str, list[float]]] = {}
    for r in main:
        for seg, e in efficiency(r, model, peaks).items():
            for key, value in e.items():
                per.setdefault(seg, {}).setdefault(key, []).append(value)
    for seg, values in per.items():
        for key, series in values.items():
            name = f"ai.{seg}" if key == "ai" else f"eff.{seg}_{key}"
            out[name] = sum(series) / len(series)
    counted = [r for r in rows if r.get("row_kind") == "flops"]
    for seg in WORK_SEGMENTS:
        flops = [v for r in counted if (v := _num(r, f"fc_flops_{seg}")) is not None]
        nbytes = [v for r in counted if (v := _num(r, f"fc_bytes_{seg}")) is not None]
        if flops:
            out[f"flops.{seg}_gflop"] = sum(flops) / len(flops) / 1e9
        if nbytes:
            out[f"flops.{seg}_gb"] = sum(nbytes) / len(nbytes) / 1e9
        ratios = [c[seg] for r in counted if seg in (c := calibration(r, model))]
        for kind in ("flops", "bytes"):
            series = [x[kind] for x in ratios if kind in x]
            if series:
                out[f"{kind}.ratio_{seg}"] = sum(series) / len(series)
    return out


#: Every key ``aggregate`` can emit, for the registry's declared list.
AGGREGATE_KEYS = (
    *(f"roofline.{k}" for k in PEAK_KEYS),
    *(f"ai.{s}" for s in WORK_SEGMENTS),
    *(f"eff.{s}_{k}" for s in WORK_SEGMENTS for k in ("tflops", "gbps", "roof_frac")),
    *(f"flops.{s}_{k}" for s in WORK_SEGMENTS for k in ("gflop", "gb")),
    *(f"{kind}.ratio_{s}" for kind in ("flops", "bytes") for s in WORK_SEGMENTS),
)
