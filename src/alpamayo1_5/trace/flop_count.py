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

"""The torch half of the work model: the structure, a counted pass, the board's peaks.

* ``work_model`` reads off the loaded model what roofline.py's formulas
  need: dense weights per stack, attention shapes, and the bytes each
  stack's weights take as loaded -- so a quantized or pruned checkpoint is
  weighed as it is, not as its config says.
* ``counted`` runs one pass under torch's FlopCounterMode and an
  operand-bytes dispatch mode, cut into the segments a profile pass uses.
  It checks the formulas; it is never timed.
* ``probe`` measures what this board delivers to the kernels the model is
  made of: a bf16 GEMM, a GEMV over a large weight, a streaming read, a
  copy, and the KV cache's concatenation.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from alpamayo1_5.trace import profile_parse as PP

#: Ops that move no data: allocation without a write, and bookkeeping. Views
#: are recognised by their schema.
_FREE = frozenset({
    "aten::empty", "aten::empty_like", "aten::empty_strided", "aten::new_empty",
    "aten::new_empty_strided", "aten::resize_", "aten::set_", "aten::record_stream",
    "aten::detach", "aten::lift_fresh",
})


def _dense(modules: Iterable[torch.nn.Module]) -> int:
    """Weights a token multiplies through: in x out of every Linear, and the
    kernel of a convolution applied once per output position (the patch
    embedding)."""
    total = 0
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, torch.nn.Linear):
                total += sub.in_features * sub.out_features
            elif isinstance(sub, (torch.nn.Conv2d, torch.nn.Conv3d)):
                total += sub.weight.numel()
    return total


def _weight_bytes(modules: Iterable[torch.nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for p in module.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel() * p.element_size()
    return total


def _head_dim(config: Any) -> int:
    return int(getattr(config, "head_dim", None)
               or config.hidden_size // config.num_attention_heads)


def work_model(model: Any) -> dict[str, Any]:
    """The loaded model's structure as roofline.work needs it (run.json ``work_model``)."""
    out: dict[str, Any] = {}
    vlm = getattr(model, "vlm", None)
    inner = getattr(vlm, "model", None)
    visual = getattr(inner, "visual", None)
    blocks = getattr(visual, "blocks", None)
    if blocks is not None and len(blocks):
        mergers = [m for m in (getattr(visual, "merger", None),
                               *(getattr(visual, "deepstack_merger_list", None) or ()))
                   if m is not None]
        embed = getattr(visual, "patch_embed", None)
        out["vision"] = {
            "dense_params": _dense(blocks) + (_dense([embed]) if embed is not None else 0),
            "merger_params": _dense(mergers),
            "layers": len(blocks),
            "hidden": int(visual.config.hidden_size),
            "weight_bytes": _weight_bytes([visual]),
        }
    language = getattr(inner, "language_model", None)
    for key, stack in (("lm", language), ("expert", getattr(model, "expert", None))):
        layers = getattr(stack, "layers", None)
        if layers is None or not len(layers):
            continue
        out[key] = {
            "dense_params": _dense(layers),
            "layers": len(layers),
            "heads": int(stack.config.num_attention_heads),
            "head_dim": _head_dim(stack.config),
            "weight_bytes": _weight_bytes(layers),
        }
    head = getattr(vlm, "lm_head", None)
    if head is not None:
        out["lm_head"] = {"dense_params": _dense([head]), "weight_bytes": _weight_bytes([head])}
    return out


def _operand_bytes(args: Any, kwargs: Any, out: Any) -> int:
    """Bytes an op reads and writes, assuming nothing is served from cache.

    A broadcast operand is charged its storage, not its logical size; a
    slice of a larger buffer, its logical size.
    """
    total = 0
    for t in tree_leaves((args, kwargs, out)):
        if isinstance(t, torch.Tensor) and t.device.type == "cuda":
            logical = t.numel() * t.element_size()
            try:
                stored = t.untyped_storage().nbytes()
            except Exception:
                stored = logical
            total += min(logical, stored)
    return total


class _Operands(TorchDispatchMode):
    def __init__(self, state: PP.SegmentState, into: dict[str, int]) -> None:
        super().__init__()
        self.state = state
        self.into = into
        self.n_ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if not func.is_view and func._schema.name not in _FREE:
            self.into[self.state.segment] += _operand_bytes(args, kwargs, out)
            self.n_ops += 1
        return out


def counted(run: Callable[[Callable[[str, str], None]], Any]) -> tuple[Any, dict[str, Any]]:
    """Run one pass with its FLOPs and operand bytes counted, by segment.

    ``run(listener)`` runs the pass with ``listener`` handed to the tracer:
    every mark moves the segment, and the FLOPs counted since the previous
    mark are credited to the segment that was current until then. Returns
    what ``run`` returned, and the ``fc_*`` columns of the pass's row.
    """
    from torch.utils.flop_counter import FlopCounterMode

    state = PP.SegmentState()
    flops: dict[str, int] = defaultdict(int)
    nbytes: dict[str, int] = defaultdict(int)
    counter = FlopCounterMode(display=False)
    last = [0]

    def listener(bucket: str, kind: str) -> None:
        total = counter.get_total_flops()
        flops[state.segment] += total - last[0]
        last[0] = total
        state.mark(bucket, kind)

    operands = _Operands(state, nbytes)
    with counter, operands:
        result = run(listener)
    flops[state.segment] += counter.get_total_flops() - last[0]
    row: dict[str, Any] = {"fc_n_ops": operands.n_ops}
    for seg in PP.SEGMENTS:
        row[f"fc_flops_{seg}"] = float(flops.get(seg, 0))
        row[f"fc_bytes_{seg}"] = float(nbytes.get(seg, 0))
    return result, row


def _seconds(fn: Callable[[], Any], reps: int) -> float:
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0 / reps


def _gemm(reps: int) -> float:
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    return 2.0 * 8192 ** 3 / _seconds(lambda: a @ b, reps) / 1e12


def _streams(reps: int) -> dict[str, float]:
    weight = torch.randn(16384, 32768, device="cuda", dtype=torch.bfloat16)
    size = weight.numel() * weight.element_size()
    rows = torch.randn(6, 16384, device="cuda", dtype=torch.bfloat16)
    return {"gemv_gbps": size / _seconds(lambda: rows @ weight, reps) / 1e9,
            "read_gbps": size / _seconds(lambda: weight.sum(), reps) / 1e9,
            "copy_gbps": 2.0 * size / _seconds(lambda: weight.clone(), reps) / 1e9}


def _kv_cat(reps: int) -> float:
    caches = [torch.randn(6, 8, 3000, 128, device="cuda", dtype=torch.bfloat16)
              for _ in range(36)]
    token = torch.randn(6, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
    moved = 2.0 * sum(c.numel() * c.element_size() for c in caches)
    return moved / _seconds(lambda: [torch.cat([c, token], dim=2) for c in caches], reps) / 1e9


def probe(reps: int = 20) -> dict[str, float]:
    """The board's peaks, from the kernels the model runs, on bf16 tensors.

    GEMM: 8192^3, the dense roof. GEMV: six rows through a 1 GiB weight, the
    decode step's shape. Read: a sum over 1 GiB. Copy: a clone of it, read
    and written. KV concatenation: 36 layers of a [6, 8, 3000, 128] cache,
    each grown by one token, the way the decode loop grows it. Each piece
    frees its buffers when it returns.
    """
    started = time.time()
    out = {"gemm_tflops": _gemm(reps), **_streams(reps), "kvcat_gbps": _kv_cat(max(reps // 4, 1))}
    torch.cuda.empty_cache()
    out["probe_s"] = time.time() - started
    return out
