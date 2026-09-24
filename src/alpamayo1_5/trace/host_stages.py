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

"""Where a clip's time goes outside the model call.

The tracer sees the model. Before the call a clip is read from the dataset
cache, four cameras are decoded on the CPU, the prompt is built and tokenized,
the images are resized and patched, and everything is copied to the GPU;
after it, results come back to the host and metrics and figures are made.
None of that was timed -- yet NVIDIA's own latency analysis of Alpamayo
counts tokenization as a component of its own, and on the Thor the data
loader alone was measured at about 2.2 s a clip.

Host work, so a plain ``perf_counter`` is the right clock: these stages either
run on the CPU or block the host until they finish (a copy from pageable
memory does).

Nothing here imports torch; the wrappers are tested with stand-ins in CI.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Iterator

#: The stages a main pass's timing row carries, in pipeline order.
STAGES = (
    "data_load_ms", "data_ego_ms", "data_cam_fetch_ms", "data_cam_decode_ms",
    "msg_build_ms", "preprocess_ms", "preprocess_image_ms", "h2d_ms",
    "model_call_ms", "extra_passes_ms",
    "result_cpu_ms", "metrics_ms", "render_ms", "flush_ms",
)

#: The stages that tile a clip's wall clock without overlapping. The data
#: sub-stages sit inside data_load and the image half inside preprocess, so
#: they are left out of the subtraction that yields clip_other_ms.
_TILING = ("data_load_ms", "msg_build_ms", "preprocess_ms", "h2d_ms", "model_call_ms",
           "extra_passes_ms", "result_cpu_ms", "metrics_ms", "render_ms", "flush_ms")


class StageClock:
    """Host milliseconds per stage for one clip. Stages add up if entered twice."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self.started = time.perf_counter()

    def add(self, name: str, ms: float) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + float(ms)

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - started) * 1000.0)

    def row(self, clip_wall_ms: float | None = None) -> dict[str, float | None]:
        """The timing-row fields. ``preprocess_text_ms`` is the processor's time
        minus its image half; ``clip_other_ms`` is the clip's wall clock minus
        every named stage and the model call, floored at zero."""
        out: dict[str, float | None] = {name: self.ms.get(name) for name in STAGES}
        if "preprocess_ms" in self.ms:
            out["preprocess_text_ms"] = max(
                self.ms["preprocess_ms"] - self.ms.get("preprocess_image_ms", 0.0), 0.0)
        else:
            out["preprocess_text_ms"] = None
        out["clip_wall_ms"] = clip_wall_ms
        if clip_wall_ms is not None:
            named = sum(self.ms.get(k, 0.0) for k in _TILING)
            out["clip_other_ms"] = max(clip_wall_ms - named, 0.0)
        else:
            out["clip_other_ms"] = None
        return out


class _Current:
    """Which clip's clock the wrappers below write to. Set per clip."""

    clock: StageClock | None = None


CURRENT = _Current()


def add_since(name: str, started: float) -> None:
    """Add the host time since ``started`` to the current clip's stage ``name``."""
    clock = CURRENT.clock
    if clock is not None:
        clock.add(name, (time.perf_counter() - started) * 1000.0)


def wrap_dataset(avdi: Any) -> Any:
    """Split the data load into the ego fetch, the camera fetches and the decodes.

    ``get_clip_feature`` is called once for egomotion and once per camera; each
    camera object then decodes its frames on the CPU. Both are wrapped on the
    instance. A reader that refuses an attribute (``__slots__``) is left
    untimed rather than failing the clip -- its decode then shows up in
    ``data_load_ms`` only.
    """
    original = avdi.get_clip_feature
    try:
        ego = avdi.features.LABELS.EGOMOTION
    except AttributeError:
        ego = None

    def get_clip_feature(clip_id: Any, feature: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        obj = original(clip_id, feature, *args, **kwargs)
        is_ego = ego is not None and feature == ego
        add_since("data_ego_ms" if is_ego else "data_cam_fetch_ms", started)
        decode = getattr(obj, "decode_images_from_timestamps", None)
        if not is_ego and decode is not None:
            def timed_decode(*a: Any, **kw: Any) -> Any:
                began = time.perf_counter()
                try:
                    return decode(*a, **kw)
                finally:
                    add_since("data_cam_decode_ms", began)
            try:
                obj.decode_images_from_timestamps = timed_decode
            except (AttributeError, TypeError):
                pass
        return obj

    avdi.get_clip_feature = get_clip_feature
    return avdi


def wrap_processor(processor: Any) -> Any:
    """Time the image half of tokenization.

    The chat template runs the image processor (resize, normalise, patch) and
    the tokenizer in one call. Wrapping ``preprocess`` on the image processor
    instance splits them: the image processor's ``__call__`` resolves
    ``self.preprocess`` through the instance.
    """
    image = getattr(processor, "image_processor", None)
    original = getattr(image, "preprocess", None)
    if original is None:
        return processor

    def preprocess(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            add_since("preprocess_image_ms", started)

    try:
        image.preprocess = preprocess
    except (AttributeError, TypeError):
        pass
    return processor
