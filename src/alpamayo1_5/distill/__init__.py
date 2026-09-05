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

"""Step distillation of the flow-matching action head.

The pretrained sampler integrates a learned velocity field with ten Euler
steps. Truncating to one step without training loses coverage: the single
step lands near the conditional mean and the K samples collapse onto each
other. The pieces here teach the head to reach in one step what its own
two-step integration reaches, sample by sample, with the VLM frozen and no
external teacher -- the objective in SnapFlow (arXiv 2604.05656).

* ``head``                    -- the projection with a second, target-time input
* ``flow_matching_shortcut``  -- the sampler that passes it, and accepts a given x0
* ``conditioning``            -- frozen-VLM prefill and reasoning tails as KV tensors
* ``forward``                 -- the differentiable student forward against a cache
* ``objective``               -- the loss
* ``pool``                    -- GPU-resident conditionings, rotated
* ``data`` / ``export``       -- clips, reasoning bank, GT actions; loadable checkpoints

Everything here is additive: no upstream file is modified, and a checkpoint
whose config names the upstream classes loads and runs exactly as before.
"""
