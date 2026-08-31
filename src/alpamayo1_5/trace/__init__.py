# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Instrumentation for recording inference runs.

Purely additive: nothing here modifies upstream files, so the fork diff against
NVlabs/alpamayo1.5 stays readable. Three pieces:

* ``metrics``     -- pure functions over arrays. Never stored, always recomputed.
* ``token_trace`` -- captures per-token logprob/entropy and exact token counts.
* ``writer``      -- the run directory: predictions.parquet, run.json, samples/.

The organizing rule is that inference is expensive and analysis is cheap, so a
run stores only what cannot be recomputed without the model. Metric definitions
will change; the raw data must outlive them.
"""
