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

"""Record a run so it can be found and compared later.

The split is: numbers that get sorted and graphed go to MLflow, files go to
Hugging Face, and a coordinate on the run points from one to the other. A run
that lands in only one of those is not findable -- metrics with no artifacts
cannot be inspected, artifacts with no metrics cannot be ranked.

Aggregates only. Per-clip and per-sample rows live in the run's parquet; the
one exception is the scene breakdown, which is a comparison axis rather than
raw data and belongs where things get compared.

Size matters here in a way it usually does not: the tracking server proxies
artifacts through a Jetson Orin Nano with one worker, so a single large upload
stalls it for everyone. Files above the ceiling are refused rather than
attempted, and the bulk goes to Hugging Face regardless.

On exit the run is checked against the recording rules and prints PASS or FAIL
with what is missing. A FAIL is a bug in the caller, not in the run -- fix the
script, because the next run will trip on the same thing.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import mlflow
except ImportError:  # only needed when a run is actually recorded
    mlflow = None

DEFAULT_TRACKING_URI = "http://ysh-jetson-orin-nano.tail4570ef.ts.net:5000"

# From the recording rules: anything larger goes to Hugging Face and MLflow
# gets only the coordinate.
MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
MAX_RUN_ARTIFACT_BYTES = 50 * 1024 * 1024

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_METRIC_KEY_RE = re.compile(r"[^0-9A-Za-z_\-./ ]")
_SOURCE_PREFIXES = ("hf:", "path:", "dvc:", "mlflow:", "s3:", "http:", "https:")
# Where an output can live and still be there next month. path: is absent on
# purpose -- it names a directory on whichever machine happened to run.
_DURABLE_URI = ("hf:", "s3:", "mlflow:", "http:", "https:")


# -- context ---------------------------------------------------------------
def env_tags(seed: int | None = None, notes: str | None = None) -> dict[str, str]:
    """Machine and invocation. Collection failures are never fatal."""
    tags = {
        "env.host": socket.gethostname(),
        "env.python": platform.python_version(),
        "entrypoint": sys.argv[0],
        "cmd": " ".join(shlex.quote(a) for a in sys.argv),
    }
    try:
        import torch

        tags["env.torch"] = torch.__version__
        tags["env.cuda"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            tags["env.gpu"] = torch.cuda.get_device_name(0)
            tags["env.gpu_count"] = str(torch.cuda.device_count())
            tags["env.compute_capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
    except Exception:
        pass
    if seed is not None:
        tags["seed"] = str(seed)
    if notes is not None:
        tags["notes"] = notes
    return {k: v for k, v in tags.items() if v}


def git_tags() -> dict[str, str]:
    """Code coordinates read from the working directory. Never raises."""

    def run(*args: str) -> str:
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            return done.stdout.strip() if done.returncode == 0 else ""
        except Exception:
            return ""

    tags = {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_remote": run("git", "config", "--get", "remote.origin.url"),
    }
    if tags["git_commit"]:
        tags["git_dirty"] = "true" if run("git", "status", "--porcelain") else "false"
    return {k: v for k, v in tags.items() if v}


# -- Hugging Face ----------------------------------------------------------
def _hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        return Path.home().joinpath(".cache/huggingface/token").read_text().strip() or None
    except OSError:
        return None


def parse_coord(coord: str) -> tuple[str, str, str]:
    """Split ``owner/name[:kind][@rev][#subpath]``."""
    coord = coord.strip()
    for prefix in ("hf:", "hf/"):
        if coord.startswith(prefix):
            coord = coord[len(prefix) :]
    subpath = ""
    if "#" in coord:
        coord, subpath = coord.split("#", 1)
    revision = "main"
    if "@" in coord:
        coord, revision = coord.rsplit("@", 1)
    return coord, revision, subpath


def hf_sha(repo: str, revision: str = "main", kind: str = "model") -> str | None:
    """Resolve a revision to its 40-character sha.

    A branch name is not a coordinate -- ``main`` points somewhere else
    tomorrow, and a run pinned to it cannot be reproduced. Returns None when the
    hub is unreachable so the caller can decide whether that is fatal.
    """
    if _SHA_RE.match(revision):
        return revision
    section = "models" if kind == "model" else "datasets"
    url = f"https://huggingface.co/api/{section}/{repo.split(':', 1)[0]}/revision/{revision}"
    request = urllib.request.Request(url)
    token = _hf_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            sha = json.load(response).get("sha", "")
        return sha if _SHA_RE.match(sha) else None
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return None


def upload_run_dir(
    paths: Iterable[Path], repo: str, path_in_repo: str, commit_message: str | None = None
) -> str | None:
    """Push a run's files to the evals repo, returning the commit sha.

    Uploads are append-only: a re-run writes a new directory rather than
    replacing one, because the previous result is what a changed number gets
    compared against.
    """
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        operations = []
        for path in paths:
            # Preserve the samples/ subdirectory; everything else sits at the top
            # of the run directory.
            leaf = f"samples/{path.name}" if path.parent.name == "samples" else path.name
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{path_in_repo}/{leaf}", path_or_fileobj=str(path)
                )
            )
        if not operations:
            print("[ml_platform] nothing to upload -- the run directory is empty",
                  file=sys.stderr)
            return None
        # Pass the token explicitly. Relying on ambient auth means a machine
        # where nobody ran `hf auth login` degrades to a local-only run that
        # still looks recorded.
        api = HfApi(token=_hf_token() or None)
        api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
        info = api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message or f"Add run {path_in_repo}",
        )
        return getattr(info, "oid", None)
    except Exception as exc:
        # Returning None is not enough on its own: the caller must record that
        # this happened, or the run claims a local path as its archive and
        # passes its own rules. See Run.check().
        print(f"[ml_platform] upload failed, outputs stay local: {exc}", file=sys.stderr)
        return None


# -- run -------------------------------------------------------------------
class Run:
    """An open MLflow run that enforces the recording rules."""

    def __init__(self, run_type: str) -> None:
        self.run_type = run_type
        self.run_id = mlflow.active_run().info.run_id
        self._metrics: set[str] = set()
        self._tags: dict[str, str] = {}
        self._artifact_bytes = 0
        self.unresolved: list[str] = []

    def tag(self, key: str, value: Any) -> None:
        self._tags[key] = str(value)
        mlflow.set_tag(key, str(value))

    def params(self, params: Mapping[str, Any] | None) -> None:
        """Everything needed to reproduce. Params are cheap; omissions are not."""
        if not params:
            return
        flat = {}
        for key, value in params.items():
            flat[key] = json.dumps(value) if isinstance(value, (list, tuple, dict)) else str(value)
        mlflow.log_params(flat)

    def metric(self, key: str, value: float, step: int | None = None) -> None:
        key = _METRIC_KEY_RE.sub("_", key)  # MLflow rejects anything outside this alphabet
        if value is None or (isinstance(value, float) and value != value):
            return
        self._metrics.add(key)
        mlflow.log_metric(key, float(value), step=step)

    def metrics(self, values: Mapping[str, float]) -> None:
        for key, value in values.items():
            self.metric(key, value)

    def score(self, value: float) -> None:
        """The one representative number this run is ranked by."""
        self.metric("score", value)

    def by_scenario(self, scores: Mapping[str, Any]) -> None:
        """Scene breakdown -- a comparison axis, so it belongs here and not only in parquet."""
        for name, value in scores.items():
            key = _METRIC_KEY_RE.sub("_", str(name))
            if isinstance(value, (tuple, list)) and len(value) == 2:
                self.metric(f"scenario.{key}.score", value[0])
                self.metric(f"scenario.{key}.count", value[1])
            else:
                self.metric(f"scenario.{key}.score", value)

    def result_path(self, uri: str) -> None:
        self.tag("output_uri", uri)
        self.tag("eval_result_path", uri)

    def artifact(self, path: str | Path, name: str | None = None) -> bool:
        """Upload one small file. Refuses anything that would strain the server."""
        path = Path(path)
        size = path.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            print(
                f"[ml_platform] SKIP {path.name} ({size / 1e6:.1f}MB > 10MB) -- "
                "put it on Hugging Face and tag output_uri instead.",
                file=sys.stderr,
            )
            return False
        if self._artifact_bytes + size > MAX_RUN_ARTIFACT_BYTES:
            print(f"[ml_platform] SKIP {path.name}: run would exceed the 50MB budget.",
                  file=sys.stderr)
            return False
        mlflow.log_artifact(str(path), artifact_path=name)
        self._artifact_bytes += size
        return True

    def checkpoint(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Checkpoints never go through MLflow. Upload to Hugging Face and record the "
            "coordinate with run.tag('output_uri', 'hf:owner/name@<sha>')."
        )

    # -- lineage -----------------------------------------------------------
    def hf_datasets(self, coords: Iterable[str]) -> None:
        self._hf_coords(coords, kind="dataset", tag_key="hf_datasets")

    def hf_models(self, coords: Iterable[str]) -> None:
        self._hf_coords(coords, kind="model", tag_key="hf_models")

    def _hf_coords(self, coords: Iterable[str], kind: str, tag_key: str) -> None:
        pinned = []
        for raw in coords:
            repo, revision, _ = parse_coord(raw)
            if kind == "model" and ":" not in repo:
                repo = f"{repo}:model"  # the hub keys model repos with this suffix
            sha = hf_sha(repo, revision, kind=kind)
            if sha is None:
                self.unresolved.append(f"{repo}@{revision}")
                sha = revision
            pinned.append(f"{repo}@{sha}")
            self.tag(f"hf.{repo}", sha)
        if pinned:
            self.tag(tag_key, json.dumps(pinned))

    def model_source(self, source: str) -> str:
        """Pin what was run.

        A variant produced by a deterministic transform of a released
        checkpoint has no repository of its own. Inventing one would make the
        coordinate false; the honest form is the source checkpoint here and the
        transform in params.
        """
        if source.startswith("hf:"):
            repo, revision, subpath = parse_coord(source)
            if ":" not in repo:
                repo = f"{repo}:model"
            sha = hf_sha(repo, revision, kind="model")
            if sha is None:
                self.unresolved.append(f"{repo}@{revision}")
                sha = revision
            source = f"hf:{repo}@{sha}" + (f"#{subpath}" if subpath else "")
            self.hf_models([f"{repo}@{sha}"])
        elif not source.startswith(_SOURCE_PREFIXES):
            source = f"path:{source}"
        self.tag("model_source", source)
        return source

    # -- validation --------------------------------------------------------
    def check(self) -> list[str]:
        """Rule violations, empty if the run is complete."""
        problems = []
        if not self._tags.get("git_commit"):
            problems.append("git_commit missing -- was this started outside a git repo?")
        if not (self._tags.get("hf_datasets") or self._tags.get("hf_models")):
            problems.append("no data coordinate: needs hf_datasets or hf_models")
        if self.run_type in ("eval", "infer") and not self._tags.get("model_source"):
            problems.append("model_source required for eval/infer")
        if self.run_type == "eval" and "score" not in self._metrics:
            problems.append("eval runs need a 'score' metric")
        if self.run_type == "infer" and not self._tags.get("output_uri"):
            problems.append("infer runs need an output_uri -- otherwise nothing was produced")
        # A local path is not an archive. When the upload fails the caller falls
        # back to path:<dir>, and treating that as satisfied hands back a PASS
        # for a run whose outputs exist on exactly one machine's disk -- the
        # failure this check exists to catch, reported as success.
        output = self._tags.get("output_uri", "")
        if self.run_type in ("eval", "infer") and output and not output.startswith(_DURABLE_URI):
            problems.append(
                f"output_uri is not somewhere the outputs survive this machine: {output}"
            )
        if self.run_type == "train" and not (self._metrics & {"val_loss", "train_loss", "loss"}):
            problems.append("train runs need val_loss / train_loss / loss")
        # Only meaningful where something was conditioned on something: a
        # training run has no conditioning source, and demanding one failed
        # every train run on a field that does not apply to it.
        if self.run_type in ("eval", "infer") and not self._tags.get("conditioning_source"):
            problems.append("conditioning_source missing -- runs cannot be compared without it")
        for coord in dict.fromkeys(self.unresolved):
            problems.append(f"unresolved revision (not a 40-char sha): {coord}")
        return problems


@contextmanager
def _run(
    experiment: str,
    run_type: str,
    run_name: str | None = None,
    params: Mapping[str, Any] | None = None,
    hf_datasets: Iterable[str] = (),
    hf_models: Iterable[str] = (),
    model: str | None = None,
    split: str | None = None,
    variant: str | None = None,
    conditioning_source: str = "generated",
    seed: int | None = None,
    notes: str | None = None,
    tracking_uri: str | None = None,
):
    if mlflow is None:
        raise RuntimeError("MLflow is not installed. pip install 'mlflow>=3'")
    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    )
    mlflow.set_experiment(experiment)
    started = time.time()
    with mlflow.start_run(run_name=run_name):
        run = Run(run_type)
        run.tag("run_type", run_type)
        run.tag("conditioning_source", conditioning_source)
        for key, value in git_tags().items():
            run.tag(key, value)
        for key, value in env_tags(seed=seed, notes=notes).items():
            run.tag(key, value)
        if variant:
            run.tag("variant", variant)
        if split:
            run.tag("eval_split", split)
        if model:
            run.model_source(model)
        run.hf_datasets(hf_datasets)
        if hf_models:
            run.hf_models(hf_models)
        run.params(params)
        try:
            yield run
        finally:
            # Everything in here talks to the server. If the run is unwinding
            # because the tailnet dropped, an exception raised here replaces the
            # real traceback with a connection error and the cause is lost.
            problems: list[str] = []
            try:
                run.tag("duration_sec", str(int(time.time() - started)))
                problems = run.check()
                # The verdict has to outlive the terminal. Printed only, there
                # is no way to look at a run six months later and tell whether
                # it satisfied the rules or which coordinate it was missing.
                run.tag("rules_check", "PASS" if not problems else "FAIL")
                if problems:
                    run.tag("rules_problems", "; ".join(problems)[:4000])
            except Exception as exc:  # noqa: BLE001 -- never mask the real error
                print(f"[ml_platform] could not finish recording: {exc}", file=sys.stderr)
            print(f"\n[ml_platform] run_id={run.run_id}  experiment={experiment}")
            if problems:
                print("[ml_platform] FAIL -- fix the script, not the run:")
                for problem in problems:
                    print(f"  - {problem}")
            else:
                print("[ml_platform] PASS -- all required coordinates recorded")
            if run._tags.get("git_dirty") == "true":
                print("[ml_platform] WARNING: repo is dirty, this run is not reproducible")


def track(experiment: str, **kwargs: Any):
    """A run that produces a model."""
    return _run(experiment, "train", **kwargs)


def evaluate(experiment: str, **kwargs: Any):
    """A run scored against labelled data."""
    return _run(experiment, "eval", **kwargs)


def infer(experiment: str, **kwargs: Any):
    """A run that produces outputs with nothing to score them against."""
    return _run(experiment, "infer", **kwargs)
