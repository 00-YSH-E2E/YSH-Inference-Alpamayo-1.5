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

Three entry points, one per kind of run::

    with track("my-project", params="params.yaml",
               hf_datasets=["YSHRobotics/CoC-Nusc@main"],
               project="my-project") as run:          # makes a model
        run.metric("val_loss", loss, step=epoch)

    with evaluate("my-project", model="hf:owner/name@main",
                  split="val14") as ev:                # scores against labels
        ev.score(0.9258)
        ev.by_scenario({"curve": (0.93, 25)})

    with infer("my-project", model="hf:owner/name@main") as ev:  # no labels
        ev.result_path("hf:owner/name-evals@<sha>#runs/<dir>/")

An eval that cannot score is an infer. Inventing a score to satisfy the eval
rules is worse than recording honestly that there was nothing to score against.

This file is the canonical copy in the ML_Platform repository under
``examples/``; every training and inference repo holds a byte-identical copy so
that a fix reaches all of them. It carries an Apache-2.0 header for the repos
that require one, and the two copies are meant to compare equal -- if they
differ, the one in ML_Platform wins.

Requires ``mlflow>=3`` at record time (import works without it) and PyYAML only
if params is given as a YAML path.
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

# The hub is on a private tailnet with no route from the internet and no
# authentication -- the bind address is the whole boundary. Point this at your
# own server with ML_PLATFORM_HOST (or MLFLOW_TRACKING_URI for the full URL);
# the default is only a convenience for the machines already on that tailnet.
DEFAULT_HOST = os.environ.get("ML_PLATFORM_HOST", "ysh-jetson-orin-nano.tail4570ef.ts.net")
DEFAULT_TRACKING_URI = f"http://{DEFAULT_HOST}:5000"

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
    """Machine and invocation. Collection failures are never fatal.

    None of this can be filled in afterwards: a run that never looked at its
    own GPU cannot be told later which one it had. Half of it is worth more
    than none, which is why each item is collected separately -- one
    try/except around the whole torch block loses the device count and the
    compute capability because the device *name* raised.
    """
    tags: dict[str, str] = {}

    def put(key: str, fn) -> None:
        try:
            value = fn()
        except Exception:
            return
        if value:
            tags[key] = str(value)

    put("env.host", socket.gethostname)
    put("env.python", platform.python_version)
    put("env.platform", lambda: f"{platform.system()} {platform.machine()}")
    put("entrypoint", lambda: sys.argv[0])
    put("cmd", lambda: " ".join(shlex.quote(a) for a in sys.argv))
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]
    if torch is not None:
        put("env.torch", lambda: torch.__version__)
        put("env.cuda", lambda: torch.version.cuda or "")
        if torch.cuda.is_available():
            put("env.gpu", lambda: torch.cuda.get_device_name(0))
            put("env.gpu_count", lambda: str(torch.cuda.device_count()))
            # What kernels actually differ on across a Thor / Orin / Pro 6000
            # fleet. The GPU name alone does not separate them reliably.
            put("env.compute_capability",
                lambda: "sm_%d%d" % torch.cuda.get_device_capability(0))
    if seed is not None:
        tags["seed"] = str(seed)
    if notes is not None:
        tags["notes"] = notes
    return {k: v for k, v in tags.items() if v}


def _strip_credentials(url: str) -> str:
    """Remove any userinfo from a git remote before it becomes a tag.

    A remote cloned with a token looks like
    ``https://x-access-token:ghp_xxx@github.com/owner/repo``. The tracking
    server has no authentication -- its bind address is the whole boundary --
    so writing that verbatim publishes a live credential to everyone who can
    reach the tailnet, and MLflow tags cannot be edited afterwards.
    """
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://{rest.rsplit('@', 1)[-1]}"
    return url


def git_tags(root: str | Path = ".") -> dict[str, str]:
    """Code coordinates for the repo at ``root``. Never raises.

    ``-C`` matters: without it this reports on whatever directory the process
    was launched from, which is not necessarily the repo holding the code
    being run.
    """

    def run(*args: str) -> str:
        try:
            done = subprocess.run(
                ("git", "-C", str(root)) + args,
                capture_output=True, text=True, timeout=10, check=False,
            )
            return done.stdout.strip() if done.returncode == 0 else ""
        except Exception:
            # A timeout is the interesting case: git can block forever on a
            # credential prompt or a dead network mount.
            return ""

    tags = {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_remote": _strip_credentials(run("config", "--get", "remote.origin.url")),
    }
    if tags["git_commit"]:
        # --porcelain counts untracked files too, which is right: a new file
        # the run depends on makes it just as unreproducible as an edit.
        tags["git_dirty"] = "true" if run("status", "--porcelain") else "false"
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


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested config to dotted keys: ``{"opt": {"lr": 1e-4}}`` -> ``opt.lr``.

    MLflow params are flat. Logging a nested dict as one JSON blob makes it
    impossible to sort or filter runs by a setting inside it, which is most of
    the point of recording settings at all.
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def _read_config(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON config. Returns {} rather than raising."""
    try:
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # imported here so the module works without PyYAML

            return yaml.safe_load(text) or {}
        return json.loads(text)
    except Exception as exc:
        print(f"[ml_platform] could not read {path}: {exc}", file=sys.stderr)
        return {}


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

    def params(self, params: Mapping[str, Any] | str | Path | None) -> None:
        """Everything needed to reproduce. Params are cheap; omissions are not.

        Accepts a mapping or a path to a YAML/JSON config. Given a path, the
        file is also attached to the run: a flattened key list answers "what
        was the setting", the file answers "what did the config look like",
        and the second question is the one asked when reproducing a run months
        later.
        """
        if params is None:
            return
        source: Path | None = None
        if isinstance(params, (str, Path)):
            source = Path(params)
            if not source.is_file():
                print(f"[ml_platform] no config at {source} -- no params recorded",
                      file=sys.stderr)
                return
            params = _read_config(source)
        if not params:
            return
        flat = {}
        for key, value in flatten(dict(params)).items():
            flat[key] = json.dumps(value) if isinstance(value, (list, tuple, dict)) else str(value)
        mlflow.log_params(flat)
        if source is not None:
            self.artifact(source, name="config")

    def metric(self, key: str, value: float, step: int | None = None) -> None:
        key = _METRIC_KEY_RE.sub("_", key)  # MLflow rejects anything outside this alphabet
        if value is None or (isinstance(value, float) and value != value):
            return
        self._metrics.add(key)
        mlflow.log_metric(key, float(value), step=step)

    def metrics(self, values: Mapping[str, float], step: int | None = None) -> None:
        """Several metrics in one request.

        One POST per key is what this avoids. The tracking server runs a single
        worker on a Jetson, so a dict of thirty keys logged one at a time is
        thirty serial round trips at the end of a run.
        """
        clean: dict[str, float] = {}
        for key, value in values.items():
            if value is None or (isinstance(value, float) and value != value):
                continue  # NaN and None are not measurements
            safe = _METRIC_KEY_RE.sub("_", str(key))
            clean[safe] = float(value)
            self._metrics.add(safe)
        if clean:
            mlflow.log_metrics(clean, step=step)

    def score(self, value: float) -> None:
        """The one representative number this run is ranked by."""
        self.metric("score", value)

    def by_scenario(self, scores: Mapping[str, Any]) -> None:
        """Scene breakdown -- a comparison axis, so it belongs here and not only in parquet.

        Keys must name a *category*, not an item. One key per clip id turns the
        metric namespace into a primary-key index: it cannot be grouped or
        sorted, and the key set differs between runs so the experiment's union
        grows without bound. Per-item values belong in the run's parquet.

        Each bucket takes a scalar, a ``(score, count)`` pair, or a mapping of
        several numbers -- ``{"score": .., "count": .., "mean_ade": ..}`` -- for
        when one number per bucket is not enough to say what happened there.
        Keep the mapping small: it multiplies by the number of buckets.
        """
        batch: dict[str, float] = {}
        for name, value in scores.items():
            key = _METRIC_KEY_RE.sub("_", str(name))
            if isinstance(value, Mapping):
                for sub, number in value.items():
                    if number is not None:
                        batch[f"scenario.{key}.{_METRIC_KEY_RE.sub('_', str(sub))}"] = number
                continue
            # (score, count) is the useful form -- a mean without its sample
            # size cannot be weighted or trusted. Anything else of length != 2
            # would previously reach float(list) and raise inside the run.
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    print(f"[ml_platform] SKIP scenario {name!r}: expected (score, count), "
                          f"got {len(value)} values", file=sys.stderr)
                    continue
                if value[0] is not None:
                    batch[f"scenario.{key}.score"] = value[0]
                if value[1] is not None:
                    batch[f"scenario.{key}.count"] = value[1]
            elif value is not None:
                batch[f"scenario.{key}.score"] = value
        self.metrics(batch)

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
        # conditioning_source is deliberately not required here. It is a useful
        # tag and a real comparison axis for trajectory models, but it is not
        # one of the recording rules' required coordinates, and a helper that
        # every project copies cannot fail a run over a field that project does
        # not have the concept of. Pass it where it means something.
        for coord in dict.fromkeys(self.unresolved):
            problems.append(f"unresolved revision (not a 40-char sha): {coord}")
        return problems


@contextmanager
def _run(
    experiment: str,
    run_type: str,
    run_name: str | None = None,
    params: Mapping[str, Any] | str | Path | None = None,
    hf_datasets: Iterable[str] = (),
    hf_models: Iterable[str] = (),
    model: str | None = None,
    split: str | None = None,
    challenge: str | None = None,
    variant: str | None = None,
    conditioning_source: str | None = None,
    project: str | None = None,
    parent_run: str | None = None,
    root: str | Path = ".",
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
        if conditioning_source:
            run.tag("conditioning_source", conditioning_source)
        for key, value in git_tags(root).items():
            run.tag(key, value)
        for key, value in env_tags(seed=seed, notes=notes).items():
            run.tag(key, value)
        # The project this belongs to. The hub can also infer it from the
        # experiment name or a repo coordinate, but both are guesses that break
        # when something is renamed; an explicit tag outranks them.
        if project:
            run.tag("project", project)
        # A sweep's runs are siblings. Without this the parent is just another
        # row and the relationship is only in the run names.
        if parent_run:
            run.tag("mlflow.parentRunId", parent_run)
        if variant:
            run.tag("variant", variant)
        if split:
            run.tag("eval_split", split)
        if challenge:
            run.tag("eval_challenge", challenge)
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
