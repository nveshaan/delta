"""MLflow run bookkeeping shared by ``few_shot.py`` and ``distill_mlp.py``.

Every pipeline job produces one root run (``tags.level = 'root'``) per
support size and modality. The root run carries the full resolved config as params, the
identity/provenance tags, the dataset-level metrics, and the artifacts. Each
``label_0_vs_k`` comparison is a nested child run (``tags.level =
'comparison'``) that copies the root's identity tags.

Runs are deduplicated by ``run_hash``: a job whose exact config already has a
finished root run of the same type prints that run and returns its stored
objective, unless ``redo`` is set, in which case the old run and its children
are soft-deleted first.

A job given ``support_sizes`` and/or ``modalities`` lists runs every
combination and returns their equally weighted mean, which it also logs on
each of those runs as ``trial_objective``.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from utils.evaluation import POOLED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODALITY_CONFIG_DIR = PROJECT_ROOT / "configs" / "modality"

# Config keys that determine the pseudolabels. few_shot.yaml defines exactly
# these (plus bookkeeping keys); distill_mlp.yaml inherits them, so hashing
# this subset of a distill config reproduces the few-shot run_hash.
PSEUDOLABEL_KEYS = ("encoder", "split", "support_size", "seed", "device", "modality", "method")
# Bookkeeping keys that never change results.
_UNHASHED_KEYS = ("mlflow", "campaign", "redo", "modalities", "support_sizes")
# Keys a multi-combination job varies; trial_hash ignores them.
_COMBINATION_KEYS = ("modality", "support_size")
# Root tags that describe one job invocation rather than the config.
_JOB_TAGS = ("git_commit", "git_dirty", "hydra_overrides", "hydra_job_num")

logger = logging.getLogger(__name__)


def flatten_params(value: Any, prefix: str = "") -> dict[str, str | int | float | bool]:
    """Flatten an OmegaConf container into MLflow-compatible parameters."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        result: dict[str, str | int | float | bool] = {}
        for key, item in value.items():
            raw_key = f"{prefix}.{key}" if prefix else str(key)
            # MLflow parameter names do not allow apostrophes or other
            # punctuation present in real dataset names. The original names
            # remain unchanged in the YAML and dataset parameters.
            safe_key = re.sub(r"[^A-Za-z0-9_. /:-]", "_", raw_key)
            result.update(flatten_params(item, safe_key))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value)}
    if value is None:
        return {prefix: "null"}
    if isinstance(value, (str, int, float, bool)):
        return {prefix: value}
    return {prefix: str(value)}


def _hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _drop_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return {name: _drop_key(item, key) for name, item in value.items() if name != key}
    if isinstance(value, list):
        return [_drop_key(item, key) for item in value]
    return value


def config_hashes(container: dict[str, Any]) -> dict[str, str]:
    """Hashes of a resolved config with bookkeeping keys removed.

    ``run_hash`` identifies the exact config (dedup and cache key),
    ``config_hash`` ignores every ``seed`` key so replicate seeds share it, and
    ``trial_hash`` ignores ``modality`` and ``support_size`` so all runs of one
    multi-combination job (e.g. one Optuna trial) share it.
    """
    base = {key: value for key, value in container.items() if key not in _UNHASHED_KEYS}
    return {
        "run_hash": _hash(base),
        "config_hash": _hash(_drop_key(base, "seed")),
        "trial_hash": _hash({key: value for key, value in base.items() if key not in _COMBINATION_KEYS}),
    }


def pseudolabel_hash(container: dict[str, Any]) -> str:
    """The few-shot ``run_hash`` of the pseudolabels a config consumes."""
    return config_hashes({key: container[key] for key in PSEUDOLABEL_KEYS})["run_hash"]


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info() -> dict[str, str]:
    """Commit, dirty flag (tracked files only), and the diff when dirty."""
    commit = (_git("rev-parse", "HEAD") or "unknown").strip()
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return {"commit": commit, "dirty": "unknown", "diff": ""}
    dirty = bool(status.strip())
    return {"commit": commit, "dirty": str(dirty).lower(), "diff": (_git("diff", "HEAD") or "") if dirty else ""}


def setup_mlflow(cfg: DictConfig):
    """Point MLflow at the configured store and return ``(mlflow, client, experiment_id)``.

    Parallel jobs may race to create the experiment; the loser re-reads it.
    """
    try:
        import mlflow
        from mlflow.exceptions import MlflowException
    except ImportError as error:
        raise RuntimeError("MLflow is required. Install the project dependencies first.") from error

    tracking_uri = str(cfg.mlflow.tracking_uri)
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        tracking_uri = "sqlite:///" + str((PROJECT_ROOT / tracking_uri.removeprefix("sqlite:///")).resolve())
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment_name = str(cfg.mlflow.experiment_name)
    if client.get_experiment_by_name(experiment_name) is None:
        artifact_location = str((PROJECT_ROOT / str(cfg.mlflow.artifact_location)).resolve())
        try:
            client.create_experiment(experiment_name, artifact_location=artifact_location)
        except MlflowException:
            if client.get_experiment_by_name(experiment_name) is None:
                raise
    experiment = mlflow.set_experiment(experiment_name)
    return mlflow, client, experiment.experiment_id


def find_finished(client, experiment_id: str, run_type: str, run_hash: str) -> list:
    """Finished root runs of ``run_type`` with this exact ``run_hash``, newest first."""
    return list(client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.level = 'root' and tags.type = '{run_type}' "
            f"and tags.run_hash = '{run_hash}' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
    ))


def delete_runs(client, experiment_id: str, runs: list) -> None:
    """Soft-delete root runs and their nested children."""
    for run in runs:
        children = client.search_runs(
            [experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'", max_results=10000
        )
        for child in children:
            client.delete_run(child.info.run_id)
        client.delete_run(run.info.run_id)


def dedup_or_redo(client, experiment_id: str, run_type: str, run_hash: str, redo: bool):
    """Return an identical finished run to reuse, or ``None`` to proceed.

    With ``redo`` the identical runs are soft-deleted and the job proceeds.
    """
    existing = find_finished(client, experiment_id, run_type, run_hash)
    if not existing:
        return None
    if not redo:
        run = existing[0]
        objective = float(run.data.metrics.get("objective", float("nan")))
        started = datetime.fromtimestamp(run.info.start_time / 1000).isoformat(sep=" ", timespec="seconds")
        print(
            f"Identical {run_type} run already exists: {run.info.run_id} "
            f"({run.info.run_name}, {started}, objective={objective:.4f}). "
            "Pass --redo to delete it and rerun.",
            flush=True,
        )
        return run
    for run in existing:
        print(f"--redo: deleting {run_type} run {run.info.run_id} ({run.info.run_name}) and its child runs", flush=True)
    delete_runs(client, experiment_id, existing)
    return None


def root_tags(container: dict[str, Any], hashes: dict[str, str], identity: dict[str, Any]) -> dict[str, str]:
    """Tags for a root run: identity, hashes, campaign, and job provenance."""
    git = git_info()
    hydra_config = HydraConfig.get()
    tags = {
        "level": "root",
        "type": str(container["mlflow"]["type"]),
        "campaign": str(container.get("campaign") or "adhoc"),
        **hashes,
        **{key: str(value) for key, value in identity.items()},
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "hydra_overrides": ", ".join(hydra_config.overrides.task)[:5000],
    }
    if hydra_config.mode == RunMode.MULTIRUN:
        tags["hydra_job_num"] = str(hydra_config.job.num)
    return tags


@contextmanager
def root_run(mlflow, container: dict[str, Any], tags: dict[str, str]) -> Iterator[Any]:
    """Start the root run and log config/provenance before any work runs.

    A failure inside the block is recorded as an ``error`` tag plus
    ``traceback.txt`` before MLflow marks the run FAILED.
    """
    with mlflow.start_run(run_name=str(container["mlflow"]["run_name"])) as run:
        try:
            mlflow.set_tags(tags)
            mlflow.log_params(flatten_params(container))
            if tags.get("git_dirty") == "true":
                diff = git_info()["diff"]
                if diff:
                    mlflow.log_text(diff, "git_diff.patch")
            yield run
        except Exception as error:
            try:
                mlflow.set_tag("error", f"{type(error).__name__}: {error}"[:5000])
                mlflow.log_text(traceback.format_exc(), "traceback.txt")
            except Exception:
                logger.exception("Could not record the failure on MLflow run %s", run.info.run_id)
            raise


def log_comparison_runs(mlflow, rows: list[dict[str, Any]], tags: dict[str, str]) -> None:
    """Log each ``label_0_vs_k`` row as a nested child of the active root run.

    ``pooled_nonzero`` rows are already on the root run as dataset-level
    metrics, so they are not repeated as children.
    """
    child_tags = {key: value for key, value in tags.items() if key not in _JOB_TAGS}
    for row in rows:
        if row["comparison"] == POOLED:
            continue
        with mlflow.start_run(run_name=f"{row['dataset']}_{row['comparison']}", nested=True):
            mlflow.set_tags({
                **child_tags,
                "level": "comparison",
                "dataset": str(row["dataset"]),
                "comparison": str(row["comparison"]),
            })
            mlflow.log_metrics({
                "auroc": float(row["auroc"]),
                "auprc": float(row["auprc"]),
                "p_at_n": float(row["precision_at_n"]),
                "n_normal": float(row["n_normal"]),
                "n_anomaly": float(row["n_anomaly"]),
            })


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["dataset", "comparison"])
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def staging_dir(prefix: str) -> Iterator[Path]:
    """A temporary artifact staging directory under ``experiments/``."""
    root = PROJECT_ROOT / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{prefix}_", dir=root) as path:
        yield Path(path)


def upload_artifacts(mlflow, client, run_id: str, run_dir: Path, required: list[str]) -> None:
    """Upload ``run_dir`` and verify every required file exists locally and remotely."""
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Artifact generation incomplete; missing: {missing}")
    mlflow.log_artifacts(str(run_dir))
    uploaded = {artifact.path for artifact in client.list_artifacts(run_id)}
    missing_remote = [name for name in required if name not in uploaded]
    if missing_remote:
        raise RuntimeError(f"MLflow artifact upload incomplete; missing: {missing_remote}")


def _combination_configs(cfg: DictConfig) -> list[DictConfig]:
    """One config per (support size, modality) of ``support_sizes`` x ``modalities``.

    A null list means the single ``support_size`` / ``modality`` of ``cfg``.
    """
    names = [str(name) for name in cfg.modalities] if cfg.get("modalities") is not None else [None]
    unknown = [name for name in names if name is not None and not (MODALITY_CONFIG_DIR / f"{name}.yaml").is_file()]
    if unknown:
        available = sorted(path.stem for path in MODALITY_CONFIG_DIR.glob("*.yaml"))
        raise ValueError(f"Unknown modalities {unknown}; available: {available}")
    sizes = [int(size) for size in cfg.support_sizes] if cfg.get("support_sizes") is not None else [None]
    configs = []
    for size in sizes:
        for name in names:
            combination_cfg = copy.deepcopy(cfg)
            if size is not None:
                OmegaConf.update(combination_cfg, "support_size", size)
            if name is not None:
                # Replace (not merge) so no dataset keys leak from the default modality.
                OmegaConf.update(
                    combination_cfg, "modality", OmegaConf.load(MODALITY_CONFIG_DIR / f"{name}.yaml"), merge=False
                )
            configs.append(combination_cfg)
    return configs


def for_each_combination(cfg: DictConfig, execute_one: Callable[[DictConfig], tuple[float, str]]) -> float:
    """Run every (support size, modality) combination and return their balanced mean.

    ``execute_one`` returns ``(objective, run_id)``. Each run's objective is the
    mean of its dataset-level AUROCs, so averaging over modalities (and then
    support sizes) weights each modality equally however many datasets it has.
    With several combinations the mean is logged on each run as
    ``trial_objective``.
    """
    configs = _combination_configs(cfg)
    if len(configs) == 1:
        return execute_one(configs[0])[0]
    results: dict[int, dict[str, float]] = {}
    run_ids = []
    for combination_cfg in configs:
        objective, run_id = execute_one(combination_cfg)
        results.setdefault(int(combination_cfg.support_size), {})[str(combination_cfg.modality.modality)] = objective
        run_ids.append(run_id)
    per_size = {size: sum(values.values()) / len(values) for size, values in results.items()}
    balanced = float(sum(per_size.values()) / len(per_size))
    import mlflow

    client = mlflow.tracking.MlflowClient()
    for run_id in run_ids:
        client.log_metric(run_id, "trial_objective", balanced)
    # Printed rather than logged: the Hydra configs disable INFO logging.
    for size, values in results.items():
        print(
            f"support_size={size}: " + ", ".join(f"{name}={value:.4f}" for name, value in values.items()),
            flush=True,
        )
    print(f"Balanced objective: {balanced:.4f}", flush=True)
    return balanced


def hydra_entry(
    cfg: DictConfig,
    execute_one: Callable[[DictConfig], tuple[float, str]],
    job_name: str,
    *,
    re_raise_multirun_failure: bool = False,
) -> float:
    """Shared ``main`` body: execute one job and handle failures.

    Single runs re-raise so the process exits non-zero; multirun jobs return
    NaN so ordinary Hydra sweeps can continue. Optuna sweeps can request
    re-raising: the Optuna sweeper then records the trial as ``FAIL`` rather
    than receiving NaN, which Optuna rejects as an objective value.
    """
    try:
        objective = for_each_combination(cfg, execute_one)
    except Exception as error:
        hydra_config = HydraConfig.get()
        if hydra_config.mode != RunMode.MULTIRUN:
            raise
        message = (
            f"{job_name} JOB FAILED\n"
            f"overrides=[{', '.join(hydra_config.overrides.task)}]\n"
            f"error={type(error).__name__}: {error}"
        )
        # Printed before a re-raise too: the Optuna sweeper reports a failed
        # trial only through Hydra logging, which the configs disable.
        print(message, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        logger.error(message)
        if re_raise_multirun_failure:
            raise
        return float("nan")
    if math.isnan(objective):
        logger.warning("%s objective is NaN", job_name)
    return objective
