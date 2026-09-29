"""Evaluate a distilled few-shot MLP on unseen test datasets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Hydra 1.3 passes a lazy help object to argparse. Python 3.14 requires
# argparse help text to be string-like during parser construction.
if sys.version_info >= (3, 14):
    _argparse_check_help = argparse.ArgumentParser._check_help

    def _hydra_argparse_check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return _argparse_check_help(self, action)

    argparse.ArgumentParser._check_help = _hydra_argparse_check_help

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.msde import DEFAULT_DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_device(value: str) -> str:
    return DEFAULT_DEVICE if value == "auto" else value


def _precision_at_n(target: np.ndarray, score: np.ndarray) -> float:
    count = int(target.sum())
    if count == 0:
        return float("nan")
    return float(target[np.argsort(-score)[:count]].mean())


def _pair_metrics(scores: torch.Tensor, source_labels: np.ndarray, dataset_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unique = sorted(int(label) for label in np.unique(source_labels))
    if 0 not in unique:
        return rows
    comparisons: list[int | str] = [label for label in unique if label != 0]
    if len(unique) > 2:
        comparisons.append("pooled")
    for anomaly_label in comparisons:
        if anomaly_label == "pooled":
            keep = source_labels != 0
            keep |= source_labels == 0
            label_text = "pooled_nonzero"
            target = (source_labels[keep] != 0).astype(int)
        else:
            keep = (source_labels == 0) | (source_labels == anomaly_label)
            label_text = f"label_0_vs_{anomaly_label}"
            target = (source_labels[keep] == anomaly_label).astype(int)
        if len(np.unique(target)) < 2:
            continue
        values = scores.detach().cpu().numpy()[keep]
        rows.append({
            "dataset": dataset_name,
            "comparison": label_text,
            "normal_label": 0,
            "anomaly_label": label_text if anomaly_label == "pooled" else int(anomaly_label),
            "auroc": float(roc_auc_score(target, values)),
            "auprc": float(average_precision_score(target, values)),
            "precision_at_n": _precision_at_n(target, values),
            "n_normal": int((target == 0).sum()),
            "n_anomaly": int((target == 1).sum()),
        })
    return rows


def _flatten_params(value: Any, prefix: str = "") -> dict[str, str | int | float | bool]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        result: dict[str, str | int | float | bool] = {}
        for key, item in value.items():
            raw_key = f"{prefix}.{key}" if prefix else str(key)
            safe_key = re.sub(r"[^A-Za-z0-9_. /:-]", "_", raw_key)
            result.update(_flatten_params(item, safe_key))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value)}
    if value is None:
        return {prefix: "null"}
    if isinstance(value, (str, int, float, bool)):
        return {prefix: value}
    return {prefix: str(value)}


def _safe_mlflow_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_. /:-]", "_", value)


def _find_checkpoint(mlflow, cfg: DictConfig, method_name: str, modality: str) -> tuple[str, str]:
    filter_string = (
        "params.type = 'few_shot' "
        f"and params.modality = '{modality}' "
        f"and params.encoder = '{cfg.encoder}' "
        f"and params.method = '{method_name}' "
        f"and params.support_size = '{int(cfg.support_size)}'"
    )
    runs = mlflow.search_runs(
        experiment_names=[str(cfg.mlflow.experiment_name)],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        output_format="pandas",
    )
    if runs.empty:
        raise FileNotFoundError(
            "No matching few-shot checkpoint found for "
            f"type=few_shot modality={modality} encoder={cfg.encoder} "
            f"method={method_name} support_size={cfg.support_size}"
        )
    artifact_run_id = str(runs.iloc[0]["params.run_id"])
    checkpoint = mlflow.artifacts.download_artifacts(run_id=artifact_run_id, artifact_path="model.pt")
    return artifact_run_id, checkpoint


def _write_predictions(path: Path, dataset, scores: torch.Tensor, source_labels: np.ndarray, binary: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "dataset", "global_label", "source_label", "binary_label", "prediction"])
        for index, (name, global_label, source_label, binary_label, score) in enumerate(
            zip(dataset.sample_dataset_names, dataset.labels, source_labels, binary, scores.tolist())
        ):
            writer.writerow([index, name, int(global_label), int(source_label), int(binary_label), float(score)])


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _log_metric_runs(
    mlflow,
    rows: list[dict[str, Any]],
    *,
    modality: str,
    method: str,
    encoder: str,
    support_size: int,
    run_type: str,
    mode: str,
    artifact_run_id: str,
) -> None:
    for row in rows:
        for source_name, metric_name in (("auroc", "auroc"), ("auprc", "auprc"), ("precision_at_n", "p@n")):
            with mlflow.start_run(run_name=f"{run_type}_{row['dataset']}_{row['comparison']}_{metric_name}"):
                mlflow.log_params({
                    "name": metric_name,
                    "modality": modality,
                    "dataset": str(row["dataset"]),
                    "method": method,
                    "encoder": encoder,
                    "support_size": support_size,
                    "type": run_type,
                    "comparison": str(row["comparison"]),
                    "mode": mode,
                    "run_id": artifact_run_id,
                })
                mlflow.log_metric("value", float(row[source_name]))


@hydra.main(version_base=None, config_path="../configs", config_name="zero_shot")
def main(cfg: DictConfig) -> None:
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("MLflow is required. Install the project dependencies first.") from error

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    hydra_config = HydraConfig.get()
    method_name = str(hydra_config.runtime.choices["method"]).lower()
    modality_name = str(cfg.modality.modality)
    device = _resolve_device(str(cfg.device))
    logger.info(
        "Starting zero-shot evaluation: modality=%s encoder=%s method=%s support_size=%s",
        modality_name, cfg.encoder, method_name, cfg.support_size,
    )

    tracking_uri = str(cfg.mlflow.tracking_uri)
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        tracking_uri = "sqlite:///" + str((PROJECT_ROOT / tracking_uri.removeprefix("sqlite:///" )).resolve())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))

    run_id, checkpoint_path = _find_checkpoint(mlflow, cfg, method_name, modality_name)
    logger.info("Using few-shot checkpoint %s", run_id)
    dataset = instantiate(cfg.modality)
    embeddings, _ = dataset.get_data()
    if len(dataset) == 0:
        raise ValueError("The selected test dataset is empty")

    model = instantiate(cfg.mlp).to(device)
    values = embeddings.to(device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        model(values[:1])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        scores = model(values).detach().cpu()

    source_labels = np.asarray([
        int(dataset.label_metadata[int(label)]["source_label"]) for label in dataset.labels
    ])
    binary_labels = (source_labels != 0).astype(int)
    rows: list[dict[str, Any]] = []
    for dataset_name in dataset.dataset_names:
        mask = np.asarray([name == dataset_name for name in dataset.sample_dataset_names])
        rows.extend(_pair_metrics(scores[mask], source_labels[mask], dataset_name))

    with mlflow.start_run(
        run_name=str(cfg.mlflow.run_name),
        tags={"type": str(cfg.mlflow.type), "modality": modality_name, "method": method_name},
    ) as root_run:
        staging_root = PROJECT_ROOT / "experiments"
        staging_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f"{cfg.mlflow.type}_", dir=staging_root))
        try:
            OmegaConf.save(cfg, run_dir / "zero_shot.yaml", resolve=True)
            (run_dir / "source_few_shot_run.json").write_text(json.dumps({"run_id": run_id}, indent=2))
            _write_predictions(run_dir / "predictions.csv", dataset, scores, source_labels, binary_labels)
            with (run_dir / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["dataset", "comparison"])
                writer.writeheader()
                writer.writerows(rows)
            mlflow.log_artifacts(str(run_dir))
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        artifact_run_id = root_run.info.run_id
    _log_metric_runs(
        mlflow,
        rows,
        modality=modality_name,
        method=method_name,
        encoder=str(cfg.encoder),
        support_size=int(cfg.support_size),
        run_type=str(cfg.mlflow.type),
        mode=str(cfg.method.get("mode") or "none"),
        artifact_run_id=artifact_run_id,
    )
    print(f"MLflow run: {artifact_run_id}\nArtifacts stored in MLflow artifact store", flush=True)


if __name__ == "__main__":
    main()
