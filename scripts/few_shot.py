"""Hydra/MLflow few-shot DELTA training pipeline.

Pipeline:

    embedding dataset -> binary support/query pseudolabels
    -> optional MSDE + GDE scores -> optional VanillaNetwork distillation

All pseudolabeling, MSDE, and GDE calls receive the complete tensor dataset;
there is no minibatch slicing in those stages.
"""

from __future__ import annotations

import csv
import argparse
import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# Hydra 1.3 passes a lazy help object to argparse. Python 3.14 tightened
# argparse's help validation to require a string, so normalize that object
# only during parser validation. This keeps Hydra's CLI/config behavior intact.
if sys.version_info >= (3, 14):
    _argparse_check_help = argparse.ArgumentParser._check_help

    def _hydra_argparse_check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return _argparse_check_help(self, action)

    argparse.ArgumentParser._check_help = _hydra_argparse_check_help

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.msde import DEFAULT_DEVICE, MeanShiftDensityEnhancement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    return DEFAULT_DEVICE if device == "auto" else device


def _flatten_params(value: Any, prefix: str = "") -> dict[str, str | int | float | bool]:
    """Flatten an OmegaConf container into MLflow-compatible parameters."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        result: dict[str, str | int | float | bool] = {}
        for key, item in value.items():
            result.update(_flatten_params(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value)}
    if value is None:
        return {prefix: "null"}
    if isinstance(value, (str, int, float, bool)):
        return {prefix: value}
    return {prefix: str(value)}


def _binary_labels(dataset) -> tuple[torch.Tensor, list[int]]:
    """Map each dataset's source normal label to 0 and all anomalies to 1."""
    normal_global_labels = {
        int(global_label)
        for global_label, metadata in dataset.label_metadata.items()
        if int(metadata["source_label"]) == 0
    }
    global_labels = torch.as_tensor(dataset.labels, dtype=torch.long)
    binary = torch.tensor(
        [0 if int(label) in normal_global_labels else 1 for label in global_labels],
        dtype=torch.long,
    )
    return binary, sorted(normal_global_labels)


def _sample_support(binary_labels: torch.Tensor, support_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if support_size <= 0:
        raise ValueError("support_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    support_parts = []
    for class_id in (0, 1):
        indices = torch.where(binary_labels == class_id)[0]
        if len(indices) == 0:
            raise ValueError(f"No samples available for binary class {class_id}")
        count = min(support_size, len(indices))
        order = torch.randperm(len(indices), generator=generator)[:count]
        support_parts.append(indices[order])
    support = torch.cat(support_parts).sort().values
    mask = torch.ones(len(binary_labels), dtype=torch.bool)
    mask[support] = False
    return support, torch.where(mask)[0]


class GDEScorer:
    """Gaussian density estimator used after MSDE normal shifting."""

    regularization = 1e-4

    def __init__(self) -> None:
        self.pca: PCA | None = None
        self.mean: np.ndarray | None = None
        self.covariance_inverse: np.ndarray | None = None

    def fit(self, embeddings: torch.Tensor) -> "GDEScorer":
        values = embeddings.detach().cpu().numpy().astype(np.float32)
        if len(values) < 2:
            raise ValueError("GDE requires at least two normal samples")
        dimension = min(256, values.shape[1], len(values) - 1)
        self.pca = PCA(n_components=dimension, random_state=42)
        projected = self.pca.fit_transform(values)
        self.mean = projected.mean(axis=0)
        centered = projected - self.mean
        covariance = (centered.T @ centered) / max(len(centered) - 1, 1)
        covariance += np.eye(dimension) * self.regularization
        self.covariance_inverse = np.linalg.inv(covariance)
        return self

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.pca is None or self.mean is None or self.covariance_inverse is None:
            raise RuntimeError("Fit GDE before scoring")
        values = embeddings.detach().cpu().numpy().astype(np.float32)
        projected = self.pca.transform(values) - self.mean
        scores = np.einsum("ij,jk,ik->i", projected, self.covariance_inverse, projected)
        return torch.from_numpy(np.sqrt(np.clip(scores, 0, None)).astype(np.float32))


def _run_msde_gde(embeddings: torch.Tensor, binary_labels: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Shift normals, fit GDE, then shift and score the complete dataset."""
    device_name = _resolve_device(str(cfg.device))
    values = embeddings.to(device=device_name, dtype=torch.float32)
    normal_embeddings = values[binary_labels.to(device=device_name) == 0]
    if len(normal_embeddings) < 2:
        raise ValueError("MSDE + GDE requires at least two normal samples")
    msde_kwargs = OmegaConf.to_container(cfg, resolve=True)
    msde_kwargs["device"] = device_name
    msde_kwargs.pop("label_aware_joint_shift", None)
    msde_kwargs["k"] = min(int(msde_kwargs["k"]), len(values) - 1)
    normal_msde = MeanShiftDensityEnhancement(**msde_kwargs)
    shifted_normals, _, _ = normal_msde(normal_embeddings)
    gde = GDEScorer().fit(shifted_normals)

    joint_msde = MeanShiftDensityEnhancement(**msde_kwargs)
    joint_labels = binary_labels.to(device=device_name) if bool(cfg.label_aware_joint_shift) else None
    shifted_all, _, _ = joint_msde(values, labels=joint_labels)
    return gde.score(shifted_all)


def _train_mlp(embeddings: torch.Tensor, targets: torch.Tensor, model_cfg: DictConfig,
               train_cfg: DictConfig, device: str) -> tuple[torch.nn.Module, list[dict[str, float]], torch.Tensor]:
    model = instantiate(model_cfg).to(device)
    values = embeddings.to(device=device, dtype=torch.float32)
    target_values = targets.to(device=device, dtype=torch.float32)
    indices = np.arange(len(values))
    labels_for_split = (target_values.detach().cpu().numpy() > np.median(target_values.detach().cpu().numpy())).astype(int)
    try:
        train_idx, validation_idx = train_test_split(
            indices, test_size=float(train_cfg.validation_fraction),
            random_state=int(train_cfg.seed), stratify=labels_for_split,
        )
    except ValueError:
        train_idx, validation_idx = train_test_split(
            indices, test_size=float(train_cfg.validation_fraction), random_state=int(train_cfg.seed)
        )
    train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    validation_idx = torch.as_tensor(validation_idx, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg.learning_rate), weight_decay=float(train_cfg.weight_decay)
    )
    best_loss = float("inf")
    best_state = None
    patience = 0
    losses: list[dict[str, float]] = []
    for epoch in tqdm(range(int(train_cfg.epochs)), desc="MLP distillation", unit="epoch"):
        model.train()
        prediction = model(values[train_idx])
        loss = torch.nn.functional.mse_loss(prediction, target_values[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = torch.nn.functional.mse_loss(
                model(values[validation_idx]), target_values[validation_idx]
            ).item()
        losses.append({"epoch": float(epoch + 1), "train_loss": float(loss.item()), "validation_loss": float(validation_loss)})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(train_cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(values).detach().cpu()
    return model, losses, predictions


def _precision_at_n(target: np.ndarray, score: np.ndarray) -> float:
    count = int(target.sum())
    if count == 0:
        return float("nan")
    return float(target[np.argsort(-score)[:count]].mean())


def _pair_metrics(scores: torch.Tensor, source_labels: np.ndarray, dataset_name: str) -> list[dict[str, Any]]:
    rows = []
    unique = sorted(int(label) for label in np.unique(source_labels))
    if 0 not in unique:
        return rows
    for anomaly_label in [label for label in unique if label != 0] + (["pooled"] if len(unique) > 2 else []):
        if anomaly_label == "pooled":
            keep = (source_labels == 0) | (source_labels != 0)
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


def _save_loss_curve(losses: list[dict[str, float]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = [item["epoch"] for item in losses]
    axis.plot(epochs, [item["train_loss"] for item in losses], label="train")
    axis.plot(epochs, [item["validation_loss"] for item in losses], label="validation")
    axis.set(xlabel="epoch", ylabel="MSE", title="MLP distillation loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _save_losses(losses: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(losses)


def _write_predictions(path: Path, dataset, binary_labels: torch.Tensor, pseudo_labels: torch.Tensor,
                       confidence: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor | None) -> None:
    source_labels = [int(dataset.label_metadata[int(label)]["source_label"]) for label in dataset.labels]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "dataset", "global_label", "source_label", "binary_label", "pseudolabel", "confidence", "target", "prediction"])
        for index, (dataset_name, global_label, source_label, binary_label) in enumerate(
            zip(dataset.sample_dataset_names, dataset.labels, source_labels, binary_labels.tolist())
        ):
            writer.writerow([
                index, dataset_name, int(global_label), source_label, int(binary_label),
                int(pseudo_labels[index]), float(confidence[index]), float(targets[index]),
                "" if predictions is None else float(predictions[index]),
            ])


@hydra.main(version_base=None, config_path="../configs", config_name="few_shot")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    logger.info("Starting few-shot pipeline: modality=%s method=%s split=%s", cfg.modality.modality, HydraConfig.get().runtime.choices["method"], cfg.modality.split)

    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("MLflow is required. Install the project dependencies first.") from error

    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    root = PROJECT_ROOT
    tracking_uri = str(cfg.mlflow.tracking_uri)
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        tracking_uri = "sqlite:///" + str((root / tracking_uri.removeprefix("sqlite:///" )).resolve())
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = str(cfg.mlflow.experiment_name)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = str((root / str(cfg.mlflow.artifact_location)).resolve())
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

    dataset_cfg = cfg.modality
    dataset = instantiate(dataset_cfg)
    embeddings, global_labels = dataset.get_data()
    logger.info("Loaded %d samples from %d datasets", len(dataset), len(dataset.dataset_names))
    binary_labels, _ = _binary_labels(dataset)
    support_indices, query_indices = _sample_support(binary_labels, int(cfg.support_size), int(cfg.seed))
    method_name = str(HydraConfig.get().runtime.choices["method"]).lower()
    method_cfg = cfg.method
    pseudolabeler = instantiate(method_cfg)
    device = _resolve_device(str(cfg.msde.device))
    if method_name in {"laplacianshot_msde", "knnvote_msde"}:
        logger.info("Generating pseudolabels with %s", method_name)
        working_embeddings, pseudo_labels, confidence, _ = pseudolabeler(
            embeddings.to(device), binary_labels.to(device), support_indices.to(device), query_indices.to(device)
        )
    else:
        support_normals = embeddings[support_indices][binary_labels[support_indices] == 0].to(device)
        support_anomalies = embeddings[support_indices][binary_labels[support_indices] == 1].to(device)
        pseudo_labels, confidence = pseudolabeler(
            support_normals, support_anomalies, embeddings[query_indices].to(device)
        )
        working_embeddings = embeddings.to(device)
    full_pseudo_labels = binary_labels.to(device).clone()
    full_pseudo_labels[query_indices.to(device)] = pseudo_labels
    full_confidence = torch.ones(len(dataset), device=device)
    full_confidence[query_indices.to(device)] = confidence

    if bool(cfg.apply_msde_gde):
        logger.info("Running MSDE + GDE scoring")
        scores = _run_msde_gde(working_embeddings, full_pseudo_labels, cfg.msde)
        score_target = scores.to(device)
    else:
        scores = full_pseudo_labels.float().cpu()
        score_target = scores.to(device)

    model = None
    losses: list[dict[str, float]] = []
    predictions = None
    if bool(cfg.distill_mlp):
        logger.info("Distilling scores into MLP")
        model, losses, predictions = _train_mlp(working_embeddings, score_target, cfg.mlp, cfg.training, device)
        final_scores = predictions
    else:
        final_scores = scores.detach().cpu()

    rows = []
    source_labels_by_sample = np.asarray([
        int(dataset.label_metadata[int(label)]["source_label"]) for label in dataset.labels
    ])
    for dataset_name in dataset.dataset_names:
        sample_mask = np.asarray([name == dataset_name for name in dataset.sample_dataset_names])
        local_labels = source_labels_by_sample[sample_mask]
        # Evaluation is deliberately based on binary pseudolabels, not on
        # post-MSDE scores or MLP predictions. Every non-zero source label has
        # prediction 1 in this temporary binary evaluation space.
        local_scores = full_pseudo_labels.detach().cpu().float()[sample_mask]
        rows.extend(_pair_metrics(local_scores, local_labels, dataset_name))

    with mlflow.start_run(run_name=str(cfg.mlflow.run_name), tags={"type": str(cfg.mlflow.type)}) as root_run:
        run_dir = root / str(cfg.output_dir) / root_run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, run_dir / "few_shot.yaml", resolve=True)
        OmegaConf.save(dataset_cfg, run_dir / "dataset.yaml", resolve=True)
        OmegaConf.save(method_cfg, run_dir / "pseudolabeler.yaml", resolve=True)
        (run_dir / "label_mapping.json").write_text(
            json.dumps({str(key): value for key, value in dataset.label_metadata.items()}, indent=2)
        )
        mlflow.log_params(_flatten_params(cfg))
        commit = _git_commit()
        mlflow.set_tag("git_commit", commit)
        mlflow.log_param("git_commit", commit)
        with mlflow.start_run(run_name="hyperparams", nested=True):
            mlflow.log_params(_flatten_params(cfg))
            with mlflow.start_run(run_name=str(cfg.modality), nested=True):
                mlflow.log_param("modality", str(cfg.modality))
                for dataset_name in dataset.dataset_names:
                    with mlflow.start_run(run_name=dataset_name, nested=True):
                        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
                        for row in dataset_rows:
                            prefix = row["comparison"].replace("/", "_")
                            for metric in ("auroc", "auprc", "precision_at_n"):
                                mlflow.log_metric(f"{prefix}.{metric}", row[metric])
                        mlflow.log_metric("n_samples", sum(name == dataset_name for name in dataset.sample_dataset_names))
        prediction_path = run_dir / "predictions.csv"
        _write_predictions(prediction_path, dataset, binary_labels, full_pseudo_labels.cpu(), full_confidence.cpu(), score_target.cpu(), predictions)
        metrics_path = run_dir / "metrics.csv"
        with metrics_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["dataset", "comparison"])
            writer.writeheader()
            writer.writerows(rows)
        if losses:
            _save_loss_curve(losses, run_dir / "loss_curve.png")
            _save_losses(losses, run_dir / "losses.csv")
            torch.save(model.state_dict(), run_dir / "model.pt")
        mlflow.log_artifacts(str(run_dir))
        mlflow.log_metric("n_samples", len(dataset))
        mlflow.log_metric("n_support", len(support_indices))
        mlflow.log_metric("n_query", len(query_indices))
        print(f"MLflow run: {root_run.info.run_id}\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
