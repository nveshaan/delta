"""Hydra/MLflow DELTA scoring and MLP distillation stage.

Pipeline:

    few-shot pseudolabels -> MLP targets -> optional VanillaNetwork distillation

The targets are the pseudolabels (``labels``), MSDE (``same_label``) + GDE
scores (``scores``), or MSDE (``zero_label``) total distances (``distances``).

The pseudolabels come from the finished ``few_shot.py`` run whose ``run_hash``
matches this config's pseudolabel keys; when there is none, they are generated
(and logged as a few-shot run) first. All MSDE and GDE calls receive the
complete tensor dataset; there is no minibatch slicing. Metrics are computed on
the final scores (MLP predictions when distilling) of the train split.
"""

from __future__ import annotations

import csv
import logging
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hydra_compat import consume_redo_flag  # patches argparse before Hydra builds its parser

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from methods.msde import DEFAULT_DEVICE, MeanShiftDensityEnhancement
from scripts.few_shot import generate_pseudolabels
from utils.evaluation import _pair_metrics, aggregate
from utils.tracking import (
    PSEUDOLABEL_KEYS,
    config_hashes,
    dedup_or_redo,
    find_finished,
    hydra_entry,
    log_comparison_runs,
    pseudolabel_hash,
    root_run,
    root_tags,
    setup_mlflow,
    staging_dir,
    upload_artifacts,
    write_metrics_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PSEUDOLABEL_ARTIFACT = "pseudolabels.pt"
# MSDE label_mode used by each MLP target ("none": MSDE is not used).
MSDE_MODES = {"labels": "none", "scores": "same_label", "distances": "zero_label"}


def _resolve_device(device: str) -> str:
    return DEFAULT_DEVICE if device == "auto" else device


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


def _msde_kwargs(cfg: DictConfig, n_samples: int, device_name: str, label_mode: str) -> dict[str, Any]:
    """MeanShiftDensityEnhancement kwargs from ``cfg`` with the given ``label_mode``."""
    msde_kwargs = OmegaConf.to_container(cfg, resolve=True)
    msde_kwargs["label_mode"] = label_mode
    msde_kwargs["device"] = device_name
    msde_kwargs["k"] = min(int(msde_kwargs["k"]), n_samples - 1)
    return msde_kwargs


def _run_msde_distances(
    embeddings: torch.Tensor,
    binary_labels: torch.Tensor,
    cfg: DictConfig,
    device_name: str,
) -> torch.Tensor:
    """Shift the complete dataset with ``zero_label`` MSDE and return each sample's total_distance."""
    values = embeddings.to(device=device_name, dtype=torch.float32)
    msde_kwargs = _msde_kwargs(cfg, len(values), device_name, MSDE_MODES["distances"])
    _, total_distance, _ = MeanShiftDensityEnhancement(**msde_kwargs)(values, labels=binary_labels.to(device=device_name))
    return total_distance.detach().float().cpu()


def _run_msde_gde(
    embeddings: torch.Tensor,
    binary_labels: torch.Tensor,
    cfg: DictConfig,
    device_name: str,
) -> torch.Tensor:
    """Shift normals, fit GDE, then shift and score the complete dataset.

    The joint shift passes the binary labels to ``same_label`` MSDE.
    """
    values = embeddings.to(device=device_name, dtype=torch.float32)
    normal_embeddings = values[binary_labels.to(device=device_name) == 0]
    if len(normal_embeddings) < 2:
        raise ValueError("MSDE + GDE requires at least two normal samples")
    msde_kwargs = _msde_kwargs(cfg, len(values), device_name, MSDE_MODES["scores"])
    normal_msde = MeanShiftDensityEnhancement(**msde_kwargs)
    shifted_normals, _, _ = normal_msde(normal_embeddings)
    gde = GDEScorer().fit(shifted_normals)

    joint_msde = MeanShiftDensityEnhancement(**msde_kwargs)
    shifted_all, _, _ = joint_msde(values, labels=binary_labels.to(device=device_name))
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


def _save_losses(losses: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(losses)


def _write_predictions(path: Path, dataset, binary_labels: torch.Tensor, pseudo_labels: torch.Tensor,
                       targets: torch.Tensor, predictions: torch.Tensor) -> None:
    source_labels = [int(dataset.label_metadata[int(label)]["source_label"]) for label in dataset.labels]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "dataset", "global_label", "source_label", "binary_label", "pseudolabel", "target", "prediction"])
        for index, (dataset_name, global_label, source_label, binary_label) in enumerate(
            zip(dataset.sample_dataset_names, dataset.labels, source_labels, binary_labels.tolist())
        ):
            writer.writerow([
                index, dataset_name, int(global_label), source_label, int(binary_label),
                int(pseudo_labels[index]), float(targets[index]), float(predictions[index]),
            ])


def _few_shot_container(container: dict[str, Any], method_name: str) -> dict[str, Any]:
    """The few-shot config that produces this distill config's pseudolabels."""
    few_shot = {key: container[key] for key in PSEUDOLABEL_KEYS}
    few_shot.update({
        "campaign": container["campaign"],
        "redo": False,
        "modalities": None,
        "support_sizes": None,
        # Mirrors few_shot.yaml's mlflow block.
        "mlflow": {
            **container["mlflow"],
            "type": "few_shot",
            "run_name": f"{method_name}_{container['modality']['modality']}",
        },
    })
    return few_shot


def _load_pseudolabels(mlflow, run_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as download_dir:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=PSEUDOLABEL_ARTIFACT, dst_path=download_dir
        )
        return torch.load(path, map_location="cpu", weights_only=True)


def _execute(cfg: DictConfig) -> tuple[float, str]:
    container = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(container)
    hydra_config = HydraConfig.get()
    method_name = str(hydra_config.runtime.choices["method"]).lower()
    mlp_targets = str(cfg.mlp_targets)
    if mlp_targets not in MSDE_MODES:
        raise ValueError(f"mlp_targets must be one of: {', '.join(MSDE_MODES)}")
    msde_mode = MSDE_MODES[mlp_targets]
    logger.info(
        "Starting distillation: modality=%s method=%s mlp_targets=%s msde_mode=%s overrides=[%s]",
        cfg.modality.modality, method_name, mlp_targets, msde_mode, ", ".join(hydra_config.overrides.task),
    )

    mlflow, client, experiment_id = setup_mlflow(cfg)
    # MSDE settings are unused for label targets, so they do not distinguish runs.
    hash_container = {key: value for key, value in container.items() if not (key == "msde" and mlp_targets == "labels")}
    hashes = config_hashes(hash_container)
    source_hash = pseudolabel_hash(container)
    existing = dedup_or_redo(client, experiment_id, str(cfg.mlflow.type), hashes["run_hash"], bool(cfg.redo))
    if existing is not None:
        return float(existing.data.metrics.get("objective", float("nan"))), existing.info.run_id
    # Generated before the distill run starts: MLflow cannot nest a second
    # root run inside it. A failure here is recorded on the few-shot run.
    if not find_finished(client, experiment_id, "few_shot", source_hash):
        print(f"No cached pseudolabels (run_hash={source_hash}); generating them", flush=True)
        generate_pseudolabels(_few_shot_container(container, method_name))
    tags = root_tags(container, hashes, {
        "modality": cfg.modality.modality,
        "encoder": cfg.modality.encoder,
        "method": method_name,
        "method_mode": cfg.method.get("mode") or "none",
        "support_size": int(cfg.support_size),
        "seed": int(cfg.seed),
        "mlp_targets": mlp_targets,
        "msde_mode": msde_mode,
        "distill_mlp": bool(cfg.distill_mlp),
        "pseudolabel_hash": source_hash,
    })

    with root_run(mlflow, container, tags) as run, staging_dir(str(cfg.mlflow.type)) as run_dir:
        source_run_id = find_finished(client, experiment_id, "few_shot", source_hash)[0].info.run_id
        mlflow.set_tag("pseudolabel_run_id", source_run_id)
        logger.info("Using pseudolabels from few-shot run %s", source_run_id)
        cache = _load_pseudolabels(mlflow, source_run_id)

        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))

        dataset = instantiate(cfg.modality)
        embeddings, _ = dataset.get_data()
        if cache["n_samples"] != len(dataset) or cache["sample_dataset_names"] != [str(name) for name in dataset.sample_dataset_names]:
            raise RuntimeError(
                f"Cached pseudolabels of run {source_run_id} do not match the loaded dataset; "
                "regenerate them with few_shot.py --redo"
            )
        device = _resolve_device(str(cfg.device))
        working = cache["working_embeddings"] if cache["working_embeddings"] is not None else embeddings
        working_embeddings = working.to(device)
        binary_labels = cache["binary_labels"]
        full_pseudo_labels = cache["pseudo_labels"].to(device)

        if mlp_targets == "labels":
            scores = full_pseudo_labels.float().cpu()
        elif mlp_targets == "distances":
            logger.info("Running %s MSDE total-distance targets", msde_mode)
            scores = _run_msde_distances(working_embeddings, full_pseudo_labels, cfg.msde, device)
        else:
            logger.info("Running %s MSDE + GDE scoring", msde_mode)
            scores = _run_msde_gde(working_embeddings, full_pseudo_labels, cfg.msde, device)
        score_target = scores.to(device)

        model = None
        losses: list[dict[str, float]] = []
        if bool(cfg.distill_mlp):
            logger.info("Distilling %s into MLP", mlp_targets)
            model, losses, final_scores = _train_mlp(working_embeddings, score_target, cfg.mlp, cfg.training, device)
            for row in losses:
                mlflow.log_metrics(
                    {"train_loss": row["train_loss"], "validation_loss": row["validation_loss"]}, step=int(row["epoch"])
                )
            mlflow.log_metrics({
                "best_validation_loss": min(row["validation_loss"] for row in losses),
                "epochs_trained": float(len(losses)),
            })
        else:
            final_scores = scores.detach().cpu()

        source_labels_by_sample = np.asarray([
            int(dataset.label_metadata[int(label)]["source_label"]) for label in dataset.labels
        ])
        rows: list[dict[str, Any]] = []
        for dataset_name in dataset.dataset_names:
            sample_mask = np.asarray([name == dataset_name for name in dataset.sample_dataset_names])
            rows.extend(_pair_metrics(final_scores[sample_mask], source_labels_by_sample[sample_mask], dataset_name))
        metrics = aggregate(rows)
        log_comparison_runs(mlflow, rows, tags)
        mlflow.log_metrics(metrics)

        OmegaConf.save(cfg, run_dir / "distill_mlp.yaml", resolve=True)
        _write_predictions(run_dir / "predictions.csv", dataset, binary_labels, full_pseudo_labels.cpu(), score_target.cpu(), final_scores)
        write_metrics_csv(run_dir / "metrics.csv", rows)
        required = ["distill_mlp.yaml", "predictions.csv", "metrics.csv"]
        if model is not None:
            _save_losses(losses, run_dir / "losses.csv")
            torch.save(model.state_dict(), run_dir / "model.pt")
            required.extend(["losses.csv", "model.pt"])
        upload_artifacts(mlflow, client, run.info.run_id, run_dir, required)
    print(f"MLflow run: {run.info.run_id} (objective={metrics['objective']:.4f})", flush=True)
    return metrics["objective"], run.info.run_id


@hydra.main(version_base=None, config_path="../configs", config_name="distill_mlp")
def main(cfg: DictConfig) -> float:
    """Run one Hydra job and return its objective (NaN for a failed multirun job)."""
    return hydra_entry(cfg, _execute, "DISTILL-MLP")


if __name__ == "__main__":
    consume_redo_flag()
    main()
