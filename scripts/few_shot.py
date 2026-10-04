"""Hydra/MLflow few-shot DELTA pseudolabeling stage.

Pipeline:

    embedding dataset -> binary support/query split -> pseudolabels

All pseudolabeling calls receive the complete tensor dataset; there is no
minibatch slicing. The pseudolabels (and, for the MSDE pseudolabelers, the
shifted embeddings) are cached as the run's ``pseudolabels.pt`` artifact and
consumed by ``distill_mlp.py``, which also calls ``generate_pseudolabels`` when
no cached run matches. Pseudolabel metrics are computed on query
samples only, since support samples carry their true labels.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
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

from methods.msde import DEFAULT_DEVICE
from utils.evaluation import _pair_metrics, aggregate
from utils.tracking import (
    PSEUDOLABEL_KEYS,
    config_hashes,
    dedup_or_redo,
    hydra_entry,
    log_comparison_runs,
    root_run,
    root_tags,
    setup_mlflow,
    staging_dir,
    upload_artifacts,
    write_metrics_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Pseudolabelers that also return MSDE-shifted embeddings for later stages.
SHIFTING_METHODS = {"laplacianshot_msde", "knnvote_msde"}
PSEUDOLABEL_ARTIFACT = "pseudolabels.pt"


def _resolve_device(device: str) -> str:
    return DEFAULT_DEVICE if device == "auto" else device


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


def _write_predictions(path: Path, dataset, binary_labels: torch.Tensor, pseudo_labels: torch.Tensor,
                       confidence: torch.Tensor, support_mask: np.ndarray) -> None:
    source_labels = [int(dataset.label_metadata[int(label)]["source_label"]) for label in dataset.labels]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "dataset", "global_label", "source_label", "binary_label", "pseudolabel", "confidence", "is_support"])
        for index, (dataset_name, global_label, source_label, binary_label) in enumerate(
            zip(dataset.sample_dataset_names, dataset.labels, source_labels, binary_labels.tolist())
        ):
            writer.writerow([
                index, dataset_name, int(global_label), source_label, int(binary_label),
                int(pseudo_labels[index]), float(confidence[index]), int(support_mask[index]),
            ])


def generate_pseudolabels(container: dict[str, Any]) -> tuple[float, str]:
    """Generate, evaluate and cache pseudolabels; return ``(objective, run_id)``.

    ``container`` is a resolved few-shot config. Only its ``PSEUDOLABEL_KEYS``
    are hashed, so ``distill_mlp.py`` finds the run by the same hash.
    """
    cfg = OmegaConf.create(container)
    hydra_config = HydraConfig.get()
    method_name = str(hydra_config.runtime.choices["method"]).lower()
    logger.info(
        "Starting few-shot pipeline: modality=%s method=%s split=%s overrides=[%s]",
        cfg.modality.modality, method_name, cfg.modality.split, ", ".join(hydra_config.overrides.task),
    )

    mlflow, client, experiment_id = setup_mlflow(cfg)
    hashes = config_hashes({key: container[key] for key in PSEUDOLABEL_KEYS})
    existing = dedup_or_redo(client, experiment_id, str(cfg.mlflow.type), hashes["run_hash"], bool(cfg.redo))
    if existing is not None:
        return float(existing.data.metrics.get("objective", float("nan"))), existing.info.run_id
    tags = root_tags(container, hashes, {
        "modality": cfg.modality.modality,
        "encoder": cfg.modality.encoder,
        "method": method_name,
        "method_mode": cfg.method.get("mode") or "none",
        "support_size": int(cfg.support_size),
        "seed": int(cfg.seed),
    })

    with root_run(mlflow, container, tags) as run, staging_dir(str(cfg.mlflow.type)) as run_dir:
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))

        dataset = instantiate(cfg.modality)
        embeddings, _ = dataset.get_data()
        logger.info("Loaded %d samples from %d datasets", len(dataset), len(dataset.dataset_names))
        binary_labels, _ = _binary_labels(dataset)
        support_indices, query_indices = _sample_support(binary_labels, int(cfg.support_size), int(cfg.seed))
        pseudolabeler = instantiate(cfg.method)
        device = _resolve_device(str(cfg.device))
        logger.info("Generating pseudolabels with %s", method_name)
        if method_name in SHIFTING_METHODS:
            working_embeddings, pseudo_labels, confidence, _ = pseudolabeler(
                embeddings.to(device), binary_labels.to(device), support_indices.to(device), query_indices.to(device)
            )
        else:
            support_normals = embeddings[support_indices][binary_labels[support_indices] == 0].to(device)
            support_anomalies = embeddings[support_indices][binary_labels[support_indices] == 1].to(device)
            pseudo_labels, confidence = pseudolabeler(
                support_normals, support_anomalies, embeddings[query_indices].to(device)
            )
            working_embeddings = None
        full_pseudo_labels = binary_labels.to(device).clone()
        full_pseudo_labels[query_indices.to(device)] = pseudo_labels
        full_confidence = torch.ones(len(dataset), device=device)
        full_confidence[query_indices.to(device)] = confidence
        full_pseudo_labels = full_pseudo_labels.cpu()
        full_confidence = full_confidence.cpu()

        support_mask = np.zeros(len(dataset), dtype=bool)
        support_mask[support_indices.numpy()] = True
        source_labels_by_sample = np.asarray([
            int(dataset.label_metadata[int(label)]["source_label"]) for label in dataset.labels
        ])
        pseudo_scores = full_pseudo_labels.float()
        rows: list[dict[str, Any]] = []
        for dataset_name in dataset.dataset_names:
            # Support samples keep their true labels, so only query samples are scored.
            sample_mask = np.asarray([name == dataset_name for name in dataset.sample_dataset_names]) & ~support_mask
            rows.extend(_pair_metrics(pseudo_scores[sample_mask], source_labels_by_sample[sample_mask], dataset_name))
        metrics = aggregate(rows)
        log_comparison_runs(mlflow, rows, tags)
        mlflow.log_metrics(metrics)

        OmegaConf.save(cfg, run_dir / "few_shot.yaml", resolve=True)
        OmegaConf.save(cfg.modality, run_dir / "dataset.yaml", resolve=True)
        OmegaConf.save(cfg.method, run_dir / "pseudolabeler.yaml", resolve=True)
        (run_dir / "label_mapping.json").write_text(
            json.dumps({str(key): value for key, value in dataset.label_metadata.items()}, indent=2)
        )
        torch.save({
            "support_indices": support_indices.cpu(),
            "query_indices": query_indices.cpu(),
            "binary_labels": binary_labels.cpu(),
            "pseudo_labels": full_pseudo_labels,
            "confidence": full_confidence,
            "n_samples": len(dataset),
            "sample_dataset_names": [str(name) for name in dataset.sample_dataset_names],
            "working_embeddings": None if working_embeddings is None else working_embeddings.detach().float().cpu(),
        }, run_dir / PSEUDOLABEL_ARTIFACT)
        _write_predictions(run_dir / "predictions.csv", dataset, binary_labels, full_pseudo_labels, full_confidence, support_mask)
        write_metrics_csv(run_dir / "metrics.csv", rows)
        upload_artifacts(mlflow, client, run.info.run_id, run_dir, [
            "few_shot.yaml", "dataset.yaml", "pseudolabeler.yaml", "label_mapping.json",
            PSEUDOLABEL_ARTIFACT, "predictions.csv", "metrics.csv",
        ])
    print(f"MLflow run: {run.info.run_id} (objective={metrics['objective']:.4f})", flush=True)
    return metrics["objective"], run.info.run_id


def _execute(cfg: DictConfig) -> tuple[float, str]:
    return generate_pseudolabels(OmegaConf.to_container(cfg, resolve=True))


@hydra.main(version_base=None, config_path="../configs", config_name="few_shot")
def main(cfg: DictConfig) -> float:
    """Run one Hydra job and return its objective (NaN for a failed multirun job)."""
    return hydra_entry(cfg, _execute, "FEW-SHOT")


if __name__ == "__main__":
    consume_redo_flag()
    main()
