"""Run an inference-only anomaly detection baseline on the raw images of a dataset and log the results to MLflow.

Example: python scripts/run_baseline.py baselines=adaclip modality=chest dataset=VinCXR
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Hydra 1.3 passes a lazy help object to argparse. Python 3.14 tightened
# argparse's help validation to require a string.
if sys.version_info >= (3, 14):
    _argparse_check_help = argparse.ArgumentParser._check_help

    def _hydra_argparse_check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return _argparse_check_help(self, action)

    argparse.ArgumentParser._check_help = _hydra_argparse_check_help

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.embeds import _balanced_cap_indices  # noqa: E402
from datasets.raw import MODALITY_DATASETS, RawDataset  # noqa: E402
from run_msde import dataset_names, detection_metrics  # noqa: E402


def raw_dataset(cfg: DictConfig, name: str) -> RawDataset:
    """The raw images of `name` in the order the embeddings were generated in (see generate_embeddings.py)."""
    raw_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "datasets" / f"raw_{cfg.modality}.yaml")
    raw_cfg.root = cfg.raw_data_root
    resolved = OmegaConf.to_container(OmegaConf.create({"data": {cfg.modality: raw_cfg}}), resolve=True)
    config = resolved["data"][cfg.modality]
    kwargs = {"seed": int(config.get("seed", 42))}
    for key in ("sample_strategy", "image_extensions"):
        if config.get(key) is not None:
            kwargs[key] = config[key]
    cls = MODALITY_DATASETS.get(cfg.modality.lower(), RawDataset)
    return cls(config["sources"][name], transform=None, **kwargs)


def run_dataset(cfg: DictConfig, name: str) -> None:
    root = Path(cfg.data_root)
    root = root if root.is_absolute() else PROJECT_ROOT / root
    all_labels = np.load(root / cfg.modality / name / "labels.npy")
    raw = raw_dataset(cfg, name)
    if not np.array_equal(np.asarray(raw.labels), all_labels):
        raise RuntimeError(f"{name}: raw images and labels.npy are out of sync ({len(raw)} vs {len(all_labels)})")
    # Same samples as run_msde.py (SingleEmbedsDataset applies the cap exactly like this).
    keep = (_balanced_cap_indices(all_labels, cfg.cap, cfg.seed)
            if cfg.cap is not None and len(all_labels) > cfg.cap else np.arange(len(all_labels)))
    labels = all_labels[keep]
    y = (labels != 0).astype(int)
    n, n_anomalies = len(y), int(y.sum())
    if n_anomalies == 0 or n_anomalies == n:
        print(f"[skip] {name}: needs both normal and anomalous samples (n={n}, anomalies={n_anomalies})")
        return

    overrides = {"class_name": cfg.class_names[cfg.modality]} if "class_name" in cfg.baselines else {}
    method = hydra.utils.instantiate(cfg.baselines, **overrides)
    score = np.empty(n, dtype=np.float64)
    for start in range(0, n, cfg.load_chunk):
        chunk = [raw.paths[i] for i in keep[start:start + cfg.load_chunk]]
        images = [Image.open(path).convert("RGB") for path in chunk]
        score[start:start + len(images)] = method.score(images)

    metrics = detection_metrics(y, score)
    subtypes = [int(c) for c in np.unique(labels) if c != 0]
    subtype_counts = {c: int((labels == c).sum()) for c in subtypes} if len(subtypes) > 1 else {}
    for c in subtype_counts:
        sub = (labels == 0) | (labels == c)
        metrics.update(detection_metrics((labels[sub] == c).astype(int), score[sub], prefix=f"subtype_{c}/"))

    label = cfg.baselines.name + ("" if cfg.baselines.get("checkpoint") is None else f"-{cfg.baselines.checkpoint}")
    with mlflow.start_run(run_name=f"{cfg.modality}-{name}-{label}"):
        mlflow.log_params({
            "dataset": name,
            "modality": cfg.modality,
            "method": cfg.baselines.name,
            "seed": cfg.seed,
            "n_samples": n,
            "n_anomalies": n_anomalies,
            **{f"n_subtype_{c}": count for c, count in subtype_counts.items()},
        })
        if cfg.baselines.get("checkpoint") is not None:
            mlflow.set_tag("checkpoint", cfg.baselines.checkpoint)
        mlflow.log_metrics(metrics, step=0)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "configs/config.yaml")
        mlflow.log_dict(OmegaConf.to_container(cfg.baselines, resolve=True), "configs/baseline.yaml")
    print(f"{cfg.modality}/{name}/{label}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "/" not in k))
    for c, count in subtype_counts.items():
        print(f"  subtype {c} (n={count}): auroc={metrics[f'subtype_{c}/auroc']:.4f} "
              f"auroc_flipped={metrics[f'subtype_{c}/auroc_flipped']:.4f}")


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    tracking_uri = cfg.mlflow.tracking_uri
    if tracking_uri.startswith("sqlite:///") and not Path(tracking_uri[10:]).is_absolute():
        db_path = PROJECT_ROOT / tracking_uri[10:]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(cfg.mlflow.experiment_name) is None:
        artifact_root = PROJECT_ROOT / "experiments" / "artifacts"
        mlflow.create_experiment(cfg.mlflow.experiment_name, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    torch.manual_seed(cfg.seed)
    for name in dataset_names(cfg):
        run_dataset(cfg, name)


if __name__ == "__main__":
    main()
