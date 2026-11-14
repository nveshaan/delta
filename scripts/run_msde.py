"""Run MSDE on the embeddings of a dataset and log the results to MLflow.

Example: python scripts/run_msde.py modality=chest encoder=CLIP dataset=VinCXR
"""
import argparse
import sys
import tempfile
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
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.embeds import SingleEmbedsDataset  # noqa: E402

COMPONENTS = ("density", "score", "weight", "stability", "metric", "bias", "step")
SUPPORT_COMPONENTS = ("bias", "step")  # used only with support > 0


def precision_at_n(y_true: np.ndarray, score: np.ndarray, n: int) -> float:
    top = np.argsort(-score, kind="stable")[:n]
    return float(y_true[top].sum() / n)


def detection_metrics(y_true: np.ndarray, score: np.ndarray, prefix: str = "") -> dict[str, float]:
    """AUROC, AUPRC and P@n (n = number of anomalies) on `score` and on `-score` (suffix `_flipped`)."""
    n_anomalies = int(y_true.sum())
    metrics = {}
    for suffix, sign in (("", 1.0), ("_flipped", -1.0)):
        directed = sign * score
        metrics[f"{prefix}auroc{suffix}"] = float(roc_auc_score(y_true, directed))
        metrics[f"{prefix}auprc{suffix}"] = float(average_precision_score(y_true, directed))
        metrics[f"{prefix}p_at_n{suffix}"] = precision_at_n(y_true, directed, n_anomalies)
    return metrics


def pseudo_labels(score: np.ndarray, anomaly_fraction: float) -> np.ndarray:
    """1 for the `anomaly_fraction` highest-scoring points, 0 for the rest."""
    top = np.argsort(-score, kind="stable")[: max(1, round(anomaly_fraction * len(score)))]
    pseudo = np.zeros(len(score), dtype=np.int64)
    pseudo[top] = 1
    return pseudo


def dataset_names(cfg: DictConfig) -> list[str]:
    if cfg.dataset is not None:
        return [cfg.dataset]
    modality_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "datasets" / f"{cfg.modality}.yaml")
    return list(modality_cfg.datasets)


def run_dataset(cfg: DictConfig, name: str) -> None:
    root = Path(cfg.data_root)
    root = root if root.is_absolute() else PROJECT_ROOT / root
    data = SingleEmbedsDataset(
        root / cfg.modality / name, encoder=cfg.encoder, cap=cfg.cap, seed=cfg.seed, as_tensor=False
    )
    embeddings, labels, indices = data.embeddings, np.asarray(data.labels), data.indices
    support = None
    if cfg.support > 0:
        normal = np.flatnonzero(labels == 0)
        if cfg.support >= len(normal):
            print(f"[skip] {name}: support={cfg.support} needs more normal samples (has {len(normal)})")
            return
        # Normal samples drawn as support are removed from the evaluated points.
        chosen = np.sort(np.random.RandomState(cfg.seed).choice(normal, cfg.support, replace=False))
        support_index = indices[chosen]
        support = torch.from_numpy(np.asarray(embeddings[chosen], dtype=np.float32))
        rest = np.setdiff1d(np.arange(len(labels)), chosen)
        embeddings, labels, indices = embeddings[rest], labels[rest], indices[rest]
    X = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    y = (labels != 0).astype(int)
    n, n_anomalies = len(y), int(y.sum())
    if n_anomalies == 0 or n_anomalies == n:
        print(f"[skip] {name}: needs both normal and anomalous samples (n={n}, anomalies={n_anomalies})")
        return

    overrides = {}
    if cfg.model.k > n - 1:
        overrides["k"] = n - 1
    if cfg.model.n_neighbors > n:
        overrides["n_neighbors"] = n
    if overrides:
        print(f"[warn] {name}: n={n} too small, overriding {overrides}")

    # A component set to null in the defaults list (e.g. step) is absent from cfg.
    components = [comp for comp in COMPONENTS
                  if cfg.get(comp) is not None and (comp not in SUPPORT_COMPONENTS or support is not None)]
    fns = {comp: None for comp in COMPONENTS}
    fns.update({comp: hydra.utils.instantiate(cfg[comp].fn) for comp in components if cfg[comp].fn is not None})
    subtypes = [int(c) for c in np.unique(labels) if c != 0]
    subtype_counts = {c: int((labels == c).sum()) for c in subtypes} if len(subtypes) > 1 else {}

    rounds = []  # metrics of every MSDE run
    scores = []  # per-sample scores of every MSDE run
    run_labels = None
    for r in range(cfg.pseudo_labels.rounds):
        model = hydra.utils.instantiate(cfg.model, **overrides)
        _, metric, _ = model(
            X, labels=run_labels, support=support, dist=cfg.dist, density_fn=fns["density"], score_fn=fns["score"],
            weight_fn=fns["weight"], stability_fn=fns["stability"],
            metric_fn=fns["metric"], bias_fn=fns["bias"], step_fn=fns["step"],
        )
        score = metric.detach().cpu().numpy()
        scores.append(score)

        metrics = detection_metrics(y, score)
        # Subtypes (datasets with several non-zero labels): each one against the normal samples (label 0) only.
        for c in subtype_counts:
            keep = (labels == 0) | (labels == c)
            metrics.update(detection_metrics((labels[keep] == c).astype(int), score[keep], prefix=f"subtype_{c}/"))
        rounds.append(metrics)

        # The next round pulls points towards the presumed-normal ones (label 0).
        run_labels = pseudo_labels(score, cfg.pseudo_labels.anomaly_fraction)

    with mlflow.start_run(run_name=f"{cfg.modality}-{name}-{cfg.encoder}"):
        mlflow.log_params({
            "dataset": name,
            "modality": cfg.modality,
            "encoder": cfg.encoder,
            "seed": cfg.seed,
            "dist": cfg.dist,
            "support": cfg.support,
            "n_samples": n,
            "n_anomalies": n_anomalies,
            "pseudo_label_rounds": cfg.pseudo_labels.rounds,
            "pseudo_label_anomaly_fraction": cfg.pseudo_labels.anomaly_fraction,
            **{f"n_subtype_{c}": count for c, count in subtype_counts.items()},
            **{comp: cfg[comp].name for comp in components},
            **{f"model.{key}": value for key, value in OmegaConf.to_container(cfg.model).items() if key != "_target_"},
            **{f"model.{key}": value for key, value in overrides.items()},
        })
        for r, metrics in enumerate(rounds):  # the latest value of each metric is the last round's
            mlflow.log_metrics(metrics, step=r)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "configs/config.yaml")
        for comp in components:
            mlflow.log_dict(OmegaConf.to_container(cfg[comp], resolve=True), f"configs/{comp}.yaml")
        # Per-sample scores of the last round (and of every round), with each sample's label and its index into
        # the dataset's labels.npy (and, with support, the support samples' indices).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.npz"
            extra = {"support_index": support_index} if support is not None else {}
            np.savez(path, score=scores[-1], score_rounds=np.stack(scores), label=labels, index=indices, **extra)
            mlflow.log_artifact(str(path), "scores")
    for r, metrics in enumerate(rounds):
        print(f"{cfg.modality}/{name}/{cfg.encoder} round {r}: "
              + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "/" not in k))
        for c, count in subtype_counts.items():
            print(f"  subtype {c} (n={count}): auroc={metrics[f'subtype_{c}/auroc']:.4f} "
                  f"auroc_flipped={metrics[f'subtype_{c}/auroc_flipped']:.4f}")


@hydra.main(version_base=None, config_path="../configs", config_name="msde")
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
