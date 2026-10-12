"""Measure structural asymmetries between normal and anomalous embeddings.

For every (modality, dataset, encoder) a class-balanced sample (label 0 versus
any non-zero label) is drawn and MSDE is run on it unlabeled, exactly as when
distance travelled is used as the anomaly score. Per-class statistics are then
written to one CSV row per combination, as ``<metric>_normal``,
``<metric>_anomaly`` and, for strictly positive metrics, ``<metric>_logratio``
(= log(anomaly / normal); positive means the anomalous class is larger).

Metric families (see the MSDE asymmetry notes):

1. Compactness   : spread, knn_within, knn_pooled, move (mean MSDE movement)
2. Intrinsic dim : twonn_id, local_pca_id, pr_dim, pca95_dim
3. Heterogeneity : n_modes, silhouette, weight_entropy, weight_cv
4. Graph         : deg_mean, intra_frac, n_components, giant_frac (fuzzy graph S)

``auroc_*`` columns score every sample with one statistic as an anomaly score
(higher = more anomalous); values above 0.5 mean "anomalies score higher", so
they read off the direction of each asymmetry directly.

Heavy: the MSDE density-weight step is O(n^2 d). Run on an MPS/CUDA machine.

Usage::

    uv run python scripts/structural_asymmetry.py
    uv run python scripts/structural_asymmetry.py --modalities oct --encoders CLIP --cap 500
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.embeds import SingleEmbedsDataset, _resolve_embeds_file  # noqa: E402
from methods.msde import (  # noqa: E402
    DEFAULT_DEVICE,
    MeanShiftDensityEnhancement,
    _build_sparse_similarity,
    _sparse_similarity_layout,
    compute_weights_from_similarity_chunked,
)

MODALITIES = ("chest", "fundus", "mri", "oct")
ENCODERS = ("CLIP", "BiomedCLIP", "MedImageInsight", "MedSigLIP", "UniMedCLIP")
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "structural_asymmetry.csv"
KEY = ["modality", "dataset", "encoder"]
POSITIVE = (
    "spread", "knn_within", "knn_pooled", "move", "twonn_id", "local_pca_id", "pr_dim",
    "pca95_dim", "n_modes", "deg_mean", "n_components", "weight_cv",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("structural_asymmetry")


# --- per-class statistics -------------------------------------------------------------


def _pairwise(X: torch.Tensor) -> torch.Tensor:
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    return D


def _kth_nn_dist(D: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, D.shape[1] - 1)
    return D.topk(k, dim=1, largest=False).values[:, -1]


def twonn_dimension(D: torch.Tensor, discard: float = 0.1) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017), regression through the origin."""
    r = D.topk(2, dim=1, largest=False).values
    r1, r2 = r[:, 0], r[:, 1]
    keep = r1 > 0
    mu = (r2[keep] / r1[keep]).double().sort().values
    n = len(mu)
    if n < 10:
        return float("nan")
    m = int(n * (1 - discard))
    mu = mu[:m]
    x = torch.log(mu)
    y = -torch.log(1 - torch.arange(1, m + 1, dtype=torch.float64) / n)
    return float((x * y).sum() / (x * x).sum())


def participation_ratio(X: torch.Tensor) -> float:
    """(sum lambda)^2 / sum lambda^2 of the covariance spectrum."""
    s = torch.linalg.svdvals(X - X.mean(0)).double() ** 2
    return float(s.sum() ** 2 / (s**2).sum())


def pca_dims(X: torch.Tensor, ratio: float = 0.95) -> int:
    s = torch.linalg.svdvals(X - X.mean(0)).double() ** 2
    return int((torch.cumsum(s, 0) / s.sum() < ratio).sum().item() + 1)


def local_pca_dimension(X: torch.Tensor, D: torch.Tensor, rng: np.random.RandomState,
                        k: int = 50, n_centers: int = 200) -> float:
    """Mean participation ratio of the covariance of k-NN neighbourhoods."""
    n = len(X)
    k = min(k, n - 1)
    if k < 5:
        return float("nan")
    centers = rng.choice(n, size=min(n_centers, n), replace=False)
    nbrs = D[centers].topk(k, dim=1, largest=False).indices
    return float(np.mean([participation_ratio(X[idx]) for idx in nbrs]))


def sub_modes(X: np.ndarray, seed: int, max_k: int = 8) -> tuple[int, float]:
    """Number of k-means clusters that maximises the silhouette, and that silhouette."""
    n = len(X)
    if n < 20:
        return 1, float("nan")
    if X.shape[1] > 50:
        X = PCA(n_components=min(50, n - 1), random_state=seed).fit_transform(X)
    best_k, best_s = 1, -1.0
    for k in range(2, min(max_k, n - 1) + 1):
        labels = KMeans(k, n_init=3, random_state=seed).fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        s = float(silhouette_score(X, labels, sample_size=min(n, 2000), random_state=seed))
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s


def weight_entropy(w: np.ndarray) -> float:
    """Entropy of the density-weight distribution, normalised to [0, 1] (1 = uniform)."""
    w = np.clip(w, 0, None)
    if len(w) < 2 or w.sum() <= 0:
        return float("nan")
    p = w / w.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(w)))


def graph_stats(S, y: np.ndarray) -> dict[str, float]:
    """Degree, intra-class edge fraction and connectivity of each class in the fuzzy graph."""
    idx = S.indices().cpu().numpy()
    vals = S.values().cpu().numpy()
    n = len(y)
    A = coo_matrix((vals, (idx[0], idx[1])), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    out: dict[str, float] = {}
    for name, cls in (("normal", 0), ("anomaly", 1)):
        members = np.flatnonzero(y == cls)
        sub = A[members][:, members]
        total = A[members].sum()
        out[f"deg_mean_{name}"] = float(A[members].sum(1).mean())
        out[f"intra_frac_{name}"] = float(sub.sum() / total) if total > 0 else float("nan")
        n_comp, comp = connected_components(sub, directed=False)
        out[f"n_components_{name}"] = float(n_comp)
        out[f"giant_frac_{name}"] = float(np.bincount(comp).max() / len(members))
    return out


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


# --- one (dataset, encoder) -----------------------------------------------------------


def balanced_sample(labels: np.ndarray, cap: int, rng: np.random.RandomState):
    """Equal numbers of normal (0) and anomalous (non-zero, uniform over subtypes) rows."""
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels != 0)
    n = min(cap, len(normal), len(anomaly))
    return (
        np.sort(rng.choice(normal, n, replace=False)),
        np.sort(rng.choice(anomaly, n, replace=False)),
    )


def analyse(X_all: np.ndarray, labels: np.ndarray, args, seed: int, device: str) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    normal_idx, anomaly_idx = balanced_sample(labels, args.cap, rng)
    idx = np.concatenate([normal_idx, anomaly_idx])
    n_per = len(normal_idx)
    y = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])
    X = torch.from_numpy(X_all[idx]).float()

    row: dict[str, float] = {
        "n_per_class": n_per,
        "n_normal_full": int((labels == 0).sum()),
        "n_anomaly_full": int((labels != 0).sum()),
        "n_anomaly_types": len(np.unique(labels[labels != 0])),
    }

    # MSDE on the pooled, unlabeled sample: the movement score and the density weights.
    k = min(args.msde_k, len(X) - 1)
    msde = MeanShiftDensityEnhancement(
        k=k, nbd_sample_count_threshold=args.msde_threshold, learning_rate=args.msde_lr,
        max_iters_shift=args.msde_iters, shift_threshold=args.msde_shift_threshold,
        device=device, use_chunking=True, weight_chunk_size=2048, enable_gradients=False,
    )
    with torch.no_grad():
        _, movement, _ = msde(X.to(device))
        movement = movement.cpu().numpy()

        X_dev = X.to(device)
        S = _build_sparse_similarity(X_dev, 15, 200, device)
        weights = compute_weights_from_similarity_chunked(
            S, len(X), args.msde_threshold, 0.3, 4, 2048, layout=_sparse_similarity_layout(S)
        ).cpu().numpy()
    graph = graph_stats(S, y)

    D_pool = _pairwise(X)
    knn_pooled = _kth_nn_dist(D_pool, args.knn).numpy()

    per_class: dict[str, dict[str, float]] = {}
    for name, cls in (("normal", 0), ("anomaly", 1)):
        m = y == cls
        Xc = X[m]
        Dc = _pairwise(Xc)
        modes, sil = sub_modes(Xc.numpy(), seed)
        per_class[name] = {
            "spread": float((Xc - Xc.mean(0)).norm(dim=1).mean()),
            "knn_within": float(_kth_nn_dist(Dc, args.knn).mean()),
            "knn_pooled": float(knn_pooled[m].mean()),
            "move": float(movement[m].mean()),
            "twonn_id": twonn_dimension(Dc),
            "local_pca_id": local_pca_dimension(Xc, Dc, rng),
            "pr_dim": participation_ratio(Xc),
            "pca95_dim": float(pca_dims(Xc)),
            "n_modes": float(modes),
            "silhouette": sil,
            "weight_entropy": weight_entropy(weights[m]),
            "weight_cv": float(weights[m].std() / max(weights[m].mean(), 1e-12)),
            "deg_mean": graph[f"deg_mean_{name}"],
            "intra_frac": graph[f"intra_frac_{name}"],
            "n_components": graph[f"n_components_{name}"],
            "giant_frac": graph[f"giant_frac_{name}"],
        }

    for metric in per_class["normal"]:
        a, b = per_class["normal"][metric], per_class["anomaly"][metric]
        row[f"{metric}_normal"], row[f"{metric}_anomaly"] = a, b
        if metric in POSITIVE:
            row[f"{metric}_logratio"] = float(np.log(b / a)) if a > 0 and b > 0 else float("nan")
        else:
            row[f"{metric}_diff"] = b - a

    # Direction read-off: AUROC of each statistic as an anomaly score (anomaly = positive).
    row["auroc_move"] = auroc(y, movement)
    row["auroc_knn_pooled"] = auroc(y, knn_pooled)
    row["auroc_neg_weight"] = auroc(y, -weights)
    return row


# --- driver ---------------------------------------------------------------------------


def iter_jobs(modalities, encoders):
    for modality in modalities:
        cfg = OmegaConf.to_container(
            OmegaConf.load(PROJECT_ROOT / "configs" / "modality" / f"{modality}.yaml"), resolve=False
        )
        root = PROJECT_ROOT / cfg.get("data_root", f"data/{modality}")
        for dataset, meta in cfg["datasets"].items():
            for encoder in encoders:
                yield modality, dataset, encoder, root / dataset, dict(meta or {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    parser.add_argument("--datasets", nargs="+", default=None, help="restrict to these dataset names")
    parser.add_argument("--cap", type=int, default=2000, help="max samples per class (balanced)")
    parser.add_argument("--knn", type=int, default=10, help="k of the local-scale statistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true", help="recompute rows already in the CSV")
    # MSDE settings default to those of configs/method/knnvote_msde.yaml.
    parser.add_argument("--msde-k", type=int, default=50)
    parser.add_argument("--msde-lr", type=float, default=0.33)
    parser.add_argument("--msde-threshold", type=int, default=70)
    parser.add_argument("--msde-iters", type=int, default=3)
    parser.add_argument("--msde-shift-threshold", type=float, default=0.01)
    args = parser.parse_args()

    device = DEFAULT_DEVICE if args.device == "auto" else args.device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(args.output) if args.output.is_file() and not args.no_resume else pd.DataFrame()
    finished = set(map(tuple, done[KEY].values)) if len(done) else set()
    rows = done.to_dict("records")

    jobs = [j for j in iter_jobs(args.modalities, args.encoders) if args.datasets is None or j[1] in args.datasets]
    LOGGER.info("%d jobs on %s (%d already done)", len(jobs), device, len(finished))
    for number, (modality, dataset, encoder, path, meta) in enumerate(jobs, 1):
        if (modality, dataset, encoder) in finished:
            continue
        try:
            _resolve_embeds_file(path, encoder)
            ds = SingleEmbedsDataset(
                path, dataset, encoder=encoder, as_tensor=False,
                include_labels=meta.get("include_labels", meta.get("labels")),
            )
        except FileNotFoundError as error:
            LOGGER.warning("skip %s/%s/%s: %s", modality, dataset, encoder, str(error)[:80])
            continue
        if not ((ds.labels == 0).any() and (ds.labels != 0).any()):
            LOGGER.warning("skip %s/%s: needs both normal and anomalous samples", modality, dataset)
            continue
        start = time.time()
        stats = analyse(ds.embeddings, ds.labels, args, args.seed, device)
        rows.append({
            "modality": modality, "dataset": dataset, "encoder": encoder,
            "tag": meta.get("tag", ""), **stats,
        })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        LOGGER.info("[%d/%d] %s/%s/%s  %.0fs  auroc_move=%.3f", number, len(jobs), modality, dataset,
                    encoder, time.time() - start, stats["auroc_move"])
    LOGGER.info("wrote %s (%d rows)", args.output, len(rows))


if __name__ == "__main__":
    main()
