"""Compare label-free per-sample scores by how well they separate normal from anomalous.

For every dataset and one encoder (default MedImageInsight) a class-balanced sample
is drawn (``--cap`` per class, ``--seeds`` seeds) and each score below is computed
without labels on the pooled sample. Labels are used only to evaluate:

* ``auroc``         : AUROC with anomaly as positive (above 0.5: anomalies score higher)
* ``separability``  : max(auroc, 1 - auroc), i.e. how well the score separates the
  classes if its direction were known (0.5 = chance, 1 = perfect)

Scores (each is one value per sample):

  Structural : centroid_dist, knn_dist_10, knn_dist_50, hubness (10-occurrence),
               local_pca_dim (participation ratio of the 50-NN covariance),
               twonn_mu (log r2/r1), graph_degree (row sum of the fuzzy graph S)
  MSDE       : msde_move (distance travelled), msde_move_unweighted (same shift with
               uniform neighbour weights, i.e. no density prior), msde_weight (density weight)
  KDE        : kde_density (Gaussian kernel, bandwidth = median 10-NN distance;
               one dense n x n kernel, cheap at the sample sizes used here)

One CSV row per (dataset, score), with seed mean and standard deviation.

Usage::

    uv run python scripts/score_comparison.py
    uv run python scripts/score_comparison.py --datasets VinCXR --cap 500 --seeds 1
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from datasets.embeds import SingleEmbedsDataset  # noqa: E402
from methods.msde import (  # noqa: E402
    DEFAULT_DEVICE,
    MeanShiftDensityEnhancement,
    _build_sparse_similarity,
    _sparse_similarity_layout,
    compute_weights_from_similarity_chunked,
)
from structural_asymmetry import MODALITIES, balanced_sample, iter_jobs  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "score_comparison.csv"
FAMILY = {
    "centroid_dist": "structural", "knn_dist_10": "structural", "knn_dist_50": "structural",
    "hubness": "structural", "local_pca_dim": "structural", "twonn_mu": "structural",
    "graph_degree": "structural", "msde_move": "msde", "msde_move_unweighted": "msde", "msde_weight": "msde", "kde_density": "kde",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("score_comparison")


def local_pca_dim(X: torch.Tensor, D: torch.Tensor, k: int = 50, chunk: int = 256) -> np.ndarray:
    """Participation ratio of the covariance of each point's k nearest neighbours."""
    k = min(k, len(X) - 1)
    nbrs = D.topk(k, dim=1, largest=False).indices
    out = []
    for start in range(0, len(X), chunk):
        patch = X[nbrs[start:start + chunk]]                      # (c, k, d)
        patch = patch - patch.mean(1, keepdim=True)
        s = torch.linalg.svdvals(patch).double() ** 2
        out.append((s.sum(1) ** 2 / (s ** 2).sum(1)).numpy())
    return np.concatenate(out)


def compute_scores(X: np.ndarray, args, device: str) -> dict[str, np.ndarray]:
    Xt = torch.from_numpy(X).float()
    n = len(Xt)
    D = torch.cdist(Xt, Xt)
    D.fill_diagonal_(float("inf"))
    sorted_d, nbr = D.topk(min(50, n - 1), dim=1, largest=False)
    k10 = min(10, sorted_d.shape[1])

    scores: dict[str, np.ndarray] = {
        "centroid_dist": (Xt - Xt.mean(0)).norm(dim=1).numpy(),
        "knn_dist_10": sorted_d[:, k10 - 1].numpy(),
        "knn_dist_50": sorted_d[:, -1].numpy(),
        "hubness": torch.bincount(nbr[:, :k10].reshape(-1), minlength=n).float().numpy(),
        "local_pca_dim": local_pca_dim(Xt, D),
        "twonn_mu": torch.log(sorted_d[:, 1].clamp_min(1e-12) / sorted_d[:, 0].clamp_min(1e-12)).numpy(),
    }

    # KDE: Gaussian kernel, bandwidth = median 10-NN distance, leave-one-out.
    # Exact duplicate embeddings give zero distances; keep the bandwidth positive.
    h = sorted_d[:, k10 - 1].median().clamp_min(1e-6)
    scores["kde_density"] = torch.logsumexp(-(D ** 2) / (2 * h ** 2), dim=1).numpy()

    Xd = Xt.to(device)
    with torch.no_grad():
        S = _build_sparse_similarity(Xd, 15, 200, device)
        scores["graph_degree"] = torch.zeros(n, device=device).index_add_(
            0, S.indices()[0], S.values()).cpu().numpy()
        scores["msde_weight"] = compute_weights_from_similarity_chunked(
            S, n, args.msde_threshold, 0.3, 4, 2048, layout=_sparse_similarity_layout(S)).cpu().numpy()
        for name, weighted in (("msde_move", True), ("msde_move_unweighted", False)):
            msde = MeanShiftDensityEnhancement(
                k=min(args.msde_k, n - 1), nbd_sample_count_threshold=args.msde_threshold,
                learning_rate=args.msde_lr, max_iters_shift=args.msde_iters,
                shift_threshold=args.msde_shift_threshold, device=device, use_chunking=True,
                weight_chunk_size=2048, enable_gradients=False,
            )
            if not weighted:
                # Uniform prior: every neighbour counts equally in the barycentre, and the
                # O(n^2) density-weight computation is skipped.
                msde._compute_base_weights = lambda X: torch.ones(len(X), device=X.device)
            scores[name] = msde(Xd)[1].cpu().numpy()
    return {name: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for name, value in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="MedImageInsight")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--cap", type=int, default=2000, help="max samples per class (balanced)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # MSDE settings default to those of configs/method/knnvote_msde.yaml.
    parser.add_argument("--msde-k", type=int, default=50)
    parser.add_argument("--msde-lr", type=float, default=0.33)
    parser.add_argument("--msde-threshold", type=int, default=70)
    parser.add_argument("--msde-iters", type=int, default=3)
    parser.add_argument("--msde-shift-threshold", type=float, default=0.01)
    args = parser.parse_args()

    device = DEFAULT_DEVICE if args.device == "auto" else args.device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in iter_jobs(args.modalities, [args.encoder]) if args.datasets is None or j[1] in args.datasets]
    LOGGER.info("%d datasets, encoder=%s, device=%s", len(jobs), args.encoder, device)

    rows = []
    for number, (modality, dataset, encoder, path, meta) in enumerate(jobs, 1):
        try:
            ds = SingleEmbedsDataset(
                path, dataset, encoder=encoder, as_tensor=False,
                include_labels=meta.get("include_labels", meta.get("labels")),
            )
        except FileNotFoundError as error:
            LOGGER.warning("skip %s/%s: %s", modality, dataset, str(error)[:80])
            continue
        if not ((ds.labels == 0).any() and (ds.labels != 0).any()):
            continue
        start = time.time()
        per_seed: dict[str, list[float]] = {name: [] for name in FAMILY}
        timing: dict[str, float] = {}
        for seed in range(args.seeds):
            rng = np.random.RandomState(seed)
            normal, anomaly = balanced_sample(ds.labels, args.cap, rng)
            X = ds.embeddings[np.concatenate([normal, anomaly])]
            y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)]
            t = time.time()
            scores = compute_scores(X, args, device)
            timing["total"] = timing.get("total", 0.0) + time.time() - t
            for name, value in scores.items():
                per_seed[name].append(float(roc_auc_score(y, value)))
        for name, values in per_seed.items():
            auroc = float(np.mean(values))
            rows.append({
                "modality": modality, "dataset": dataset, "encoder": encoder, "score": name,
                "family": FAMILY[name], "n_per_class": len(normal), "auroc": auroc,
                "auroc_std": float(np.std(values)),
                "separability": max(auroc, 1 - auroc),
                "direction": "anomaly_higher" if auroc >= 0.5 else "anomaly_lower",
            })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        LOGGER.info("[%d/%d] %s/%s  %.0fs", number, len(jobs), modality, dataset, time.time() - start)

    df = pd.DataFrame(rows)
    LOGGER.info("mean separability:\n%s", df.groupby("score")["separability"].mean().sort_values(ascending=False).round(3))
    LOGGER.info("wrote %s (%d rows)", args.output, len(df))


if __name__ == "__main__":
    main()
