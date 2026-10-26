"""KDE vs MSDE movement as label-free anomaly scores, on every encoder x dataset.

For each (encoder, dataset) a class-balanced sample (``--cap`` per class, ``--seeds`` seeds) is
drawn, both scores are computed without labels, and the AUROC (anomaly = positive) is taken.
``separability`` = max(auroc, 1 - auroc).

  kde_density : Gaussian kernel, bandwidth = median 10-NN distance, leave-one-out log-density
  msde_move   : distance travelled under MSDE (src/msde.py)
  msde_pseudo : second MSDE pass. The msde_move scores are turned into pseudo labels (the
                ``--pseudo-fraction`` of points that moved most, or least with
                ``--pseudo-direction lower``, get label 1, the rest 0) and passed as ``labels`` to an
                MSDE with ``label_mode="zero_label"``, so every point is pulled only towards
                pseudo-normal points; its distance travelled is the score.

A bar chart of the separability of the three scores for every encoder x dataset is written to
a multi-page PDF (one page per dataset, one panel per encoder).

Usage::

    uv run python scripts/score_comparison.py
    uv run python scripts/score_comparison.py --encoders CLIP --datasets VinCXR --seeds 1
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.embeds import SingleEmbedsDataset  # noqa: E402
from msde import MeanShiftDensityEnhancement  # noqa: E402

MODALITIES = ("chest", "fundus", "mri", "oct")
ENCODERS = ("CLIP", "BiomedCLIP", "MedImageInsight", "MedSigLIP", "UniMedCLIP")
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "score_comparison.csv"
DEFAULT_PDF = PROJECT_ROOT / "assets" / "score_comparison.pdf"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("score_comparison")


def balanced_sample(labels: np.ndarray, cap: int, rng: np.random.RandomState):
    """Equal numbers of normal (0) and anomalous (non-zero) rows."""
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels != 0)
    n = min(cap, len(normal), len(anomaly))
    return np.sort(rng.choice(normal, n, replace=False)), np.sort(rng.choice(anomaly, n, replace=False))


def iter_jobs(modalities, encoders):
    for modality in modalities:
        cfg = OmegaConf.to_container(
            OmegaConf.load(PROJECT_ROOT / "configs" / "modality" / f"{modality}.yaml"), resolve=False
        )
        root = PROJECT_ROOT / cfg.get("data_root", f"data/{modality}")
        for dataset, meta in cfg["datasets"].items():
            for encoder in encoders:
                yield modality, dataset, encoder, root / dataset, dict(meta or {})


def compute_scores(X: np.ndarray, args, device: str) -> dict[str, np.ndarray]:
    Xt = torch.from_numpy(X).float()
    n = len(Xt)
    D = torch.cdist(Xt, Xt)
    D.fill_diagonal_(float("inf"))
    k10 = min(10, n - 1)
    # KDE: leave-one-out Gaussian kernel; keep the bandwidth positive for duplicate embeddings.
    h = D.topk(k10, dim=1, largest=False).values[:, -1].median().clamp_min(1e-6)
    kde = torch.logsumexp(-(D ** 2) / (2 * h ** 2), dim=1).numpy()

    def make_msde(label_mode):
        return MeanShiftDensityEnhancement(
            k=min(args.msde_k, n - 1), nbd_sample_count_threshold=args.msde_threshold,
            learning_rate=args.msde_lr, max_iters_shift=args.msde_iters,
            shift_threshold=args.msde_shift_threshold, device=device, label_mode=label_mode,
        )

    _, movement, _ = make_msde("same_label")(Xt)
    movement = movement.cpu().numpy()
    # Pseudo labels: 1 for the pseudo-anomalous fraction of the points, 0 for the rest.
    order = np.argsort(movement if args.pseudo_direction == "lower" else -movement, kind="stable")
    pseudo = np.zeros(n, dtype=np.int64)
    pseudo[order[: max(1, int(round(args.pseudo_fraction * n)))]] = 1
    _, movement2, _ = make_msde("zero_label")(Xt, labels=torch.from_numpy(pseudo))
    scores = {"kde_density": kde, "msde_move": movement, "msde_pseudo": movement2.cpu().numpy()}
    return {name: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for name, v in scores.items()}


SCORE_LABELS = {"kde_density": "KDE", "msde_move": "MSDE", "msde_pseudo": "MSDE pseudo-label"}
SCORE_COLORS = {"kde_density": "#4C72B0", "msde_move": "#C44E52", "msde_pseudo": "#55A868"}


def plot_bars(df: pd.DataFrame, path: Path) -> None:
    """One page per dataset, one bar chart (separability of each score) per encoder."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    scores = [s for s in SCORE_LABELS if s in set(df["score"])]
    encoders = [e for e in ENCODERS if e in set(df["encoder"])]
    with PdfPages(path) as pdf:
        for (modality, dataset), group in df.groupby(["modality", "dataset"], sort=False):
            fig, axes = plt.subplots(1, len(encoders), figsize=(3 * len(encoders), 3.2), sharey=True, squeeze=False)
            for ax, encoder in zip(axes[0], encoders):
                sub = group[group["encoder"] == encoder].set_index("score")
                values = [sub["separability"].get(s, np.nan) for s in scores]
                ax.bar(range(len(scores)), values, color=[SCORE_COLORS[s] for s in scores])
                for i, v in enumerate(values):
                    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
                ax.axhline(0.5, color="k", ls="--", lw=0.8)
                ax.set_ylim(0.4, 1.05)
                ax.set_xticks(range(len(scores)), [SCORE_LABELS[s] for s in scores], rotation=30, ha="right", fontsize=7)
                ax.set_title(encoder, fontsize=9)
            axes[0][0].set_ylabel("max(AUROC, 1 - AUROC)")
            fig.suptitle(f"{modality} / {dataset}", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--cap", type=int, default=2000, help="max samples per class (balanced)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="bar chart of every encoder x dataset")
    parser.add_argument("--plot-only", action="store_true", help="only redraw the PDF from --output")
    parser.add_argument("--pseudo-fraction", type=float, default=0.5,
                        help="fraction of points labelled pseudo-anomalous (1) from the first MSDE scores")
    parser.add_argument("--pseudo-direction", choices=("higher", "lower"), default="higher",
                        help="'higher': the points that moved most are pseudo-anomalous")
    # Same MSDE settings as the earlier score comparison.
    parser.add_argument("--msde-k", type=int, default=50)
    parser.add_argument("--msde-lr", type=float, default=0.33)
    parser.add_argument("--msde-threshold", type=int, default=70)
    parser.add_argument("--msde-iters", type=int, default=3)
    parser.add_argument("--msde-shift-threshold", type=float, default=0.01)
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_bars(pd.read_csv(args.output), args.pdf)
        return
    jobs = [j for j in iter_jobs(args.modalities, args.encoders) if args.datasets is None or j[1] in args.datasets]
    LOGGER.info("%d jobs, device=%s", len(jobs), device)

    rows = []
    for number, (modality, dataset, encoder, path, meta) in enumerate(jobs, 1):
        try:
            ds = SingleEmbedsDataset(
                path, dataset, encoder=encoder, as_tensor=False,
                include_labels=meta.get("include_labels", meta.get("labels")),
            )
        except FileNotFoundError as error:
            LOGGER.warning("skip %s/%s/%s: %s", modality, dataset, encoder, str(error)[:80])
            continue
        if not ((ds.labels == 0).any() and (ds.labels != 0).any()):
            continue
        start = time.time()
        per_seed: dict[str, list[float]] = {}
        for seed in range(args.seeds):
            normal, anomaly = balanced_sample(ds.labels, args.cap, np.random.RandomState(seed))
            X = ds.embeddings[np.concatenate([normal, anomaly])]
            y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)]
            for name, value in compute_scores(X, args, device).items():
                per_seed.setdefault(name, []).append(float(roc_auc_score(y, value)))
        for name, values in per_seed.items():
            auroc = float(np.mean(values))
            rows.append({
                "modality": modality, "dataset": dataset, "encoder": encoder, "score": name,
                "n_per_class": len(normal), "auroc": auroc, "auroc_std": float(np.std(values)),
                "separability": max(auroc, 1 - auroc),
            })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        LOGGER.info("[%d/%d] %s/%s/%s  %.0fs", number, len(jobs), modality, dataset, encoder, time.time() - start)

    df = pd.DataFrame(rows)
    LOGGER.info("mean separability by score:\n%s", df.groupby("score")["separability"].mean().round(4))
    LOGGER.info("by encoder:\n%s", df.pivot_table(index="encoder", columns="score", values="separability").round(4))
    LOGGER.info("wrote %s (%d rows)", args.output, len(df))
    plot_bars(df, args.pdf)
    LOGGER.info("wrote %s", args.pdf)


if __name__ == "__main__":
    main()
