"""KDE vs MSDE movement as label-free anomaly scores, on every encoder x dataset.

For each (encoder, dataset) a class-balanced sample (``--cap`` per class, ``--seeds`` seeds) is
drawn, both scores are computed without labels, and the AUROC (anomaly = positive) is taken.
``separability`` = max(auroc, 1 - auroc).

  kde_density : Gaussian kernel, bandwidth = median 10-NN distance, leave-one-out log-density
  msde_stage_1 : distance travelled under the initial, unlabelled MSDE pass.
  msde_stage_2 .. msde_stage_10 : repeated MSDE passes. At each stage, the previous stage's
                movement scores are converted to pseudo labels and passed as ``labels`` to an
                MSDE with ``label_mode="zero_label"``; the distance travelled is the score.

A plot of mean separability against MSDE stage, with KDE as a horizontal baseline, is written
to a PDF.

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
STAGE_COUNT = 10
STAGE_SCORES = tuple(f"msde_stage_{stage}" for stage in range(1, STAGE_COUNT + 1))
LEGACY_SCORE_NAMES = {"msde_move": "msde_stage_1", "msde_pseudo": "msde_stage_2"}
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
    scores = {"kde_density": kde, "msde_stage_1": movement.cpu().numpy()}
    for stage in range(2, STAGE_COUNT + 1):
        # Label 1 marks pseudo-anomalies; zero_label makes every point seek only label-0 neighbours.
        movement_np = movement.cpu().numpy()
        order = np.argsort(movement_np if args.pseudo_direction == "lower" else -movement_np, kind="stable")
        pseudo = np.zeros(n, dtype=np.int64)
        pseudo[order[: max(1, int(round(args.pseudo_fraction * n)))]] = 1
        _, movement, _ = make_msde("zero_label")(Xt, labels=torch.from_numpy(pseudo))
        scores[f"msde_stage_{stage}"] = movement.cpu().numpy()
    return {name: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for name, v in scores.items()}


SCORE_LABELS = {"kde_density": "KDE", **{name: f"MSDE {stage}" for stage, name in enumerate(STAGE_SCORES, 1)}}


def canonicalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["score"].replace(LEGACY_SCORE_NAMES)
    return df


def plot_stage_curve(df: pd.DataFrame, path: Path) -> None:
    """Plot mean stage separability with the mean KDE separability as baseline."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    df = canonicalize_scores(df)
    staged = df[df["score"].isin(STAGE_SCORES)].copy()
    staged["stage"] = staged["score"].str.rsplit("_", n=1).str[-1].astype(int)
    stage_means = staged.groupby("stage")["separability"].mean()
    kde_mean = df.loc[df["score"] == "kde_density", "separability"].mean()
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(stage_means.index, stage_means.values, marker="o", color="#C44E52", label="MSDE")
        ax.axhline(kde_mean, color="#4C72B0", ls="--", label=f"KDE ({kde_mean:.3f})")
        ax.set(xlabel="MSDE stage", ylabel="Mean separability", xticks=range(1, STAGE_COUNT + 1),
               ylim=(0.4, 1.0), title="Mean separability by MSDE stage")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
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
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="stage separability plot")
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
    cached = canonicalize_scores(pd.read_csv(args.output)) if args.output.exists() else pd.DataFrame()
    if args.plot_only:
        plot_stage_curve(cached, args.pdf)
        return
    jobs = [j for j in iter_jobs(args.modalities, args.encoders) if args.datasets is None or j[1] in args.datasets]
    LOGGER.info("%d jobs, device=%s", len(jobs), device)

    rows = []
    for number, (modality, dataset, encoder, path, meta) in enumerate(jobs, 1):
        cached_job = cached[
            (cached["modality"] == modality) & (cached["dataset"] == dataset) &
            (cached["encoder"] == encoder)
        ] if not cached.empty else pd.DataFrame()
        required_scores = {"kde_density", *STAGE_SCORES}
        if set(cached_job["score"]) >= required_scores:
            rows.extend(cached_job[cached_job["score"].isin(required_scores)].to_dict("records"))
            LOGGER.info("[%d/%d] %s/%s/%s  cached", number, len(jobs), modality, dataset, encoder)
            continue
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
            row = {
                "modality": modality, "dataset": dataset, "encoder": encoder, "score": name,
                "n_per_class": len(normal), "auroc": auroc, "auroc_std": float(np.std(values)),
                "separability": max(auroc, 1 - auroc),
            }
            cached_row = cached_job[cached_job["score"] == name]
            rows.append(cached_row.iloc[-1].to_dict() if not cached_row.empty else row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        LOGGER.info("[%d/%d] %s/%s/%s  %.0fs", number, len(jobs), modality, dataset, encoder, time.time() - start)

    df = pd.DataFrame(rows)
    LOGGER.info("mean separability by score:\n%s", df.groupby("score")["separability"].mean().round(4))
    LOGGER.info("by encoder:\n%s", df.pivot_table(index="encoder", columns="score", values="separability").round(4))
    LOGGER.info("wrote %s (%d rows)", args.output, len(df))
    plot_stage_curve(df, args.pdf)
    LOGGER.info("wrote %s", args.pdf)


if __name__ == "__main__":
    main()
