"""Plot and summarise ``assets/structural_asymmetry.csv`` (scripts/structural_asymmetry.py).

Figures (PNG and PDF, written to ``assets/structural_asymmetry/``):

1. ``auroc_move_heatmap``  : MSDE movement as an anomaly score (AUROC) for every
   dataset x encoder. Red above 0.5 (anomalies move more, the regime MSDE
   assumes), blue below 0.5 (the direction is flipped).
2. ``direction_vs_scale``  : AUROC of movement against the log ratio of local scale
   (anomaly / normal k-NN distance). Shows how much of the direction is explained by
   relative compactness.
3. ``asymmetry_signature`` : per-dataset signed asymmetry (anomaly vs normal) of each
   structural statistic, averaged over encoders; columns share one colour scale
   within a family only through their own symmetric normalisation.
4. ``direction_drivers``   : Spearman correlation of every asymmetry statistic with
   the AUROC of movement across all rows.

It also writes ``structural_asymmetry_summary.csv`` (one row per dataset, averaged
over encoders) with a direction call: ``anomaly_sparse`` (mean AUROC > 0.55),
``anomaly_dense`` (< 0.45) or ``ambiguous``.

Usage::

    uv run python plots/structural_asymmetry.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zero_shot_encoder_method_consistency import PALETTE, PROJECT_ROOT, _apply_publication_style

DEFAULT_INPUT = PROJECT_ROOT / "assets" / "structural_asymmetry.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "structural_asymmetry"
MODALITY_ORDER = ("chest", "fundus", "mri", "oct")
MODALITY_TITLES = {"chest": "Chest X-ray", "fundus": "Fundus", "mri": "MRI", "oct": "OCT"}
MODALITY_COLORS = {
    "chest": PALETTE["blue_main"],
    "fundus": PALETTE["green_3"],
    "mri": PALETTE["red_strong"],
    "oct": PALETTE["violet"],
}
ENCODER_ORDER = ("CLIP", "BiomedCLIP", "MedImageInsight", "MedSigLIP", "UniMedCLIP")
FLIPPED, SPARSE = 0.45, 0.55
LABELS = {
    "spread": "Spread",
    "knn_within": "k-NN dist (in class)",
    "knn_pooled": "k-NN dist (pooled)",
    "move": "MSDE movement",
    "twonn_id": "TwoNN dim.",
    "local_pca_id": "Local PCA dim.",
    "pr_dim": "Participation ratio",
    "pca95_dim": "PCA-95% dim.",
    "n_modes": "Sub-modes",
    "silhouette": "Silhouette",
    "weight_entropy": "Weight entropy",
    "weight_cv": "Weight CV",
    "deg_mean": "Graph degree",
    "intra_frac": "Intra-class edges",
    "n_components": "Components",
    "giant_frac": "Giant component",
}
FAMILIES = {
    "Compactness": ("spread", "knn_within", "knn_pooled", "move"),
    "Intrinsic dim.": ("twonn_id", "local_pca_id", "pr_dim", "pca95_dim"),
    "Heterogeneity": ("n_modes", "silhouette", "weight_entropy", "weight_cv"),
    "Graph": ("deg_mean", "intra_frac", "n_components", "giant_frac"),
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _asym_column(df: pd.DataFrame, metric: str) -> str:
    return f"{metric}_logratio" if f"{metric}_logratio" in df else f"{metric}_diff"


def _dataset_order(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(modality, dataset) sorted by modality, then by mean AUROC descending."""
    mean = df.groupby(["modality", "dataset"])["auroc_move"].mean().reset_index()
    mean["m"] = mean["modality"].map(MODALITY_ORDER.index)
    mean = mean.sort_values(["m", "auroc_move"], ascending=[True, False])
    return list(zip(mean["modality"], mean["dataset"]))


def _save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def _separators(ax, order, horizontal: bool = True) -> None:
    for i in range(1, len(order)):
        if order[i][0] != order[i - 1][0]:
            (ax.axhline if horizontal else ax.axvline)(i - 0.5, color="black", lw=2.0)


def _short(name: str, width: int = 34) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def _label(modality: str, dataset: str) -> str:
    return f"{MODALITY_TITLES[modality]} · {_short(dataset, 28)}"


def plot_auroc_heatmap(df: pd.DataFrame, out: Path) -> list[Path]:
    order = _dataset_order(df)
    encoders = [e for e in ENCODER_ORDER if e in set(df["encoder"])]
    pivot = df.pivot_table(index=["modality", "dataset"], columns="encoder", values="auroc_move").reindex(order)[encoders]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(order) + 1.6))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(encoders)), encoders, rotation=30, ha="right")
    ax.set_yticks(range(len(order)), [_label(m, d) for m, d in order], fontsize=11)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v - 0.5) > 0.22 else "black")
    _separators(ax, order)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("AUROC of MSDE movement\n(< 0.5: anomalies move less)")
    ax.set_title("MSDE movement as anomaly score", fontsize=12)
    return _save(fig, out / "auroc_move_heatmap")


def plot_direction_vs_scale(df: pd.DataFrame, out: Path) -> list[Path]:
    from scipy.stats import spearmanr

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, metric, title in (
        (axes[0], "knn_within", "Local scale (k-NN distance)"),
        (axes[1], "twonn_id", "Intrinsic dimension (TwoNN)"),
    ):
        x = df[f"{metric}_logratio"]
        for m in MODALITY_ORDER:
            sub = df[df["modality"] == m]
            ax.scatter(sub[f"{metric}_logratio"], sub["auroc_move"], s=34, alpha=0.8, color=MODALITY_COLORS[m],
                       edgecolor="white", linewidth=0.5, label=MODALITY_TITLES[m])
        ax.axhline(0.5, color="#555555", lw=1.2, ls="--")
        ax.axvline(0, color="#555555", lw=1.2, ls="--")
        rho = spearmanr(x, df["auroc_move"], nan_policy="omit").statistic
        ax.text(0.04, 0.95, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top")
        ax.set_xlabel("log(anomaly / normal)")
        ax.set_title(title, fontsize=14)
    axes[0].set_ylabel("AUROC of MSDE movement")
    axes[0].legend(loc="lower right", fontsize=11, handletextpad=0.2)
    return _save(fig, out / "direction_vs_scale")


def plot_signature(df: pd.DataFrame, out: Path) -> list[Path]:
    order = _dataset_order(df)
    metrics = [m for fam in FAMILIES.values() for m in fam if _asym_column(df, m) in df]
    columns = [_asym_column(df, m) for m in metrics]
    mean = df.groupby(["modality", "dataset"])[columns].mean().reindex(order)
    scale = mean.abs().quantile(0.9).replace(0, 1)
    norm = (mean / scale).clip(-1, 1)
    fig, ax = plt.subplots(figsize=(0.62 * len(metrics) + 4.8, 0.36 * len(order) + 2.6))
    im = ax.imshow(norm.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), [LABELS[m] for m in metrics], rotation=40, ha="right", fontsize=11)
    ax.set_yticks(range(len(order)), [_label(m, d) for m, d in order], fontsize=11)
    for i in range(mean.shape[0]):
        for j, col in enumerate(columns):
            v = mean.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(norm.iat[i, j]) > 0.6 else "black")
    start = 0
    for fam, members in FAMILIES.items():
        count = sum(m in metrics for m in members)
        if start:
            ax.axvline(start - 0.5, color="black", lw=2.0)
        ax.text(start + count / 2 - 0.5, -1.2, fam, ha="center", va="bottom", fontsize=12, fontweight="bold")
        start += count
    _separators(ax, order)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("anomaly − normal\n(per-column scaled; red = anomaly larger)")
    return _save(fig, out / "asymmetry_signature")


def plot_drivers(df: pd.DataFrame, out: Path) -> list[Path]:
    rows = []
    for metric, label in LABELS.items():
        col = _asym_column(df, metric)
        if col in df and metric != "move":
            rows.append((label, df[[col, "auroc_move"]].corr(method="spearman").iat[0, 1], metric))
    rows.sort(key=lambda r: r[1])
    family = {m: f for f, ms in FAMILIES.items() for m in ms}
    fam_colors = dict(zip(FAMILIES, (PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["green_3"], PALETTE["violet"])))
    fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(rows) + 1.4))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[fam_colors[family[r[2]]] for r in rows])
    ax.axvline(0, color="black", lw=1.5)
    ax.set_xlabel("Spearman ρ with AUROC of MSDE movement")
    ax.set_xlim(-1, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_colors.values()]
    ax.legend(handles, fam_colors, loc="lower right", fontsize=11)
    return _save(fig, out / "direction_drivers")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in LABELS if _asym_column(df, m) in df]
    agg = df.groupby(["modality", "dataset"]).agg(
        auroc_move_mean=("auroc_move", "mean"),
        auroc_move_min=("auroc_move", "min"),
        auroc_move_max=("auroc_move", "max"),
        n_anomaly_types=("n_anomaly_types", "first"),
        **{_asym_column(df, m): (_asym_column(df, m), "mean") for m in metrics},
    ).reset_index()
    agg["direction"] = np.select(
        [agg["auroc_move_mean"] > SPARSE, agg["auroc_move_mean"] < FLIPPED], ["anomaly_sparse", "anomaly_dense"], "ambiguous"
    )
    # Encoders disagree on the sign of the call when AUROC straddles 0.5.
    agg["encoders_flipped"] = df.assign(f=df["auroc_move"] < 0.5).groupby(["modality", "dataset"])["f"].sum().to_numpy()
    agg["m"] = agg["modality"].map(MODALITY_ORDER.index)
    return agg.sort_values(["m", "auroc_move_mean"], ascending=[True, False]).drop(columns="m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    _apply_publication_style()
    paths = []
    for plot in (plot_auroc_heatmap, plot_direction_vs_scale, plot_signature, plot_drivers):
        paths += plot(df, args.output_dir)
    summary = summarise(df)
    summary_path = args.output_dir.parent / "structural_asymmetry_summary.csv"
    summary.to_csv(summary_path, index=False)
    for path in [*paths, summary_path]:
        LOGGER.info("wrote %s", path)
    LOGGER.info("direction calls: %s", summary["direction"].value_counts().to_dict())


if __name__ == "__main__":
    main()
