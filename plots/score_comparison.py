"""Radar plots of ``assets/score_comparison.csv`` (scripts/score_comparison.py).

Radial value is separability = max(AUROC, 1 - AUROC): how well a label-free score
separates normal from anomalous, ignoring its direction. The dashed ring is chance
(0.5). PNG and PDF outputs are written to ``assets/score_comparison/``:

1. ``score_radar_modality``   : one spoke per score, one polygon per modality
   (mean over the datasets of that modality).
2. ``score_radar_datasets``   : one radar per modality, one spoke per dataset, one
   polygon per selected score.

Usage::

    uv run python plots/score_comparison.py
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

DEFAULT_INPUT = PROJECT_ROOT / "assets" / "score_comparison.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "score_comparison"
MODALITY_ORDER = ("chest", "fundus", "mri", "oct")
MODALITY_TITLES = {"chest": "Chest X-ray", "fundus": "Fundus", "mri": "MRI", "oct": "OCT"}
MODALITY_COLORS = {
    "chest": PALETTE["blue_main"], "fundus": PALETTE["green_3"],
    "mri": PALETTE["red_strong"], "oct": PALETTE["violet"],
}
LABELS = {
    "kde_density": "KDE density", "knn_dist_50": "k-NN dist (50)", "centroid_dist": "Centroid dist",
    "knn_dist_10": "k-NN dist (10)", "msde_move": "MSDE movement", "local_pca_dim": "Local PCA dim.",
    "twonn_mu": "TwoNN ratio", "graph_degree": "Graph degree", "hubness": "Hubness",
    "msde_weight": "MSDE weight",
}
# Scores drawn in the per-modality dataset radars (a readable subset).
SELECTED = ("kde_density", "knn_dist_50", "msde_move", "local_pca_dim", "msde_weight")
SELECTED_STYLE = {
    "kde_density": (PALETTE["blue_main"], "-"), "knn_dist_50": (PALETTE["teal"], "-"),
    "msde_move": (PALETTE["red_strong"], "-"), "local_pca_dim": ("#E69F00", "-"),
    "msde_weight": ("#7F7F7F", "--"),
}
RMIN, RMAX = 0.4, 1.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def _radar_axes(ax, labels: list[str], fontsize: float = 11, radial_labels: bool = True) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, labels, fontsize=fontsize)
    ax.set_ylim(RMIN, RMAX)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if radial_labels:
        ax.set_yticklabels(["0.5", "", "0.7", "", "0.9", ""], fontsize=9, color="#555555")
        ax.set_rlabel_position(180 / len(labels))
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", pad=14)
    ax.grid(color="#CFCECE", linewidth=0.8)
    ax.spines["polar"].set_color("#CFCECE")
    ax.spines["polar"].set_linewidth(1.0)
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 0.5), color="#555555", lw=1.2, ls="--")
    return angles


def _polygon(ax, angles, values, color, ls="-", label=None, fill=0.10, lw=2.2):
    angles = np.r_[angles, angles[:1]]
    values = np.r_[values, values[:1]]
    ax.plot(angles, values, color=color, ls=ls, lw=lw, marker="o", ms=3.5, label=label)
    ax.fill(angles, values, color=color, alpha=fill)


def plot_modality_radar(df: pd.DataFrame, out: Path) -> list[Path]:
    mean = df.groupby(["modality", "score"])["separability"].mean().unstack(0)
    order = df.groupby("score")["separability"].mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(7.4, 7.4), subplot_kw={"projection": "polar"})
    angles = _radar_axes(ax, [LABELS[s] for s in order], fontsize=12)
    for modality in MODALITY_ORDER:
        _polygon(ax, angles, mean.loc[order, modality].to_numpy(), MODALITY_COLORS[modality],
                 label=MODALITY_TITLES[modality])
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=12)
    ax.set_title("Mean separability per score (dashed = chance)", pad=26, fontsize=13)
    return _save(fig, out / "score_radar_modality")


def plot_dataset_radars(df: pd.DataFrame, out: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.5), subplot_kw={"projection": "polar"})
    for ax, modality in zip(axes.ravel(), MODALITY_ORDER):
        sub = df[df["modality"] == modality]
        pivot = sub.pivot(index="dataset", columns="score", values="separability")
        pivot = pivot.loc[pivot[list(SELECTED)].mean(axis=1).sort_values(ascending=False).index]
        names = [d if len(d) <= 22 else d[:21] + "…" for d in pivot.index]
        angles = _radar_axes(ax, names, fontsize=10, radial_labels=False)
        for score in SELECTED:
            color, ls = SELECTED_STYLE[score]
            _polygon(ax, angles, pivot[score].to_numpy(), color, ls=ls, label=LABELS[score],
                     fill=0.0 if ls == "--" else 0.07, lw=1.9)
        ax.set_title(MODALITY_TITLES[modality], pad=22, fontsize=14, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(SELECTED), fontsize=12, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.045, "rings at 0.5 (dashed, chance), 0.6, 0.7, 0.8, 0.9, 1.0 separability",
             ha="center", fontsize=11, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, out / "score_radar_datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    _apply_publication_style()
    for path in plot_modality_radar(df, args.output_dir) + plot_dataset_radars(df, args.output_dir):
        LOGGER.info("wrote %s", path)


if __name__ == "__main__":
    main()
