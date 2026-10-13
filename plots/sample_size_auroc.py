"""Plot how the AUROC of MSDE movement depends on the number of samples used.

Reads ``assets/sample_size_sweep.csv`` (scripts/sample_size_sweep.py) and writes,
as PNG and PDF, to ``assets/sample_size/``:

1. ``sample_size_curves``      : one panel per dataset, AUROC against samples per
   class (log axis), one line per encoder (mean over seeds, band = min to max).
2. ``sample_size_correlation`` : per dataset x encoder, the Spearman correlation
   between log(samples) and the seed-averaged AUROC (left), and the AUROC change
   from the smallest to the largest sample size (right).

``assets/sample_size_correlation.csv`` holds the numbers behind figure 2. Datasets
with a small minority class are swept over fewer sizes, so their correlation rests
on few points.

Usage::

    uv run python plots/sample_size_auroc.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from zero_shot_encoder_method_consistency import PALETTE, PROJECT_ROOT, _apply_publication_style

DEFAULT_INPUT = PROJECT_ROOT / "assets" / "sample_size_sweep.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "sample_size"
MODALITY_ORDER = ("chest", "fundus", "mri", "oct")
MODALITY_TITLES = {"chest": "Chest X-ray", "fundus": "Fundus", "mri": "MRI", "oct": "OCT"}
ENCODER_ORDER = ("CLIP", "BiomedCLIP", "MedImageInsight", "MedSigLIP", "UniMedCLIP")
ENCODER_COLORS = dict(zip(ENCODER_ORDER, (
    PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["green_3"], PALETTE["violet"], "#E69F00",
)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _short(name: str, width: int = 26) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def _save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def _dataset_order(df: pd.DataFrame) -> list[tuple[str, str]]:
    mean = df.groupby(["modality", "dataset"])["auroc_move"].mean().reset_index()
    mean["m"] = mean["modality"].map(MODALITY_ORDER.index)
    mean = mean.sort_values(["m", "auroc_move"], ascending=[True, False])
    return list(zip(mean["modality"], mean["dataset"]))


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman(log n, seed-mean AUROC) and AUROC gain per (dataset, encoder)."""
    mean = df.groupby(["modality", "dataset", "encoder", "n_per_class"])["auroc_move"].mean().reset_index()
    rows = []
    for (modality, dataset, encoder), g in mean.groupby(["modality", "dataset", "encoder"]):
        g = g.sort_values("n_per_class")
        rho = spearmanr(np.log(g["n_per_class"]), g["auroc_move"]).statistic if len(g) >= 3 else np.nan
        rows.append({
            "modality": modality, "dataset": dataset, "encoder": encoder, "n_sizes": len(g),
            "n_min": int(g["n_per_class"].iloc[0]), "n_max": int(g["n_per_class"].iloc[-1]),
            "auroc_at_n_min": g["auroc_move"].iloc[0], "auroc_at_n_max": g["auroc_move"].iloc[-1],
            "spearman_logn_auroc": rho, "auroc_gain": g["auroc_move"].iloc[-1] - g["auroc_move"].iloc[0],
        })
    return pd.DataFrame(rows)


def plot_curves(df: pd.DataFrame, out: Path) -> list[Path]:
    order = _dataset_order(df)
    ncols = 5
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, (modality, dataset) in zip(axes, order):
        sub = df[(df["modality"] == modality) & (df["dataset"] == dataset)]
        for encoder in ENCODER_ORDER:
            g = sub[sub["encoder"] == encoder].groupby("n_per_class")["auroc_move"].agg(["mean", "min", "max"])
            if g.empty:
                continue
            ax.plot(g.index, g["mean"], marker="o", ms=3.5, lw=1.8, color=ENCODER_COLORS[encoder], label=encoder)
            ax.fill_between(g.index, g["min"], g["max"], color=ENCODER_COLORS[encoder], alpha=0.18, linewidth=0)
        ax.axhline(0.5, color="#555555", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{MODALITY_TITLES[modality]} · {_short(dataset, 20)}", fontsize=10.5)
        ax.tick_params(labelsize=10)
    for ax in axes[len(order):]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel("AUROC", fontsize=12)
    for ax in axes[-ncols:]:
        ax.set_xlabel("samples per class", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=1, fontsize=12, bbox_to_anchor=(0.99, 0.02))
    fig.tight_layout()
    return _save(fig, out / "sample_size_curves")


def plot_correlation(corr: pd.DataFrame, order: list[tuple[str, str]], out: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.36 * len(order) + 2.0), sharey=True)
    specs = (
        ("spearman_logn_auroc", "Spearman ρ (log n, AUROC)", -1.0, 1.0, "{:+.1f}"),
        ("auroc_gain", "AUROC gain, smallest → largest n", -0.3, 0.3, "{:+.2f}"),
    )
    for ax, (column, label, lo, hi, fmt) in zip(axes, specs):
        pivot = corr.pivot_table(index=["modality", "dataset"], columns="encoder", values=column).reindex(order)[list(ENCODER_ORDER)]
        im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto")
        ax.set_xticks(range(len(ENCODER_ORDER)), ENCODER_ORDER, rotation=35, ha="right", fontsize=11)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.6 * hi else "black")
        for i in range(1, len(order)):
            if order[i][0] != order[i - 1][0]:
                ax.axhline(i - 0.5, color="black", lw=2.0)
        ax.tick_params(axis="y", left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02).set_label(label)
    axes[0].set_yticks(range(len(order)), [f"{MODALITY_TITLES[m]} · {_short(d, 24)}" for m, d in order], fontsize=11)
    fig.tight_layout()
    return _save(fig, out / "sample_size_correlation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    _apply_publication_style()
    corr = correlations(df)
    corr_path = args.output_dir.parent / "sample_size_correlation.csv"
    corr.to_csv(corr_path, index=False)
    paths = plot_curves(df, args.output_dir) + plot_correlation(corr, _dataset_order(df), args.output_dir)
    for path in [*paths, corr_path]:
        LOGGER.info("wrote %s", path)
    LOGGER.info(
        "median Spearman %.2f; AUROC gain median %.3f; rows with gain > 0.05: %d/%d",
        corr["spearman_logn_auroc"].median(), corr["auroc_gain"].median(),
        int((corr["auroc_gain"] > 0.05).sum()), len(corr),
    )


if __name__ == "__main__":
    main()
