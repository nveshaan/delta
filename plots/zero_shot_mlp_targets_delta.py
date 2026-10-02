"""Plot each MLP target's zero-shot composite gain over ``labels``.

Modality composite scores are computed as in
``zero_shot_encoder_method_consistency.py``, except that ``mlp_targets`` is
kept separate: each dataset uses ``pooled_nonzero`` when subtype comparisons
exist and ``label_0_vs_1`` otherwise. AUROC, AUPRC, and precision-at-n are
averaged over runs to form the dataset composite. Dataset composites are then
averaged within each modality. For every encoder, method, support size, and
modality, the ``labels`` score is subtracted from the ``scores`` and
``label_scores`` scores. Because the comparison is paired within a modality,
differences in difficulty between modalities cancel out. Only the effect of the
target remains.

The figure is a grid with one row per encoder and one column per method. Each
panel shows the mean difference across modalities as a line with circle
markers, the individual modality differences as small points shaped by
modality, and a zero line for parity with ``labels``. Each panel has its own
y-axis range. PNG and PDF outputs are written to
``assets/`` by default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from zero_shot_encoder_method_consistency import (
    DEFAULT_TRACKING_URI,
    GROUP_COLUMNS,
    MARKERS,
    METRICS,
    PALETTE,
    PROJECT_ROOT,
    _apply_publication_style,
    _choose_dataset_comparisons,
    load_zero_shot_runs,
)


DEFAULT_OUTPUT_STEM = PROJECT_ROOT / "assets" / "zero_shot_mlp_targets_delta"
ENCODERS = ("MedImageInsight", "MedSigLIP")
METHODS = ("laplacianshot", "laplacianshot_msde")
BASELINE_TARGET = "labels"
COMPARED_TARGETS = ("scores", "label_scores")
TARGET_COLORS = {"scores": PALETTE["red_strong"], "label_scores": PALETTE["teal"]}
# Horizontal offset, in support-size units, so the two targets do not overlap.
DODGE = {"scores": -0.6, "label_scores": 0.6}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def modality_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """Return the composite score for every pair/support/target/modality combination."""
    runs = runs[runs["encoder"].isin(ENCODERS) & runs["method"].isin(METHODS)]
    if runs.empty:
        raise ValueError(f"No runs found for encoders {ENCODERS} and methods {METHODS}")
    runs = _choose_dataset_comparisons(runs)

    # Average metrics at dataset level, then create the composite score.
    dataset_group = [*GROUP_COLUMNS[:-1], "mlp_targets"]
    dataset_scores = runs.groupby(dataset_group, as_index=False, dropna=False)[list(METRICS)].mean()
    dataset_scores["composite"] = dataset_scores[list(METRICS)].mean(axis=1)

    # Equal weight for every dataset within each modality.
    modality_group = ["encoder", "method", "support_size", "mlp_targets", "modality"]
    return dataset_scores.groupby(modality_group, as_index=False, dropna=False)["composite"].mean()


def paired_differences(runs: pd.DataFrame) -> pd.DataFrame:
    """Return per-modality composite differences of each target against ``labels``."""
    scores = modality_scores(runs).pivot_table(
        index=["encoder", "method", "support_size", "modality"],
        columns="mlp_targets",
        values="composite",
    )
    missing = [target for target in (BASELINE_TARGET, *COMPARED_TARGETS) if target not in scores]
    if missing:
        raise ValueError(f"Runs are missing mlp_targets: {', '.join(missing)}")
    differences = scores[list(COMPARED_TARGETS)].sub(scores[BASELINE_TARGET], axis=0)
    return differences.dropna().reset_index().melt(
        id_vars=["encoder", "method", "support_size", "modality"],
        var_name="mlp_targets",
        value_name="delta",
    )


def plot_differences(differences: pd.DataFrame, output_stem: Path = DEFAULT_OUTPUT_STEM) -> list[Path]:
    """Create an encoder × method grid of paired differences and save PNG/PDF outputs."""
    _apply_publication_style()
    fig, axes = plt.subplots(
        len(ENCODERS),
        len(METHODS),
        figsize=(7.5 * len(METHODS), 6.0 * len(ENCODERS)),
        squeeze=False,
        sharex=True,
        sharey=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.15, h_pad=0.15, hspace=0.12, wspace=0.12)
    support_sizes = sorted(differences["support_size"].unique())
    # Circles are reserved for the mean lines, so modalities use the other markers.
    modalities = sorted(differences["modality"].unique())
    modality_markers = {modality: MARKERS[1 + index % (len(MARKERS) - 1)] for index, modality in enumerate(modalities)}

    for row, encoder in enumerate(ENCODERS):
        for col, method in enumerate(METHODS):
            axis = axes[row, col]
            panel = differences[(differences["encoder"] == encoder) & (differences["method"] == method)]
            axis.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--", zorder=1)
            for target in COMPARED_TARGETS:
                series = panel[panel["mlp_targets"] == target]
                for modality, points in series.groupby("modality"):
                    axis.scatter(
                        points["support_size"] + DODGE[target],
                        points["delta"],
                        s=26,
                        color=TARGET_COLORS[target],
                        marker=modality_markers[modality],
                        alpha=0.45,
                        linewidths=0,
                        zorder=2,
                    )
                means = series.groupby("support_size")["delta"].mean()
                axis.plot(
                    means.index + DODGE[target],
                    means.to_numpy(),
                    linewidth=2.0,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    zorder=3,
                    color=TARGET_COLORS[target],
                    marker="o",
                )
            axis.set_title(f"{encoder} · {method}")
            axis.set_xticks(support_sizes)
            axis.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.55)

    for axis in axes[-1]:
        axis.set_xlabel("Support size")
    fig.supylabel(f"Δ composite vs {BASELINE_TARGET}", fontsize=plt.rcParams["axes.labelsize"])

    target_handles = [
        Line2D([0], [0], linewidth=2.0, markersize=7, markeredgecolor="white", label=f"{target} (mean)",
               color=TARGET_COLORS[target], marker="o")
        for target in COMPARED_TARGETS
    ]
    modality_handles = [
        Line2D([0], [0], marker=modality_markers[modality], color="#777777", alpha=0.7, linestyle="None",
               markersize=6, label=modality)
        for modality in modalities
    ]
    fig.legend(
        handles=[*target_handles, *modality_handles],
        loc="outside lower center",
        ncol=len(target_handles) + len(modality_handles),
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.suptitle(f"Zero-shot composite gain of MLP targets over {BASELINE_TARGET}")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default="delta")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_zero_shot_runs(tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)
    differences = paired_differences(runs)
    outputs = plot_differences(differences, output_stem=args.output_stem)
    LOGGER.info("Computed %d paired modality differences", len(differences))
    LOGGER.info("Saved %s", ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
