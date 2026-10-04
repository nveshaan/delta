"""Plot zero-shot composite gains of ``scores`` and ``distances`` over ``labels``.

Modality composite scores are computed as in
``zero_shot_encoder_method_consistency.py``, except that ``mlp_targets`` is
kept separate: each dataset uses ``pooled_nonzero`` when subtype comparisons
exist and ``label_0_vs_1`` otherwise. AUROC, AUPRC, and precision-at-n are
averaged over runs to form the dataset composite. Dataset composites are then
averaged within each modality. For every encoder, method, support size, and
modality, the ``labels`` score is subtracted from the ``scores`` (same_label
MSDE + GDE) and ``distances`` (zero_label MSDE) scores. Because the comparison
is paired within a modality, differences in difficulty between modalities
cancel out. Only the effect of the target remains.

The figure is a grid with one row per encoder and one column per method (all
those with runs, or the ones passed with ``--encoders`` / ``--methods``). Each
panel shows the mean difference across modalities as a line with circle
markers (solid for ``scores``, dashed for ``distances``), the individual
modality differences as small points shaped by modality, and a zero line for
parity with ``labels``. Each panel has its own y-axis range. PNG and PDF
outputs are written to ``assets/`` by default.
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
BASELINE_TARGET = "labels"
COMPARED_TARGETS = ("scores", "distances")
TARGET_STYLES = {
    "scores": {"color": PALETTE["teal"], "linestyle": "-"},
    "distances": {"color": PALETTE["violet"], "linestyle": "--"},
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def modality_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """Return the composite score for every encoder/method/support/target/modality combination."""
    runs = _choose_dataset_comparisons(runs)

    # Average metrics at dataset level, then create the composite score.
    dataset_group = [*GROUP_COLUMNS[:-1], "mlp_targets"]
    dataset_scores = runs.groupby(dataset_group, as_index=False, dropna=False)[list(METRICS)].mean()
    dataset_scores["composite"] = dataset_scores[list(METRICS)].mean(axis=1)

    # Equal weight for every dataset within each modality.
    modality_group = ["encoder", "method", "support_size", "mlp_targets", "modality"]
    return dataset_scores.groupby(modality_group, as_index=False, dropna=False)["composite"].mean()


def paired_differences(runs: pd.DataFrame) -> pd.DataFrame:
    """Return per-modality composite differences of ``scores``/``distances`` against ``labels``."""
    scores = modality_scores(runs).pivot_table(
        index=["encoder", "method", "support_size", "modality"],
        columns="mlp_targets",
        values="composite",
    )
    if BASELINE_TARGET not in scores:
        raise ValueError(f"Runs are missing the {BASELINE_TARGET!r} baseline")
    present = [target for target in COMPARED_TARGETS if target in scores]
    if not present:
        raise ValueError(f"Runs have none of the compared targets: {', '.join(COMPARED_TARGETS)}")
    differences = scores[present].sub(scores[BASELINE_TARGET], axis=0).reset_index().melt(
        id_vars=["encoder", "method", "support_size", "modality"],
        var_name="mlp_targets",
        value_name="delta",
    ).dropna(subset=["delta"])
    if differences.empty:
        raise ValueError(
            f"No encoder/method/support size/modality has both {BASELINE_TARGET!r} and one of "
            f"{', '.join(COMPARED_TARGETS)}"
        )
    return differences


def plot_differences(differences: pd.DataFrame, output_stem: Path = DEFAULT_OUTPUT_STEM) -> list[Path]:
    """Create an encoder × method grid of paired differences and save PNG/PDF outputs."""
    targets = [target for target in COMPARED_TARGETS if target in set(differences["mlp_targets"])]
    encoders = sorted(differences["encoder"].unique())
    methods = sorted(differences["method"].unique())
    _apply_publication_style()
    fig, axes = plt.subplots(
        len(encoders),
        len(methods),
        figsize=(7.5 * len(methods), 6.0 * len(encoders)),
        squeeze=False,
        sharex=True,
        sharey=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.15, h_pad=0.15, hspace=0.12, wspace=0.12)
    support_sizes = sorted(differences["support_size"].unique())
    # Horizontal offset, in support-size units, so the targets do not overlap.
    dodge = {target: (index - (len(targets) - 1) / 2) * 1.2 for index, target in enumerate(targets)}
    # Circles are reserved for the mean lines, so modalities use the other markers.
    modalities = sorted(differences["modality"].unique())
    modality_markers = {modality: MARKERS[1 + index % (len(MARKERS) - 1)] for index, modality in enumerate(modalities)}

    for row, encoder in enumerate(encoders):
        for col, method in enumerate(methods):
            axis = axes[row, col]
            panel = differences[(differences["encoder"] == encoder) & (differences["method"] == method)]
            axis.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--", zorder=1)
            for target in targets:
                series = panel[panel["mlp_targets"] == target]
                for modality, points in series.groupby("modality"):
                    axis.scatter(
                        points["support_size"] + dodge[target],
                        points["delta"],
                        s=26,
                        color=TARGET_STYLES[target]["color"],
                        marker=modality_markers[modality],
                        alpha=0.45,
                        linewidths=0,
                        zorder=2,
                    )
                means = series.groupby("support_size")["delta"].mean()
                axis.plot(
                    means.index + dodge[target],
                    means.to_numpy(),
                    linewidth=2.0,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    zorder=3,
                    marker="o",
                    **TARGET_STYLES[target],
                )
            axis.set_title(f"{encoder} · {method}")
            axis.set_xticks(support_sizes)
            axis.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.55)

    for axis in axes[-1]:
        axis.set_xlabel("Support size")
    fig.supylabel(f"Δ composite vs {BASELINE_TARGET}", fontsize=plt.rcParams["axes.labelsize"])

    target_handles = [
        Line2D([0], [0], linewidth=2.0, markersize=7, markeredgecolor="white", label=f"{target} (mean)",
               marker="o", **TARGET_STYLES[target])
        for target in targets
    ]
    modality_handles = [
        Line2D([0], [0], marker=modality_markers[modality], color="#777777", alpha=0.7, linestyle="None",
               markersize=6, label=modality)
        for modality in modalities
    ]
    fig.legend(
        handles=[*target_handles, *modality_handles],
        loc="outside lower center",
        ncol=min(len(target_handles) + len(modality_handles), 6),
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
    parser.add_argument("--campaign", default="adhoc", help="zero-shot runs to plot (adhoc: grid, optuna: tuned)")
    parser.add_argument("--encoders", nargs="+", help="encoders to plot (default: all with runs)")
    parser.add_argument("--methods", nargs="+", help="methods to plot (default: all with runs)")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_zero_shot_runs(
        tracking_uri=args.tracking_uri, experiment_name=args.experiment_name, campaign=args.campaign
    )
    if args.encoders:
        runs = runs[runs["encoder"].isin(args.encoders)]
    if args.methods:
        runs = runs[runs["method"].isin(args.methods)]
    if runs.empty:
        raise ValueError(f"No runs found for encoders {args.encoders} and methods {args.methods}")
    differences = paired_differences(runs)
    outputs = plot_differences(differences, output_stem=args.output_stem)
    LOGGER.info("Computed %d paired modality differences", len(differences))
    LOGGER.info("Saved %s", ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
