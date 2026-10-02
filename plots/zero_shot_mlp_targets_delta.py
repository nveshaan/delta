"""Plot zero-shot composite gains of MSDE-based MLP targets over ``labels``.

Modality composite scores are computed as in
``zero_shot_encoder_method_consistency.py``, except that ``mlp_targets`` is
kept separate: each dataset uses ``pooled_nonzero`` when subtype comparisons
exist and ``label_0_vs_1`` otherwise. AUROC, AUPRC, and precision-at-n are
averaged over runs to form the dataset composite. Dataset composites are then
averaged within each modality. For every encoder, method, support size, and
modality, the ``labels`` score is subtracted from the score of each
``mlp_targets`` × ``msde_mode`` target. Runs logged before ``msde_mode``
existed are mapped onto it: ``scores`` -> ``scores · msde`` and
``label_scores`` -> ``scores · same_label``. Targets without runs are skipped.
Because the comparison is paired within a modality, differences in difficulty
between modalities cancel out. Only the effect of the target remains.

Two figures are written:

1. ``<stem>``: all targets -- ``scores`` and ``distances`` with ``msde``,
   ``same_label``, and ``zero_label``, plus ``scores · no_msde``.
2. ``<stem>_selected``: ``scores · same_label`` and ``distances · zero_label``.

Each figure is a grid with one row per encoder and one column per method. Each
panel shows the mean difference across modalities as a line with circle
markers (solid for ``scores``, dashed for ``distances``; color is the
``msde_mode``), the individual modality differences as small points shaped by
modality, and a zero line for parity with ``labels``. Each panel has its own
y-axis range. PNG and PDF outputs are written to ``assets/`` by default.
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
    _normalise_tracking_uri,
)


DEFAULT_OUTPUT_STEM = PROJECT_ROOT / "assets" / "zero_shot_mlp_targets_delta"
ENCODERS = ("MedImageInsight", "MedSigLIP")
METHODS = ("laplacianshot", "laplacianshot_msde")
BASELINE_TARGET = "labels"
# Targets are "<mlp_targets> · <msde_mode>".
ALL_TARGETS = (
    "scores · no_msde",
    "scores · msde",
    "scores · same_label",
    "scores · zero_label",
    "distances · msde",
    "distances · same_label",
    "distances · zero_label",
)
SELECTED_TARGETS = ("scores · same_label", "distances · zero_label")
MODE_COLORS = {
    "no_msde": PALETTE["blue_secondary"],
    "msde": PALETTE["red_strong"],
    "same_label": PALETTE["teal"],
    "zero_label": PALETTE["violet"],
}
TARGET_LINESTYLES = {"scores": "-", "distances": "--"}
# Runs logged before msde_mode existed encode it in mlp_targets alone.
LEGACY_TARGETS = {"labels": "labels", "scores": "scores · msde", "label_scores": "scores · same_label"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_runs_by_target(
    *, tracking_uri: str = DEFAULT_TRACKING_URI, experiment_name: str = "delta"
) -> pd.DataFrame:
    """Query finished zero-shot metric runs, keyed by target.

    The ``mlp_targets`` column holds ``labels`` for pseudolabel targets and
    ``"<mlp_targets> · <msde_mode>"`` for MSDE-based targets, so the rest of the
    script can compare them as plain targets.
    """
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("MLflow is required to query the experiment runs") from error

    mlflow.set_tracking_uri(_normalise_tracking_uri(tracking_uri))
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="params.type = 'zero_shot' and attributes.status = 'FINISHED'",
        output_format="pandas",
    )
    if runs.empty:
        raise ValueError(f"No finished zero-shot runs found in MLflow experiment {experiment_name!r}")

    msde_mode = runs["params.msde_mode"] if "params.msde_mode" in runs else pd.Series(None, index=runs.index)
    mlp_targets = runs["params.mlp_targets"]
    legacy = mlp_targets.map(LEGACY_TARGETS)
    new = mlp_targets.where(mlp_targets == "labels", mlp_targets + " · " + msde_mode.astype(str))
    runs["params.mlp_targets"] = new.where(msde_mode.notna(), legacy)

    renamed = runs.rename(
        columns={
            "params.encoder": "encoder",
            "params.method": "method",
            "params.modality": "modality",
            "params.dataset": "dataset",
            "params.comparison": "comparison",
            "params.support_size": "support_size",
            "params.mlp_targets": "mlp_targets",
            "metrics.auroc": "auroc",
            "metrics.auprc": "auprc",
            "metrics.p_at_n": "p_at_n",
        }
    )
    required = [*GROUP_COLUMNS, "mlp_targets", *METRICS]
    missing = [column for column in required if column not in renamed]
    if missing:
        raise ValueError(f"MLflow runs are missing required columns: {', '.join(missing)}")

    selected = renamed[required].copy()
    selected["support_size"] = pd.to_numeric(selected["support_size"], errors="coerce")
    for metric in METRICS:
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected.dropna(subset=required)
    selected["support_size"] = selected["support_size"].astype(int)
    if selected.empty:
        raise ValueError("Finished zero-shot runs contain no complete metric rows")
    return selected


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
    if BASELINE_TARGET not in scores:
        raise ValueError(f"Runs are missing the {BASELINE_TARGET!r} baseline")
    present = [target for target in ALL_TARGETS if target in scores]
    if not present:
        raise ValueError(f"Runs have none of the compared targets: {', '.join(ALL_TARGETS)}")
    differences = scores[present].sub(scores[BASELINE_TARGET], axis=0)
    return differences.reset_index().melt(
        id_vars=["encoder", "method", "support_size", "modality"],
        var_name="mlp_targets",
        value_name="delta",
    ).dropna(subset=["delta"])


def _target_style(target: str) -> dict[str, str]:
    mlp_targets, msde_mode = target.split(" · ")
    return {"color": MODE_COLORS[msde_mode], "linestyle": TARGET_LINESTYLES[mlp_targets]}


def plot_differences(
    differences: pd.DataFrame,
    compared_targets: tuple[str, ...] = ALL_TARGETS,
    output_stem: Path = DEFAULT_OUTPUT_STEM,
) -> list[Path]:
    """Create an encoder × method grid of paired differences and save PNG/PDF outputs."""
    targets = [target for target in compared_targets if target in set(differences["mlp_targets"])]
    if not targets:
        LOGGER.warning("Skipping %s: no runs for %s", output_stem.name, ", ".join(compared_targets))
        return []
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
    # Horizontal offset, in support-size units, so the targets do not overlap.
    spacing = 0.4 if len(targets) > 3 else 1.2
    dodge = {target: (index - (len(targets) - 1) / 2) * spacing for index, target in enumerate(targets)}
    # Circles are reserved for the mean lines, so modalities use the other markers.
    modalities = sorted(differences["modality"].unique())
    modality_markers = {modality: MARKERS[1 + index % (len(MARKERS) - 1)] for index, modality in enumerate(modalities)}

    for row, encoder in enumerate(ENCODERS):
        for col, method in enumerate(METHODS):
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
                        color=_target_style(target)["color"],
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
                    **_target_style(target),
                )
            axis.set_title(f"{encoder} · {method}")
            axis.set_xticks(support_sizes)
            axis.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.55)

    for axis in axes[-1]:
        axis.set_xlabel("Support size")
    fig.supylabel(f"Δ composite vs {BASELINE_TARGET}", fontsize=plt.rcParams["axes.labelsize"])

    target_handles = [
        Line2D([0], [0], linewidth=2.0, markersize=7, markeredgecolor="white", label=f"{target} (mean)",
               marker="o", **_target_style(target))
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
    parser.add_argument(
        "--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM,
        help="Stem for the all-targets figure; the selected-targets figure adds '_selected'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs_by_target(tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)
    differences = paired_differences(runs)
    outputs = plot_differences(differences, ALL_TARGETS, output_stem=args.output_stem)
    selected_stem = args.output_stem.with_name(f"{args.output_stem.name}_selected")
    outputs += plot_differences(differences, SELECTED_TARGETS, output_stem=selected_stem)
    LOGGER.info("Computed %d paired modality differences", len(differences))
    LOGGER.info("Saved %s", ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
