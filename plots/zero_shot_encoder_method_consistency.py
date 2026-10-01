"""Plot zero-shot performance versus consistency for encoder/method pairs.

The MLflow metric runs produced by ``scripts/zero_shot.py`` are aggregated in
the following order:

1. Average AUROC, AUPRC, and precision-at-n over the three ``mlp_targets``.
2. For each dataset, use ``pooled_nonzero`` when subtype comparisons exist;
   otherwise use ``label_0_vs_1``. Average the three metrics over runs at the
   dataset level and form their composite score.
3. Average dataset composite scores within each modality.
4. Average modality scores for the final mean, and calculate their population
   standard deviation for the consistency axis.

The figure contains one scatter panel per support size. Each point is one
encoder/method pair; color identifies the encoder and marker identifies the
method. PNG and PDF outputs are written to ``assets/`` by default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'experiments' / 'mlruns.db'}"
DEFAULT_OUTPUT_STEM = PROJECT_ROOT / "assets" / "zero_shot_encoder_method_consistency"

METRICS = ("auroc", "auprc", "p_at_n")
GROUP_COLUMNS = (
    "encoder",
    "method",
    "support_size",
    "modality",
    "dataset",
    "comparison",
)
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}
COLORS = list(PALETTE.values())
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _normalise_tracking_uri(tracking_uri: str) -> str:
    """Resolve project-relative SQLite URIs like the training scripts do."""
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        relative_path = tracking_uri.removeprefix("sqlite:///")
        return f"sqlite:///{(PROJECT_ROOT / relative_path).resolve()}"
    return tracking_uri


def load_zero_shot_runs(
    *, tracking_uri: str = DEFAULT_TRACKING_URI, experiment_name: str = "delta"
) -> pd.DataFrame:
    """Query finished zero-shot metric runs from MLflow."""
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("MLflow is required to query the experiment runs") from error

    mlflow.set_tracking_uri(_normalise_tracking_uri(tracking_uri))
    filter_string = "params.type = 'zero_shot' and attributes.status = 'FINISHED'"
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=filter_string,
        output_format="pandas",
    )
    if runs.empty:
        raise ValueError(f"No finished zero-shot runs found in MLflow experiment {experiment_name!r}")

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


def _choose_dataset_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep pooled subtype comparisons, or binary label 0-vs-1 otherwise."""
    comparison_by_dataset = runs.groupby("dataset")["comparison"].agg(set)

    def chosen(dataset: str) -> str:
        comparisons = comparison_by_dataset[dataset]
        if "pooled_nonzero" in comparisons:
            return "pooled_nonzero"
        if "label_0_vs_1" in comparisons:
            return "label_0_vs_1"
        raise ValueError(f"Dataset {dataset!r} has no supported comparison")

    result = runs.copy()
    result["selected_comparison"] = result["dataset"].map(chosen)
    return result[result["comparison"] == result["selected_comparison"]].drop(
        columns="selected_comparison"
    )


def aggregate_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """Return final mean and consistency for every support/pair combination."""
    runs = _choose_dataset_comparisons(runs)

    # The first reduction averages the three MLP targets before any dataset
    # or modality receives a weight.
    mlp_group = list(GROUP_COLUMNS)
    by_mlp_target = runs.groupby(mlp_group, as_index=False, dropna=False)[list(METRICS)].mean()

    # Average metrics at dataset level, then create the composite score.
    dataset_group = ["encoder", "method", "support_size", "modality", "dataset"]
    dataset_scores = by_mlp_target.groupby(dataset_group, as_index=False, dropna=False)[list(METRICS)].mean()
    dataset_scores["composite"] = dataset_scores[list(METRICS)].mean(axis=1)

    # Equal weight for every dataset within each modality.
    modality_group = ["encoder", "method", "support_size", "modality"]
    modality_scores = dataset_scores.groupby(
        modality_group, as_index=False, dropna=False
    )["composite"].mean()

    # Equal weight for every modality in each support-size mean. The
    # population standard deviation describes observed cross-modality
    # consistency.
    pair_group = ["encoder", "method", "support_size"]
    final = modality_scores.groupby(pair_group, as_index=False, dropna=False)["composite"].agg(
        mean="mean", std=lambda values: float(np.std(values.to_numpy(), ddof=0)), n_modalities="count"
    )

    # For the all-support-size row, first average each modality across
    # support sizes, then recompute the cross-modality statistics. Averaging
    # support-size standard deviations would not produce a pooled statistic.
    overall_modality_scores = modality_scores.groupby(
        ["encoder", "method", "modality"], as_index=False, dropna=False
    )["composite"].mean()
    overall = overall_modality_scores.groupby(
        ["encoder", "method"], as_index=False, dropna=False
    )["composite"].agg(
        mean="mean", std=lambda values: float(np.std(values.to_numpy(), ddof=0)), n_modalities="count"
    )
    overall["support_size"] = "all"
    final = pd.concat([final, overall], ignore_index=True)
    return final.sort_values(pair_group).reset_index(drop=True)


def add_all_support_size_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Return scores, including the all-support-size rows from aggregation."""
    return scores.copy()


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 15,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )


def plot_scores(scores: pd.DataFrame, output_stem: Path = DEFAULT_OUTPUT_STEM) -> list[Path]:
    """Create one scatter panel per support size and save PNG/PDF outputs."""
    _apply_publication_style()
    support_sizes = sorted(
        scores["support_size"].unique(),
        key=lambda value: (value == "all", int(value) if value != "all" else 0),
    )
    if not support_sizes:
        raise ValueError("No support sizes are available to plot")

    ncols = min(3, len(support_sizes))
    nrows = int(np.ceil(len(support_sizes) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()
    encoders = sorted(scores["encoder"].unique())
    methods = sorted(scores["method"].unique())
    encoder_colors = {encoder: COLORS[index % len(COLORS)] for index, encoder in enumerate(encoders)}
    method_markers = {method: MARKERS[index % len(MARKERS)] for index, method in enumerate(methods)}

    x_values = scores["mean"].to_numpy()
    y_values = scores["std"].to_numpy()
    x_margin = max((x_values.max() - x_values.min()) * 0.08, 0.01)
    y_margin = max((y_values.max() - y_values.min()) * 0.12, 0.005)
    x_limits = (x_values.min() - x_margin, x_values.max() + x_margin)
    y_limits = (max(0.0, y_values.min() - y_margin), y_values.max() + y_margin)

    for axis, support_size in zip(axes, support_sizes):
        panel = scores[scores["support_size"] == support_size]
        for _, row in panel.iterrows():
            axis.scatter(
                row["mean"],
                row["std"],
                s=88,
                color=encoder_colors[row["encoder"]],
                marker=method_markers[row["method"]],
                edgecolors="white",
                linewidths=0.7,
                alpha=0.9,
                zorder=3,
            )
        title = "All support sizes" if support_size == "all" else f"support_size = {support_size}"
        axis.set_title(title)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.55)

    for axis in axes[len(support_sizes) :]:
        axis.set_visible(False)
    for axis in axes[:ncols]:
        axis.set_xlabel("Mean composite score")
    fig.text(
        0.015,
        0.5,
        "Composite score standard deviation",
        rotation="vertical",
        va="center",
        ha="center",
    )

    encoder_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=encoder_colors[name],
               markeredgecolor="white", markersize=8, label=name)
        for name in encoders
    ]
    method_handles = [
        Line2D([0], [0], marker=method_markers[name], color="#333333", linestyle="None",
               markersize=8, label=name)
        for name in methods
    ]
    fig.legend(
        handles=encoder_handles + method_handles,
        loc="lower center",
        ncol=max(len(encoder_handles), len(method_handles)),
        bbox_to_anchor=(0.5, -0.015),
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.suptitle("Zero-shot encoder × method performance and consistency", y=1.01)
    fig.tight_layout(rect=(0.04, 0.08, 1, 0.98), pad=1.2)

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
    scores = add_all_support_size_scores(aggregate_scores(runs))
    outputs = plot_scores(scores, output_stem=args.output_stem)
    LOGGER.info("Aggregated %d encoder/method/support-size points", len(scores))
    LOGGER.info("Saved %s", ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
