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

With ``--campaign optuna`` (experiment set 2) only the Optuna best parameters
are plotted: for every method/encoder in ``experiments/optuna_best``, the
``scores`` and ``distances`` runs are those of the stage-2 best trial and the
``labels`` runs those of the stage-1 best parameters (``labels`` has no stage-2
study). Each best configuration is matched to its distill_mlp runs by their
logged parameters, and their ``trial_hash`` selects the zero-shot runs. A best
configuration without zero-shot runs is an error that prints the
``zero_shot.py`` command to evaluate it.

Usage::

    # Experiment set 1 (default --campaign adhoc).
    uv run python plots/zero_shot_mlp_targets_delta.py

    # Experiment set 2 (Optuna best parameters).
    uv run python plots/zero_shot_mlp_targets_delta.py --campaign optuna
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
import yaml

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
    load_zero_shot_runs,
)


DEFAULT_OUTPUT_STEM = PROJECT_ROOT / "assets" / "zero_shot_mlp_targets_delta"
DEFAULT_OPTUNA_OUTPUT_STEM = PROJECT_ROOT / "assets" / "zero_shot_mlp_targets_delta_optuna"
DEFAULT_OPTUNA_BEST_ROOT = PROJECT_ROOT / "experiments" / "optuna_best"
ZERO_SHOT_SWEEP = "modality=chest,fundus,mri,oct support_size=5,10,20,30,50"
BASELINE_TARGET = "labels"
COMPARED_TARGETS = ("scores", "distances")
TARGET_STYLES = {
    "scores": {"color": PALETTE["teal"], "linestyle": "-"},
    "distances": {"color": PALETTE["violet"], "linestyle": "--"},
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def best_configurations(
    optuna_best_root: Path, encoders: list[str] | None = None, methods: list[str] | None = None
) -> dict[tuple[str, str, str], dict[str, object]]:
    """The best parameters of every method/encoder/target in ``optuna_best_root``.

    Keys are ``(method, encoder, mlp_targets)``; values are the ``method.*``
    (stage 1) and, for ``scores``/``distances``, ``msde.*`` (stage 2) parameters.
    """
    configurations: dict[tuple[str, str, str], dict[str, object]] = {}
    for stage1_path in sorted(optuna_best_root.glob("*/*/method/best_params.yaml")):
        encoder_dir = stage1_path.parent.parent
        method, encoder = encoder_dir.parent.name, encoder_dir.name
        if (methods and method not in methods) or (encoders and encoder not in encoders):
            continue
        method_params = yaml.safe_load(stage1_path.read_text(encoding="utf-8"))["params"]
        configurations[(method, encoder, BASELINE_TARGET)] = dict(method_params)
        for target in COMPARED_TARGETS:
            target_path = encoder_dir / target / "best_params.yaml"
            if not target_path.is_file():
                LOGGER.warning("No stage-2 best parameters at %s", target_path)
                continue
            best = yaml.safe_load(target_path.read_text(encoding="utf-8"))
            if best["fixed_params"] and best["fixed_params"] != method_params:
                raise ValueError(f"{target_path} was tuned with method parameters other than {stage1_path}")
            configurations[(method, encoder, target)] = {**method_params, **best["params"]}
    if not configurations:
        raise ValueError(f"No Optuna best parameters found under {optuna_best_root}")
    return configurations


def _matches(column: pd.Series, value: object) -> pd.Series:
    """Rows whose logged MLflow parameter (a string) equals ``value``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.Series(np.isclose(pd.to_numeric(column, errors="coerce"), value, rtol=1e-12, atol=0.0),
                         index=column.index)
    return column == str(value)


def best_trial_hashes(
    configurations: dict[tuple[str, str, str], dict[str, object]],
    *,
    tracking_uri: str,
    experiment_name: str,
) -> dict[tuple[str, str, str], str]:
    """The ``trial_hash`` of the Optuna distill_mlp runs of every best configuration."""
    import mlflow

    mlflow.set_tracking_uri(_normalise_tracking_uri(tracking_uri))
    hashes: dict[tuple[str, str, str], str] = {}
    for (method, encoder, target), params in configurations.items():
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=(
                "tags.type = 'distill_mlp' and tags.level = 'root' and tags.campaign = 'optuna' "
                f"and tags.method = '{method}' and tags.encoder = '{encoder}' "
                f"and tags.mlp_targets = '{target}' and attributes.status = 'FINISHED'"
            ),
            output_format="pandas",
        )
        for key, value in params.items():
            if runs.empty:
                break
            runs = runs[_matches(runs[f"params.{key}"], value)]
        trial_hashes = sorted(runs["tags.trial_hash"].unique()) if not runs.empty else []
        if len(trial_hashes) != 1:
            raise ValueError(
                f"Expected one distill_mlp trial for the best {method}/{encoder}/{target} parameters, "
                f"found {len(trial_hashes)}: {trial_hashes}"
            )
        hashes[(method, encoder, target)] = trial_hashes[0]
        LOGGER.info("Best %s/%s/%s: trial_hash %s (%d runs)", method, encoder, target, trial_hashes[0], len(runs))
    return hashes


def load_best_zero_shot_runs(
    *, tracking_uri: str, experiment_name: str, optuna_best_root: Path,
    encoders: list[str] | None, methods: list[str] | None,
) -> pd.DataFrame:
    """Zero-shot runs of the Optuna best configurations; fail if any is not evaluated."""
    hashes = best_trial_hashes(
        best_configurations(optuna_best_root, encoders, methods),
        tracking_uri=tracking_uri, experiment_name=experiment_name,
    )
    try:
        runs = load_zero_shot_runs(
            tracking_uri=tracking_uri, experiment_name=experiment_name, campaign="optuna",
            trial_hashes=hashes.values(), with_trial_hash=True,
        )
    except ValueError:
        runs = pd.DataFrame(columns=["trial_hash"])
    evaluated = set(runs["trial_hash"])
    missing = {key: trial_hash for key, trial_hash in hashes.items() if trial_hash not in evaluated}
    if missing:
        commands = "\n".join(
            f"uv run python scripts/zero_shot.py -m method={method} encoder={encoder} "
            f"mlp_targets={target} trial_hash={trial_hash} {ZERO_SHOT_SWEEP}"
            for (method, encoder, target), trial_hash in missing.items()
        )
        raise ValueError(f"{len(missing)} best configurations have no zero-shot runs; evaluate them with:\n{commands}")
    return runs.drop(columns="trial_hash")


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
    parser.add_argument("--optuna-best-root", type=Path, default=DEFAULT_OPTUNA_BEST_ROOT,
                        help="exported Optuna best parameters (--campaign optuna)")
    parser.add_argument("--output-stem", type=Path, default=None,
                        help="default: assets/zero_shot_mlp_targets_delta[_optuna]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.campaign == "optuna":
        runs = load_best_zero_shot_runs(
            tracking_uri=args.tracking_uri, experiment_name=args.experiment_name,
            optuna_best_root=args.optuna_best_root, encoders=args.encoders, methods=args.methods,
        )
        output_stem = args.output_stem or DEFAULT_OPTUNA_OUTPUT_STEM
    else:
        runs = load_zero_shot_runs(
            tracking_uri=args.tracking_uri, experiment_name=args.experiment_name, campaign=args.campaign
        )
        output_stem = args.output_stem or DEFAULT_OUTPUT_STEM
    if args.encoders:
        runs = runs[runs["encoder"].isin(args.encoders)]
    if args.methods:
        runs = runs[runs["method"].isin(args.methods)]
    if runs.empty:
        raise ValueError(f"No runs found for encoders {args.encoders} and methods {args.methods}")
    differences = paired_differences(runs)
    outputs = plot_differences(differences, output_stem=output_stem)
    LOGGER.info("Computed %d paired modality differences", len(differences))
    LOGGER.info("Saved %s", ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
