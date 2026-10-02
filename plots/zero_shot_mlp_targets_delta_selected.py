"""Plot the selected zero-shot MLP-target gains over ``labels`` across experiments.

This draws only the ``<stem>_selected`` figure of
``zero_shot_mlp_targets_delta.py`` (``scores · same_label`` and
``distances · zero_label``), but takes its runs from two MLflow experiments:
the ``labels`` baseline from ``delta`` and the ``scores``/``distances``
targets from ``msde50``. Composite scores, paired differences, and the figure
layout are unchanged. PNG and PDF outputs are written to ``assets/`` by default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from zero_shot_encoder_method_consistency import DEFAULT_TRACKING_URI, PROJECT_ROOT
from zero_shot_mlp_targets_delta import (
    BASELINE_TARGET,
    SELECTED_TARGETS,
    load_runs_by_target,
    paired_differences,
    plot_differences,
)


DEFAULT_OUTPUT_STEM = PROJECT_ROOT / "assets" / "zero_shot_mlp_targets_delta_selected"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_runs(*, tracking_uri: str, labels_experiment: str, targets_experiment: str) -> pd.DataFrame:
    """Combine ``labels`` runs from one experiment with MSDE-target runs from another."""
    labels = load_runs_by_target(tracking_uri=tracking_uri, experiment_name=labels_experiment)
    labels = labels[labels["mlp_targets"] == BASELINE_TARGET]
    if labels.empty:
        raise ValueError(f"No {BASELINE_TARGET!r} runs found in MLflow experiment {labels_experiment!r}")

    targets = load_runs_by_target(tracking_uri=tracking_uri, experiment_name=targets_experiment)
    targets = targets[targets["mlp_targets"].str.startswith(("scores · ", "distances · "))]
    if targets.empty:
        raise ValueError(f"No scores/distances runs found in MLflow experiment {targets_experiment!r}")
    return pd.concat([labels, targets], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--labels-experiment", default="delta")
    parser.add_argument("--targets-experiment", default="msde50")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(
        tracking_uri=args.tracking_uri,
        labels_experiment=args.labels_experiment,
        targets_experiment=args.targets_experiment,
    )
    differences = paired_differences(runs)
    outputs = plot_differences(differences, SELECTED_TARGETS, output_stem=args.output_stem)
    LOGGER.info("Computed %d paired modality differences", len(differences))
    LOGGER.info("Saved %s", ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
