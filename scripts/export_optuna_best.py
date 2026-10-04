"""Export the best completed trial from a persisted Optuna study.

Stage ``method`` studies are named ``<method>_<encoder>_method`` and exported
to ``<output_root>/<method>/<encoder>/method/best_params.yaml``; stage ``msde``
studies are named ``<method>_<encoder>_<mlp_targets>_msde`` and exported to
``<output_root>/<method>/<encoder>/<mlp_targets>/best_params.yaml``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT / 'experiments' / 'optuna.db'}"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "optuna_best"
STAGES = ("method", "msde")


def study_name(stage: str, method: str, encoder: str, mlp_targets: str | None = None) -> str:
    """The study name set by ``configs/experiment/optuna_<stage>.yaml``."""
    if stage == "method":
        return f"{method}_{encoder}_method"
    if stage == "msde":
        if mlp_targets is None:
            raise ValueError("The msde stage needs mlp_targets")
        return f"{method}_{encoder}_{mlp_targets}_msde"
    raise ValueError(f"stage must be one of: {', '.join(STAGES)}")


def best_params_path(stage: str, method: str, encoder: str, mlp_targets: str | None = None,
                     output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    subdir = "method" if stage == "method" else mlp_targets
    if subdir is None:
        raise ValueError("The msde stage needs mlp_targets")
    return output_root / method / encoder / subdir / "best_params.yaml"


def load_best_params(stage: str, method: str, encoder: str, mlp_targets: str | None = None,
                     output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    path = best_params_path(stage, method, encoder, mlp_targets, output_root)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; run the {stage} stage first")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def export_best(
    stage: str,
    method: str,
    encoder: str,
    mlp_targets: str | None = None,
    *,
    storage: str = DEFAULT_STORAGE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    name = study_name(stage, method, encoder, mlp_targets)
    study = optuna.load_study(study_name=name, storage=storage)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"Study {name!r} has no completed trials")

    best = study.best_trial
    output_path = best_params_path(stage, method, encoder, mlp_targets, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_name": study.study_name,
        "stage": stage,
        "method": method,
        "encoder": encoder,
        "mlp_targets": mlp_targets,
        "trial_number": best.number,
        "objective": float(best.value),
        "params": best.params,
        # The sweeper records fixed overrides as user attributes; for the msde
        # stage these are the stage-1 method parameters (stored as strings).
        "fixed_params": {
            key: yaml.safe_load(str(value)) for key, value in best.user_attrs.items() if key.startswith("method.")
        },
        "completed_trials": len(completed),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--method", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--mlp-targets", choices=("scores", "distances"), default=None,
                        help="required for --stage msde")
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.stage == "msde" and args.mlp_targets is None:
        parser.error("--stage msde requires --mlp-targets")
    output_path = export_best(
        args.stage,
        args.method,
        args.encoder,
        args.mlp_targets if args.stage == "msde" else None,
        storage=args.storage,
        output_root=args.output_root,
    )
    print(f"Best Optuna params written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
