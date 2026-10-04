"""Export the best completed trial from a persisted Optuna study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import optuna
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT / 'experiments' / 'optuna.db'}"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "optuna_best"


def export_best(
    method: str,
    encoder: str,
    mlp_targets: str,
    *,
    storage: str = DEFAULT_STORAGE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    study_name = f"{method}_{encoder}_{mlp_targets}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"Study {study_name!r} has no completed trials")

    best = study.best_trial
    output_dir = output_root / method / encoder / mlp_targets
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_params.yaml"
    payload = {
        "study_name": study.study_name,
        "method": method,
        "encoder": encoder,
        "mlp_targets": mlp_targets,
        "trial_number": best.number,
        "objective": float(best.value),
        "params": best.params,
        "completed_trials": len(completed),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--mlp-targets", required=True, choices=("labels", "scores", "distances"))
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    output_path = export_best(
        args.method,
        args.encoder,
        args.mlp_targets,
        storage=args.storage,
        output_root=args.output_root,
    )
    print(f"Best Optuna params written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
