"""Run one stage of the two-stage DELTA Optuna search and export its best parameters.

Stage ``method`` tunes the pseudolabeler with ``few_shot.py`` (one study per
method x encoder). Stage ``msde`` fixes the method's stage-1 best parameters
and tunes MSDE with ``distill_mlp.py`` (one study per method x encoder x
mlp_targets). ``labels`` targets use no MSDE, so ``--stage msde
--mlp-targets labels`` runs a single distillation with the stage-1 parameters
instead of a study.

Each study has a budget of ``n_trials`` completed trials (the experiment
config, or ``hydra.sweeper.n_trials=N``). Rerunning resumes: a finished study is
skipped (its best parameters are re-exported), a partial one runs only its
missing trials, and trials left RUNNING by a killed process are marked FAIL.
Run one study at a time: the stale-trial cleanup assumes no other process is
running the same study.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_optuna_best import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STORAGE,
    STAGES,
    experiment_config,
    export_best,
    load_best_params,
    study_name,
)


def _method_overrides(method: str, encoder: str, output_root: Path) -> list[str]:
    """The stage-1 best parameters as ``method.<key>=<value>`` overrides."""
    best = load_best_params("method", method, encoder, output_root=output_root)
    print(
        f"Using stage-1 params of {best['study_name']} trial {best['trial_number']} "
        f"(objective={best['objective']:.4f})",
        flush=True,
    )
    # Floats print as their shortest round-trip repr, so the overrides rebuild the
    # stage-1 config exactly and its cached pseudolabels are found by hash.
    return [f"{key}={value}" for key, value in best["params"].items()]


def _prepare_study(name: str, storage: str) -> tuple[int, int]:
    """Fail stale RUNNING trials; return ``(completed trials, all trials)``."""
    import optuna

    try:
        study = optuna.load_study(study_name=name, storage=storage)
    except KeyError:
        return 0, 0
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            print(f"Marking stale RUNNING trial {trial.number} of {name} as FAIL", flush=True)
            study._storage.set_trial_state_values(trial._trial_id, optuna.trial.TrialState.FAIL)
    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    return completed, len(study.trials)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--method", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--mlp-targets", choices=("labels", "scores", "distances"), default=None,
                        help="required for --stage msde")
    parser.add_argument("--storage", default=DEFAULT_STORAGE, help="Optuna storage URL")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args, extra = parser.parse_known_args()
    if args.stage == "msde" and args.mlp_targets is None:
        parser.error("--stage msde requires --mlp-targets")

    base = [f"method={args.method}", f"encoder={args.encoder}"]
    if args.stage == "method":
        script, mlp_targets = "few_shot.py", None
        command = ["-m", "+experiment=optuna_method", *base]
    else:
        script, mlp_targets = "distill_mlp.py", args.mlp_targets
        fixed = _method_overrides(args.method, args.encoder, args.output_root)
        if mlp_targets == "labels":
            # Nothing to search: one job over the experiment's support sizes x modalities.
            command = ["+experiment=optuna_msde", *base, "mlp_targets=labels", *fixed]
        else:
            command = ["-m", "+experiment=optuna_msde", *base, f"mlp_targets={mlp_targets}", *fixed]
    command.append(f"hydra.sweeper.storage={args.storage}")
    if mlp_targets != "labels":
        sweeper = experiment_config(args.stage)
        budget = int(sweeper.n_trials)
        for arg in extra:
            if arg.startswith("hydra.sweeper.n_trials="):
                budget = int(arg.split("=", 1)[1])
        extra = [arg for arg in extra if not arg.startswith("hydra.sweeper.n_trials=")]
        name = study_name(args.stage, args.method, args.encoder, mlp_targets)
        completed, existing = _prepare_study(name, args.storage)
        remaining = budget - completed
        if remaining <= 0:
            print(f"{name}: {completed}/{budget} trials complete; skipping", flush=True)
            return _export(args, mlp_targets, budget, 0)
        print(f"{name}: {completed}/{budget} trials complete; running {remaining}", flush=True)
        command.append(f"hydra.sweeper.n_trials={remaining}")
        # Offset the seed by the trials already in the study: a resumed study
        # would otherwise replay the seeded sampler's first draws, which during
        # the random startup phase repeats the earlier trials exactly.
        if not any(arg.startswith("hydra.sweeper.sampler.seed=") for arg in extra):
            command.append(f"hydra.sweeper.sampler.seed={int(sweeper.sampler.seed) + existing}")
    # Extra arguments are passed through as Hydra overrides.
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *command, *extra]
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if mlp_targets == "labels":
        return result.returncode

    return _export(args, mlp_targets, budget, result.returncode)


def _export(args: argparse.Namespace, mlp_targets: str | None, budget: int, returncode: int) -> int:
    # Export even when the study process exits nonzero: completed trials from
    # a partially finished study are still useful and remain reproducible.
    try:
        output_path = export_best(
            args.stage, args.method, args.encoder, mlp_targets,
            storage=args.storage, output_root=args.output_root, n_trials=budget,
        )
        print(f"Best Optuna params written to {output_path}")
    except Exception as error:
        print(f"Could not export best Optuna params: {error}", file=sys.stderr)
        if returncode == 0:
            return 1
    return returncode

if __name__ == "__main__":
    raise SystemExit(main())
