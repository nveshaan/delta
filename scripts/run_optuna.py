"""Run one stage of the two-stage DELTA Optuna search and export its best parameters.

Stage ``method`` tunes the pseudolabeler with ``few_shot.py`` (one study per
method x encoder). Stage ``msde`` fixes the method's stage-1 best parameters
and tunes MSDE with ``distill_mlp.py`` (one study per method x encoder x
mlp_targets). ``labels`` targets use no MSDE, so ``--stage msde
--mlp-targets labels`` runs a single distillation with the stage-1 parameters
instead of a study.
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
    export_best,
    load_best_params,
    study_name,
)

EXPERIMENT_DIR = PROJECT_ROOT / "configs" / "experiment"


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


def _sampler_seed(stage: str, name: str, storage: str) -> int:
    """The configured sampler seed, offset by the trials already in the study.

    A resumed study would otherwise replay the seeded sampler's first draws,
    which during the random startup phase repeats the earlier trials exactly.
    """
    import optuna
    from omegaconf import OmegaConf

    seed = int(OmegaConf.load(EXPERIMENT_DIR / f"optuna_{stage}.yaml").hydra.sweeper.sampler.seed)
    try:
        existing = len(optuna.load_study(study_name=name, storage=storage).trials)
    except KeyError:
        existing = 0
    return seed + existing


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
    if mlp_targets != "labels" and not any(arg.startswith("hydra.sweeper.sampler.seed=") for arg in extra):
        name = study_name(args.stage, args.method, args.encoder, mlp_targets)
        command.append(f"hydra.sweeper.sampler.seed={_sampler_seed(args.stage, name, args.storage)}")
    # Extra arguments are passed through as Hydra overrides, e.g.
    # hydra.sweeper.n_trials=10.
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *command, *extra]
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if mlp_targets == "labels":
        return result.returncode

    # Export even when the study process exits nonzero: completed trials from
    # a partially finished study are still useful and remain reproducible.
    try:
        output_path = export_best(
            args.stage, args.method, args.encoder, mlp_targets,
            storage=args.storage, output_root=args.output_root,
        )
        print(f"Best Optuna params written to {output_path}")
    except Exception as error:
        print(f"Could not export best Optuna params: {error}", file=sys.stderr)
        if result.returncode == 0:
            return 1
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
