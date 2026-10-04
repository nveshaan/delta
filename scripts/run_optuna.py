"""Run one DELTA Optuna study and export its best parameters."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_optuna_best import export_best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--mlp-targets", required=True, choices=("labels", "scores", "distances"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "distill_mlp.py"),
        "-m",
        "+experiment=optuna",
        f"method={args.method}",
        f"encoder={args.encoder}",
        f"mlp_targets={args.mlp_targets}",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT)

    # Export even when the study process exits nonzero: completed trials from
    # a partially finished study are still useful and remain reproducible.
    try:
        output_path = export_best(
            args.method,
            args.encoder,
            args.mlp_targets,
            output_root=args.output_root or Path(__file__).resolve().parents[1] / "experiments" / "optuna_best",
        )
        print(f"Best Optuna params written to {output_path}")
    except Exception as error:
        print(f"Could not export best Optuna params: {error}", file=sys.stderr)
        if result.returncode == 0:
            return 1
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
