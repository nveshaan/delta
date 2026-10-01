#!/usr/bin/env python3
"""Patch the installed MLflow UI to request 10,000 runs per page.

Run after `uv sync`:
    .venv/bin/python scripts/patch_mlflow_ui.py
"""

from pathlib import Path
import re

import mlflow


def main() -> None:
    package_dir = Path(mlflow.__file__).resolve().parent
    build_dir = package_dir / "server" / "js" / "build"
    index_path = build_dir / "index.html"
    index = index_path.read_text()

    match = re.search(r'static-files/static/js/(main\.[^"?]+\.js)(?:\?[^" ]*)?', index)
    if match is None:
        raise RuntimeError(f"Could not find the MLflow main bundle in {index_path}")

    bundle_path = build_dir / "static" / "js" / match.group(1)
    bundle = bundle_path.read_text()
    old = re.search(r'const f=(?:100|1000|10000|50000),p="GET_EXPERIMENT_API"', bundle)
    if old is None:
        raise RuntimeError(f"Could not find the MLflow run-fetch constant in {bundle_path}")

    bundle = bundle[: old.start()] + 'const f=10000,p="GET_EXPERIMENT_API"' + bundle[old.end() :]
    bundle_path.write_text(bundle)

    # Ensure a browser does not reuse a cached copy of the old hashed bundle.
    script_name = match.group(1)
    replacement = f"static-files/static/js/{script_name}?runs=10000"
    index = index[: match.start()] + replacement + index[match.end() :]
    index_path.write_text(index)

    print(f"Patched MLflow {mlflow.__version__}: {bundle_path}")


if __name__ == "__main__":
    main()
