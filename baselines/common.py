"""Shared helpers for the inference-only baselines: device choice, vendored repos and checkpoints."""
import os
import subprocess
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def choose_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def prepare_repo(url: str, repo_dir: str) -> Path:
    """Clone the authors' repo into `repo_dir` if missing and put it first on `sys.path`."""
    if "pkg_resources" not in sys.modules:  # the repos do `from pkg_resources import packaging`, gone from new setuptools
        import packaging.version
        sys.modules["pkg_resources"] = types.SimpleNamespace(packaging=packaging)
    path = Path(repo_dir)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if not path.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def select_checkpoint(checkpoint: str, checkpoints: dict, method: str):
    """The entry of `checkpoints` named `checkpoint`; fails clearly if the authors released none."""
    if checkpoint not in checkpoints:
        raise ValueError(f"{method}: unknown checkpoint {checkpoint!r}, choose from {list(checkpoints)}")
    if checkpoints[checkpoint] is None:
        available = [key for key, value in checkpoints.items() if value is not None]
        raise ValueError(f"{method}: no released {checkpoint!r} checkpoint, available: {available}")
    return checkpoints[checkpoint]
