"""Optuna search spaces for the two-stage sweep.

Used as hydra-optuna-sweeper's ``custom_search_space``: ``method`` samples the
selected method's pseudolabeler parameters (``few_shot.py``), ``msde`` samples
the MSDE parameters of ``mlp_targets=scores/distances`` (``distill_mlp.py``).
The ranges live in ``configs/search_space.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

SEARCH_SPACE_FILE = Path(__file__).resolve().parents[1] / "configs" / "search_space.yaml"


def _suggest(trial, overrides: list[str]) -> None:
    """Sample Hydra sweep ``overrides`` into ``trial`` (they become job overrides)."""
    from hydra_plugins.hydra_optuna_sweeper._impl import create_params_from_overrides

    distributions, _, _ = create_params_from_overrides(overrides)
    for name, distribution in distributions.items():
        trial._suggest(name, distribution)


def method(cfg: DictConfig, trial) -> None:
    """Stage 1: the selected method's pseudolabeler parameters."""
    space = OmegaConf.load(SEARCH_SPACE_FILE)
    name = str(cfg.hydra.runtime.choices.method)
    if name not in space.method:
        raise ValueError(f"No search space for method {name!r} in {SEARCH_SPACE_FILE}")
    _suggest(trial, [f"method.{key}={value}" for key, value in space.method[name].items()])


def msde(cfg: DictConfig, trial) -> None:
    """Stage 2: the MSDE parameters (the method's parameters are fixed overrides)."""
    if str(cfg.mlp_targets) == "labels":
        raise ValueError("mlp_targets=labels does not use MSDE; there is nothing to search")
    space = OmegaConf.load(SEARCH_SPACE_FILE)
    _suggest(trial, [f"msde.{key}={value}" for key, value in space.msde.items()])
