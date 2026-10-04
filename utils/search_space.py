"""Optuna search space for ``distill_mlp.py`` sweeps.

Used as hydra-optuna-sweeper's ``custom_search_space``: every trial samples the
selected method's pseudolabeler parameters, plus the MSDE parameters when
``mlp_targets`` is ``scores`` or ``distances``. The ranges live in
``configs/search_space.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

SEARCH_SPACE_FILE = Path(__file__).resolve().parents[1] / "configs" / "search_space.yaml"


def suggest(cfg: DictConfig, trial) -> None:
    """Sample this sweep's parameters into ``trial`` (they become job overrides)."""
    from hydra_plugins.hydra_optuna_sweeper._impl import create_params_from_overrides

    space = OmegaConf.load(SEARCH_SPACE_FILE)
    method = str(cfg.hydra.runtime.choices.method)
    if method not in space.method:
        raise ValueError(f"No search space for method {method!r} in {SEARCH_SPACE_FILE}")
    overrides = [f"method.{key}={value}" for key, value in space.method[method].items()]
    if str(cfg.mlp_targets) != "labels":
        overrides += [f"msde.{key}={value}" for key, value in space.msde.items()]
    distributions, _, _ = create_params_from_overrides(overrides)
    for name, distribution in distributions.items():
        trial._suggest(name, distribution)
