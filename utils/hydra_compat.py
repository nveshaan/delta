"""Hydra CLI helpers shared by the few-shot and distillation scripts.

Import this module before ``hydra``: it patches argparse at import time.
"""

from __future__ import annotations

import argparse
import sys

# Hydra 1.3 passes a lazy help object to argparse. Python 3.14 tightened
# argparse's help validation to require a string, so normalize that object
# only during parser validation. This keeps Hydra's CLI/config behavior intact.
if sys.version_info >= (3, 14) and not getattr(argparse.ArgumentParser, "_delta_help_patched", False):
    _argparse_check_help = argparse.ArgumentParser._check_help

    def _hydra_argparse_check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return _argparse_check_help(self, action)

    argparse.ArgumentParser._check_help = _hydra_argparse_check_help
    argparse.ArgumentParser._delta_help_patched = True


def _patch_optuna_sweeper() -> None:
    """Let hydra-optuna-sweeper 1.4.0.dev run on hydra-core 1.3.

    The plugin targets Hydra 1.4, which instantiates plugins non-recursively:
    it builds its sampler itself and passes ``_execution_whitelist_``. Hydra
    1.3 instantiates the sweeper config recursively, so the sampler arrives
    already built (an optuna sampler, or a ``functools.partial`` for the grid
    sampler) and ``_execution_whitelist_`` is unknown. Only calls carrying
    ``_execution_whitelist_`` (i.e. the plugin's) are changed: built samplers
    pass through and the argument is dropped.

    ``hydra.utils.instantiate`` is patched rather than the plugin module,
    because Hydra's plugin scan re-executes plugin modules, which re-imports
    the original function. Remove this once hydra-core 1.4 is adopted.
    """
    import hydra.utils
    from omegaconf import DictConfig

    if getattr(hydra.utils.instantiate, "_delta_patched", False):
        return
    original = hydra.utils.instantiate

    def instantiate(config, *args, **kwargs):
        if "_execution_whitelist_" in kwargs:
            kwargs.pop("_execution_whitelist_")
            if not isinstance(config, (DictConfig, dict)):
                return config
        return original(config, *args, **kwargs)

    instantiate._delta_patched = True
    hydra.utils.instantiate = instantiate


_patch_optuna_sweeper()


def consume_redo_flag() -> None:
    """Translate ``--redo`` into the Hydra override ``redo=true``.

    Hydra rejects unknown flags, so the flag is removed from ``sys.argv``
    before ``@hydra.main`` parses it.
    """
    if "--redo" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--redo"]
        sys.argv.append("redo=true")
