"""Vanilla score-distillation network used by DELTA."""

from __future__ import annotations

import torch
from torch import nn


class VanillaNetwork(nn.Module):
    """Two-layer MLP that predicts a non-negative distilled anomaly score."""

    def __init__(
        self,
        in_dim: int | None = None,
        hidden: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if (in_dim is not None and in_dim <= 0) or hidden <= 1:
            raise ValueError("in_dim must be positive when specified and hidden must be greater than 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.net = nn.Sequential(
            # LazyLinear allows the same Hydra target to support encoders with
            # different embedding dimensions. The layer initializes on the
            # first full-batch forward pass.
            nn.LazyLinear(hidden) if in_dim is None else nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one non-negative score per input embedding."""
        return self.net(x).squeeze(1)
