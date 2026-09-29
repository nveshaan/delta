"""PyTorch LaplacianShot pseudolabeling.

The propagated scores are the final pseudolabel source:

    CLIP tensors -> KNN graph -> Laplacian -> F -> argmax/confidence

No post-propagation CLIP-space KNN vote is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


DEFAULT_DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


@dataclass
class LaplacianShotConfig:
    """Runtime options for LaplacianShot."""

    k_nn: int = 15
    lambda_laplacian: float = 0.5
    confidence: float = 0.75
    solver_iterations: int = 500
    solver_tolerance: float = 1e-5
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.k_nn <= 0:
            raise ValueError("k_nn must be positive")
        if self.lambda_laplacian < 0:
            raise ValueError("lambda_laplacian must be non-negative")
        if self.solver_iterations <= 0:
            raise ValueError("solver_iterations must be positive")
        if self.solver_tolerance <= 0:
            raise ValueError("solver_tolerance must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, mps, cuda, cpu")


def _resolve_device(device: str) -> torch.device:
    return torch.device(DEFAULT_DEVICE if device == "auto" else device)


def _build_knn_affinity(embeddings: torch.Tensor, k_nn: int) -> torch.Tensor:
    """Build a symmetric sparse cosine-affinity matrix in PyTorch."""
    n_nodes = len(embeddings)
    if n_nodes < 2:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=embeddings.device),
            torch.empty(0, dtype=embeddings.dtype, device=embeddings.device),
            (n_nodes, n_nodes), device=embeddings.device,
        ).coalesce()

    normalized = F.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.T
    k_actual = min(k_nn + 1, n_nodes)
    values, indices = similarities.topk(k_actual, dim=1, largest=True)
    values = values[:, 1:].clamp_min(0.0)
    indices = indices[:, 1:]
    rows = torch.arange(n_nodes, device=embeddings.device).unsqueeze(1).expand_as(indices)

    # Add both directions so the graph is symmetric. coalesce() combines
    # duplicate edges, which is equivalent to accumulating their affinity.
    row = torch.cat([rows.reshape(-1), indices.reshape(-1)])
    col = torch.cat([indices.reshape(-1), rows.reshape(-1)])
    weight = torch.cat([values.reshape(-1), values.reshape(-1)])
    return torch.sparse_coo_tensor(
        torch.stack([row, col]), weight, (n_nodes, n_nodes),
        device=embeddings.device,
    ).coalesce()


def _sparse_matmul(matrix: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix, value)


def _conjugate_gradient(
    apply_operator,
    rhs: torch.Tensor,
    iterations: int,
    tolerance: float,
) -> torch.Tensor:
    """Solve a positive-definite matrix equation for multiple RHS columns."""
    solution = torch.zeros_like(rhs)
    residual = rhs - apply_operator(solution)
    direction = residual.clone()
    residual_norm = (residual * residual).sum(dim=0)
    initial_norm = residual_norm.clamp_min(torch.finfo(rhs.dtype).eps)

    for _ in tqdm(range(iterations), desc="LaplacianShot solver", unit="iter"):
        operator_direction = apply_operator(direction)
        denominator = (direction * operator_direction).sum(dim=0).clamp_min(1e-12)
        step = residual_norm / denominator
        solution = solution + direction * step
        residual = residual - operator_direction * step
        new_norm = (residual * residual).sum(dim=0)
        if torch.sqrt(new_norm.max()) <= tolerance * torch.sqrt(initial_norm.max()):
            break
        direction = residual + direction * (new_norm / residual_norm.clamp_min(1e-12))
        residual_norm = new_norm
    return solution


def _laplacian_propagate(
    affinity: torch.Tensor,
    initialization: torch.Tensor,
    lambda_laplacian: float,
    solver_iterations: int,
    solver_tolerance: float,
) -> torch.Tensor:
    """Solve ``F = (I + lambda * (D-W))^-1 Z`` with PyTorch CG."""
    degree = torch.sparse.sum(affinity, dim=1).to_dense()

    def apply_operator(value: torch.Tensor) -> torch.Tensor:
        return value + lambda_laplacian * (
            degree[:, None] * value - _sparse_matmul(affinity, value)
        )

    propagated = _conjugate_gradient(
        apply_operator, initialization, solver_iterations, solver_tolerance
    )
    return propagated / propagated.sum(dim=1, keepdim=True).clamp_min(1e-12)


def pseudolabel(
    seed_normals: torch.Tensor,
    seed_anomalies: torch.Tensor,
    unlabeled: torch.Tensor,
    config: LaplacianShotConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info("LaplacianShot pseudolabeling: %d seeds, %d query samples", len(seed_normals) + len(seed_anomalies), len(unlabeled))
    """Generate labels/confidences directly from Laplacian propagation.

    Inputs and outputs are PyTorch tensors. Output labels are ``0`` for normal
    and ``1`` for anomaly; confidence is ``max(F_unlabeled, dim=1)``.
    """
    config = config or LaplacianShotConfig()
    device = _resolve_device(config.device)
    seed_normals = seed_normals.to(device=device, dtype=torch.float32)
    seed_anomalies = seed_anomalies.to(device=device, dtype=torch.float32)
    unlabeled = unlabeled.to(device=device, dtype=torch.float32)
    if len(seed_normals) == 0 or len(seed_anomalies) == 0:
        raise ValueError("At least one normal and one anomaly seed are required")
    if len(unlabeled) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    torch.manual_seed(config.seed)
    all_embeddings = torch.cat([seed_normals, seed_anomalies, unlabeled], dim=0)
    normalized = F.normalize(all_embeddings, dim=1)
    normal_prototype = F.normalize(normalized[:len(seed_normals)].mean(dim=0, keepdim=True), dim=1)
    anomaly_start = len(seed_normals)
    anomaly_prototype = F.normalize(normalized[anomaly_start:anomaly_start + len(seed_anomalies)].mean(dim=0, keepdim=True), dim=1)
    logits = torch.cat([
        normalized @ normal_prototype.T,
        normalized @ anomaly_prototype.T,
    ], dim=1)
    initialization = torch.softmax(logits, dim=1)

    # Hard-pin the labeled anchors before propagation.
    initialization[:len(seed_normals)] = torch.tensor([1.0, 0.0], device=device)
    initialization[len(seed_normals):len(seed_normals) + len(seed_anomalies)] = torch.tensor(
        [0.0, 1.0], device=device
    )
    affinity = _build_knn_affinity(all_embeddings, config.k_nn)
    propagated = _laplacian_propagate(
        affinity, initialization, config.lambda_laplacian,
        config.solver_iterations, config.solver_tolerance,
    )

    propagated_unlabeled = propagated[len(seed_normals) + len(seed_anomalies):]
    confidence, labels = propagated_unlabeled.max(dim=1)
    return labels, confidence


class LaplacianShotPseudolabeler:
    """Hydra-instantiable callable LaplacianShot labeler."""

    def __init__(self, **kwargs: object) -> None:
        self.config = LaplacianShotConfig(**kwargs)

    def __call__(
        self,
        seed_normals: torch.Tensor,
        seed_anomalies: torch.Tensor,
        unlabeled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return pseudolabel(seed_normals, seed_anomalies, unlabeled, self.config)


def filter_by_confidence(
    labels: torch.Tensor, confidences: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return labels and mask for propagated scores above ``threshold``."""
    keep = confidences >= threshold
    return labels[keep], keep
