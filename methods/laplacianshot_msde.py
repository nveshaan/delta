"""PyTorch EM refinement with LaplacianShot pseudo-labeling and MSDE."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from .msde import DEFAULT_DEVICE, MeanShiftDensityEnhancement, _sparse_mm_supported


EMMode = Literal["no_msde", "msde", "label_aware_msde"]


@dataclass
class LaplacianShotMSDEConfig:
    """Configuration for alternating LaplacianShot propagation and MSDE."""

    laplacian_k: int = 20
    laplacian_lambda: float = 0.7
    laplacian_iterations: int = 20
    laplacian_tolerance: float = 1e-4
    mode: EMMode = "label_aware_msde"
    em_rounds: int = 5
    em_convergence_tolerance: float = 0.01
    gate_sharpness: float = 1.0
    msde_k: int = 50
    msde_learning_rate: float = 0.33
    msde_nbd_sample_count_threshold: int = 70
    msde_shift_threshold: float = 0.01
    msde_inner_iterations: int = 3
    n_classes: int | None = None
    confidence: float = 0.75
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.mode not in {"no_msde", "msde", "label_aware_msde"}:
            raise ValueError(f"Unknown EM mode: {self.mode}")
        for name in ("laplacian_k", "laplacian_iterations", "em_rounds", "msde_k", "msde_inner_iterations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.laplacian_lambda < 0 or self.gate_sharpness < 0:
            raise ValueError("laplacian_lambda and gate_sharpness must be non-negative")
        if self.laplacian_tolerance <= 0 or self.em_convergence_tolerance < 0:
            raise ValueError("propagation tolerance must be positive and EM tolerance non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, mps, cuda, cpu")


def _resolve_device(device: str) -> torch.device:
    return torch.device(DEFAULT_DEVICE if device == "auto" else device)


def _query_affinity(query: torch.Tensor, k: int) -> torch.Tensor:
    """Build the symmetric unweighted query KNN graph in PyTorch."""
    n_query = len(query)
    if n_query < 2:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=query.device),
            torch.empty(0, dtype=query.dtype, device=query.device),
            (n_query, n_query), device=query.device,
        ).coalesce()
    normalized = F.normalize(query, dim=1)
    similarities = normalized @ normalized.T
    k_actual = min(k + 1, n_query)
    _, neighbours = similarities.topk(k_actual, dim=1, largest=True)
    neighbours = neighbours[:, 1:]
    rows = torch.arange(n_query, device=query.device).unsqueeze(1).expand_as(neighbours)
    row = torch.cat([rows.reshape(-1), neighbours.reshape(-1)])
    col = torch.cat([neighbours.reshape(-1), rows.reshape(-1)])
    values = torch.ones(len(row), dtype=query.dtype, device=query.device)
    return torch.sparse_coo_tensor(
        torch.stack([row, col]), values, (n_query, n_query), device=query.device
    ).coalesce()


def _laplacian_pseudolabel(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    n_classes: int,
    config: LaplacianShotMSDEConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the compact LaplacianShot bound-optimization update."""
    if len(query) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=query.device),
            torch.empty(0, dtype=torch.float32, device=query.device),
        )

    prototypes = torch.zeros((n_classes, support.shape[1]), device=query.device)
    missing = torch.ones(n_classes, dtype=torch.bool, device=query.device)
    for class_id in range(n_classes):
        mask = support_labels == class_id
        if mask.any():
            prototypes[class_id] = support[mask].mean(dim=0)
            missing[class_id] = False

    unary = torch.cdist(query, prototypes).square()
    unary[:, missing] = float("inf")
    affinity = _query_affinity(query, config.laplacian_k)
    if _sparse_mm_supported(query.device):
        degree = torch.sparse.sum(affinity, dim=1).to_dense().clamp_min(1).unsqueeze(1)

        def smooth(values: torch.Tensor) -> torch.Tensor:
            return torch.sparse.mm(affinity, values) / degree
    else:
        dense = affinity.to_dense()
        degree = dense.sum(dim=1, keepdim=True).clamp_min(1)

        def smooth(values: torch.Tensor) -> torch.Tensor:
            return dense @ values / degree

    assignments = torch.softmax(-unary, dim=1)
    for _ in tqdm(range(config.laplacian_iterations), desc="Laplacian propagation", unit="iter", leave=False):
        updated = torch.softmax(
            -unary + config.laplacian_lambda * smooth(assignments), dim=1
        )
        delta = (updated - assignments).abs().max()
        assignments = updated
        if float(delta) < config.laplacian_tolerance:
            break
    confidence, labels = assignments.max(dim=1)
    return labels, confidence


def refine(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    support_indices: torch.Tensor,
    query_indices: torch.Tensor,
    config: LaplacianShotMSDEConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alternate corrected LaplacianShot pseudolabeling and MSDE shifts.

    Returns ``(refined_embeddings, final_pseudolabels, final_confidences,
    movement_history)``. Every returned value is a PyTorch tensor on the
    configured device.
    """
    config = config or LaplacianShotMSDEConfig()
    logger.info("LaplacianShot-MSDE refinement: %d samples, mode=%s", len(embeddings), config.mode)
    device = _resolve_device(config.device)
    working = embeddings.to(device=device, dtype=torch.float32).clone()
    labels = labels.to(device=device, dtype=torch.long)
    support_indices = torch.as_tensor(support_indices, dtype=torch.long, device=device)
    query_indices = torch.as_tensor(query_indices, dtype=torch.long, device=device)
    if len(support_indices) == 0 or len(query_indices) == 0:
        raise ValueError("support_indices and query_indices must both be non-empty")
    if len(working) != len(labels):
        raise ValueError("embeddings and labels must have the same number of rows")

    n_classes = config.n_classes or int(labels.max().item()) + 1
    support_labels = labels[support_indices]
    final_labels = torch.empty(0, dtype=torch.long, device=device)
    final_confidence = torch.empty(0, dtype=torch.float32, device=device)
    history: list[torch.Tensor] = []

    torch.manual_seed(config.seed)
    with torch.no_grad():
        for _ in tqdm(range(config.em_rounds), desc="LaplacianShot-MSDE", unit="round"):
            final_labels, final_confidence = _laplacian_pseudolabel(
                working[support_indices], support_labels, working[query_indices],
                n_classes, config,
            )
            joint_labels = labels.clone()
            joint_labels[query_indices] = final_labels

            if config.mode == "no_msde":
                shifted = working
            else:
                msde = MeanShiftDensityEnhancement(
                    k=min(config.msde_k, len(working) - 1),
                    nbd_sample_count_threshold=config.msde_nbd_sample_count_threshold,
                    learning_rate=config.msde_learning_rate,
                    max_iters_shift=config.msde_inner_iterations,
                    shift_threshold=config.msde_shift_threshold,
                    device=str(device),
                    use_chunking=True,
                    weight_chunk_size=2048,
                    enable_gradients=False,
                )
                msde_labels = joint_labels if config.mode == "label_aware_msde" else None
                shifted, _, _ = msde(working, labels=msde_labels)
            gate = torch.ones(len(working), device=device)
            gate[query_indices] = final_confidence.pow(config.gate_sharpness)
            previous = working
            working = gate[:, None] * shifted + (1.0 - gate[:, None]) * previous
            movement = (working[query_indices] - previous[query_indices]).norm(dim=1).mean()
            history.append(movement)
            if float(movement) < config.em_convergence_tolerance:
                break

    return working, final_labels, final_confidence, torch.stack(history)


def filter_by_confidence(
    labels: torch.Tensor, confidences: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return labels and mask for confidence-filtered pseudolabels."""
    keep = confidences >= threshold
    return labels[keep], keep


class LaplacianShotMSDEPseudolabeler:
    """Hydra-instantiable callable EM LaplacianShot + MSDE algorithm."""

    def __init__(self, **kwargs: object) -> None:
        self.config = LaplacianShotMSDEConfig(**kwargs)

    def __call__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        support_indices: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return refine(embeddings, labels, support_indices, query_indices, self.config)
