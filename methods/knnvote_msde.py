"""PyTorch EM refinement with seed KNN voting and MSDE."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from .msde import DEFAULT_DEVICE, MeanShiftDensityEnhancement


EMMode = Literal["no_msde", "msde", "label_aware_msde"]


@dataclass
class KNNVoteMSDEConfig:
    """Configuration for alternating KNN voting and MSDE."""

    k_vote: int = 5
    em_rounds: int = 5
    em_convergence_tolerance: float = 0.01
    gate_sharpness: float = 1.0
    msde_k: int = 50
    msde_learning_rate: float = 0.33
    msde_nbd_sample_count_threshold: int = 70
    msde_shift_threshold: float = 0.01
    msde_inner_iterations: int = 3
    confidence: float = 0.8
    seed: int = 42
    device: str = "auto"
    mode: EMMode = "label_aware_msde"

    def __post_init__(self) -> None:
        if self.k_vote <= 0 or self.em_rounds <= 0 or self.msde_k <= 0 or self.msde_inner_iterations <= 0:
            raise ValueError("k_vote, em_rounds, msde_k, and msde_inner_iterations must be positive")
        if self.em_convergence_tolerance < 0 or self.gate_sharpness < 0:
            raise ValueError("EM tolerance and gate sharpness must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.mode not in {"no_msde", "msde", "label_aware_msde"}:
            raise ValueError(f"Unknown EM mode: {self.mode}")
        if self.device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, mps, cuda, cpu")


def _resolve_device(device: str) -> torch.device:
    return torch.device(DEFAULT_DEVICE if device == "auto" else device)


def _knn_vote(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    k_vote: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign each query point the majority label among its support neighbours."""
    if len(query) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=query.device),
            torch.empty(0, dtype=torch.float32, device=query.device),
        )
    support_normalized = F.normalize(support, dim=1)
    query_normalized = F.normalize(query, dim=1)
    similarities = query_normalized @ support_normalized.T
    neighbour_count = min(k_vote, len(support))
    neighbour_indices = similarities.topk(neighbour_count, dim=1, largest=True).indices
    neighbour_labels = support_labels[neighbour_indices]

    predictions = []
    confidences = []
    for row in neighbour_labels:
        classes, counts = torch.unique(row, return_counts=True)
        winner = counts.argmax()
        predictions.append(classes[winner])
        confidences.append(counts[winner].float() / neighbour_count)
    return torch.stack(predictions).long(), torch.stack(confidences)


def refine(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    support_indices: torch.Tensor,
    query_indices: torch.Tensor,
    config: KNNVoteMSDEConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alternate current-space KNN voting and confidence-gated MSDE shifts.

    Returns ``(refined_embeddings, final_pseudolabels, final_confidences,
    movement_history)`` as PyTorch tensors.
    """
    config = config or KNNVoteMSDEConfig()
    logger.info("KNNVote-MSDE refinement: %d samples, mode=%s", len(embeddings), config.mode)
    device = _resolve_device(config.device)
    working = embeddings.to(device=device, dtype=torch.float32).clone()
    labels = labels.to(device=device, dtype=torch.long)
    support_indices = torch.as_tensor(support_indices, dtype=torch.long, device=device)
    query_indices = torch.as_tensor(query_indices, dtype=torch.long, device=device)
    if len(support_indices) == 0 or len(query_indices) == 0:
        raise ValueError("support_indices and query_indices must both be non-empty")
    if len(working) != len(labels):
        raise ValueError("embeddings and labels must have the same number of rows")

    support_labels = labels[support_indices]
    final_labels = torch.empty(0, dtype=torch.long, device=device)
    final_confidence = torch.empty(0, dtype=torch.float32, device=device)
    history: list[torch.Tensor] = []
    torch.manual_seed(config.seed)

    with torch.no_grad():
        for _ in tqdm(range(config.em_rounds), desc="KNNVote-MSDE", unit="round"):
            final_labels, final_confidence = _knn_vote(
                working[support_indices], support_labels, working[query_indices], config.k_vote
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
    """Return labels and mask for confidence-filtered KNN pseudolabels."""
    keep = confidences >= threshold
    return labels[keep], keep


class KNNVoteMSDEPseudolabeler:
    """Hydra-instantiable callable KNN-vote + MSDE algorithm."""

    def __init__(self, **kwargs: object) -> None:
        self.config = KNNVoteMSDEConfig(**kwargs)

    def __call__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        support_indices: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return refine(embeddings, labels, support_indices, query_indices, self.config)
