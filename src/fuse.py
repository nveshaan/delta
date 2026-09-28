"""FUSE pseudolabeling and its configurable ablations.

The module keeps FUSE isolated to pseudolabel generation.  Input embeddings
remain in their original space; the temporary FUSE embedding is discarded
after seed-neighbour voting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


FuseMode = Literal[
    "knn_only",
    "fuse_no_modularity",
    "fuse_no_supervised",
    "fuse_no_random_walk",
    "fuse",
]

DEFAULT_DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


@dataclass
class FuseConfig:
    """Runtime options for FUSE and the five supported ablations."""

    mode: FuseMode = "fuse"
    k_nn: int = 15
    k_vote: int = 5
    confidence: float = 0.8
    k_embed: int = 150
    learning_rate: float = 0.05
    iterations: int = 200
    lambda_modularity: float = 1.0
    lambda_supervised: float = 1.0
    lambda_random_walk: float = 2.0
    random_walks: int = 10
    walk_length: int = 5
    max_labeled_steps: int = 3
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.mode not in {
            "knn_only", "fuse_no_modularity", "fuse_no_supervised",
            "fuse_no_random_walk", "fuse",
        }:
            raise ValueError(f"Unknown FUSE mode: {self.mode}")
        for name in ("k_nn", "k_vote", "k_embed", "iterations", "random_walks", "walk_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")


class FusePseudolabeler:
    """Hydra-instantiable callable wrapper around :func:`pseudolabel`."""

    def __init__(self, **kwargs: object) -> None:
        self.config = FuseConfig(**kwargs)

    def __call__(
        self,
        seed_normals: torch.Tensor,
        seed_anomalies: torch.Tensor,
        unlabeled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return pseudolabel(seed_normals, seed_anomalies, unlabeled, self.config)


def _build_knn_graph(embeddings: torch.Tensor, k_nn: int) -> list[list[int]]:
    """Build a symmetric cosine KNN adjacency list."""
    if len(embeddings) < 2:
        return [[] for _ in range(len(embeddings))]
    n_neighbors = min(k_nn + 1, len(embeddings))
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.T
    indices = similarities.topk(n_neighbors, dim=1, largest=True).indices.cpu().numpy()
    adjacency = [set() for _ in range(len(embeddings))]
    for i in range(len(embeddings)):
        for j in indices[i, 1:]:
            adjacency[i].add(int(j))
            adjacency[int(j)].add(i)
    return [list(neighbours) for neighbours in adjacency]


def _labeled_random_walks(
    adjacency: list[list[int]], label_mask: np.ndarray, *, random_walks: int,
    walk_length: int, max_labeled_steps: int, seed: int,
) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    walks = {i: [] for i in range(len(adjacency))}
    for i in range(len(adjacency)):
        for _ in range(random_walks):
            node = i
            labeled_steps = 0
            for _ in range(max(walk_length - 1, 0)):
                neighbours = adjacency[node]
                if not neighbours:
                    break
                labeled_neighbours = [j for j in neighbours if label_mask[j]]
                if labeled_neighbours and labeled_steps < max_labeled_steps:
                    node = int(labeled_neighbours[rng.integers(len(labeled_neighbours))])
                    labeled_steps += 1
                else:
                    node = int(neighbours[rng.integers(len(neighbours))])
                if label_mask[node]:
                    walks[i].append(node)
    return walks


def _fuse_embedding(
    embeddings: torch.Tensor,
    label_mask: np.ndarray,
    labels: np.ndarray,
    adjacency: list[list[int]],
    config: FuseConfig,
) -> torch.Tensor:
    """Optimize the temporary FUSE embedding."""
    n_nodes = len(embeddings)
    rows, cols = [], []
    for i, neighbours in enumerate(adjacency):
        rows.extend([i] * len(neighbours))
        cols.extend(neighbours)
    device = torch.device(DEFAULT_DEVICE if config.device == "auto" else config.device)
    embeddings = embeddings.to(device=device, dtype=torch.float32)
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
    cols_t = torch.as_tensor(cols, dtype=torch.long, device=device)

    def adjacency_times(value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(value)
        if len(rows_t):
            result.index_add_(0, rows_t, value[cols_t])
        return result

    degree = torch.as_tensor(
        [len(neighbours) for neighbours in adjacency], dtype=torch.float32, device=device
    )
    degree_sum = float(degree.sum().item())
    if degree_sum == 0:
        return torch.zeros((n_nodes, min(config.k_embed, n_nodes)), device=device)

    torch.manual_seed(config.seed)
    embedding = torch.randn(
        (n_nodes, min(config.k_embed, n_nodes)), dtype=torch.float32, device=device
    )
    embedding, _ = torch.linalg.qr(embedding, mode="reduced")
    walks = _labeled_random_walks(
        adjacency, label_mask, random_walks=config.random_walks,
        walk_length=config.walk_length, max_labeled_steps=config.max_labeled_steps,
        seed=config.seed,
    )
    normal_nodes = np.where(label_mask & (labels == 0))[0]
    anomaly_nodes = np.where(label_mask & (labels == 1))[0]
    modularity_weight = 0.0 if config.mode == "fuse_no_modularity" else config.lambda_modularity
    supervised_weight = 0.0 if config.mode == "fuse_no_supervised" else config.lambda_supervised
    random_walk_weight = 0.0 if config.mode == "fuse_no_random_walk" else config.lambda_random_walk

    for _ in range(config.iterations):
        attention: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if random_walk_weight:
            for i, visited in walks.items():
                nodes = torch.as_tensor(visited, dtype=torch.long, device=device).unique()
                if len(nodes):
                    similarities = embedding[i] @ embedding[nodes].T
                    attention[i] = (nodes, torch.softmax(similarities, dim=0))

        graph_gradient = torch.zeros_like(embedding)
        if modularity_weight:
            graph_gradient = (
                adjacency_times(embedding)
                - degree[:, None] * embedding.sum(dim=0, keepdim=True) / degree_sum
            ) / degree_sum

        supervised_gradient = torch.zeros_like(embedding)
        if supervised_weight:
            target = embedding.clone()
            if len(normal_nodes):
                normal_index = torch.as_tensor(normal_nodes, dtype=torch.long, device=device)
                target[normal_index] = embedding[normal_index].mean(dim=0)
            if len(anomaly_nodes):
                anomaly_index = torch.as_tensor(anomaly_nodes, dtype=torch.long, device=device)
                target[anomaly_index] = embedding[anomaly_index].mean(dim=0)
            supervised_gradient = embedding - target

        walk_gradient = torch.zeros_like(embedding)
        if random_walk_weight:
            for i, (nodes, values) in attention.items():
                walk_gradient[i] = embedding[i] - (values[:, None] * embedding[nodes]).sum(dim=0)

        embedding += config.learning_rate * (
            modularity_weight * graph_gradient
            - supervised_weight * supervised_gradient
            - random_walk_weight * walk_gradient
        )
        embedding, _ = torch.linalg.qr(embedding, mode="reduced")

    return embedding[:, : min(config.k_embed, n_nodes)].detach()


def _seed_knn_vote(seed_embedding: torch.Tensor, seed_labels: torch.Tensor,
                   unlabeled_embedding: torch.Tensor, k_vote: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(unlabeled_embedding) == 0:
        return (torch.empty(0, dtype=torch.long, device=unlabeled_embedding.device),
                torch.empty(0, dtype=torch.float32, device=unlabeled_embedding.device))
    seed_embedding = torch.nn.functional.normalize(seed_embedding, dim=1)
    unlabeled_embedding = torch.nn.functional.normalize(unlabeled_embedding, dim=1)
    similarities = unlabeled_embedding @ seed_embedding.T
    indices = similarities.topk(min(k_vote, len(seed_embedding)), dim=1, largest=True).indices
    vote_mean = seed_labels[indices].float().mean(dim=1)
    labels = (vote_mean >= 0.5).long()
    confidence = torch.where(labels == 1, vote_mean, 1.0 - vote_mean)
    return labels, confidence


def pseudolabel(
    seed_normals: torch.Tensor,
    seed_anomalies: torch.Tensor,
    unlabeled: torch.Tensor,
    config: FuseConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pseudolabels and confidences for one joint unlabeled pool.

    ``knn_only`` performs seed KNN voting directly in CLIP space. The other
    modes build the joint graph and differ only in which FUSE objective terms
    are enabled.
    """
    config = config or FuseConfig()
    if len(unlabeled) == 0:
        return (torch.empty(0, dtype=torch.long, device=unlabeled.device),
                torch.empty(0, dtype=torch.float32, device=unlabeled.device))
    device = torch.device(DEFAULT_DEVICE if config.device == "auto" else config.device)
    seed_normals = seed_normals.to(device=device, dtype=torch.float32)
    seed_anomalies = seed_anomalies.to(device=device, dtype=torch.float32)
    unlabeled = unlabeled.to(device=device, dtype=torch.float32)
    all_embeddings = torch.cat([seed_normals, seed_anomalies, unlabeled], dim=0)
    seed_labels = torch.cat([
        torch.zeros(len(seed_normals), dtype=torch.long, device=device),
        torch.ones(len(seed_anomalies), dtype=torch.long, device=device),
    ])
    if len(seed_labels) == 0:
        raise ValueError("At least one normal and one anomaly seed are required")
    if config.mode == "knn_only":
        return _seed_knn_vote(seed_embedding=all_embeddings[:len(seed_labels)], seed_labels=seed_labels,
                              unlabeled_embedding=unlabeled, k_vote=config.k_vote)

    label_mask = np.zeros(len(all_embeddings), dtype=bool)
    label_mask[:len(seed_labels)] = True
    labels = np.zeros(len(all_embeddings), dtype=int)
    labels[:len(seed_labels)] = seed_labels.cpu().numpy()
    adjacency = _build_knn_graph(all_embeddings, config.k_nn)
    fused = _fuse_embedding(all_embeddings, label_mask, labels, adjacency, config)
    return _seed_knn_vote(
        fused[:len(seed_labels)], seed_labels, fused[len(seed_labels):], config.k_vote
    )


def filter_by_confidence(
    labels: torch.Tensor, confidences: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep pseudolabels meeting the configured confidence threshold."""
    keep = confidences >= threshold
    return labels[keep], keep
