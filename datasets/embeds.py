"""Embeddings PyTorch dataset loaders for pre-computed feature representations.

Exposes pre-computed embeddings (<encoder>_embeds.npy) and labels (labels.npy)
saved by ``scripts/generate_embeddings.py`` as PyTorch datasets.

Modality dataset configs are located in ``configs/modality/<modality>.yaml``,
where each dataset includes a tag ('train' or 'test') and a sample cap limit
matching the experimental protocol in stage-2 distillation and evaluation.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_name(name: str) -> str:
    """Sanitize name to match the encoder filename pattern in generate_embeddings."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.") or "dataset"


def _resolve_embeds_file(dataset_dir: Path, encoder: str) -> Path:
    """Find the embedding file matching the requested encoder."""
    candidates = [
        dataset_dir / f"{encoder}_embeds.npy",
        dataset_dir / f"{_safe_name(encoder)}_embeds.npy",
        dataset_dir / f"{encoder.upper()}_embeds.npy",
        dataset_dir / f"{encoder.lower()}_embeds.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Fallback: case-insensitive search in dataset_dir
    lowered = encoder.lower()
    for file in dataset_dir.glob("*_embeds.npy"):
        stem = file.name.removesuffix("_embeds.npy").lower()
        if stem == lowered or stem == _safe_name(lowered):
            return file

    available = [f.name.removesuffix("_embeds.npy") for f in dataset_dir.glob("*_embeds.npy")]
    raise FileNotFoundError(
        f"Embeddings for encoder {encoder!r} not found in {dataset_dir}. "
        f"Available encoders in this directory: {available}"
    )


def _balanced_cap_indices(labels: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Select at most ``cap`` samples, balancing labels as evenly as possible."""
    if cap < 0:
        raise ValueError(f"cap must be non-negative, got {cap}")
    if cap >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    if cap == 0 or len(labels) == 0:
        return np.empty((0,), dtype=np.int64)

    rng = np.random.RandomState(seed)
    label_values = np.sort(np.unique(labels))
    shuffled_by_label: dict[Any, np.ndarray] = {}
    for label in label_values:
        label_indices = np.flatnonzero(labels == label)
        shuffled_by_label[label.item() if hasattr(label, "item") else label] = rng.permutation(
            label_indices
        )

    # Round-robin selection gives every available label the same contribution
    # before a label is exhausted, with deterministic tie-breaking by label.
    selected: list[int] = []
    positions = {label: 0 for label in shuffled_by_label}
    while len(selected) < cap:
        made_progress = False
        for label in shuffled_by_label:
            position = positions[label]
            candidates = shuffled_by_label[label]
            if position >= len(candidates):
                continue
            selected.append(int(candidates[position]))
            positions[label] += 1
            made_progress = True
            if len(selected) == cap:
                break
        if not made_progress:
            break

    return np.sort(np.asarray(selected, dtype=np.int64))


def _balanced_binary_indices(labels: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Select an equal number of zero and non-zero samples.

    Every non-zero label gets the same number of samples, and the total number
    of non-zero samples equals the number of zero samples.
    """
    nonzero_labels = sorted(int(label) for label in np.unique(labels) if label != 0)
    if cap <= 0 or not nonzero_labels or not np.any(labels == 0):
        return np.empty((0,), dtype=np.int64)

    per_nonzero_label = cap // (2 * len(nonzero_labels))
    if per_nonzero_label == 0:
        return np.empty((0,), dtype=np.int64)

    rng = np.random.RandomState(seed)
    selected: list[int] = []
    groups = [np.flatnonzero(labels == 0)]
    groups.extend(np.flatnonzero(labels == label) for label in nonzero_labels)
    for group_index, group in enumerate(groups):
        required = per_nonzero_label * len(nonzero_labels) if group_index == 0 else per_nonzero_label
        if len(group) < required:
            return np.empty((0,), dtype=np.int64)
        selected.extend(int(index) for index in rng.permutation(group)[:required])
    return np.sort(np.asarray(selected, dtype=np.int64))


def _automatic_cap(
    loaded: list[tuple[str, "SingleEmbedsDataset"]], seed: int, max_total_size: int
) -> dict[str, np.ndarray]:
    """Build equal-size, class-balanced selections within a total-size budget."""
    if not loaded:
        return {}

    dataset_count = len(loaded)
    min_dataset_size = min(len(dataset) for _, dataset in loaded)
    label_counts = []
    max_balanced_sizes = []
    for _, dataset in loaded:
        nonzero_labels = [int(label) for label in np.unique(dataset.labels) if label != 0]
        label_counts.append(len(nonzero_labels))
        if nonzero_labels and np.any(dataset.labels == 0):
            zero_count = int(np.sum(dataset.labels == 0))
            nonzero_count = min(int(np.sum(dataset.labels == label)) for label in nonzero_labels)
            max_balanced_sizes.append(
                2 * len(nonzero_labels) * min(zero_count // len(nonzero_labels), nonzero_count)
            )
        else:
            max_balanced_sizes.append(0)
    label_lcm = int(np.lcm.reduce([count for count in label_counts if count], initial=1))
    per_dataset_limit = min(
        [min_dataset_size, max_total_size // dataset_count, *max_balanced_sizes]
    )
    common_cap = (per_dataset_limit // (2 * label_lcm)) * (2 * label_lcm)

    selections: dict[str, np.ndarray] = {}
    for offset, (dataset_name, dataset) in enumerate(loaded):
        selections[dataset_name] = _balanced_binary_indices(
            dataset.labels, common_cap, seed + offset
        )
    return selections


def _test_cap_indices(
    loaded: list[tuple[str, "SingleEmbedsDataset"]], seed: int, max_total_size: int
) -> dict[str, np.ndarray]:
    """Keep all test samples up to one global maximum, without balancing."""
    total_size = sum(len(dataset) for _, dataset in loaded)
    if total_size <= max_total_size:
        return {
            dataset_name: np.arange(len(dataset), dtype=np.int64)
            for dataset_name, dataset in loaded
        }

    rng = np.random.RandomState(seed)
    selected_global = np.sort(rng.permutation(total_size)[:max_total_size])
    selections: dict[str, np.ndarray] = {}
    offset = 0
    for dataset_name, dataset in loaded:
        local_mask = (selected_global >= offset) & (selected_global < offset + len(dataset))
        selections[dataset_name] = selected_global[local_mask] - offset
        offset += len(dataset)
    return selections


class SingleEmbedsDataset(Dataset):
    """PyTorch Dataset for a single dataset folder's pre-computed embeddings and labels."""

    def __init__(
        self,
        dataset_dir: str | Path,
        dataset_name: str | None = None,
        encoder: str = "CLIP",
        cap: int | None = None,
        include_labels: list[int] | tuple[int, ...] | set[int] | None = None,
        seed: int = 42,
        as_tensor: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = dataset_name or self.dataset_dir.name
        self.encoder = encoder
        self.cap = cap
        self.include_labels = include_labels
        self.seed = seed
        self.as_tensor = as_tensor
        self.transform = transform

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        embeds_path = _resolve_embeds_file(self.dataset_dir, self.encoder)
        labels_path = self.dataset_dir / "labels.npy"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        embeds = np.load(embeds_path)
        labels = np.load(labels_path)

        if len(embeds) != len(labels):
            raise ValueError(
                f"Embeddings and labels length mismatch in {self.dataset_dir}: "
                f"{len(embeds)} embeds vs {len(labels)} labels"
            )

        if include_labels is not None:
            include_labels_array = np.asarray(list(include_labels))
            keep = np.isin(labels, include_labels_array)
            embeds = embeds[keep]
            labels = labels[keep]

        if self.cap is not None and len(embeds) > self.cap:
            indices = _balanced_cap_indices(labels, self.cap, self.seed)
            embeds = embeds[indices]
            labels = labels[indices]

        self.embeddings = embeds
        self.labels = labels

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | np.ndarray, int | torch.Tensor]:
        emb = self.embeddings[index]
        lbl = int(self.labels[index])

        if self.as_tensor:
            emb_tensor = torch.from_numpy(emb).float()
            if self.transform is not None:
                emb_tensor = self.transform(emb_tensor)
            return emb_tensor, lbl

        return emb, lbl


class ModalityEmbedsDataset(Dataset):
    """Aggregate pre-computed embedding datasets for a modality.

    Filters datasets by split ('train', 'test', or 'all') using tags specified in
    configs/modality/<modality>.yaml. When capping is enabled, every
    dataset in the selected split contributes equally, with zero and non-zero
    labels balanced within each dataset.

    Labels are made globally unique while datasets are merged. ``dataset_label_mapping``
    and ``label_metadata`` expose the relationship between merged labels and their
    source datasets.
    """

    def __init__(
        self,
        modality: str | None = None,
        datasets: Mapping[str, Mapping[str, Any]] | None = None,
        data_root: str | Path | None = None,
        split: str | None = "train",
        encoder: str = "CLIP",
        cap: bool | None = None,
        max_dataset_size: int = 10000,
        seed: int = 42,
        as_tensor: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        config_path: str | Path | None = None,
        **sections: Mapping[str, Any],
    ):
        super().__init__()
        self.modality = modality
        self.split = split
        self.encoder = encoder
        self.cap_enabled = cap
        if max_dataset_size <= 0:
            raise ValueError("max_dataset_size must be positive")
        self.max_dataset_size = max_dataset_size
        self.seed = seed
        self.as_tensor = as_tensor
        self.transform = transform

        # The maps are public so callers can interpret labels returned by this dataset.
        self.dataset_label_mapping: dict[str, dict[int, int]] = {}
        self.label_metadata: dict[int, dict[str, Any]] = {}
        self.label_to_dataset = self.label_metadata
        self.label_mapping = self.label_metadata

        cfg_dict: dict[str, Any] = {}
        if config_path is not None or (datasets is None and modality is not None):
            resolved_config_path = (
                Path(config_path)
                if config_path is not None
                else PROJECT_ROOT / "configs" / "modality" / f"{modality}.yaml"
            )
            if resolved_config_path.is_file():
                loaded = OmegaConf.load(resolved_config_path)
                # The modality configs are also Hydra targets and use these
                # interpolations. Resolve them when this dataset is created
                # directly, outside Hydra composition.
                loaded.encoder = self.encoder
                loaded.split = self.split
                loaded.seed = self.seed
                cfg_dict = OmegaConf.to_container(loaded, resolve=True)

        if datasets is None:
            datasets = cfg_dict.get("datasets", {})
        if data_root is None:
            data_root = cfg_dict.get("data_root")
        if data_root is None and modality is not None:
            data_root = f"data/{modality}"

        root_path = Path(data_root) if data_root is not None else PROJECT_ROOT / f"data/{self.modality}"
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path

        self.root_path = root_path

        if self.cap_enabled is None:
            self.cap_enabled = bool(cfg_dict.get("cap", False))
        if max_dataset_size == 10000 and "max_dataset_size" in cfg_dict:
            self.max_dataset_size = int(cfg_dict["max_dataset_size"])
            if self.max_dataset_size <= 0:
                raise ValueError("max_dataset_size must be positive")

        all_embeds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        self.dataset_names: list[str] = []
        self.sample_dataset_names: list[str] = []

        split_filter = split.lower() if split is not None else "all"

        loaded: list[tuple[str, SingleEmbedsDataset]] = []
        for dataset_name, dataset_meta in datasets.items():
            meta = dict(dataset_meta) if isinstance(dataset_meta, Mapping) else {}
            tag = str(meta.get("tag", "train")).lower()

            if split_filter not in {"all", "both"} and tag != split_filter:
                continue

            dataset_dir = self.root_path / dataset_name
            if not dataset_dir.is_dir():
                continue

            include_labels = meta.get("include_labels", meta.get("labels"))
            if include_labels is not None:
                include_labels = [int(label) for label in include_labels]

            sub_ds = SingleEmbedsDataset(
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                encoder=self.encoder,
                include_labels=include_labels,
                seed=self.seed,
                as_tensor=False,
            )

            if len(sub_ds) > 0:
                loaded.append((dataset_name, sub_ds))

        if self.cap_enabled and split_filter == "test":
            # Test evaluation only has a global size limit. Unlike training,
            # test datasets and labels do not need equal contributions.
            selections = _test_cap_indices(loaded, self.seed, self.max_dataset_size)
        elif self.cap_enabled:
            selections = _automatic_cap(loaded, self.seed, self.max_dataset_size)
        else:
            selections = {
                dataset_name: np.arange(len(sub_ds), dtype=np.int64)
                for dataset_name, sub_ds in loaded
            }

        for dataset_name, sub_ds in loaded:
            selected_indices = selections[dataset_name]
            if self.cap_enabled:
                sub_embeds = sub_ds.embeddings[selected_indices]
                sub_labels = sub_ds.labels[selected_indices]
            else:
                sub_embeds = sub_ds.embeddings
                sub_labels = sub_ds.labels

            if len(sub_embeds) > 0:
                source_labels = sorted(int(label) for label in np.unique(sub_labels))
                label_map = {
                    source_label: len(self.label_metadata) + offset
                    for offset, source_label in enumerate(source_labels)
                }
                remapped_labels = np.asarray(
                    [label_map[int(label)] for label in sub_labels], dtype=np.int64
                )
                self.dataset_label_mapping[dataset_name] = label_map
                for source_label, merged_label in label_map.items():
                    self.label_metadata[merged_label] = {
                        "dataset": dataset_name,
                        "source_label": source_label,
                    }
                all_embeds.append(sub_embeds)
                all_labels.append(remapped_labels)
                self.dataset_names.append(dataset_name)
                self.sample_dataset_names.extend([dataset_name] * len(sub_embeds))

        if all_embeds:
            self.embeddings = np.concatenate(all_embeds, axis=0)
            self.labels = np.concatenate(all_labels, axis=0)
        else:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.labels = np.empty((0,), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | np.ndarray, int | torch.Tensor]:
        emb = self.embeddings[index]
        lbl = int(self.labels[index])

        if self.as_tensor:
            emb_tensor = torch.from_numpy(emb).float()
            if self.transform is not None:
                emb_tensor = self.transform(emb_tensor)
            return emb_tensor, lbl

        return emb, lbl

    def get_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the complete dataset as (embeddings, labels) tensors."""
        emb_tensor = torch.from_numpy(self.embeddings).float()
        lbl_tensor = torch.from_numpy(self.labels).long()
        if self.transform is not None:
            emb_tensor = self.transform(emb_tensor)
        return emb_tensor, lbl_tensor


class ChestEmbedsDataset(ModalityEmbedsDataset):
    """Aggregate chest embedding datasets from configs/modality/chest.yaml."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("modality", None)
        super().__init__(modality="chest", **kwargs)


class FundusEmbedsDataset(ModalityEmbedsDataset):
    """Aggregate fundus embedding datasets from configs/modality/fundus.yaml."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("modality", None)
        super().__init__(modality="fundus", **kwargs)


class MRIEmbedsDataset(ModalityEmbedsDataset):
    """Aggregate MRI embedding datasets from configs/modality/mri.yaml."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("modality", None)
        super().__init__(modality="mri", **kwargs)


class OCTEmbedsDataset(ModalityEmbedsDataset):
    """Aggregate OCT embedding datasets from configs/modality/oct.yaml."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("modality", None)
        super().__init__(modality="oct", **kwargs)


MODALITY_EMBEDS_DATASETS: dict[str, type[ModalityEmbedsDataset]] = {
    "chest": ChestEmbedsDataset,
    "fundus": FundusEmbedsDataset,
    "mri": MRIEmbedsDataset,
    "oct": OCTEmbedsDataset,
}


def build_embeds_dataset(
    modality: str,
    split: str = "train",
    encoder: str = "CLIP",
    cap: bool | None = None,
    max_dataset_size: int = 10000,
    seed: int = 42,
    as_tensor: bool = True,
    **kwargs: Any,
) -> ModalityEmbedsDataset:
    """Build a modality embedding dataset from configured YAML."""
    cls = MODALITY_EMBEDS_DATASETS.get(modality.lower(), ModalityEmbedsDataset)
    return cls(
        modality=modality,
        split=split,
        encoder=encoder,
        cap=cap,
        max_dataset_size=max_dataset_size,
        seed=seed,
        as_tensor=as_tensor,
        **kwargs,
    )


def main() -> None:
    """Smoke test CLI for loading and inspecting pre-computed embeddings."""
    parser = argparse.ArgumentParser(description="Load pre-computed embeddings dataset.")
    parser.add_argument(
        "--modality",
        choices=["chest", "fundus", "mri", "oct"],
        default="chest",
        help="Medical imaging modality (default: chest)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="train",
        help="Split filter based on dataset tag (default: train)",
    )
    parser.add_argument(
        "--encoder",
        default="CLIP",
        help="Encoder name matching saved embeds (default: CLIP)",
    )
    parser.add_argument(
        "--cap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable automatic equal-dataset and balanced-label capping",
    )
    parser.add_argument(
        "--max-dataset-size",
        type=int,
        default=10000,
        help="Maximum total samples in the merged dataset when capping (default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader check (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic capping (default: 42)",
    )
    args = parser.parse_args()

    print(f"\n--- Loading {args.modality} embeddings ---")
    print(f"  Encoder         : {args.encoder}")
    print(f"  Split           : {args.split}")
    print(f"  Cap enabled     : {args.cap}")
    print(f"  Max dataset size: {args.max_dataset_size}")

    dataset = build_embeds_dataset(
        modality=args.modality,
        split=args.split,
        encoder=args.encoder,
        cap=args.cap,
        max_dataset_size=args.max_dataset_size,
        seed=args.seed,
    )

    print(f"\n  Loaded {len(dataset.dataset_names)} dataset(s): {dataset.dataset_names}")
    print(f"  Total samples   : {len(dataset)}")
    if len(dataset) > 0:
        print(f"  Embedding shape : {dataset.embeddings.shape}")
        unique, counts = np.unique(dataset.labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  Label distribution: {dist}")

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        batch_embs, batch_lbls = next(iter(loader))
        print(f"  DataLoader batch: embs={tuple(batch_embs.shape)}, labels={tuple(batch_lbls.shape)}")
    else:
        print("  Warning: Dataset is empty.")


if __name__ == "__main__":
    main()
