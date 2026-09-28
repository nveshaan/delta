"""Embeddings PyTorch dataset loaders for pre-computed feature representations.

Exposes pre-computed embeddings (<encoder>_embeds.npy) and labels (labels.npy)
saved by ``scripts/generate_embeddings.py`` as PyTorch datasets.

Modality dataset configs are located in ``configs/datasets/embeds_<modality>.yaml``,
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


class SingleEmbedsDataset(Dataset):
    """PyTorch Dataset for a single dataset folder's pre-computed embeddings and labels."""

    def __init__(
        self,
        dataset_dir: str | Path,
        dataset_name: str | None = None,
        encoder: str = "CLIP",
        pool_anomalies: bool = False,
        cap: int | None = None,
        seed: int = 42,
        as_tensor: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = dataset_name or self.dataset_dir.name
        self.encoder = encoder
        self.pool_anomalies = pool_anomalies
        self.cap = cap
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

        if self.cap is not None and len(embeds) > self.cap:
            rng = np.random.RandomState(self.seed)
            indices = sorted(rng.choice(len(embeds), self.cap, replace=False))
            embeds = embeds[indices]
            labels = labels[indices]

        if self.pool_anomalies:
            labels = np.where(labels >= 1, 1, 0).astype(labels.dtype)

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
    configs/datasets/embeds_<modality>.yaml, applies per-dataset caps, and optionally
    pools anomalies into a binary label (0 = normal, 1 = anomalous).
    """

    def __init__(
        self,
        modality: str | None = None,
        datasets: Mapping[str, Mapping[str, Any]] | None = None,
        data_root: str | Path | None = None,
        split: str | None = "train",
        encoder: str = "CLIP",
        pool_anomalies: bool = False,
        cap: int | None = None,
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
        self.pool_anomalies = pool_anomalies
        self.cap_override = cap
        self.seed = seed
        self.as_tensor = as_tensor
        self.transform = transform

        cfg_dict: dict[str, Any] = {}
        if config_path is not None or (datasets is None and modality is not None):
            resolved_config_path = (
                Path(config_path)
                if config_path is not None
                else PROJECT_ROOT / "configs" / "datasets" / f"embeds_{modality}.yaml"
            )
            if resolved_config_path.is_file():
                loaded = OmegaConf.load(resolved_config_path)
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

        all_embeds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        self.dataset_names: list[str] = []
        self.sample_dataset_names: list[str] = []

        split_filter = split.lower() if split is not None else "all"

        for dataset_name, dataset_meta in datasets.items():
            meta = dict(dataset_meta) if isinstance(dataset_meta, Mapping) else {}
            tag = str(meta.get("tag", "train")).lower()

            if split_filter not in {"all", "both"} and tag != split_filter:
                continue

            dataset_cap = self.cap_override if self.cap_override is not None else meta.get("cap")
            if dataset_cap is not None:
                dataset_cap = int(dataset_cap)

            dataset_dir = self.root_path / dataset_name
            if not dataset_dir.is_dir():
                continue

            sub_ds = SingleEmbedsDataset(
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                encoder=self.encoder,
                pool_anomalies=self.pool_anomalies,
                cap=dataset_cap,
                seed=self.seed,
                as_tensor=False,
            )

            if len(sub_ds) > 0:
                all_embeds.append(sub_ds.embeddings)
                all_labels.append(sub_ds.labels)
                self.dataset_names.append(dataset_name)
                self.sample_dataset_names.extend([dataset_name] * len(sub_ds))

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
    """Aggregate chest embedding datasets from configs/datasets/embeds_chest.yaml."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("modality", None)
        super().__init__(modality="chest", **kwargs)


class FundusEmbedsDataset(ModalityEmbedsDataset):
    """Aggregate fundus embedding datasets from configs/datasets/embeds_fundus.yaml."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("modality", None)
        super().__init__(modality="fundus", **kwargs)


class MRIEmbedsDataset(ModalityEmbedsDataset):
    """Aggregate MRI embedding datasets from configs/datasets/embeds_mri.yaml."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("modality", None)
        super().__init__(modality="mri", **kwargs)


class OCTEmbedsDataset(ModalityEmbedsDataset):
    """Aggregate OCT embedding datasets from configs/datasets/embeds_oct.yaml."""

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
    pool_anomalies: bool = False,
    cap: int | None = None,
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
        pool_anomalies=pool_anomalies,
        cap=cap,
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
        "--pool-anomalies",
        action="store_true",
        help="Pool all non-zero labels to 1 (binary anomaly detection mode)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Optional override for per-dataset sample cap",
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
    print(f"  Pool anomalies  : {args.pool_anomalies}")
    print(f"  Cap override    : {args.cap}")

    dataset = build_embeds_dataset(
        modality=args.modality,
        split=args.split,
        encoder=args.encoder,
        pool_anomalies=args.pool_anomalies,
        cap=args.cap,
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
