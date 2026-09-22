"""Chest raw-data dataset factory; see ``configs/datasets/raw_chest.yaml``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .raw_common import IMAGE_EXTENSIONS, RawDataset, Transform, pil_to_tensor, select_split


class ChestRawDataset(RawDataset):
    """Aggregate a named chest training/evaluation source from the YAML config."""

    def __init__(self, sources=None, transform: Transform = None, seed: int = 42,
                 sample_strategy: str = "python_sorted", image_extensions=tuple(IMAGE_EXTENSIONS),
                 split: str = "train", **sections: Mapping[str, Any]):
        selected = sources if sources is not None else select_split(sections, split)["sources"]
        super().__init__(selected, transform=transform, seed=seed, sample_strategy=sample_strategy,
                         image_extensions=image_extensions)


def build_dataset(config: Mapping[str, Any], split: str | None = None, transform: Transform = None) -> ChestRawDataset:
    """Build the flat configured source collection."""
    split = split or "sources"
    section = select_split(config, split)
    return ChestRawDataset(section["sources"], transform=transform, seed=int(config.get("seed", 42)),
                           sample_strategy=config.get("sample_strategy", "python_sorted"),
                           image_extensions=config.get("image_extensions", tuple(IMAGE_EXTENSIONS)))


def main() -> None:
    """Load one configured split and print one DataLoader batch shape."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "raw_chest.yaml"
    config = OmegaConf.load(config_path)
    dataset = instantiate(config, split="train", transform=pil_to_tensor)
    images, labels = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    print(f"chest: image batch shape={tuple(images.shape)}, labels shape={tuple(labels.shape)}")


if __name__ == "__main__":
    main()
