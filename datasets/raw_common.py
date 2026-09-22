"""Composable raw-image PyTorch datasets used by the modality configs.

The source experiments used a few small, repeated loaders.  This module keeps
their semantics explicit: folders are non-recursive, samples are sorted before
deterministic caps are applied, and every loaded image is converted to RGB.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
Transform = Callable[[Image.Image], Any] | None


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Small dependency-light transform used by the loader smoke-test mains."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def select_split(sections: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    """Resolve a split, or return the nested ``sources`` config unchanged."""
    if "sources" in sections:
        return sections
    if split in sections:
        return sections[split]
    if split.startswith("eval_"):
        return sections["eval"][split.removeprefix("eval_")]
    if split.startswith("eval/"):
        return sections["eval"][split.removeprefix("eval/")]
    raise KeyError(f"Unknown dataset split {split!r}; expected train or eval_<name>")


def _as_dict(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Accept plain mappings as well as OmegaConf DictConfig objects."""
    return dict(value)


def list_images(folder: str | Path, extensions: Sequence[str] = tuple(IMAGE_EXTENSIONS)) -> list[Path]:
    """Return sorted, non-recursive image files; fail early for a bad config path."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    allowed = {extension.lower() for extension in extensions}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in allowed)


def capped(paths: Sequence[Path], max_samples: int | None, seed: int,
           sample_strategy: str = "python_sorted") -> list[Path]:
    """Apply the precise deterministic cap strategy used by the source script."""
    result = list(paths)
    if max_samples is None or len(result) <= max_samples:
        return result
    if sample_strategy == "python_sorted":
        return sorted(random.Random(seed).sample(result, max_samples))
    if sample_strategy == "python_ordered":
        return random.Random(seed).sample(result, max_samples)
    if sample_strategy == "numpy_sorted":
        indices = sorted(np.random.RandomState(seed).choice(len(result), max_samples, replace=False))
        return [result[index] for index in indices]
    raise ValueError(f"Unknown sample strategy: {sample_strategy}")


def fraction_cap(paths: Sequence[Path], fraction: float, seed: int,
                 sample_strategy: str = "python_sorted") -> list[Path]:
    """Keep floor(N * fraction), but retain one sample for non-empty folders."""
    if not paths:
        return []
    return capped(paths, max(1, int(len(paths) * fraction)), seed, sample_strategy)


class ImageListDataset(Dataset):
    """A PIL-to-PyTorch dataset for explicit image paths and binary labels."""

    def __init__(self, paths: Sequence[str | Path], labels: Sequence[int], transform: Transform = None):
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths = [Path(path) for path in paths]
        self.labels = [int(label) for label in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return (self.transform(image) if self.transform else image), self.labels[index]


class FolderDataset(ImageListDataset):
    """One non-recursive class folder with a fixed label and optional cap."""

    def __init__(
        self,
        folder: str | Path,
        label: int,
        transform: Transform = None,
        max_samples: int | None = None,
        sample_fraction: float | None = None,
        seed: int = 42,
        sample_strategy: str = "python_sorted",
        extensions: Sequence[str] = tuple(IMAGE_EXTENSIONS),
    ):
        paths = list_images(folder, extensions)
        if sample_fraction is not None:
            paths = fraction_cap(paths, sample_fraction, seed, sample_strategy)
        paths = capped(paths, max_samples, seed, sample_strategy)
        super().__init__(paths, [label] * len(paths), transform)


def _resolve_image(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / path


def _entries_to_paths(entries: Any, images_dir: Path, default_label: int | None = None) -> tuple[list[Path], list[int]]:
    """Handle the dict and list layouts seen in MedIAnomaly data.json files."""
    paths: list[Path] = []
    labels: list[int] = []
    if isinstance(entries, Mapping):
        for label, names in entries.items():
            part_paths, part_labels = _entries_to_paths(names, images_dir, int(label))
            paths.extend(part_paths)
            labels.extend(part_labels)
        return paths, labels
    for entry in entries:
        if isinstance(entry, str):
            name, label = entry, default_label
        elif isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("img_path") or entry.get("image") or entry.get("filename")
            label = entry.get("label", entry.get("class", default_label))
        else:
            raise TypeError(f"Unsupported data.json entry: {entry!r}")
        if name is None or label is None:
            raise ValueError(f"MedIAnomaly entry has no image name or label: {entry!r}")
        paths.append(_resolve_image(images_dir, str(name)))
        labels.append(int(label))
    return paths, labels


class MedIAnomalyDataset(ImageListDataset):
    """Load one labelled ``data.json`` split from ``<root>/images``.

    ``split: test`` preserves the test-only evaluation behaviour in the supplied
    scripts.  The fallback normal_test/abnormal_test layout is also supported.
    """

    def __init__(self, root: str | Path, split: str = "test", transform: Transform = None):
        root_path = Path(root)
        with (root_path / "data.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
        images_dir = root_path / "images"
        if split in payload:
            paths, labels = _entries_to_paths(payload[split], images_dir)
        elif split == "test" and ("normal_test" in payload or "abnormal_test" in payload):
            normal, _ = _entries_to_paths(payload.get("normal_test", []), images_dir, 0)
            abnormal, _ = _entries_to_paths(payload.get("abnormal_test", []), images_dir, 1)
            paths, labels = normal + abnormal, [0] * len(normal) + [1] * len(abnormal)
        else:
            raise KeyError(f"No {split!r} split in {root_path / 'data.json'}")
        super().__init__(paths, labels, transform)


def _csv_columns(fieldnames: Sequence[str], name_column: str | None, label_column: str | None) -> tuple[str, str]:
    lowered = {column.strip().lower(): column for column in fieldnames}
    name = name_column or next((lowered[key] for key in ("image", "filename", "name", "img", "file") if key in lowered), fieldnames[0])
    label = label_column or next((lowered[key] for key in ("label", "class", "glaucoma", "disease", "target", "anomaly") if key in lowered), fieldnames[-1])
    return name, label


def _resolve_csv_image(images_dir: Path, filename: str) -> Path | None:
    candidate = images_dir / filename
    if candidate.is_file():
        return candidate
    if candidate.suffix:
        return None
    for extension in IMAGE_EXTENSIONS:
        alternative = candidate.with_suffix(extension)
        if alternative.is_file():
            return alternative
    return None


class CSVLabelDataset(ImageListDataset):
    """CSV image/label loader that preserves numeric and categorical classes."""

    NORMAL_VALUES = {"0", "normal", "n", "false"}

    def __init__(self, images_dir: str | Path, csv_path: str | Path, transform: Transform = None,
                 name_column: str | None = None, label_column: str | None = None):
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {csv_path}")
            name_column, label_column = _csv_columns(reader.fieldnames, name_column, label_column)
            rows = list(reader)
        paths, raw_labels = [], []
        for row in rows:
            path = _resolve_csv_image(Path(images_dir), row[name_column])
            if path is None:
                continue
            paths.append(path)
            raw_labels.append(str(row[label_column]).strip())

        # Keep integer class ids as-is. For textual classes, reserve 0 for the
        # normal values and assign deterministic ids to the remaining classes.
        categorical = sorted({value.lower() for value in raw_labels
                               if value.lower() not in self.NORMAL_VALUES
                               and not value.lstrip("+-").isdigit()})
        categorical_ids = {value: index + 1 for index, value in enumerate(categorical)}
        labels = []
        for value in raw_labels:
            lowered = value.lower()
            if lowered in self.NORMAL_VALUES:
                labels.append(0)
            elif lowered in categorical_ids:
                labels.append(categorical_ids[lowered])
            else:
                labels.append(int(value))
        super().__init__(paths, labels, transform)


class SymptomCSVDataset(ImageListDataset):
    """Chest emphysema loader: any positive radiologist symptom means anomaly."""

    def __init__(self, images_dir: str | Path, csv_path: str | Path, symptom_columns: Sequence[str],
                 mode: str, transform: Transform = None):
        if mode not in {"normal", "abnormal"}:
            raise ValueError("mode must be 'normal' or 'abnormal'")
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        paths: list[Path] = []
        for row in rows:
            abnormal = any(str(row.get(column, "")).strip().lower() in {"true", "1"} for column in symptom_columns)
            if abnormal != (mode == "abnormal"):
                continue
            path = _resolve_csv_image(Path(images_dir), row["filename"])
            if path is not None:
                paths.append(path)
        label = 0 if mode == "normal" else 1
        super().__init__(paths, [label] * len(paths), transform)


class RawDataset(ImageListDataset):
    """Combine declarative source entries from a raw-modality Hydra config."""

    def __init__(self, sources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
                 transform: Transform = None, seed: int = 42,
                 sample_strategy: str = "python_sorted",
                 image_extensions: Sequence[str] = tuple(IMAGE_EXTENSIONS)):
        def flatten_sources(node: Any):
            """Yield leaf source configs from the nested dataset/subtype map."""
            if isinstance(node, Mapping):
                if "loader" in node:
                    yield node
                elif "datasets" in node:
                    yield from flatten_sources(node["datasets"])
                else:
                    for value in node.values():
                        yield from flatten_sources(value)
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                for value in node:
                    yield from flatten_sources(value)
            else:
                raise TypeError(f"Expected a source mapping or sequence, got {type(node).__name__}")

        sources = list(flatten_sources(sources))
        paths: list[Path] = []
        labels: list[int] = []
        for source in sources:
            cfg = _as_dict(source)
            loader = cfg.pop("loader")
            path_spec = cfg.pop("paths", None)
            if path_spec is not None:
                if loader == "folder":
                    cfg["folder"] = path_spec[0] if isinstance(path_spec, Sequence) and not isinstance(path_spec, str) else path_spec["folder"]
                elif loader == "mediany_test":
                    cfg["root"] = path_spec[0] if isinstance(path_spec, Sequence) and not isinstance(path_spec, str) else path_spec["root"]
                elif loader in {"csv_labels", "symptom_csv"}:
                    cfg["images_dir"] = path_spec["images_dir"]
                    cfg["csv_path"] = path_spec["csv_path"]
            cfg.pop("name", None)
            cfg.pop("notes", None)
            cfg.setdefault("seed", seed)
            cfg.setdefault("sample_strategy", sample_strategy)
            cfg.setdefault("extensions", image_extensions)
            if loader == "folder":
                dataset = FolderDataset(transform=transform, **cfg)
            elif loader == "mediany_test":
                cfg.pop("seed", None)
                cfg.pop("sample_strategy", None)
                cfg.pop("extensions", None)
                dataset = MedIAnomalyDataset(transform=transform, split="test", **cfg)
            elif loader == "csv_labels":
                cfg.pop("seed", None)
                cfg.pop("sample_strategy", None)
                cfg.pop("extensions", None)
                dataset = CSVLabelDataset(transform=transform, **cfg)
            elif loader == "symptom_csv":
                cfg.pop("seed", None)
                cfg.pop("sample_strategy", None)
                cfg.pop("extensions", None)
                dataset = SymptomCSVDataset(transform=transform, **cfg)
            else:
                raise ValueError(f"Unknown raw loader {loader!r}")
            paths.extend(dataset.paths)
            labels.extend(dataset.labels)
        # A metadata CSV can list the same image in multiple source files
        # (notably emphysema train/test). Keep one sample and reject conflicting
        # labels instead of silently training on contradictory duplicates.
        unique_paths: list[Path] = []
        unique_labels: list[int] = []
        seen: dict[Path, int] = {}
        for path, label in zip(paths, labels):
            if path in seen:
                if seen[path] != label:
                    raise ValueError(f"Conflicting labels for duplicate image: {path}")
                continue
            seen[path] = label
            unique_paths.append(path)
            unique_labels.append(label)
        super().__init__(unique_paths, unique_labels, transform)
