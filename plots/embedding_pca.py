"""Plot 2D PCA projections of the pre-computed embeddings of every dataset.

For each modality and encoder, one multi-page PDF is written to
``assets/embeddings/<modality>/<encoder>.pdf``:

1. A pooled page. PCA is fitted on all datasets of the modality together (train
   and test tags, no cap). One panel is colored by source dataset; the other is
   colored by normal (label 0) versus anomalous (any non-zero label).
2. One page per dataset. PCA is fitted on that dataset alone, and points are
   colored by their source label.

Every sample is used. ``include_labels`` in ``configs/modality/<modality>.yaml``
is respected, but ``cap`` and ``max_dataset_size`` are ignored. PCA moments are
accumulated in float64 chunks, so the pooled fit never holds a concatenated
copy of all embeddings in memory. Axis labels give the variance explained by
each component. Scatter points are rasterized at 200 dpi to keep the PDFs small.

Usage::

    uv run python plots/embedding_pca.py

    # A subset of modalities and encoders.
    uv run python plots/embedding_pca.py --modalities oct mri --encoders CLIP MedSigLIP
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from zero_shot_encoder_method_consistency import PALETTE, PROJECT_ROOT, _apply_publication_style


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "assets" / "embeddings"
MODALITIES = ("chest", "fundus", "mri", "oct")
MODALITY_TITLES = {"chest": "Chest X-ray", "fundus": "Fundus", "mri": "MRI", "oct": "OCT"}
NORMAL_COLOR = PALETTE["blue_main"]
ANOMALY_COLOR = PALETTE["red_strong"]
CHUNK_SIZE = 16384
RASTER_DPI = 200
# House palette reordered for contrast between neighbours, then Okabe-Ito hues.
DISTINCT_COLORS = (
    PALETTE["blue_main"],
    PALETTE["red_strong"],
    PALETTE["green_3"],
    PALETTE["violet"],
    "#E69F00",
    PALETTE["teal"],
    "#56B4E9",
    "#CC79A7",
    "#8C564B",
    "#D55E00",
    "#7F7F7F",
    "#BCBD22",
    "#000000",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class Moments:
    """Sufficient statistics for PCA: sample count, sum, and Gram matrix."""

    count: int
    total: np.ndarray
    gram: np.ndarray

    def __add__(self, other: Moments) -> Moments:
        return Moments(self.count + other.count, self.total + other.total, self.gram + other.gram)


@dataclass
class Projection:
    """2D PCA coordinates of one dataset with its labels."""

    name: str
    tag: str
    coords: np.ndarray
    labels: np.ndarray


def _categorical_colors(count: int) -> list[str]:
    """Return ``count`` distinct colors, house palette first."""
    colors = list(DISTINCT_COLORS)
    if count <= len(colors):
        return colors[:count]
    cmap = plt.get_cmap("tab20")
    return colors + [matplotlib.colors.to_hex(cmap(index % 20)) for index in range(count - len(colors))]


def _label_colors(labels: np.ndarray) -> dict[int, str]:
    """Label 0 (normal) keeps the normal color; non-zero labels get the others."""
    values = sorted(int(label) for label in np.unique(labels))
    others = [color for color in _categorical_colors(len(values) + 1) if color != NORMAL_COLOR]
    nonzero = [label for label in values if label != 0]
    mapping = {label: others[index] for index, label in enumerate(nonzero)}
    if 0 in values:
        mapping[0] = NORMAL_COLOR
    return mapping


def _load_modality_config(modality: str) -> dict:
    """Read the dataset list and data root of one modality without resolving Hydra keys."""
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "modality" / f"{modality}.yaml")
    datasets = OmegaConf.to_container(config.datasets, resolve=False)
    data_root = Path(config.get("data_root", f"data/{modality}"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    return {"datasets": datasets, "data_root": data_root}


def _default_encoders() -> list[str]:
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "embeddings.yaml")
    return list(config.encoders.keys())


def _load_dataset(dataset_dir: Path, encoder: str, include_labels: list[int] | None):
    """Load embeddings and labels of one dataset, or ``None`` when missing."""
    embeds_path = next(
        (
            path
            for path in sorted(dataset_dir.glob("*_embeds.npy"))
            if path.name.removesuffix("_embeds.npy").lower() == encoder.lower()
        ),
        None,
    )
    if embeds_path is None:
        return None
    embeds = np.load(embeds_path, mmap_mode="r")
    labels = np.load(dataset_dir / "labels.npy")
    if len(embeds) != len(labels):
        raise ValueError(f"{dataset_dir}: {len(embeds)} embeds vs {len(labels)} labels")
    if include_labels is not None:
        keep = np.flatnonzero(np.isin(labels, include_labels))
        embeds = embeds[keep]
        labels = labels[keep]
    return embeds, labels


def _moments(embeds: np.ndarray) -> Moments:
    """Accumulate count, sum, and Gram matrix in float64 chunks."""
    dim = embeds.shape[1]
    total = np.zeros(dim, dtype=np.float64)
    gram = np.zeros((dim, dim), dtype=np.float64)
    for start in range(0, len(embeds), CHUNK_SIZE):
        chunk = np.asarray(embeds[start : start + CHUNK_SIZE], dtype=np.float64)
        total += chunk.sum(axis=0)
        gram += chunk.T @ chunk
    return Moments(len(embeds), total, gram)


def _fit_pca(moments: Moments, components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean, the top principal axes, and their explained variance ratios."""
    mean = moments.total / moments.count
    covariance = moments.gram / moments.count - np.outer(mean, mean)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:components]
    axes = eigenvectors[:, order].T
    # Fix the sign so the largest loading of each axis is positive.
    signs = np.sign(axes[np.arange(components), np.abs(axes).argmax(axis=1)])
    axes *= signs[:, None]
    ratios = eigenvalues[order] / max(np.clip(eigenvalues, 0.0, None).sum(), np.finfo(float).tiny)
    return mean, axes, ratios


def _project(embeds: np.ndarray, mean: np.ndarray, axes: np.ndarray) -> np.ndarray:
    coords = np.empty((len(embeds), axes.shape[0]), dtype=np.float32)
    for start in range(0, len(embeds), CHUNK_SIZE):
        chunk = np.asarray(embeds[start : start + CHUNK_SIZE], dtype=np.float64)
        coords[start : start + len(chunk)] = (chunk - mean) @ axes.T
    return coords


def _marker_size(count: int) -> float:
    return float(np.clip(40000.0 / max(count, 1), 0.6, 16.0))


def _scatter(axis, coords: np.ndarray, colors: np.ndarray, seed: int) -> None:
    """Scatter in random order so no group is always drawn on top."""
    order = np.random.default_rng(seed).permutation(len(coords))
    axis.scatter(
        coords[order, 0],
        coords[order, 1],
        c=colors[order],
        s=_marker_size(len(coords)),
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )


def _style_axis(axis, ratios: np.ndarray) -> None:
    axis.set_xlabel(f"PC1 ({ratios[0]:.1%})")
    axis.set_ylabel(f"PC2 ({ratios[1]:.1%})")
    axis.tick_params(labelsize=11)


def _legend_handles(items: list[tuple[str, str]]) -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=color, label=label)
        for label, color in items
    ]


def _plot_pooled(pdf: PdfPages, modality: str, encoder: str, projections: list[Projection], ratios, seed):
    coords = np.concatenate([projection.coords for projection in projections])
    dataset_colors = _categorical_colors(len(projections))
    by_dataset = np.concatenate(
        [np.full(len(projection.coords), color) for projection, color in zip(projections, dataset_colors)]
    )
    labels = np.concatenate([projection.labels for projection in projections])
    by_anomaly = np.where(labels == 0, NORMAL_COLOR, ANOMALY_COLOR)

    figure, (left, right) = plt.subplots(1, 2, figsize=(16, 7.6), sharex=True, sharey=True)
    _scatter(left, coords, by_dataset, seed)
    _scatter(right, coords, by_anomaly, seed)
    for axis in (left, right):
        _style_axis(axis, ratios)
    left.set_title("By dataset", fontsize=14)
    right.set_title("Normal vs anomalous", fontsize=14)

    left.legend(
        handles=_legend_handles(
            [
                (f"{projection.name} [{projection.tag}] (n={len(projection.coords):,})", color)
                for projection, color in zip(projections, dataset_colors)
            ]
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=10,
    )
    right.legend(
        handles=_legend_handles(
            [
                (f"normal, label 0 (n={int(np.sum(labels == 0)):,})", NORMAL_COLOR),
                (f"anomalous, label ≠ 0 (n={int(np.sum(labels != 0)):,})", ANOMALY_COLOR),
            ]
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=10,
    )
    figure.suptitle(
        f"{MODALITY_TITLES[modality]} · {encoder} · pooled ({len(projections)} datasets, n={len(coords):,})",
        fontsize=16,
    )
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight", dpi=RASTER_DPI)
    plt.close(figure)


def _plot_dataset(pdf: PdfPages, modality: str, encoder: str, projection: Projection, ratios, seed):
    colors_by_label = _label_colors(projection.labels)
    colors = np.asarray([colors_by_label[int(label)] for label in projection.labels])

    figure, axis = plt.subplots(figsize=(8.5, 7.6))
    _scatter(axis, projection.coords, colors, seed)
    _style_axis(axis, ratios)
    items = []
    for label, color in sorted(colors_by_label.items()):
        count = int(np.sum(projection.labels == label))
        name = "0 (normal)" if label == 0 else str(label)
        items.append((f"{name} (n={count:,})", color))
    axis.legend(
        handles=_legend_handles(items),
        title="label",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=min(len(items), 4),
        fontsize=10,
        title_fontsize=11,
    )
    axis.set_title(
        f"{MODALITY_TITLES[modality]} · {encoder}\n{projection.name} [{projection.tag}] "
        f"(n={len(projection.coords):,})",
        fontsize=14,
    )
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight", dpi=RASTER_DPI)
    plt.close(figure)


def plot_modality_encoder(modality: str, encoder: str, output_root: Path, seed: int) -> Path | None:
    """Write the pooled and per-dataset PCA pages of one modality/encoder pair."""
    config = _load_modality_config(modality)
    entries = []
    for name, meta in config["datasets"].items():
        meta = meta or {}
        include_labels = meta.get("include_labels", meta.get("labels"))
        if include_labels is not None:
            include_labels = [int(label) for label in include_labels]
        dataset_dir = config["data_root"] / name
        if not dataset_dir.is_dir():
            LOGGER.warning("%s/%s: dataset directory missing, skipped", modality, name)
            continue
        entries.append((name, str(meta.get("tag", "train")), dataset_dir, include_labels))

    # Pass 1: per-dataset PCA, and moments for the pooled fit.
    pooled_moments: Moments | None = None
    projections: list[Projection] = []
    dataset_ratios: list[np.ndarray] = []
    loaded_entries = []
    for name, tag, dataset_dir, include_labels in entries:
        loaded = _load_dataset(dataset_dir, encoder, include_labels)
        if loaded is None:
            LOGGER.warning("%s/%s: no %s embeddings, skipped", modality, name, encoder)
            continue
        embeds, labels = loaded
        if len(embeds) < 3:
            LOGGER.warning("%s/%s: only %d samples, skipped", modality, name, len(embeds))
            continue
        moments = _moments(embeds)
        mean, axes, ratios = _fit_pca(moments)
        projections.append(Projection(name, tag, _project(embeds, mean, axes), labels))
        dataset_ratios.append(ratios)
        pooled_moments = moments if pooled_moments is None else pooled_moments + moments
        loaded_entries.append((dataset_dir, include_labels))
        LOGGER.info("%s/%s/%s: %d samples", modality, encoder, name, len(embeds))

    if not projections:
        LOGGER.warning("%s/%s: no datasets with embeddings, nothing written", modality, encoder)
        return None

    # Pass 2: project every dataset onto the pooled axes.
    pooled_mean, pooled_axes, pooled_ratios = _fit_pca(pooled_moments)
    pooled_projections = []
    for projection, (dataset_dir, include_labels) in zip(projections, loaded_entries):
        embeds, _ = _load_dataset(dataset_dir, encoder, include_labels)
        pooled_projections.append(
            Projection(projection.name, projection.tag, _project(embeds, pooled_mean, pooled_axes), projection.labels)
        )

    output_path = output_root / modality / f"{encoder}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _plot_pooled(pdf, modality, encoder, pooled_projections, pooled_ratios, seed)
        for projection, ratios in zip(projections, dataset_ratios):
            _plot_dataset(pdf, modality, encoder, projection, ratios, seed)
    LOGGER.info("Wrote %s (%d pages)", output_path, len(projections) + 1)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument(
        "--encoders", nargs="+", default=None, help="Default: every encoder in configs/embeddings.yaml"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the scatter draw order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_publication_style()
    encoders = args.encoders or _default_encoders()
    for modality in args.modalities:
        for encoder in encoders:
            plot_modality_encoder(modality, encoder, args.output_root, args.seed)


if __name__ == "__main__":
    main()
