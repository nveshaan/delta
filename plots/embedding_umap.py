"""Plot spherical views of UMAP projections of the pre-computed embeddings.

Two backends are available (``--backend``):

``mlx-vis`` (default, Apple Silicon GPU)
    Described below. Not a true spherical embedding.

``umap-learn``
    Fits UMAP with ``metric="cosine"`` and ``output_metric="haversine"``, which
    optimizes the 2-D output directly on S^2 (output columns are polar angle
    theta and azimuth phi). This is a genuine spherical embedding and is shown on
    the full sphere. It runs on CPU and is slower on large sets. Requires
    ``umap-learn``.

mlx-vis backend details: ``mlx-vis`` computes UMAP neighbours with Euclidean distance and (as far as its
documentation shows) does not expose ``metric`` or ``output_metric`` parameters.
Inputs are therefore L2-normalized before fitting. Euclidean distance between
normalized vectors is monotonic with cosine distance, so neighbour sets match
cosine UMAP; edge weights, which use the distance values, differ slightly.

This is NOT a UMAP optimized on the sphere. A flat 2-D UMAP layout is wrapped
onto the upper hemisphere with an azimuthal-equidistant map (distance from the
layout's median -> polar angle theta in [0, pi/2], planar angle -> phi):

    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)

The hemisphere (rather than the full sphere) avoids pinching the layout's
periphery into a single pole. Radii are scaled by their 99th percentile so a few
outliers do not compress the rest. Cluster adjacency and gaps are properties of
the 2-D UMAP layout, not of the original embedding geometry.

Outputs (``--formats``, default ``video``; use ``--formats pdf video html`` for all):

* ``pdf``: one multi-page PDF at ``assets/embeddings/<modality>/<encoder>_umap.pdf``.
  The first page contains pooled views colored by dataset and by normal versus
  anomalous label; the remaining pages show one dataset at a time.
* ``video``: rotating-camera MP4s in ``assets/embeddings/<modality>/<encoder>_umap/``
  (``pooled.mp4`` plus one ``<dataset>.mp4`` each). The camera makes one full
  turn in azimuth while the elevation sweeps up and down once, so both poles and
  the whole sphere are seen. Falls back to GIF if ``ffmpeg`` is not on PATH.
  Rendering is CPU-bound matplotlib: expect roughly seconds per second of video
  per ~10k points; lower ``--fps``/``--video-seconds``/``--video-dpi`` to speed up.
* ``html``: self-contained interactive Plotly HTML files in the same directory as
  the videos. Open them in a browser to rotate, zoom, hover, and toggle traces.

Usage::

    uv run python plots/embedding_umap.py
    uv run python plots/embedding_umap.py --modalities oct mri --encoders CLIP MedSigLIP
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
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
RASTER_DPI = 200
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


def _categorical_colors(count: int) -> list[str]:
    colors = list(DISTINCT_COLORS)
    if count <= len(colors):
        return colors[:count]
    cmap = plt.get_cmap("tab20")
    return colors + [matplotlib.colors.to_hex(cmap(index % 20)) for index in range(count - len(colors))]


def _label_colors(labels: np.ndarray) -> dict[int, str]:
    values = sorted(int(label) for label in np.unique(labels))
    others = [color for color in _categorical_colors(len(values) + 1) if color != NORMAL_COLOR]
    mapping = {label: others[index] for index, label in enumerate(label for label in values if label != 0)}
    if 0 in values:
        mapping[0] = NORMAL_COLOR
    return mapping


def _load_modality_config(modality: str) -> dict:
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
    embeds = np.asarray(np.load(embeds_path, mmap_mode="r"), dtype=np.float32)
    labels = np.load(dataset_dir / "labels.npy")
    if len(embeds) != len(labels):
        raise ValueError(f"{dataset_dir}: {len(embeds)} embeds vs {len(labels)} labels")
    if include_labels is not None:
        keep = np.flatnonzero(np.isin(labels, include_labels))
        embeds, labels = embeds[keep], labels[keep]
    return embeds, labels


def _normalize_for_cosine(embeds: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("Cannot compute cosine geometry for zero-norm embedding")
    return embeds / norms


def _spherical_coordinates(umap_coords: np.ndarray) -> np.ndarray:
    """Map planar UMAP coordinates to (theta, phi) on the upper hemisphere."""
    centered = umap_coords - np.median(umap_coords, axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1)
    scale = max(float(np.percentile(radius, 99)), float(np.finfo(np.float32).eps))
    theta = 0.5 * np.pi * np.minimum(radius / scale, 1.0)
    phi = np.arctan2(centered[:, 1], centered[:, 0])
    return np.column_stack((theta, phi)).astype(np.float32)


def _to_cartesian(spherical: np.ndarray) -> np.ndarray:
    theta, phi = spherical.T
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    ).astype(np.float32)


BACKENDS = ("mlx-vis", "umap-learn")


def _fit_projection(
    embeds: np.ndarray, seed: int, n_neighbors: int, n_epochs: int, backend: str = "mlx-vis"
) -> np.ndarray:
    normalized = _normalize_for_cosine(embeds)
    neighbors = min(n_neighbors, len(normalized) - 1)
    if backend == "umap-learn":
        try:
            import umap
        except ImportError as error:
            raise ImportError("--backend umap-learn requires `pip install umap-learn`") from error
        spherical = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            n_epochs=n_epochs,
            metric="cosine",
            output_metric="haversine",
            random_state=seed,
        ).fit_transform(normalized)
        # umap-learn's haversine output: column 0 = polar angle, column 1 = azimuth.
        return _to_cartesian(np.asarray(spherical, dtype=np.float32))
    if backend != "mlx-vis":
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    from mlx_vis import UMAP

    umap_coords = UMAP(
        n_components=2,
        n_neighbors=neighbors,
        n_epochs=n_epochs,
        random_state=seed,
    ).fit_transform(normalized)
    return _to_cartesian(_spherical_coordinates(umap_coords))


def _view_label(backend: str) -> str:
    if backend == "umap-learn":
        return "spherical UMAP (cosine, haversine output)"
    return "UMAP (L2-normalized), hemisphere view"


def _marker_size(count: int) -> float:
    return float(np.clip(40000.0 / max(count, 1), 0.6, 16.0))


def _legend_handles(items: list[tuple[str, str]]) -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=color, label=label)
        for label, color in items
    ]


def _scatter(axis, coords: np.ndarray, colors: np.ndarray, seed: int, full_sphere: bool = False) -> None:
    sphere_radius = 1.0
    point_radius = 1.01
    polar_max = np.pi if full_sphere else 0.5 * np.pi
    u, v = np.meshgrid(
        np.linspace(0, 2 * np.pi, 257),
        np.linspace(0, polar_max, 129 if full_sphere else 65),
    )
    sphere_x = sphere_radius * np.cos(u) * np.sin(v)
    sphere_y = sphere_radius * np.sin(u) * np.sin(v)
    sphere_z = sphere_radius * np.cos(v)
    light = np.array([-0.45, 0.55, 0.7])
    light /= np.linalg.norm(light)
    center_x = (sphere_x[:-1, :-1] + sphere_x[1:, 1:]) / 2
    center_y = (sphere_y[:-1, :-1] + sphere_y[1:, 1:]) / 2
    center_z = (sphere_z[:-1, :-1] + sphere_z[1:, 1:]) / 2
    brightness = 0.45 + 0.55 * np.clip(
        center_x * light[0] + center_y * light[1] + center_z * light[2], 0, 1
    )
    facecolors = np.empty((*brightness.shape, 4))
    facecolors[..., :3] = brightness[..., None]
    facecolors[..., 3] = 1.0
    axis.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        facecolors=facecolors,
        rcount=sphere_x.shape[0],
        ccount=sphere_x.shape[1],
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=1.0,
        zorder=0,
    )

    order = np.random.default_rng(seed).permutation(len(coords))
    rendered_coords = coords[order] * point_radius
    points = axis.scatter(
        rendered_coords[:, 0],
        rendered_coords[:, 1],
        rendered_coords[:, 2],
        c=colors[order],
        s=_marker_size(len(coords)),
        alpha=0.6,
        linewidths=0,
        depthshade=False,
        zorder=10,
        rasterized=True,
    )
    axis._umap_coords = coords[order]
    axis._umap_rendered_coords = rendered_coords
    axis._umap_colors = colors[order]
    axis._umap_points = points
    axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1 if full_sphere else 0, 1))
    axis.set_box_aspect((1, 1, 1) if full_sphere else (2, 2, 1))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")


def _set_visible_points(axis, elev: float, azim: float) -> None:
    """Show only points on the camera-facing hemisphere.

    Matplotlib's 3D collections do not depth-test scatter points against
    surfaces, so far-side points must be culled explicitly for video frames.
    """
    elevation = np.deg2rad(elev)
    azimuth = np.deg2rad(azim)
    camera_direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    coords = axis._umap_coords
    visible = coords @ camera_direction >= 0
    visible_coords = axis._umap_rendered_coords[visible]
    axis._umap_points._offsets3d = (
        visible_coords[:, 0],
        visible_coords[:, 1],
        visible_coords[:, 2],
    )
    axis._umap_points.set_facecolors(axis._umap_colors[visible])


def _pooled_figure(modality: str, encoder: str, projections, seed: int, backend: str = "mlx-vis"):
    coords = np.concatenate([projection[2] for projection in projections])
    dataset_colors = _categorical_colors(len(projections))
    by_dataset = np.concatenate(
        [np.full(len(projection[2]), color) for projection, color in zip(projections, dataset_colors)]
    )
    labels = np.concatenate([projection[1] for projection in projections])
    by_anomaly = np.where(labels == 0, NORMAL_COLOR, ANOMALY_COLOR)

    figure = plt.figure(figsize=(16, 7.6))
    left = figure.add_subplot(1, 2, 1, projection="3d")
    right = figure.add_subplot(1, 2, 2, projection="3d")
    _scatter(left, coords, by_dataset, seed, backend == "umap-learn")
    _scatter(right, coords, by_anomaly, seed, backend == "umap-learn")
    left.set_title("By dataset")
    right.set_title("Normal vs anomalous")
    left.legend(
        handles=_legend_handles(
            [(f"{projection[0]} [{projection[3]}] (n={len(projection[2]):,})", color)
             for projection, color in zip(projections, dataset_colors)]
        ),
        loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=10,
    )
    right.legend(
        handles=_legend_handles(
            [
                (f"normal, label 0 (n={int(np.sum(labels == 0)):,})", NORMAL_COLOR),
                (f"anomalous, label != 0 (n={int(np.sum(labels != 0)):,})", ANOMALY_COLOR),
            ]
        ),
        loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=10,
    )
    figure.suptitle(
        f"{MODALITY_TITLES[modality]} · {encoder} · {_view_label(backend)} · "
        f"pooled ({len(projections)} datasets, n={len(coords):,})"
    )
    figure.tight_layout()
    return figure, [left, right]


def _dataset_figure(modality: str, encoder: str, projection, seed: int, backend: str = "mlx-vis"):
    name, labels, coords, tag = projection
    colors_by_label = _label_colors(labels)
    colors = np.asarray([colors_by_label[int(label)] for label in labels])
    figure = plt.figure(figsize=(8.5, 7.6))
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    _scatter(axis, coords, colors, seed, backend == "umap-learn")
    items = [
        (f"{'0 (normal)' if label == 0 else label} (n={int(np.sum(labels == label)):,})", color)
        for label, color in sorted(colors_by_label.items())
    ]
    axis.legend(
        handles=_legend_handles(items), title="label", loc="upper center",
        bbox_to_anchor=(0.5, -0.08), ncol=min(len(items), 4), fontsize=10, title_fontsize=11,
    )
    axis.set_title(
        f"{MODALITY_TITLES[modality]} · {encoder} · {_view_label(backend)}\n"
        f"{name} [{tag}] (n={len(coords):,})"
    )
    figure.tight_layout()
    return figure, [axis]


def _plot_pooled(pdf: PdfPages, modality: str, encoder: str, projections, seed: int, backend: str = "mlx-vis") -> None:
    figure, _ = _pooled_figure(modality, encoder, projections, seed, backend)
    pdf.savefig(figure, bbox_inches="tight", dpi=RASTER_DPI)
    plt.close(figure)


def _plot_dataset(pdf: PdfPages, modality: str, encoder: str, projection, seed: int, backend: str = "mlx-vis") -> None:
    figure, _ = _dataset_figure(modality, encoder, projection, seed, backend)
    pdf.savefig(figure, bbox_inches="tight", dpi=RASTER_DPI)
    plt.close(figure)


def _camera_angles(frame: int, n_frames: int, full_sphere: bool) -> tuple[float, float]:
    """Camera (elevation, azimuth) in degrees for a seamlessly looping flight.

    Azimuth makes one full turn. Elevation sweeps through one sine cycle, so the
    camera looks down on the north pole, along the equator, up at the south pole
    and back. On the full sphere that is +-60 degrees; the hemisphere (mlx-vis
    backend) has nothing below the equator, so its camera stays above it.
    """
    phase = 2.0 * np.pi * frame / n_frames
    azim = -60.0 + 360.0 * frame / n_frames
    if full_sphere:
        elev = 60.0 * np.sin(phase)
    else:
        elev = 45.0 + 35.0 * np.sin(phase)
    return float(elev), float(azim)


def _save_video(figure, axes, path: Path, full_sphere: bool, fps: int, seconds: float, dpi: int) -> Path:
    """Animate ``axes`` with a rotating camera and write ``path`` (.mp4, or .gif without ffmpeg)."""
    n_frames = max(int(round(fps * seconds)), 1)
    for axis in axes:
        axis.set_axis_off()   # hide panes/ticks so only the sphere grid and points remain

    def update(frame: int):
        elev, azim = _camera_angles(frame, n_frames, full_sphere)
        for axis in axes:
            axis.view_init(elev=elev, azim=azim)
            _set_visible_points(axis, elev, azim)
        return []

    animation = FuncAnimation(figure, update, frames=n_frames, blit=False)
    if shutil.which("ffmpeg"):
        output = path.with_suffix(".mp4")
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",   # yuv420p needs even dimensions
            ],
        )
    else:
        LOGGER.warning("ffmpeg not found on PATH; writing GIF instead of MP4")
        output = path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)

    def progress(frame: int, total: int) -> None:
        if frame % max(total // 4, 1) == 0:
            LOGGER.info("  %s: frame %d/%d", output.name, frame, total)

    animation.save(output, writer=writer, dpi=dpi, progress_callback=progress)
    return output


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "dataset"


def _save_coordinate_cache(
    output_dir: Path, pooled_projections, dataset_projections
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "pooled.npz",
        coords=np.concatenate([projection[2] for projection in pooled_projections]),
        labels=np.concatenate([projection[1] for projection in pooled_projections]),
        dataset=np.concatenate(
            [np.full(len(projection[2]), projection[0]) for projection in pooled_projections]
        ),
        tag=np.concatenate(
            [np.full(len(projection[2]), projection[3]) for projection in pooled_projections]
        ),
    )
    for name, labels, coords, tag in dataset_projections:
        np.savez_compressed(
            output_dir / f"{_safe_name(name)}.npz",
            coords=coords,
            labels=labels,
            dataset=np.full(len(coords), name),
            tag=np.full(len(coords), tag),
        )


def _plotly_sphere(full_sphere: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polar_max = np.pi if full_sphere else 0.5 * np.pi
    u, v = np.meshgrid(
        np.linspace(0, 2 * np.pi, 129),
        np.linspace(0, polar_max, 65 if full_sphere else 33),
    )
    return np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)


def _plotly_surface(full_sphere: bool):
    import plotly.graph_objects as go

    sphere_x, sphere_y, sphere_z = _plotly_sphere(full_sphere)
    return go.Surface(
        x=sphere_x,
        y=sphere_y,
        z=sphere_z,
        surfacecolor=np.zeros_like(sphere_x),
        colorscale=[[0.0, "#F5F5F5"], [1.0, "#F5F5F5"]],
        showscale=False,
        opacity=1.0,
        hoverinfo="skip",
        lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.15, "roughness": 0.9},
        lightposition={"x": -100, "y": 100, "z": 150},
        name="sphere",
        showlegend=False,
    )


def _plotly_points(
    coords: np.ndarray,
    colors: np.ndarray,
    hover_text: list[str],
    name: str,
    legendgroup: str | None = None,
    legendgrouptitle: str | None = None,
):
    import plotly.graph_objects as go

    return go.Scatter3d(
        x=coords[:, 0] * 1.01,
        y=coords[:, 1] * 1.01,
        z=coords[:, 2] * 1.01,
        mode="markers",
        marker={"size": 3, "color": colors, "opacity": 0.75},
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        name=name,
        legendgroup=legendgroup,
        legendgrouptitle={"text": legendgrouptitle} if legendgrouptitle else None,
    )


def _sync_scene_cameras_script() -> str:
    return """
const graph = document.getElementById('{plot_id}');
let syncingCameras = false;
graph.on('plotly_relayout', (event) => {
    if (syncingCameras) return;
    const camera = event['scene.camera'] || event['scene2.camera'];
    if (!camera) return;
    syncingCameras = true;
    const target = event['scene.camera']
        ? {'scene2.camera': camera}
        : {'scene.camera': camera};
    Plotly.relayout(graph, target).finally(() => { syncingCameras = false; });
});
"""


def _plotly_layout(title: str, full_sphere: bool) -> dict:
    axis_range = [-1.1, 1.1]
    axis = {
        "range": axis_range,
        "showbackground": False,
        "showgrid": False,
        "showline": False,
        "showticklabels": False,
        "ticks": "",
        "title": "",
        "zeroline": False,
    }
    return {
        "title": title,
        "scene": {
            "xaxis": axis,
            "yaxis": axis,
            "zaxis": {**axis, "range": [-1.1, 1.1] if full_sphere else [0, 1.1]},
            "aspectmode": "cube",
        },
        "margin": {"l": 0, "r": 0, "t": 48, "b": 0},
        "template": "plotly_white",
        "showlegend": True,
    }


def _save_html(
    modality: str,
    encoder: str,
    pooled_projections,
    dataset_projections,
    output_dir: Path,
    backend: str,
) -> list[Path]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError("--formats html requires `plotly`") from error

    full_sphere = backend == "umap-learn"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    pooled_coords = np.concatenate([projection[2] for projection in pooled_projections])
    pooled_labels = np.concatenate([projection[1] for projection in pooled_projections])
    dataset_colors = _categorical_colors(len(pooled_projections))
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("By dataset", "Normal vs anomalous"),
    )
    for column in (1, 2):
        figure.add_trace(_plotly_surface(full_sphere), row=1, col=column)
    for index, (projection, color) in enumerate(zip(pooled_projections, dataset_colors)):
        name, labels, coords, tag = projection
        count = len(coords)
        hover = [f"dataset: {name} [{tag}]<br>label: {label}" for label in labels]
        figure.add_trace(
            _plotly_points(
                coords,
                np.full(count, color),
                hover,
                f"{name} [{tag}]",
                legendgroup="dataset",
                legendgrouptitle="By dataset" if index == 0 else None,
            ),
            row=1,
            col=1,
        )
    for index, (label, color, label_name) in enumerate((
        (0, NORMAL_COLOR, "0 (normal)"),
        (None, ANOMALY_COLOR, "anomalous (label != 0)"),
    )):
        mask = pooled_labels == 0 if label == 0 else pooled_labels != 0
        hover = [f"label: {value}" for value in pooled_labels[mask]]
        figure.add_trace(
            _plotly_points(
                pooled_coords[mask],
                np.full(int(mask.sum()), color),
                hover,
                label_name,
                legendgroup="subtype",
                legendgrouptitle="Subtypes" if index == 0 else None,
            ),
            row=1,
            col=2,
        )
    figure.update_layout(
        title=f"{MODALITY_TITLES[modality]} · {encoder} · {_view_label(backend)} · pooled",
        scene=_plotly_layout("", full_sphere)["scene"],
        scene2=_plotly_layout("", full_sphere)["scene"],
        margin={"l": 0, "r": 0, "t": 72, "b": 0},
        template="plotly_white",
        showlegend=True,
    )
    pooled_path = output_dir / "pooled.html"
    figure.write_html(
        pooled_path,
        include_plotlyjs=True,
        full_html=True,
        post_script=_sync_scene_cameras_script(),
    )
    written.append(pooled_path)

    for name, labels, coords, tag in dataset_projections:
        colors_by_label = _label_colors(labels)
        traces = [_plotly_surface(full_sphere)]
        for index, (label, color) in enumerate(sorted(colors_by_label.items())):
            mask = labels == label
            subtype = "0 (normal)" if label == 0 else f"label {label}"
            hover = [f"label: {value}" for value in labels[mask]]
            traces.append(
                _plotly_points(
                    coords[mask],
                    np.full(int(mask.sum()), color),
                    hover,
                    subtype,
                    legendgroup="subtype",
                    legendgrouptitle="Subtypes" if index == 0 else None,
                )
            )
        figure = go.Figure(traces)
        figure.update_layout(
            **_plotly_layout(
                f"{MODALITY_TITLES[modality]} · {encoder} · {_view_label(backend)}<br>{name} [{tag}]",
                full_sphere,
            )
        )
        dataset_path = output_dir / f"{_safe_name(name)}.html"
        figure.write_html(dataset_path, include_plotlyjs=True, full_html=True)
        written.append(dataset_path)
    return written


def plot_modality_encoder(
    modality: str,
    encoder: str,
    output_root: Path,
    seed: int,
    n_neighbors: int,
    n_epochs: int,
    backend: str = "mlx-vis",
    formats: tuple[str, ...] = ("video",),
    fps: int = 24,
    video_seconds: float = 10.0,
    video_dpi: int = 120,
) -> Path | None:
    config = _load_modality_config(modality)
    projections = []
    for name, meta in config["datasets"].items():
        meta = meta or {}
        include_labels = meta.get("include_labels", meta.get("labels"))
        include_labels = None if include_labels is None else [int(label) for label in include_labels]
        dataset_dir = config["data_root"] / name
        if not dataset_dir.is_dir():
            LOGGER.warning("%s/%s: dataset directory missing, skipped", modality, name)
            continue
        loaded = _load_dataset(dataset_dir, encoder, include_labels)
        if loaded is None:
            LOGGER.warning("%s/%s: no %s embeddings, skipped", modality, name, encoder)
            continue
        embeds, labels = loaded
        if len(embeds) < 3:
            LOGGER.warning("%s/%s: only %d samples, skipped", modality, name, len(embeds))
            continue
        tag = str(meta.get("tag", "train"))
        LOGGER.info("%s/%s/%s: fitting %d samples", modality, encoder, name, len(embeds))
        projections.append((name, labels, embeds, tag))

    if not projections:
        LOGGER.warning("%s/%s: no datasets with embeddings, nothing written", modality, encoder)
        return None

    pooled_embeds = np.concatenate([projection[2] for projection in projections])
    pooled_coords = _fit_projection(pooled_embeds, seed, n_neighbors, n_epochs, backend)
    pooled_projections = []
    offset = 0
    for name, labels, embeds, tag in projections:
        end = offset + len(embeds)
        pooled_projections.append((name, labels, pooled_coords[offset:end], tag))
        offset = end

    dataset_projections = []
    for name, labels, embeds, tag in projections:
        dataset_projections.append((name, labels, _fit_projection(embeds, seed, n_neighbors, n_epochs, backend), tag))

    stem = f"{encoder}_umap" if backend == "mlx-vis" else f"{encoder}_umap_{backend.replace('-', '_')}"
    output_path = output_root / modality / f"{stem}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "pdf" in formats:
        with PdfPages(output_path) as pdf:
            _plot_pooled(pdf, modality, encoder, pooled_projections, seed, backend)
            for projection in dataset_projections:
                _plot_dataset(pdf, modality, encoder, projection, seed, backend)
        LOGGER.info("Wrote %s (%d pages)", output_path, len(projections) + 1)

    if "video" in formats:
        video_dir = output_path.with_suffix("")
        video_dir.mkdir(parents=True, exist_ok=True)
        full_sphere = backend == "umap-learn"
        figure, axes = _pooled_figure(modality, encoder, pooled_projections, seed, backend)
        written = _save_video(figure, axes, video_dir / "pooled", full_sphere, fps, video_seconds, video_dpi)
        plt.close(figure)
        LOGGER.info("Wrote %s", written)
        for projection in dataset_projections:
            figure, axes = _dataset_figure(modality, encoder, projection, seed, backend)
            written = _save_video(
                figure, axes, video_dir / _safe_name(projection[0]), full_sphere, fps, video_seconds, video_dpi
            )
            plt.close(figure)
            LOGGER.info("Wrote %s", written)
    if "html" in formats:
        html_dir = output_path.with_suffix("")
        _save_coordinate_cache(html_dir, pooled_projections, dataset_projections)
        written = _save_html(
            modality, encoder, pooled_projections, dataset_projections, html_dir, backend
        )
        for path in written:
            LOGGER.info("Wrote %s", path)
    else:
        _save_coordinate_cache(output_path.with_suffix(""), pooled_projections, dataset_projections)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--encoders", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--backend", choices=BACKENDS, default="mlx-vis")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "video", "html"), default=["video"])
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video-seconds", type=float, default=10.0, help="duration of one full camera loop")
    parser.add_argument("--video-dpi", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_publication_style()
    encoders = args.encoders or _default_encoders()
    for modality in args.modalities:
        for encoder in encoders:
            plot_modality_encoder(
                modality, encoder, args.output_root, args.seed, args.n_neighbors, args.n_epochs, args.backend,
                tuple(args.formats), args.fps, args.video_seconds, args.video_dpi,
            )


if __name__ == "__main__":
    main()