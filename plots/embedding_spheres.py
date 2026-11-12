"""Create an interactive two-sphere structural-metric visualization.

The first sphere shows cached spherical UMAP coordinates colored by dataset
(pooled mode) or subtype. The second sphere uses the same coordinates but colors
points by a selectable per-sample metric computed from the original embeddings.

UMAP coordinate caches are produced by ``plots/embedding_umap.py``. Generate
them first, for example::

    uv run python plots/embedding_umap.py --modalities chest --encoders CLIP \
        --backend umap-learn --formats html

Then create an interactive file::

    uv run python scripts/embedding_spheres.py --modality chest \
        --encoder CLIP --dataset pooled --backend umap-learn --metric kde_density
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "plots"))

from embedding_umap import (  # noqa: E402
    ANOMALY_COLOR,
    MODALITIES,
    MODALITY_TITLES,
    NORMAL_COLOR,
    _categorical_colors,
    _label_colors,
    _safe_name,
    _view_label,
)

LOGGER = logging.getLogger("embedding_spheres")
DEFAULT_METRICS = (
    "spread", "knn_within", "knn_pooled", "move", "twonn_id", "local_pca_id",
    "pr_dim", "pca95_dim", "n_modes", "silhouette", "weight_entropy", "weight_cv",
    "deg_mean", "intra_frac", "n_components", "giant_frac", "kde_density",
)
METRIC_LABELS = {metric: metric.replace("_", " ") for metric in DEFAULT_METRICS}


def _config(modality: str) -> dict:
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "modality" / f"{modality}.yaml")
    root = Path(config.get("data_root", f"data/{modality}"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return {"root": root, "datasets": OmegaConf.to_container(config.datasets, resolve=False)}


def _default_encoders() -> list[str]:
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "embeddings.yaml")
    return list(config.encoders.keys())


def _source_embeddings(
    modality: str, encoder: str, dataset: str, indices: np.ndarray | None = None
):
    from datasets.embeds import _resolve_embeds_file

    config = _config(modality)
    meta = config["datasets"].get(dataset) or {}
    path = config["root"] / dataset
    embeds = np.asarray(np.load(_resolve_embeds_file(path, encoder), mmap_mode="r"), dtype=np.float32)
    labels = np.load(path / "labels.npy")
    include = meta.get("include_labels", meta.get("labels"))
    if include is not None:
        keep = np.flatnonzero(np.isin(labels, [int(value) for value in include]))
        embeds, labels = embeds[keep], labels[keep]
    if indices is not None:
        embeds, labels = embeds[indices], labels[indices]
    return embeds, labels, str(meta.get("tag", "train"))


def _cache_dir(modality: str, encoder: str, backend: str) -> Path:
    stem = f"{encoder}_umap" if backend == "mlx-vis" else f"{encoder}_umap_{backend.replace('-', '_')}"
    return PROJECT_ROOT / "assets" / "embeddings" / modality / stem


def _sample_indices(length: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.sort(rng.choice(length, size=max_points, replace=False))


def _load_selection(
    modality: str,
    encoder: str,
    dataset: str,
    backend: str,
    max_points: int = 5000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    directory = _cache_dir(modality, encoder, backend)
    if dataset == "pooled":
        cache = directory / "pooled.npz"
        if not cache.is_file():
            raise FileNotFoundError(f"Missing UMAP cache {cache}; run plots/embedding_umap.py first")
        pooled = np.load(cache, allow_pickle=False)
        config = _config(modality)
        loaded = []
        for name, meta in config["datasets"].items():
            try:
                X, y, tag = _source_embeddings(modality, encoder, name)
            except FileNotFoundError:
                continue
            loaded.append((name, X, y, tag))
        if not loaded:
            raise FileNotFoundError(f"No source embeddings found for {modality}/{encoder}")
        quotas = [max(1, max_points * len(y) // sum(len(item[2]) for item in loaded)) for _, _, y, _ in loaded]
        while sum(quotas) > max_points:
            largest = int(np.argmax(quotas))
            if quotas[largest] <= 1:
                break
            quotas[largest] -= 1
        embeds, labels, datasets, tags, coordinates = [], [], [], [], []
        offset = 0
        for (name, X, y, tag), quota in zip(loaded, quotas):
            local = _sample_indices(len(y), quota, rng)
            embeds.append(np.asarray(X[local], dtype=np.float32))
            labels.append(y[local])
            coordinates.append(pooled["coords"][offset + local])
            datasets.extend([name] * len(local))
            tags.extend([tag] * len(local))
            offset += len(y)
        return (
            np.concatenate(embeds),
            np.concatenate(labels),
            np.concatenate(coordinates),
            np.asarray(datasets),
            np.asarray(tags),
        )

    cache = directory / f"{_safe_name(dataset)}.npz"
    if not cache.is_file():
        raise FileNotFoundError(f"Missing UMAP cache {cache}; run plots/embedding_umap.py first")
    X, y, tag = _source_embeddings(modality, encoder, dataset)
    local = _sample_indices(len(y), max_points, rng)
    cached = np.load(cache, allow_pickle=False)
    return (
        np.asarray(X[local], dtype=np.float32),
        y[local],
        cached["coords"][local],
        np.full(len(local), dataset),
        np.full(len(local), tag),
    )


def _point_metrics(X: np.ndarray, labels: np.ndarray, seed: int, device: str) -> dict[str, np.ndarray]:
    """Compute metric fields in the original embedding space, never UMAP space."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    from methods.msde import DEFAULT_DEVICE, MeanShiftDensityEnhancement

    Xt = torch.from_numpy(X).float()
    n = len(Xt)
    D = torch.cdist(Xt, Xt)
    D.fill_diagonal_(float("inf"))
    k = min(10, n - 1)
    nearest = D.topk(min(50, n - 1), dim=1, largest=False)
    scores: dict[str, np.ndarray] = {
        "spread": np.zeros(n, dtype=np.float32),
        "knn_pooled": nearest.values[:, k - 1].numpy(),
        "twonn_id": torch.log(
            nearest.values[:, 1].clamp_min(1e-12) / nearest.values[:, 0].clamp_min(1e-12)
        ).numpy(),
    }
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < 2:
            continue
        Xc = Xt[members]
        Dc = torch.cdist(Xc, Xc)
        Dc.fill_diagonal_(float("inf"))
        kc = min(10, len(members) - 1)
        scores["spread"][members] = (Xc - Xc.mean(0)).norm(dim=1).numpy()
        scores.setdefault("knn_within", np.zeros(n, dtype=np.float32))[members] = (
            Dc.topk(kc, dim=1, largest=False).values[:, -1].numpy()
        )

    h = nearest.values[:, k - 1].median().clamp_min(1e-6)
    scores["kde_density"] = torch.logsumexp(-(D**2) / (2 * h**2), dim=1).numpy()
    scores["deg_mean"] = torch.zeros(n).index_add_(
        0, nearest.indices[:, :k].reshape(-1), torch.ones(n * k)
    ).numpy()
    scores["intra_frac"] = np.zeros(n, dtype=np.float32)
    neighbor_labels = labels[nearest.indices[:, :k].numpy()]
    scores["intra_frac"] = np.mean(neighbor_labels == labels[:, None], axis=1).astype(np.float32)
    neighbors = nearest.indices[:, :k]
    scores["move"] = np.zeros(n, dtype=np.float32)
    msde_device = DEFAULT_DEVICE if device == "auto" else device
    if n > 15:
        msde = MeanShiftDensityEnhancement(
            k=min(50, n - 1),
            nbd_sample_count_threshold=70,
            learning_rate=0.33,
            max_iters_shift=3,
            shift_threshold=0.01,
            device=msde_device,
            use_chunking=True,
            weight_chunk_size=2048,
            enable_gradients=False,
        )
        with torch.no_grad():
            scores["move"] = msde(Xt.to(msde_device))[1].cpu().numpy()
    else:
        LOGGER.warning("Skipping MSDE movement for a dataset with only %d sampled points", n)

    # Group statistics from the original space are broadcast to their subtype.
    for name in ("pr_dim", "pca95_dim", "n_modes", "silhouette", "weight_entropy", "weight_cv",
                 "local_pca_id", "n_components", "giant_frac"):
        scores[name] = np.zeros(n, dtype=np.float32)
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        Xc = Xt[members]
        centered = Xc - Xc.mean(0)
        singular = torch.linalg.svdvals(centered).double() ** 2
        pr = singular.sum() ** 2 / (singular**2).sum().clamp_min(1e-12)
        pca95 = (torch.cumsum(singular, 0) / singular.sum().clamp_min(1e-12) < 0.95).sum() + 1
        within = scores["knn_within"][members]
        weights = 1.0 / np.maximum(within, 1e-12)
        weights /= weights.sum()
        modes, silhouette = 1, 0.0
        if len(members) >= 20:
            max_k = min(8, len(members) - 1)
            best = (-1.0, 1)
            for cluster_count in range(2, max_k + 1):
                cluster_labels = KMeans(cluster_count, n_init=3, random_state=seed).fit_predict(Xc.numpy())
                value = float(np.mean(silhouette_samples(Xc.numpy(), cluster_labels)))
                if value > best[0]:
                    best = (value, cluster_count)
            silhouette, modes = best
        local_id = np.zeros(len(members), dtype=np.float32)
        local_k = min(10, len(members) - 1)
        if local_k >= 5:
            member_distances = torch.cdist(Xc, Xc)
            member_distances.fill_diagonal_(float("inf"))
            neighbor_indices = member_distances.topk(local_k, dim=1, largest=False).indices
            for row, neighbors in enumerate(neighbor_indices):
                patch = Xc[neighbors] - Xc[neighbors].mean(0)
                spectrum = torch.linalg.svdvals(patch).double() ** 2
                local_id[row] = float(spectrum.sum() ** 2 / (spectrum**2).sum().clamp_min(1e-12))
        adjacency = np.zeros((len(members), len(members)), dtype=bool)
        local_neighbors = np.argsort(torch.cdist(Xc, Xc).numpy(), axis=1)[:, 1 : local_k + 1]
        adjacency[np.arange(len(members))[:, None], local_neighbors] = True
        n_components, components = connected_components(coo_matrix(adjacency), directed=False)
        giant_fraction = np.bincount(components).max() / len(members)
        values = {
            "pr_dim": float(pr),
            "pca95_dim": float(pca95),
            "n_modes": float(modes),
            "silhouette": float(silhouette),
            "weight_entropy": float(-(weights * np.log(np.maximum(weights, 1e-12))).sum() / np.log(len(weights))),
            "weight_cv": float(weights.std() / max(weights.mean(), 1e-12)),
            "local_pca_id": local_id,
            "n_components": float(n_components),
            "giant_frac": float(giant_fraction),
        }
        for metric, value in values.items():
            scores[metric][members] = value if np.ndim(value) else float(value)
    return {name: np.nan_to_num(value).astype(np.float32) for name, value in scores.items()}


def _surface(full_sphere: bool):
    import plotly.graph_objects as go

    polar = np.pi if full_sphere else 0.5 * np.pi
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 129), np.linspace(0, polar, 65))
    return go.Surface(
        x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v),
        surfacecolor=np.zeros_like(u), colorscale=[[0, "#F5F5F5"], [1, "#F5F5F5"]],
        showscale=False, hoverinfo="skip", opacity=1.0, showlegend=False,
    )


def _scene(full_sphere: bool) -> dict:
    axis = dict(range=[-1.1, 1.1], showbackground=False, showgrid=False,
                showline=False, showticklabels=False, ticks="", title="", zeroline=False)
    return dict(
        xaxis=axis, yaxis=axis,
        zaxis={**axis, "range": [-1.1, 1.1] if full_sphere else [0, 1.1]},
        aspectmode="cube",
    )


def _sync_script() -> str:
    return """
const graph = document.getElementById('{plot_id}');
let syncing = false;
graph.on('plotly_relayout', (event) => {
  if (syncing) return;
  const camera = event['scene.camera'] || event['scene2.camera'];
  if (!camera) return;
  syncing = true;
  const target = event['scene.camera'] ? {'scene2.camera': camera} : {'scene.camera': camera};
  Plotly.relayout(graph, target).finally(() => { syncing = false; });
});
"""


def create_html(
    modality: str, encoder: str, dataset: str, backend: str, metric: str, output: Path, device: str
) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    X, labels, coords, datasets, tags = _load_selection(modality, encoder, dataset, backend)
    metrics = _point_metrics(X, labels, 42, device)
    if metric not in metrics:
        raise ValueError(f"Metric {metric!r} is unavailable; choose from {sorted(metrics)}")
    full_sphere = backend == "umap-learn"
    figure = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Groups", "Metric"),
    )
    figure.add_trace(_surface(full_sphere), row=1, col=1)
    figure.add_trace(_surface(full_sphere), row=1, col=2)
    if dataset == "pooled":
        colors = _categorical_colors(len(np.unique(datasets)))
        mapping = {name: colors[i] for i, name in enumerate(np.unique(datasets))}
        color_values = np.asarray([mapping[name] for name in datasets])
        legend_names = list(np.unique(datasets))
        hover = [f"dataset: {d} [{t}]<br>subtype: {y}" for d, t, y in zip(datasets, tags, labels)]
    else:
        label_colors = _label_colors(labels)
        color_values = np.asarray([label_colors[int(label)] for label in labels])
        legend_names = [f"0 (normal)" if label == 0 else f"label {label}" for label in sorted(label_colors)]
        hover = [f"subtype: {y}" for y in labels]
    first = go.Scatter3d(
        x=coords[:, 0] * 1.01, y=coords[:, 1] * 1.01, z=coords[:, 2] * 1.01,
        mode="markers", marker={"size": 3, "color": color_values, "opacity": 0.8},
        text=hover, hovertemplate="%{text}<extra></extra>", name="groups",
    )
    metric_values = metrics[metric]
    second = go.Scatter3d(
        x=coords[:, 0] * 1.01, y=coords[:, 1] * 1.01, z=coords[:, 2] * 1.01,
        mode="markers",
        marker={"size": 3, "color": metric_values, "colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": METRIC_LABELS[metric]}, "opacity": 0.8},
        text=hover, hovertemplate="%{text}<br>metric: %{marker.color:.4g}<extra></extra>",
        name="metric",
    )
    figure.add_trace(first, row=1, col=1)
    figure.add_trace(second, row=1, col=2)
    for name in legend_names:
        legend_color = mapping[name] if dataset == "pooled" else label_colors[
            0 if name.startswith("0 ") else int(name.removeprefix("label "))
        ]
        figure.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker={"size": 6, "color": legend_color},
                name=name, showlegend=True, hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    figure.update_layout(
        title=f"{MODALITY_TITLES[modality]} · {encoder} · {dataset} · {_view_label(backend)}",
        scene=_scene(full_sphere), scene2=_scene(full_sphere), showlegend=True,
        margin={"l": 0, "r": 0, "t": 52, "b": 0}, template="plotly_white",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_buttons = []
    for name in metrics:
        values = metrics[name].tolist()
        metric_buttons.append(dict(
            label=METRIC_LABELS.get(name, name),
            method="restyle",
            args=[{"marker.color": [values], "marker.colorbar.title.text": [METRIC_LABELS.get(name, name)]}, [3]],
        ))
    figure.update_layout(updatemenus=[dict(buttons=metric_buttons, x=0.01, y=1.02, xanchor="left",
                                           yanchor="top", direction="down")])
    figure.write_html(output, include_plotlyjs=True, full_html=True, post_script=_sync_script())
    return output


def _dashboard_entry(
    modality: str,
    encoder: str,
    dataset: str,
    backend: str,
    device: str,
    max_points: int,
) -> dict:
    X, labels, coords, datasets, tags = _load_selection(
        modality, encoder, dataset, backend, max_points=max_points
    )
    metric_values = _metrics_per_dataset(X, labels, datasets, device)
    return {
        "modality": modality,
        "encoder": encoder,
        "dataset": dataset,
        "coords": np.asarray(coords, dtype=np.float32).tolist(),
        "labels": np.asarray(labels).tolist(),
        "datasets": np.asarray(datasets).tolist(),
        "tags": np.asarray(tags).tolist(),
        "metrics": {name: values.tolist() for name, values in metric_values.items()},
    }


def _metrics_per_dataset(
    embeds: np.ndarray, labels: np.ndarray, datasets: np.ndarray, device: str
) -> dict[str, np.ndarray]:
    """Compute each metric independently within every source dataset."""
    metrics: dict[str, np.ndarray] = {}
    for dataset_name in np.unique(datasets):
        members = np.flatnonzero(datasets == dataset_name)
        if len(members) < 2:
            LOGGER.warning("Skipping metrics for %s: only %d sampled points", dataset_name, len(members))
            continue
        dataset_metrics = _point_metrics(embeds[members], labels[members], 42, device)
        for name, values in dataset_metrics.items():
            if name not in metrics:
                metrics[name] = np.zeros(len(embeds), dtype=np.float32)
            metrics[name][members] = values
    if not metrics:
        raise ValueError("No dataset contains at least two sampled points for metric computation")
    return metrics


def create_dashboard_html(
    entries: list[dict], backend: str, initial_metric: str, output: Path
) -> Path:
    """Write one browser-side dashboard containing every requested selection."""
    from plotly.offline import get_plotlyjs

    payload = {
        f"{entry['modality']}|{entry['encoder']}|{entry['dataset']}": entry
        for entry in entries
    }
    metric_labels = {name: METRIC_LABELS.get(name, name) for name in DEFAULT_METRICS}
    plotly_js = get_plotlyjs()
    payload_json = json.dumps(payload, separators=(",", ":"))
    labels_json = json.dumps(metric_labels, separators=(",", ":"))
    backend_json = json.dumps(backend)
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Embedding spheres</title>
<style>
body {{ margin: 0; font-family: sans-serif; background: #fff; }}
#controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 12px 18px; }}
label {{ font-weight: 600; }}
select {{ min-width: 170px; padding: 4px; }}
#status {{ color: #666; }}
#plots {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; height: calc(100vh - 64px); }}
.plot {{ min-width: 0; min-height: 500px; }}
</style>
</head>
<body>
<div id="controls">
  <label>Modality <select id="modality"></select></label>
  <label>Encoder <select id="encoder"></select></label>
  <label>Dataset <select id="dataset"></select></label>
  <label>Metric <select id="metric"></select></label>
  <span id="status"></span>
</div>
<div id="plots"><div id="groups" class="plot"></div><div id="metric_plot" class="plot"></div></div>
<script>{plotly_js}</script>
<script>
const DATA = {payload_json};
const METRIC_LABELS = {labels_json};
const BACKEND = {backend_json};
const entries = Object.values(DATA);
const modalitySelect = document.getElementById('modality');
const encoderSelect = document.getElementById('encoder');
const datasetSelect = document.getElementById('dataset');
const metricSelect = document.getElementById('metric');
const status = document.getElementById('status');
let syncing = false;
let currentKey = null;

function options(select, values) {{
  select.replaceChildren(...values.map(value => new Option(value, value)));
}}
function modalities() {{
  return [...new Set(entries.map(entry => entry.modality))].sort();
}}
function encoders() {{
  return [...new Set(entries.filter(entry => entry.modality === modalitySelect.value)
    .map(entry => entry.encoder))].sort();
}}
function datasets() {{
  return [...new Set(entries.filter(entry => entry.modality === modalitySelect.value &&
    entry.encoder === encoderSelect.value).map(entry => entry.dataset))]
    .sort((a, b) => a === 'pooled' ? -1 : b === 'pooled' ? 1 : a.localeCompare(b));
}}
function surface(fullSphere) {{
  const u = [], v = [], x = [], y = [], z = [];
  const rows = 49, cols = 97, polar = fullSphere ? Math.PI : Math.PI / 2;
  for (let i = 0; i < rows; i++) {{
    const rowX = [], rowY = [], rowZ = [];
    const theta = polar * i / (rows - 1);
    for (let j = 0; j < cols; j++) {{
      const phi = 2 * Math.PI * j / (cols - 1);
      rowX.push(Math.cos(phi) * Math.sin(theta));
      rowY.push(Math.sin(phi) * Math.sin(theta));
      rowZ.push(Math.cos(theta));
    }}
    x.push(rowX); y.push(rowY); z.push(rowZ);
  }}
  return {{type: 'surface', x, y, z, surfacecolor: x.map(row => row.map(() => 0)),
    colorscale: [[0, '#F5F5F5'], [1, '#F5F5F5']], showscale: false,
    hoverinfo: 'skip', opacity: 1, lighting: {{ambient: 0.55, diffuse: 0.8,
    specular: 0.15, roughness: 0.9}}, showlegend: false}};
}}
function scene(fullSphere) {{
  const axis = {{range: [-1.1, 1.1], showbackground: false, showgrid: false,
    showline: false, showticklabels: false, ticks: '', title: '', zeroline: false}};
  return {{xaxis: axis, yaxis: axis,
    zaxis: Object.assign({{}}, axis, {{range: fullSphere ? [-1.1, 1.1] : [0, 1.1]}}),
    aspectmode: 'cube'}};
}}
function pointTrace(d, mask, colors, name, hover, showScale, metricName) {{
  const xyz = mask.map(i => d.coords[i]);
  const trace = {{type: 'scatter3d', mode: 'markers',
    x: xyz.map(p => p[0] * 1.01), y: xyz.map(p => p[1] * 1.01),
    z: xyz.map(p => p[2] * 1.01), marker: {{size: 3, color: colors, opacity: 0.8}},
    text: hover, hovertemplate: '%{{text}}<extra></extra>', name}};
  if (showScale) {{
    trace.marker.colorscale = 'Viridis'; trace.marker.showscale = true;
    trace.marker.colorbar = {{title: METRIC_LABELS[metricName]}};
  }}
  return trace;
}}
function updatePlots() {{
  const key = [modalitySelect.value, encoderSelect.value, datasetSelect.value].join('|');
  const d = DATA[key]; currentKey = key;
  const fullSphere = BACKEND === 'umap-learn';
  const groupValues = datasetSelect.value === 'pooled'
    ? d.datasets
    : d.labels.map(value => String(value));
  const groupNames = [...new Set(groupValues)];
  const colors = ['#0F4D92', '#B64342', '#8BCF8B', '#9A4D8E', '#E69F00',
    '#42949E', '#56B4E9', '#CC79A7', '#8C564B', '#D55E00'];
  const groupTraces = [surface(fullSphere)];
  groupNames.forEach((group, index) => {{
    const mask = groupValues.map((value, i) => value === group ? i : -1).filter(i => i >= 0);
    const hover = mask.map(i => d.dataset[i] === 'pooled'
      ? `dataset: ${{d.datasets[i]}} [${{d.tags[i]}}]<br>subtype: ${{d.labels[i]}}`
      : `subtype: ${{d.labels[i]}}`);
    groupTraces.push(pointTrace(d, mask, colors[index % colors.length],
      datasetSelect.value === 'pooled' ? group : `subtype ${{group}}`, hover, false, ''));
  }});
  const metricName = metricSelect.value;
  const all = d.coords.map((_, i) => i);
  const metricHover = all.map(i => d.dataset[i] === 'pooled'
    ? `dataset: ${{d.datasets[i]}} [${{d.tags[i]}}]<br>subtype: ${{d.labels[i]}}`
    : `subtype: ${{d.labels[i]}}`);
  const metricTrace = pointTrace(d, all, d.metrics[metricName], 'metric', metricHover, true, metricName);
  Plotly.react('groups', groupTraces, {{scene: scene(fullSphere), showlegend: true,
    margin: {{l: 0, r: 0, t: 36, b: 0}}, title: 'Groups'}});
  Plotly.react('metric_plot', [surface(fullSphere), metricTrace], {{scene: scene(fullSphere),
    showlegend: false, margin: {{l: 0, r: 0, t: 36, b: 0}}, title: METRIC_LABELS[metricName]}});
  status.textContent = `${{d.coords.length.toLocaleString()}} points`;
}}
function refreshEncoders() {{
  const old = encoderSelect.value; options(encoderSelect, encoders());
  if (encoders().includes(old)) encoderSelect.value = old;
  refreshDatasets();
}}
function refreshDatasets() {{
  const old = datasetSelect.value; options(datasetSelect, datasets());
  if (datasets().includes(old)) datasetSelect.value = old;
  updatePlots();
}}
modalitySelect.addEventListener('change', refreshEncoders);
encoderSelect.addEventListener('change', refreshDatasets);
datasetSelect.addEventListener('change', updatePlots);
metricSelect.addEventListener('change', updatePlots);
function sync(source, target) {{
  document.getElementById(source).on('plotly_relayout', event => {{
    if (syncing) return;
    const camera = event['scene.camera']; if (!camera) return;
    syncing = true; Plotly.relayout(target, {{'scene.camera': camera}})
      .finally(() => syncing = false);
  }});
}}
options(metricSelect, Object.keys(METRIC_LABELS).map(name => METRIC_LABELS[name]));
Object.keys(METRIC_LABELS).forEach((name, i) => metricSelect.options[i].value = name);
metricSelect.value = {json.dumps(initial_metric)};
options(modalitySelect, modalities());
refreshEncoders();
sync('groups', 'metric_plot'); sync('metric_plot', 'groups');
</script>
</body>
</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--encoder", nargs="+", default=None)
    parser.add_argument("--dataset", nargs="+", default=None, help="generation scope; HTML still selects datasets")
    parser.add_argument("--backend", choices=("mlx-vis", "umap-learn"), default="umap-learn")
    parser.add_argument("--metric", default="kde_density", choices=DEFAULT_METRICS, help="initial HTML metric")
    parser.add_argument("--device", default="auto", help="metric device: auto, cpu, mps, or cuda")
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="maximum sampled points per dataset in each HTML selection (default: 5000)",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "assets" / "embedding_spheres.html")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    encoders = args.encoder or _default_encoders()
    entries = []
    for modality in args.modality:
        datasets = args.dataset or ["pooled", *_config(modality)["datasets"].keys()]
        for encoder in encoders:
            for dataset in datasets:
                LOGGER.info("Preparing %s/%s/%s", modality, encoder, dataset)
                try:
                    entries.append(
                        _dashboard_entry(
                            modality,
                            encoder,
                            dataset,
                            args.backend,
                            args.device,
                            args.max_points,
                        )
                    )
                except FileNotFoundError as error:
                    LOGGER.warning("Skipping %s/%s/%s: %s", modality, encoder, dataset, error)
    if not entries:
        parser.error("no cached selections found; run plots/embedding_umap.py first")
    LOGGER.info("Wrote %s", create_dashboard_html(entries, args.backend, args.metric, args.output))


if __name__ == "__main__":
    main()
