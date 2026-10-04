"""Per-dataset anomaly metrics and their run-level aggregation."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

POOLED = "pooled_nonzero"


def _precision_at_n(target: np.ndarray, score: np.ndarray) -> float:
    count = int(target.sum())
    if count == 0:
        return float("nan")
    return float(target[np.argsort(-score)[:count]].mean())


def _pair_metrics(scores: torch.Tensor, source_labels: np.ndarray, dataset_name: str) -> list[dict[str, Any]]:
    rows = []
    unique = sorted(int(label) for label in np.unique(source_labels))
    if 0 not in unique:
        return rows
    for anomaly_label in [label for label in unique if label != 0] + (["pooled"] if len(unique) > 2 else []):
        if anomaly_label == "pooled":
            keep = (source_labels == 0) | (source_labels != 0)
            label_text = POOLED
            target = (source_labels[keep] != 0).astype(int)
        else:
            keep = (source_labels == 0) | (source_labels == anomaly_label)
            label_text = f"label_0_vs_{anomaly_label}"
            target = (source_labels[keep] == anomaly_label).astype(int)
        if len(np.unique(target)) < 2:
            continue
        values = scores.detach().cpu().numpy()[keep]
        rows.append({
            "dataset": dataset_name,
            "comparison": label_text,
            "normal_label": 0,
            "anomaly_label": label_text if anomaly_label == "pooled" else int(anomaly_label),
            "auroc": float(roc_auc_score(target, values)),
            "auprc": float(average_precision_score(target, values)),
            "precision_at_n": _precision_at_n(target, values),
            "n_normal": int((target == 0).sum()),
            "n_anomaly": int((target == 1).sum()),
        })
    return rows


def dataset_level(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick each dataset's normal-vs-all-anomalies row.

    That is the ``pooled_nonzero`` row, or the single ``label_0_vs_k`` row of a
    binary dataset (``_pair_metrics`` only pools when there are several
    anomaly labels, and for one label the two comparisons coincide).
    """
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    selected = {}
    for dataset_name, dataset_rows in by_dataset.items():
        pooled = [row for row in dataset_rows if row["comparison"] == POOLED]
        if pooled:
            selected[dataset_name] = pooled[0]
        elif len(dataset_rows) == 1:
            selected[dataset_name] = dataset_rows[0]
    return selected


def _metric_key(dataset_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-/ ]", "_", dataset_name)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Run-level metrics: dataset-level values plus their unweighted means.

    ``objective`` (= ``mean_auroc``) is NaN when no dataset has a usable row.
    """
    selected = dataset_level(rows)
    metrics: dict[str, float] = {}
    for dataset_name, row in selected.items():
        key = _metric_key(dataset_name)
        metrics[f"{key}/auroc"] = float(row["auroc"])
        metrics[f"{key}/auprc"] = float(row["auprc"])
        metrics[f"{key}/p_at_n"] = float(row["precision_at_n"])
    for name, column in (("mean_auroc", "auroc"), ("mean_auprc", "auprc"), ("mean_p_at_n", "precision_at_n")):
        values = [float(row[column]) for row in selected.values()]
        metrics[name] = float(np.mean(values)) if values else float("nan")
    metrics["objective"] = metrics["mean_auroc"]
    return metrics
