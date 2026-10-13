"""AUROC of MSDE movement as a function of the number of samples used.

For every (modality, dataset, encoder) a class-balanced sample of ``n`` normal and
``n`` anomalous rows is drawn for each ``n`` in ``--sizes`` (capped by the smaller
class), repeated for ``--seeds`` seeds. MSDE is run unlabeled on the pooled sample
and the AUROC of the movement is recorded, exactly as in
``scripts/structural_asymmetry.py``. One CSV row per (dataset, encoder, n, seed).

Usage::

    uv run python scripts/sample_size_sweep.py
    uv run python scripts/sample_size_sweep.py --modalities oct --encoders CLIP --seeds 1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from datasets.embeds import SingleEmbedsDataset  # noqa: E402
from methods.msde import DEFAULT_DEVICE, MeanShiftDensityEnhancement  # noqa: E402
from structural_asymmetry import ENCODERS, MODALITIES, balanced_sample, iter_jobs  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "sample_size_sweep.csv"
KEY = ["modality", "dataset", "encoder", "n_per_class", "seed"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("sample_size_sweep")


def movement_auroc(X: np.ndarray, y: np.ndarray, args, device: str) -> float:
    msde = MeanShiftDensityEnhancement(
        k=min(args.msde_k, len(X) - 1), nbd_sample_count_threshold=args.msde_threshold,
        learning_rate=args.msde_lr, max_iters_shift=args.msde_iters,
        shift_threshold=args.msde_shift_threshold, device=device, use_chunking=True,
        weight_chunk_size=2048, enable_gradients=False,
    )
    with torch.no_grad():
        _, movement, _ = msde(torch.from_numpy(X).float().to(device))
    return float(roc_auc_score(y, movement.cpu().numpy()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[25, 50, 100, 250, 500, 1000, 2000],
                        help="samples per class")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true")
    # MSDE settings default to those of configs/method/knnvote_msde.yaml.
    parser.add_argument("--msde-k", type=int, default=50)
    parser.add_argument("--msde-lr", type=float, default=0.33)
    parser.add_argument("--msde-threshold", type=int, default=70)
    parser.add_argument("--msde-iters", type=int, default=3)
    parser.add_argument("--msde-shift-threshold", type=float, default=0.01)
    args = parser.parse_args()

    device = DEFAULT_DEVICE if args.device == "auto" else args.device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(args.output) if args.output.is_file() and not args.no_resume else pd.DataFrame()
    finished = set(map(tuple, done[KEY].values)) if len(done) else set()
    rows = done.to_dict("records")

    jobs = [j for j in iter_jobs(args.modalities, args.encoders) if args.datasets is None or j[1] in args.datasets]
    LOGGER.info("%d jobs on %s, sizes=%s, seeds=%d", len(jobs), device, args.sizes, args.seeds)
    for number, (modality, dataset, encoder, path, meta) in enumerate(jobs, 1):
        try:
            ds = SingleEmbedsDataset(
                path, dataset, encoder=encoder, as_tensor=False,
                include_labels=meta.get("include_labels", meta.get("labels")),
            )
        except FileNotFoundError as error:
            LOGGER.warning("skip %s/%s/%s: %s", modality, dataset, encoder, str(error)[:80])
            continue
        if not ((ds.labels == 0).any() and (ds.labels != 0).any()):
            continue
        available = min(int((ds.labels == 0).sum()), int((ds.labels != 0).sum()))
        # Sizes beyond the smaller class collapse to it; keep that point once.
        sizes = sorted({min(n, available) for n in args.sizes})
        start = time.time()
        for n in sizes:
            for seed in range(args.seeds):
                if (modality, dataset, encoder, n, seed) in finished:
                    continue
                rng = np.random.RandomState(1000 * seed + n)
                normal, anomaly = balanced_sample(ds.labels, n, rng)
                X = ds.embeddings[np.concatenate([normal, anomaly])]
                y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)]
                rows.append({
                    "modality": modality, "dataset": dataset, "encoder": encoder,
                    "n_per_class": n, "seed": seed, "n_total": len(X),
                    "auroc_move": movement_auroc(X, y, args, device),
                })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        LOGGER.info("[%d/%d] %s/%s/%s  %d sizes  %.0fs", number, len(jobs), modality, dataset, encoder,
                    len(sizes), time.time() - start)
    LOGGER.info("wrote %s (%d rows)", args.output, len(rows))


if __name__ == "__main__":
    main()
