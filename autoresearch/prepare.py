"""
FROZEN — the autoresearch agent must NOT modify this file.

Builds the fixed train/validation split and defines the evaluation metric, so that every
experiment in results.tsv is comparable. The split is SPATIAL (held-out blocks or a held-out
site), never random pixels.

Usage:
    python autoresearch/prepare.py --synthetic                 # toy data, no downloads (smoke test)
    python autoresearch/prepare.py                             # real chips from data/processed/chips/
    python autoresearch/prepare.py --holdout-site yungas_bo    # transferability: validate on an unseen site

Real chips: one .npz per chip in data/processed/chips/ with keys
    x (C, H, W) float32, y (H, W) int (BRIDGE legend ids, 0 = nodata),
    origin (2,) projected coords of the chip corner in metres, site (str), bands (list[str]).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bridge_lc.evaluation.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    macro_f1,
    mean_iou,
)
from bridge_lc.evaluation.spatial_cv import spatial_folds  # noqa: E402

CACHE = Path(__file__).parent / "cache"
DATASET = CACHE / "dataset.npz"
META = CACHE / "meta.json"

TIME_BUDGET_S = 300  # wall-clock training budget per experiment (5 min)
SEED = 42
VAL_FOLD = 0
BLOCK_SIZE_M = 10_000
N_FOLDS = 5
IGNORE_INDEX = 0


# --------------------------------------------------------------------------- data building
def _synthetic(n_side: int = 768, chip: int = 64, n_classes: int = 8, seed: int = SEED):
    """Elevation/moisture-structured toy landscape with class-dependent spectra."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n_side, 0:n_side] / n_side
    elev = 0.6 * xx + 0.25 * np.sin(6 * yy) * np.cos(4 * xx) + 0.05 * rng.standard_normal((n_side,) * 2)
    moist = 0.5 + 0.3 * np.sin(5 * xx + 3 * yy) + 0.05 * rng.standard_normal((n_side,) * 2)
    e_bin = np.digitize(elev, np.quantile(elev, np.linspace(0, 1, n_classes // 2 + 1)[1:-1]))
    labels = (e_bin * 2 + (moist > np.median(moist))).astype(np.int64) + 1  # 1..n_classes
    flip = rng.random(labels.shape) < 0.05  # label noise, like imperfect reference maps
    labels[flip] = rng.integers(1, n_classes + 1, flip.sum())
    labels[rng.random(labels.shape) < 0.02] = 0  # sprinkle nodata

    # Overlapping class spectra + strong noise, so per-pixel spectra alone are ambiguous.
    n_bands = 6
    spectra = rng.uniform(0.15, 0.3, size=(n_classes + 1, n_bands))
    image = spectra[labels].transpose(2, 0, 1) + 0.12 * rng.standard_normal((n_bands, n_side, n_side))
    dem = elev + 0.15 * rng.standard_normal(elev.shape)  # noisy DEM-like channel
    image = np.concatenate([image, dem[None]], 0).astype(np.float32)

    xs, ys, origins = [], [], []
    for r in range(0, n_side - chip + 1, chip):
        for c in range(0, n_side - chip + 1, chip):
            xs.append(image[:, r : r + chip, c : c + chip])
            ys.append(labels[r : r + chip, c : c + chip])
            origins.append((c * 30.0, r * 30.0))  # pretend 30 m pixels
    bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2", "ELEVATION"]
    return np.stack(xs), np.stack(ys), np.asarray(origins), np.array(["synthetic"] * len(xs)), bands


def _real(chips_dir: Path):
    files = sorted(chips_dir.glob("*.npz"))
    if not files:
        sys.exit(f"No chips found in {chips_dir}. Run the chipping pipeline or use --synthetic.")
    xs, ys, origins, sites, bands = [], [], [], [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        xs.append(d["x"].astype(np.float32))
        ys.append(d["y"].astype(np.int64))
        origins.append(d["origin"])
        sites.append(str(d["site"]))
        bands = bands or list(d["bands"])
    return np.stack(xs), np.stack(ys), np.asarray(origins), np.asarray(sites), bands


def build(synthetic: bool, holdout_site: str | None) -> None:
    x, y, origins, sites, bands = _synthetic() if synthetic else _real(ROOT / "data/processed/chips")
    if holdout_site:
        is_val = sites == holdout_site
        split = f"holdout_site={holdout_site}"
    else:
        block = BLOCK_SIZE_M if not synthetic else 64 * 30 * 3
        folds = spatial_folds(origins[:, 0], origins[:, 1], block, N_FOLDS, SEED)
        is_val = folds == VAL_FOLD
        split = f"spatial_blocks(fold={VAL_FOLD}/{N_FOLDS})"
    if not is_val.any() or is_val.all():
        sys.exit("Degenerate split: check holdout site / block size.")

    num_classes = int(y.max()) + 1  # includes nodata 0
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(
        DATASET,
        x_train=x[~is_val],
        y_train=y[~is_val],
        x_val=x[is_val],
        y_val=y[is_val],
    )
    META.write_text(
        json.dumps(
            {
                "synthetic": synthetic,
                "split": split,
                "bands": bands,
                "num_classes": num_classes,
                "n_train": int((~is_val).sum()),
                "n_val": int(is_val.sum()),
                "time_budget_s": TIME_BUDGET_S,
            },
            indent=2,
        )
    )
    print(
        f"Wrote {DATASET} — {split}, train={(~is_val).sum()} val={is_val.sum()} chips, "
        f"bands={bands}, classes={num_classes}"
    )


# --------------------------------------------------------------------------- API for train.py
def load():
    """Return (x_train, y_train, x_val, meta). y_val is withheld on purpose — use evaluate()."""
    if not DATASET.exists():
        sys.exit("Run `python autoresearch/prepare.py [--synthetic]` first.")
    d = np.load(DATASET)
    return d["x_train"], d["y_train"], d["x_val"], json.loads(META.read_text())


def evaluate(pred_val: np.ndarray, verbose: bool = True) -> float:
    """THE metric: macro-F1 over classes present in the spatially held-out validation set."""
    d = np.load(DATASET)
    y_val = d["y_val"]
    if pred_val.shape != y_val.shape:
        raise ValueError(f"pred shape {pred_val.shape} != y_val shape {y_val.shape}")
    k = int(max(y_val.max(), pred_val.max()))
    cm = confusion_matrix(y_val, pred_val, k, IGNORE_INDEX)
    score = macro_f1(cm, IGNORE_INDEX)
    if verbose:
        for row in classification_report(cm):
            print(f"  class {row['id']:>3}  f1={row['f1']:.3f}  iou={row['iou']:.3f}  n={row['support']}")
        print(f"val_miou: {mean_iou(cm, IGNORE_INDEX):.4f}")
        print(f"val_macro_f1: {score:.4f}")
    return score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--holdout-site", default=None)
    args = ap.parse_args()
    build(args.synthetic, args.holdout_site)
