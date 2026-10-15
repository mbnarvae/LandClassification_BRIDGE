"""
AGENT-EDITABLE — this is the only file the autoresearch agent modifies.

Contract (do not break):
  * data comes from `prepare.load()`; never read y_val or touch the cache directly
  * training must respect TIME_BUDGET_S from prepare.py
  * finish by calling `prepare.evaluate(pred_val)`, which prints `val_macro_f1: X.XXXX`

Baseline: pixel-wise Random Forest on bands + spectral indices + local texture.
Ideas for the agent (see program.md): feature engineering (terrain, indices, seasonal deltas),
gradient boosting, frozen Prithvi-EO-2.0 embeddings + linear/MLP probe, LoRA fine-tuning,
class re-weighting, test-time augmentation, spatial smoothing of predictions...
"""

from __future__ import annotations

import time

import numpy as np
from prepare import TIME_BUDGET_S, evaluate, load
from sklearn.ensemble import RandomForestClassifier

SEED = 0
MAX_TRAIN_PIXELS = 200_000
EPS = 1e-6


def features(x: np.ndarray, bands: list[str]) -> np.ndarray:
    """(N, C, H, W) → (N, F, H, W)."""
    b = {name: x[:, i] for i, name in enumerate(bands)}
    feats = [x]
    if {"NIR_NARROW", "RED"} <= b.keys():
        feats.append(((b["NIR_NARROW"] - b["RED"]) / (b["NIR_NARROW"] + b["RED"] + EPS))[:, None])
    if {"NIR_NARROW", "SWIR_1"} <= b.keys():
        feats.append(((b["NIR_NARROW"] - b["SWIR_1"]) / (b["NIR_NARROW"] + b["SWIR_1"] + EPS))[:, None])
    return np.concatenate(feats, 1)


def to_pixels(f: np.ndarray) -> np.ndarray:
    return f.transpose(0, 2, 3, 1).reshape(-1, f.shape[1])


def main() -> None:
    t0 = time.time()
    x_train, y_train, x_val, meta = load()
    bands = meta["bands"]

    f_train, f_val = features(x_train, bands), features(x_val, bands)
    X, y = to_pixels(f_train), y_train.ravel()
    keep = y != 0
    X, y = X[keep], y[keep]
    rng = np.random.default_rng(SEED)
    if len(y) > MAX_TRAIN_PIXELS:
        idx = rng.choice(len(y), MAX_TRAIN_PIXELS, replace=False)
        X, y = X[idx], y[idx]

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(X, y)
    train_s = time.time() - t0
    if train_s > TIME_BUDGET_S:
        print(f"WARNING: exceeded time budget ({train_s:.0f}s > {TIME_BUDGET_S}s)")

    pred = model.predict(to_pixels(f_val)).reshape(x_val.shape[0], *x_val.shape[2:])
    evaluate(pred)
    print(f"train_seconds: {train_s:.1f}")
    print(f"total_seconds: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
