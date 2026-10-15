"""Spatially blocked cross-validation (avoid optimistic random pixel splits)."""

from __future__ import annotations

import numpy as np


def block_ids(x: np.ndarray, y: np.ndarray, block_size: float) -> np.ndarray:
    """Assign each sample (projected coords, metres) to a square spatial block id."""
    bx = np.floor((x - x.min()) / block_size).astype(np.int64)
    by = np.floor((y - y.min()) / block_size).astype(np.int64)
    return by * (bx.max() + 1) + bx


def spatial_folds(
    x: np.ndarray, y: np.ndarray, block_size: float, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Fold index (0..n_folds-1) per sample; all samples in a block share a fold."""
    blocks = block_ids(x, y, block_size)
    uniq = np.unique(blocks)
    rng = np.random.default_rng(seed)
    fold_of_block = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % n_folds, strict=True))
    return np.vectorize(fold_of_block.__getitem__)(blocks)


def iter_folds(folds: np.ndarray):
    """Yield (train_idx, test_idx) for each fold."""
    for k in np.unique(folds):
        yield np.flatnonzero(folds != k), np.flatnonzero(folds == k)
