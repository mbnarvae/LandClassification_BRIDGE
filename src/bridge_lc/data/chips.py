"""Tile co-registered image/label arrays into fixed-size chips for FM training."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def iter_chip_windows(
    height: int, width: int, size: int, stride: int | None = None
) -> Iterator[tuple[int, int]]:
    """Yield (row, col) upper-left corners of chips fully inside the raster."""
    stride = stride or size
    for r in range(0, height - size + 1, stride):
        for c in range(0, width - size + 1, stride):
            yield r, c


def make_chips(
    image: np.ndarray,
    labels: np.ndarray,
    size: int = 224,
    stride: int | None = None,
    nodata: int = 0,
    min_valid_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut ``image`` (C, H, W) and ``labels`` (H, W) into chips.

    Returns ``(x, y, origins)`` with shapes (N, C, size, size), (N, size, size), (N, 2).
    Chips with fewer than ``min_valid_frac`` labelled pixels are dropped.
    """
    if image.shape[-2:] != labels.shape:
        raise ValueError(f"image {image.shape} and labels {labels.shape} are not aligned")
    xs, ys, origins = [], [], []
    for r, c in iter_chip_windows(*labels.shape, size, stride):
        y = labels[r : r + size, c : c + size]
        if (y != nodata).mean() < min_valid_frac:
            continue
        xs.append(image[:, r : r + size, c : c + size])
        ys.append(y)
        origins.append((r, c))
    if not xs:
        return (
            np.empty((0, image.shape[0], size, size), image.dtype),
            np.empty((0, size, size), labels.dtype),
            np.empty((0, 2), int),
        )
    return np.stack(xs), np.stack(ys), np.asarray(origins)


class ChipDataset:
    """Minimal torch-compatible dataset over in-memory chips (avoids importing torch here)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        import torch

        x, y = self.x[i], self.y[i]
        if self.transform is not None:
            x, y = self.transform(x, y)
        return torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
