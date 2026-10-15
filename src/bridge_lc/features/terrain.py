"""Terrain descriptors from a projected DEM (metres). Pure numpy; for flow routing use pysheds/richdem."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (radians, clockwise from north) via central differences."""
    dz_dy, dz_dx = np.gradient(dem.astype(np.float64), res)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    # Row index increases southward, so north-facing slopes have dz_dy > 0.
    aspect = np.mod(np.arctan2(-dz_dx, dz_dy), 2 * np.pi)
    return slope.astype(np.float32), aspect.astype(np.float32)


def tpi(dem: np.ndarray, radius: int = 5) -> np.ndarray:
    """Topographic position index: elevation minus mean of a (2r+1)² neighbourhood."""
    k = 2 * radius + 1
    padded = np.pad(dem.astype(np.float64), radius, mode="edge")
    mean = sliding_window_view(padded, (k, k)).mean(axis=(-1, -2))
    return (dem - mean).astype(np.float32)


def twi(flow_acc: np.ndarray, slope_deg: np.ndarray, res: float) -> np.ndarray:
    """Topographic wetness index ln(a / tan β), from flow accumulation (cells) and slope.

    Compute ``flow_acc`` with a proper D8/D-inf routing (pysheds, richdem, WhiteboxTools).
    """
    a = (flow_acc + 1.0) * res
    tan_b = np.tan(np.radians(np.maximum(slope_deg, 0.1)))
    return np.log(a / tan_b).astype(np.float32)


def terrain_stack(dem: np.ndarray, res: float, flow_acc: np.ndarray | None = None) -> dict:
    slope, aspect = slope_aspect(dem, res)
    out = {
        "elevation": dem.astype(np.float32),
        "slope": slope,
        "aspect_sin": np.sin(aspect),
        "aspect_cos": np.cos(aspect),
        "tpi": tpi(dem),
    }
    if flow_acc is not None:
        out["twi"] = twi(flow_acc, slope, res)
    return out
