"""Spectral indices on reflectance arrays (any shape; bands passed explicitly)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return (a - b) / (a + b + _EPS)


def ndvi(nir, red):
    return _nd(nir, red)


def ndwi(green, nir):
    """McFeeters NDWI — open water."""
    return _nd(green, nir)


def ndmi(nir, swir1):
    """Moisture index — useful for bofedales and páramo wetness."""
    return _nd(nir, swir1)


def nbr(nir, swir2):
    return _nd(nir, swir2)


def ndsi(green, swir1):
    """Normalised difference snow index — glaciers / snow."""
    return _nd(green, swir1)


def evi(nir, red, blue, g=2.5, c1=6.0, c2=7.5, l=1.0):  # noqa: E741
    nir, red, blue = (x.astype(np.float32) for x in (nir, red, blue))
    return g * (nir - red) / (nir + c1 * red - c2 * blue + l + _EPS)


INDEX_FUNCS = {
    "ndvi": (ndvi, ("NIR_NARROW", "RED")),
    "ndwi": (ndwi, ("GREEN", "NIR_NARROW")),
    "ndmi": (ndmi, ("NIR_NARROW", "SWIR_1")),
    "nbr": (nbr, ("NIR_NARROW", "SWIR_2")),
    "ndsi": (ndsi, ("GREEN", "SWIR_1")),
    "evi": (evi, ("NIR_NARROW", "RED", "BLUE")),
}


def compute_indices(image: np.ndarray, band_names: list[str], names: list[str]) -> np.ndarray:
    """Compute indices from a (C, H, W) stack. Returns (len(names), H, W)."""
    idx = {b: i for i, b in enumerate(band_names)}
    out = []
    for name in names:
        func, bands = INDEX_FUNCS[name]
        out.append(func(*(image[idx[b]] for b in bands)))
    return np.stack(out)
