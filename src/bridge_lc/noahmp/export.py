"""Export classifications for Noah-MP (HRLDAS / WRF geo_em LU_INDEX). Requires the ``geo`` extra."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bridge_lc.legend import Legend, load_legend
from bridge_lc.noahmp.crosswalk import to_noahmp


def remap_geotiff(src: Path, dst: Path, scheme: str = "noahmp_ext", legend: Legend | None = None):
    """Remap a classified GeoTIFF to Noah-MP codes, preserving georeferencing."""
    import rasterio

    legend = legend or load_legend()
    with rasterio.open(src) as ds:
        data = ds.read(1)
        profile = ds.profile | {"dtype": "uint8", "nodata": 0, "compress": "deflate"}
    with rasterio.open(dst, "w", **profile) as out:
        out.write(to_noahmp(data, legend, scheme).astype(np.uint8), 1)


def dominant_category(codes: np.ndarray, factor: int, nodata: int = 0) -> np.ndarray:
    """Aggregate a fine Noah-MP code raster to a coarser grid by majority (block size ``factor``)."""
    h, w = (codes.shape[0] // factor) * factor, (codes.shape[1] // factor) * factor
    blocks = codes[:h, :w].reshape(h // factor, factor, w // factor, factor).swapaxes(1, 2)
    blocks = blocks.reshape(h // factor, w // factor, -1)
    n = int(codes.max()) + 1
    counts = np.apply_along_axis(lambda b: np.bincount(b, minlength=n), -1, blocks)
    counts[..., nodata] = 0
    return counts.argmax(-1)


def write_lu_index(geo_em: Path, codes_on_grid: np.ndarray, out: Path, num_categories: int = 20):
    """Write ``LU_INDEX`` (and update ``NUM_LAND_CAT``) in a copy of a WRF/HRLDAS geo_em file.

    ``codes_on_grid`` must already be on the geo_em mass grid (south_north, west_east).
    With extended classes (>20) Noah-MP must be compiled/configured with the matching MPTABLE.
    """
    import xarray as xr

    ds = xr.open_dataset(geo_em)
    lu = ds["LU_INDEX"]
    if lu.shape[-2:] != codes_on_grid.shape:
        raise ValueError(f"grid mismatch: geo_em {lu.shape[-2:]} vs codes {codes_on_grid.shape}")
    ds["LU_INDEX"] = lu.copy(data=codes_on_grid.reshape(lu.shape).astype(lu.dtype))
    ds.attrs["NUM_LAND_CAT"] = max(num_categories, int(codes_on_grid.max()))
    ds.attrs["BRIDGE_LANDCLASS"] = "LU_INDEX replaced by BRIDGE land classification"
    ds.to_netcdf(out)
