"""Cloud-masked seasonal median composites of HLS v2 (Prithvi's input). Requires the ``geo`` extra.

Reading HLS assets needs NASA Earthdata credentials in ``~/.netrc`` (machine urs.earthdata.nasa.gov).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Prithvi band name → HLS asset name per sensor.
HLS_BANDS = {
    "hls_s30": {
        "BLUE": "B02",
        "GREEN": "B03",
        "RED": "B04",
        "NIR_NARROW": "B8A",
        "SWIR_1": "B11",
        "SWIR_2": "B12",
    },
    "hls_l30": {
        "BLUE": "B02",
        "GREEN": "B03",
        "RED": "B04",
        "NIR_NARROW": "B05",
        "SWIR_1": "B06",
        "SWIR_2": "B07",
    },
}
HLS_SCALE = 1e-4  # stored as int16 reflectance × 10000
HLS_FILL = -9999

# HLS v2 Fmask bits. Snow/ice (bit 4) and water (bit 5) are deliberately NOT masked:
# glaciers and lakes are classes we need to map.
FMASK_CLOUD = 1 << 1
FMASK_ADJACENT = 1 << 2
FMASK_SHADOW = 1 << 3


def fmask_valid(fmask: np.ndarray, mask_adjacent: bool = True) -> np.ndarray:
    bad = FMASK_CLOUD | FMASK_SHADOW | (FMASK_ADJACENT if mask_adjacent else 0)
    return (fmask.astype(np.int32) & bad) == 0


def configure_earthdata_gdal() -> None:
    """GDAL settings so rasterio can stream HLS COGs with Earthdata cookies."""
    cookies = str(Path.home() / ".urs_cookies")
    os.environ.setdefault("GDAL_HTTP_COOKIEFILE", cookies)
    os.environ.setdefault("GDAL_HTTP_COOKIEJAR", cookies)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "TIF")
    os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "NO")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def seasonal_composite(
    bbox: tuple[float, float, float, float],
    crs: str,
    start: str,
    end: str,
    resolution: float = 30.0,
    max_cloud_pct: float = 80.0,
    bands: list[str] | None = None,
    chunks: int = 1024,
):
    """Median composite of HLS S30+L30 between ``start`` and ``end`` over ``bbox`` (EPSG:4326).

    Returns an xarray DataArray (band, y, x) of reflectance ×10000 as float32 (NaN = no clear obs),
    plus a DataArray with the number of clear observations per pixel (useful QA in cloudy Andes).
    """
    import xarray as xr
    from odc.stac import load
    from pystac_client import Client

    from bridge_lc.data.stac import CMR_LPCLOUD

    configure_earthdata_gdal()
    bands = bands or list(HLS_BANDS["hls_s30"])
    client = Client.open(CMR_LPCLOUD)

    per_sensor = []
    for product, collection in (("hls_s30", "HLSS30_2.0"), ("hls_l30", "HLSL30_2.0")):
        items = [
            it
            for it in client.search(collections=[collection], bbox=bbox, datetime=f"{start}/{end}").items()
            if it.properties.get("eo:cloud_cover", 0) <= max_cloud_pct
        ]
        if not items:
            continue
        mapping = HLS_BANDS[product]
        ds = load(
            items,
            bands=[mapping[b] for b in bands] + ["Fmask"],
            bbox=bbox,
            crs=crs,
            resolution=resolution,
            chunks={"x": chunks, "y": chunks},
            groupby="solar_day",
            nodata=HLS_FILL,
        )
        ds = ds.rename({mapping[b]: b for b in bands})
        valid = xr.apply_ufunc(fmask_valid, ds["Fmask"], dask="parallelized", output_dtypes=[bool])
        stack = ds[bands].to_array("band").astype("float32")
        stack = stack.where((stack != HLS_FILL) & valid)
        per_sensor.append(stack)

    if not per_sensor:
        raise RuntimeError(f"No HLS scenes for {bbox} {start}/{end}")
    stack = xr.concat(per_sensor, dim="time")
    n_clear = stack.isel(band=0).notnull().sum("time").astype("int16").rename("n_clear")
    return stack.median("time", skipna=True), n_clear
