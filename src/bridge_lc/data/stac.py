"""STAC search for optical/SAR imagery and DEMs (requires the ``geo`` extra).

Endpoints:
- HLS v2 (Prithvi's native input): NASA CMR-STAC LPCLOUD, collections ``HLSS30_2.0`` / ``HLSL30_2.0``.
  Needs an Earthdata login (~/.netrc) to read the assets.
- Sentinel-2 L2A, Sentinel-1 RTC, Copernicus DEM: Microsoft Planetary Computer.
"""

from __future__ import annotations

from bridge_lc.data.aoi import Site

CMR_LPCLOUD = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"

COLLECTIONS = {
    "hls_s30": (CMR_LPCLOUD, "HLSS30_2.0"),
    "hls_l30": (CMR_LPCLOUD, "HLSL30_2.0"),
    "sentinel2": (PLANETARY_COMPUTER, "sentinel-2-l2a"),
    "sentinel1": (PLANETARY_COMPUTER, "sentinel-1-rtc"),
    "cop_dem_30": (PLANETARY_COMPUTER, "cop-dem-glo-30"),
}


def search(
    site: Site,
    product: str,
    start: str,
    end: str,
    max_cloud_pct: float | None = None,
    limit: int = 500,
):
    """Return a list of STAC items for ``product`` over ``site`` between ``start`` and ``end``."""
    from pystac_client import Client

    url, collection = COLLECTIONS[product]
    modifier = None
    if url == PLANETARY_COMPUTER:
        import planetary_computer

        modifier = planetary_computer.sign_inplace
    client = Client.open(url, modifier=modifier)

    query = {}
    if max_cloud_pct is not None and product == "sentinel2":
        query["eo:cloud_cover"] = {"lt": max_cloud_pct}
    results = client.search(
        collections=[collection],
        bbox=site.bbox,
        datetime=f"{start}/{end}",
        query=query or None,
        max_items=limit,
    )
    items = list(results.items())
    if max_cloud_pct is not None and product.startswith("hls"):
        items = [i for i in items if i.properties.get("eo:cloud_cover", 100) < max_cloud_pct]
    return items
