"""Basin polygons (user shapefile / GeoPackage) used as experiment units. Requires the ``geo`` extra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bridge_lc.config import REPO_ROOT


@dataclass
class Basin:
    id: str
    geometry: object  # shapely geometry in EPSG:4326
    utm_crs: str  # local metric CRS used for the basin's raster grid
    attrs: dict

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return tuple(self.geometry.bounds)


def utm_epsg(lon: float, lat: float) -> str:
    """WGS84 / UTM zone EPSG code for a point."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def load_basins(
    path: str | Path,
    id_field: str | None = None,
    buffer_m: float = 0.0,
    expected: int | None = None,
) -> list[Basin]:
    """Read basins, reproject to EPSG:4326, optionally buffer (in metres, in each basin's UTM CRS).

    ``id_field``: attribute holding a unique basin id; if None, uses the row index (``basin_00``…).
    ``expected``: if given, raise unless exactly this many basins are found.
    """
    import geopandas as gpd

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS (.prj missing?)")
    gdf = gdf.to_crs("EPSG:4326")
    if gdf.geometry.is_empty.any() or (~gdf.geometry.is_valid).any():
        gdf["geometry"] = gdf.geometry.make_valid()
    if expected is not None and len(gdf) != expected:
        raise ValueError(f"Expected {expected} basins, found {len(gdf)} in {path}")
    if id_field is not None:
        if id_field not in gdf.columns:
            raise KeyError(f"id_field '{id_field}' not in {list(gdf.columns)}")
        if gdf[id_field].duplicated().any():
            raise ValueError(f"id_field '{id_field}' is not unique")

    basins = []
    for i, row in gdf.reset_index(drop=True).iterrows():
        c = row.geometry.centroid
        crs = utm_epsg(c.x, c.y)
        geom = row.geometry
        if buffer_m:
            geom = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(crs).buffer(buffer_m).to_crs(4326).iloc[0]
        bid = str(row[id_field]) if id_field else f"basin_{i:02d}"
        attrs = {k: v for k, v in row.items() if k != "geometry"}
        basins.append(Basin(id=bid, geometry=geom, utm_crs=crs, attrs=attrs))
    return basins
