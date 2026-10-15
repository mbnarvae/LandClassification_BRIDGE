"""Test-bed sites (areas of interest) defined in configs/sites/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field

from bridge_lc.config import CONFIG_DIR, load_yaml

SITES_DIR = CONFIG_DIR / "sites"


@dataclass
class Site:
    id: str
    name: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat (EPSG:4326)
    crs: str
    country: str = ""
    elevation_range_m: tuple[float, float] | None = None
    target_ecosystems: list[str] = field(default_factory=list)
    seasons: dict[str, list[str]] = field(default_factory=dict)
    label_sources: list[str] = field(default_factory=list)
    notes: str = ""

    def geometry(self):
        """Shapely polygon of the bbox (requires the ``geo`` extra)."""
        from shapely.geometry import box

        return box(*self.bbox)


def load_site(site_id: str) -> Site:
    raw = load_yaml(SITES_DIR / f"{site_id}.yaml")
    raw["bbox"] = tuple(raw["bbox"])
    if raw.get("elevation_range_m"):
        raw["elevation_range_m"] = tuple(raw["elevation_range_m"])
    return Site(**raw)


def list_sites() -> list[Site]:
    return [load_site(p.stem) for p in sorted(SITES_DIR.glob("*.yaml"))]
