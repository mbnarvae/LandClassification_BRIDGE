"""Hierarchical BRIDGE legend and lookups to Noah-MP vegetation categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bridge_lc.config import CONFIG_DIR, load_yaml

DEFAULT_LEGEND = CONFIG_DIR / "legend" / "bridge_andes_v0.yaml"

# Noah-MP modified MODIS-IGBP 20-category scheme.
IGBP20 = {
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-Up",
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated",
    17: "Water",
    18: "Wooded Tundra",
    19: "Mixed Tundra",
    20: "Barren Tundra",
}


@dataclass(frozen=True)
class LegendClass:
    id: int
    name: str
    level1: str
    igbp: int
    noahmp_ext: int
    rationale: str = ""


@dataclass
class Legend:
    name: str
    version: str
    nodata: int
    classes: list[LegendClass] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        """Number of real classes (excluding nodata)."""
        return len(self.classes)

    @property
    def ids(self) -> list[int]:
        return [c.id for c in self.classes]

    def by_name(self, name: str) -> LegendClass:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(name)

    def lookup(self, target: str) -> np.ndarray:
        """Dense lookup array mapping legend id → ``target`` code ('igbp' | 'noahmp_ext' | 'level1').

        For 'level1' the result maps to the index of the group in ``self.level1_groups`` (1-based).
        Unmapped ids (incl. nodata) map to 0.
        """
        lut = np.zeros(max(self.ids) + 1, dtype=np.int32)
        groups = self.level1_groups
        for c in self.classes:
            lut[c.id] = groups.index(c.level1) + 1 if target == "level1" else getattr(c, target)
        return lut

    @property
    def level1_groups(self) -> list[str]:
        return sorted({c.level1 for c in self.classes})

    @property
    def extended_classes(self) -> list[LegendClass]:
        """Classes that require a new Noah-MP parameter set (code > 20)."""
        return [c for c in self.classes if c.noahmp_ext > 20]

    def validate(self) -> None:
        ids = self.ids
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate class ids in legend")
        if self.nodata in ids:
            raise ValueError("nodata value collides with a class id")
        for c in self.classes:
            if c.igbp not in IGBP20:
                raise ValueError(f"{c.name}: igbp code {c.igbp} not in IGBP-20")
            if c.noahmp_ext <= 20 and c.noahmp_ext not in IGBP20:
                raise ValueError(f"{c.name}: invalid noahmp_ext {c.noahmp_ext}")


def load_legend(path: str | Path = DEFAULT_LEGEND) -> Legend:
    raw = load_yaml(path)
    legend = Legend(
        name=raw["name"],
        version=str(raw["version"]),
        nodata=int(raw.get("nodata", 0)),
        classes=[LegendClass(**c) for c in raw["classes"]],
    )
    legend.validate()
    return legend
