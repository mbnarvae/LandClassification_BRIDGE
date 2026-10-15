"""Reference label harmonisation: map external products onto the BRIDGE legend.

Mappings live in ``configs/legend/label_mappings/<source>.yaml`` as ``{source_code: bridge_class_name}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bridge_lc.config import CONFIG_DIR, load_yaml
from bridge_lc.legend import Legend

MAPPINGS_DIR = CONFIG_DIR / "legend" / "label_mappings"


def load_mapping(source: str, legend: Legend) -> dict[int, int]:
    """Return ``{source_code: bridge_id}`` for a label source (e.g. 'esa_worldcover')."""
    raw = load_yaml(MAPPINGS_DIR / f"{source}.yaml")
    return {int(k): legend.by_name(v).id for k, v in raw["mapping"].items() if v is not None}


def remap(arr: np.ndarray, mapping: dict[int, int], nodata: int = 0) -> np.ndarray:
    """Vectorised remap; unmapped values become ``nodata``."""
    lut = np.full(max(max(mapping), int(arr.max())) + 1, nodata, dtype=np.int32)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut[arr]


def available_sources() -> list[str]:
    return sorted(p.stem for p in Path(MAPPINGS_DIR).glob("*.yaml"))
