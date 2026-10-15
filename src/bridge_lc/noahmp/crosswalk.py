"""Crosswalk BRIDGE legend → Noah-MP vegetation categories, and MPTABLE templates."""

from __future__ import annotations

import numpy as np

from bridge_lc.legend import IGBP20, Legend

# MPTABLE.TBL parameters that most directly control the ecohydrological behaviour we expect to change
# for Andean classes. Values must come from literature/field data — never guessed.
MPTABLE_PARAMS = [
    "CH2OP",  # max intercepted water per unit LAI+SAI (cloud forest fog interception!)
    "DLEAF",  # characteristic leaf dimension
    "Z0MVT",  # momentum roughness length
    "HVT",
    "HVB",  # canopy top / bottom height
    "RC",  # tree crown radius
    "RHOL_VIS",
    "RHOL_NIR",
    "TAUL_VIS",
    "TAUL_NIR",  # leaf optical properties
    "XL",  # leaf/stem orientation index
    "CWPVT",  # canopy wind parameter
    "VCMX25",  # max carboxylation rate at 25 °C
    "MP",  # slope of conductance–photosynthesis relationship
    "ROOT_DEPTH_LAYERS",  # NROOT (in VEGPARM) — rooting depth
    "LAI_M01..LAI_M12",  # monthly LAI climatology
    "SAI_M01..SAI_M12",
]


def to_noahmp(classification: np.ndarray, legend: Legend, scheme: str = "noahmp_ext") -> np.ndarray:
    """Remap a raster of legend ids to Noah-MP codes. ``scheme``: 'igbp' or 'noahmp_ext'."""
    if scheme not in ("igbp", "noahmp_ext"):
        raise ValueError(scheme)
    lut = legend.lookup(scheme)
    return lut[np.clip(classification, 0, len(lut) - 1)]


def mptable_stub(legend: Legend) -> str:
    """Text template listing the new Noah-MP categories and the parameters to fill in."""
    lines = [
        "! BRIDGE extended Noah-MP categories (append to the MODIS section of MPTABLE.TBL)",
        "! Start each from its IGBP 'parent' parameters, then replace with literature/field values.",
        "",
    ]
    for c in legend.extended_classes:
        lines.append(f"! {c.noahmp_ext:>3}  {c.name:<28} parent IGBP {c.igbp} ({IGBP20[c.igbp]})")
        lines.append(f"!      rationale: {c.rationale}")
    lines += ["", "! Parameters to calibrate per new category:"]
    lines += [f"!   {p}" for p in MPTABLE_PARAMS]
    return "\n".join(lines) + "\n"
