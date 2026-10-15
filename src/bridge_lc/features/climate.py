"""Climate & ecohydrological descriptors from monthly climatologies (12, H, W).

Inputs typically come from CHELSA / TerraClimate / CHIRPS (precip) and ERA5-Land / TerraClimate (PET).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def aridity_index(precip_monthly: np.ndarray, pet_monthly: np.ndarray) -> np.ndarray:
    """UNEP aridity index P/PET (annual sums). <0.65 = drylands; páramo is typically >1."""
    return (precip_monthly.sum(0) / (pet_monthly.sum(0) + _EPS)).astype(np.float32)


def precip_seasonality(precip_monthly: np.ndarray) -> np.ndarray:
    """Coefficient of variation of monthly precipitation (BIO15-like, as a fraction)."""
    mean = precip_monthly.mean(0)
    return (precip_monthly.std(0) / (mean + _EPS)).astype(np.float32)


def seasonality_index(precip_monthly: np.ndarray) -> np.ndarray:
    """Walsh & Lawler seasonality index (0 = uniform, >1.2 = extreme seasonality)."""
    annual = precip_monthly.sum(0)
    return (np.abs(precip_monthly - annual / 12).sum(0) / (annual + _EPS)).astype(np.float32)


def dry_months(precip_monthly: np.ndarray, pet_monthly: np.ndarray) -> np.ndarray:
    """Number of months with P < PET (water-limited months)."""
    return (precip_monthly < pet_monthly).sum(0).astype(np.float32)


def cloud_frequency(cloud_mask_stack: np.ndarray) -> np.ndarray:
    """Fraction of cloudy observations (T, H, W boolean) — proxy for cloud-forest immersion."""
    return cloud_mask_stack.mean(0).astype(np.float32)
