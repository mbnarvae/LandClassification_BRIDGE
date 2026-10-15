"""Config loading: YAML files merged on top of configs/default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(*overrides: str | Path) -> dict[str, Any]:
    """Load ``configs/default.yaml`` and merge each override file in order."""
    cfg = load_yaml(CONFIG_DIR / "default.yaml")
    for path in overrides:
        cfg = deep_merge(cfg, load_yaml(path))
    return cfg
