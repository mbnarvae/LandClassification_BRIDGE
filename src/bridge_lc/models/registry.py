"""Name → builder registry so configs/autoresearch can swap models by name."""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return deco


def build_model(name: str, num_classes: int, **cfg):
    # Import builders lazily so heavy deps are only needed for the model actually used.
    from bridge_lc.models import baselines, landsegmenter, prithvi  # noqa: F401

    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes, **cfg)


def list_models() -> list[str]:
    from bridge_lc.models import baselines, landsegmenter, prithvi  # noqa: F401

    return sorted(_REGISTRY)
