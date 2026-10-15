"""LandSegmenter adapter — placeholder until weights/repo and input spec are confirmed.

Pattern for adding any new geospatial foundation model (Clay, SatMAE, DOFA, Galileo, AnySat, …):
1. Implement a builder returning an ``nn.Module`` mapping (B, C, [T,] H, W) → (B, num_classes, H, W).
2. Decorate it with ``@register("<name>")`` and add ``configs/models/<name>.yaml``.
3. Document weights, licence and input normalisation in docs/science/pipeline.md.
"""

from __future__ import annotations

from bridge_lc.models.registry import register


@register("landsegmenter")
def landsegmenter(num_classes: int, weights: str | None = None, **_):
    raise NotImplementedError(
        "LandSegmenter adapter not implemented yet: confirm the official repo, weights licence and "
        "band/normalisation spec, then build it here (see module docstring)."
    )
