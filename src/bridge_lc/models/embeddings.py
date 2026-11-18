"""Frozen patch embeddings from several foundation models on the same HLS tiles (EXP004).

Every model sees the same input — HLS v2, 6 bands, 2 seasonal frames, 224 px tiles — and returns, for each
requested layer, one feature vector per 16×16-pixel patch (14×14 grid per tile, 480 m at 30 m pixels), so
the unsupervised analysis of EXP003 can be repeated unchanged for every encoder.

- Prithvi-EO-2.0 is multi-temporal: both frames go in together; tokens are averaged over the frames.
- The other encoders are single-image: each frame is encoded separately, then features are averaged.
- Hierarchical encoders (Swin, ResNet) are average-pooled from their own stride onto the 16-px grid.

Normalisation is per model (``EMBED_SPECS[...]["norm"]``): Prithvi/HLS statistics, or TerraMind's own
Sentinel-2 L2A pretraining statistics. DOFA, Satlas and SSL4EO-L use the HLS statistics (as in EXP002);
their own pretraining scaling differs, so treat those two reference encoders with caution.

Requires the ``fm`` extra (torch, terratorch). Verified with terratorch 1.2.11.
"""

from __future__ import annotations

import numpy as np

from bridge_lc.models.prithvi import HLS_BANDS, PRITHVI_V2_MEAN, PRITHVI_V2_STD

PATCH = 16

# kind: how outputs come back from the TerraTorch backbone
#   prithvi    list of every block, tokens (B, 1 + T·g², D) with a class token, frames together
#   terramind  list of every block, tokens (B, g², D), no class token
#   vit_cls    list at ``out_indices``, tokens (B, 1 + g², D) with a class token (DOFA)
#   chw / hwc  list at ``out_indices``, feature maps (B, C, h, w) or channels-last (B, h, w, C)
# layers: (mid, last) block for ViTs; for out_indices models, the out_indices themselves.
EMBED_SPECS: dict[str, dict] = {
    "prithvi_eo2_300_tl": {"backbone": "prithvi_eo_v2_300_tl", "kind": "prithvi", "layers": [11, 23]},
    "prithvi_eo2_600_tl": {"backbone": "prithvi_eo_v2_600_tl", "kind": "prithvi", "layers": [15, 31]},
    "terramind_v1_small": {"backbone": "terramind_v1_small", "kind": "terramind", "layers": [5, 11]},
    "terramind_v1_base": {"backbone": "terramind_v1_base", "kind": "terramind", "layers": [5, 11]},
    "dofa_base": {"backbone": "dofa_base_patch16_224", "kind": "vit_cls", "layers": [5, 11]},
    "dofa_large": {"backbone": "dofa_large_patch16_224", "kind": "vit_cls", "layers": [11, 23]},
    "satlas_swin_b_landsat": {"backbone": "satlas_swin_b_landsat_mi_ms", "kind": "hwc", "layers": [5, 7]},
    "ssl4eol_resnet50": {"backbone": "ssl4eol_resnet50_landsat_oli_sr_moco", "kind": "chw", "layers": [3, 4]},
}
for _name, _spec in EMBED_SPECS.items():
    _spec.setdefault("norm", "terramind" if _spec["kind"] == "terramind" else "prithvi")

DEFAULT_MODELS = [
    "prithvi_eo2_300_tl",
    "prithvi_eo2_600_tl",
    "terramind_v1_small",
    "terramind_v1_base",
    "dofa_base",
    "dofa_large",
]


def norm_stats(name: str, bands: list[str] = HLS_BANDS) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and std (reflectance ×10000) used to standardise the input of ``name``."""
    if EMBED_SPECS[name]["norm"] == "terramind":
        from terratorch.models.backbones.terramind.model.terramind_register import (
            PRETRAINED_BANDS,
            v1_pretraining_mean,
            v1_pretraining_std,
        )

        key = "untok_sen2l2a@224"
        idx = [PRETRAINED_BANDS[key].index(b) for b in bands]
        return (
            np.array(v1_pretraining_mean[key], "float32")[idx],
            np.array(v1_pretraining_std[key], "float32")[idx],
        )
    idx = [HLS_BANDS.index(b) for b in bands]
    return np.array(PRITHVI_V2_MEAN, "float32")[idx], np.array(PRITHVI_V2_STD, "float32")[idx]


def to_patch_grid(t, kind: str, grid: int, n_frames: int = 1):
    """One backbone output → (B, grid, grid, D) torch tensor (frames averaged for ``prithvi``).

    Encoders whose native patch grid differs from ``grid`` (Prithvi 600M uses 14-px patches; Swin and
    ResNet stages have their own strides) are average-pooled onto it.
    """
    import torch.nn.functional as F

    B = t.shape[0]
    if kind in ("prithvi", "terramind", "vit_cls"):
        tok = t if kind == "terramind" else t[:, 1:, :]
        frames = n_frames if kind == "prithvi" else 1
        g = int(round((tok.shape[1] / frames) ** 0.5))
        if frames * g * g != tok.shape[1]:
            raise ValueError(f"{kind}: {tok.shape[1]} tokens do not form {frames} square grid(s)")
        tok = tok.reshape(B, frames, g, g, -1).mean(1)  # (B, g, g, D)
        t = tok.permute(0, 3, 1, 2)
    elif kind == "hwc":
        t = t.permute(0, 3, 1, 2)
    elif kind != "chw":
        raise ValueError(f"Unknown output kind {kind}")
    if t.shape[-1] != grid or t.shape[-2] != grid:
        if t.device.type == "mps":  # MPS only pools between divisible sizes (16 → 14 is not)
            t = t.cpu()
        t = F.adaptive_avg_pool2d(t, grid)
    return t.permute(0, 2, 3, 1)


class PatchEmbedder:
    """Frozen encoder: ``(B, C, T, H, W)`` standardised tiles → ``{layer: (B, H/16, W/16, D)}``."""

    def __init__(
        self,
        name: str,
        n_frames: int = 2,
        bands: list[str] = HLS_BANDS,
        pretrained: bool = True,
        device: str = "cpu",
    ):
        from terratorch.registry import BACKBONE_REGISTRY

        if name not in EMBED_SPECS:
            raise KeyError(f"Unknown model '{name}'. Available: {sorted(EMBED_SPECS)}")
        self.name, self.spec, self.n_frames, self.device = name, EMBED_SPECS[name], n_frames, device
        self.layers = list(self.spec["layers"])
        kind, bb = self.spec["kind"], self.spec["backbone"]
        if kind == "prithvi":
            kw = {"num_frames": n_frames, "bands": list(bands), "coords_encoding": []}
        elif kind == "terramind":
            kw = {"modalities": ["S2L2A"], "bands": {"S2L2A": list(bands)}}
        else:
            kw = {"model_bands": list(bands), "out_indices": self.layers}
        self.model = BACKBONE_REGISTRY.build(bb, pretrained=pretrained, **kw).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.mean, self.std = norm_stats(name, bands)

    def standardise(self, x: np.ndarray, fill: float) -> np.ndarray:
        """``(C, T, H, W)`` reflectance ×10000 → standardised; ``fill`` pixels become the band mean (0)."""
        m, s = self.mean[:, None, None, None], self.std[:, None, None, None]
        return ((np.where(x == fill, m, x) - m) / s).astype("float32")

    def __call__(self, xb, amp: bool = False) -> dict:
        import torch

        B, C, T, H, W = xb.shape
        grid = H // PATCH
        kind = self.spec["kind"]
        dtype = torch.float16 if self.device.startswith("cuda") else torch.bfloat16
        with torch.no_grad(), torch.autocast(self.device.split(":")[0], dtype=dtype, enabled=amp):
            if kind == "prithvi":
                outs = self.model(xb)
                picked = {ly: outs[ly] for ly in self.layers}
            else:  # single-image encoders: frames go through as extra batch items
                xs = xb.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                outs = self.model(xs)
                if kind == "terramind":
                    picked = {ly: outs[ly] for ly in self.layers}
                else:
                    picked = dict(zip(self.layers, outs, strict=True))
        res = {}
        for ly, t in picked.items():
            g = to_patch_grid(t.float(), kind, grid, n_frames=T)
            if kind != "prithvi":
                g = g.reshape(B, T, grid, grid, -1).mean(1)
            res[ly] = g
        return res


def _selfcheck(names: list[str]) -> None:
    """Build each encoder with random weights and check it returns the 14×14 patch grid (no download)."""
    import torch

    x = torch.zeros(1, len(HLS_BANDS), 2, 224, 224)
    for n in names:
        out = PatchEmbedder(n, pretrained=False)(x)
        print(n, {ly: tuple(v.shape) for ly, v in out.items()}, flush=True)


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    _selfcheck(list(EMBED_SPECS) if arg == "all" else arg.split(","))
