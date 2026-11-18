"""Patch-grid alignment of frozen encoder outputs (EXP004). Needs torch; skipped without it."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bridge_lc.models.embeddings import EMBED_SPECS, norm_stats, to_patch_grid  # noqa: E402


def test_prithvi_tokens_average_frames_and_drop_cls():
    B, T, g, D = 2, 2, 14, 8
    tok = torch.arange(T, dtype=torch.float32).repeat_interleave(g * g)[None, :, None].expand(B, -1, D)
    t = torch.cat([torch.full((B, 1, D), 99.0), tok], 1)  # class token first
    out = to_patch_grid(t, "prithvi", g, n_frames=T)
    assert out.shape == (B, g, g, D)
    assert torch.allclose(out, torch.full_like(out, 0.5))  # mean of frame 0 and frame 1, class token ignored


def test_14px_patches_pooled_to_16px_grid():
    t = torch.randn(1, 1 + 2 * 16 * 16, 4)  # Prithvi 600M: 224 / 14 = 16 tokens per side
    assert to_patch_grid(t, "prithvi", 14, n_frames=2).shape == (1, 14, 14, 4)


def test_channels_last_and_strided_maps():
    assert to_patch_grid(torch.randn(3, 7, 7, 5), "hwc", 14).shape == (3, 14, 14, 5)
    assert to_patch_grid(torch.randn(3, 5, 28, 28), "chw", 14).shape == (3, 14, 14, 5)


def test_vit_cls_token_count_checked():
    with pytest.raises(ValueError):
        to_patch_grid(torch.randn(1, 1 + 10, 4), "vit_cls", 14)


def test_prithvi_norm_stats_match_hls_bands():
    mean, std = norm_stats("prithvi_eo2_300_tl")
    assert mean.shape == std.shape == (6,) and np.all(std > 0)
    assert all(spec["norm"] in ("prithvi", "terramind") for spec in EMBED_SPECS.values())
