"""Prithvi-EO-2.0 segmentation models via TerraTorch (requires the ``fm`` extra).

Verified against terratorch 1.2.11 (repo moved to https://github.com/torchgeo/terratorch).
For training, prefer the YAML configs in experiments/ with ``terratorch fit``; this builder is for
notebooks, autoresearch and inference.
"""

from __future__ import annotations

from bridge_lc.models.registry import register

HLS_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
# Pretraining normalisation, HLS reflectance ×10000.
PRITHVI_V2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_V2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
# Transformer blocks feeding the decoder.
SELECT_INDICES = {"300": [5, 11, 17, 23], "600": [7, 15, 23, 31], "100": [2, 5, 8, 11], "tiny": [2, 5, 8, 11]}

LORA_DEFAULT = {
    "method": "LORA",
    "replace_qkv": "qkv",
    "peft_config_kwargs": {
        "target_modules": ["qkv.q_linear", "qkv.v_linear", "mlp.fc1", "mlp.fc2"],
        "r": 16,
        "lora_alpha": 16,
    },
}


@register("prithvi_eo2")
def prithvi_eo2(
    num_classes: int,
    backbone: str = "prithvi_eo_v2_300_tl",
    bands: list[str] | None = None,
    num_frames: int = 2,
    decoder: str = "UNetDecoder",
    decoder_channels: list[int] | None = None,
    strategy: str = "lora",
    lora: dict | None = None,
    pretrained: bool = True,
    **_,
):
    """Prithvi-EO-2.0 encoder + UNet decoder for segmentation.

    Input (B, C=6, T=num_frames, H, W) normalised with PRITHVI_V2_MEAN/STD; ``.output`` is (B, K, H, W).
    ``num_classes`` includes the nodata class 0 (pass ``legend.num_classes + 1``).
    ``strategy``: 'lora' (default; ≈ full FT accuracy, better transfer), 'full_finetune', or
    'frozen_probe' (cheap diagnostic only — clearly worse).
    """
    from terratorch.models import EncoderDecoderFactory

    size = next((k for k in SELECT_INDICES if f"_{k}" in backbone), "300")
    peft = None
    if strategy == "lora":
        peft = LORA_DEFAULT if lora is None else LORA_DEFAULT | lora
    elif strategy not in ("full_finetune", "frozen_probe"):
        raise ValueError(f"Unknown strategy {strategy}")

    model = EncoderDecoderFactory().build_model(
        task="segmentation",
        backbone=backbone,
        backbone_pretrained=pretrained,
        backbone_bands=bands or HLS_BANDS,
        backbone_num_frames=num_frames,
        backbone_coords_encoding=[],
        necks=[
            {"name": "SelectIndices", "indices": SELECT_INDICES[size]},
            {"name": "ReshapeTokensToImage", "effective_time_dim": num_frames},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        decoder=decoder,
        decoder_channels=decoder_channels or [512, 256, 128, 64],
        head_dropout=0.1,
        num_classes=num_classes,
        peft_config=peft,
    )
    if strategy == "frozen_probe":
        for p in model.encoder.parameters():
            p.requires_grad = False
    return model
