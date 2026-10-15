# Experiment log

Human-readable summary of experiments (manual and autoresearch). Newest first.
Per-run numbers live in `autoresearch/results.tsv`; keep this file to conclusions and decisions.

## 2026-10-08 — exp001_prithvi_8basins (setup)
- Pipeline built and smoke-tested on fake chips (runs A/B/C: fit + test OK on Apple MPS).
- Verified TerraTorch 1.2.11 API: `backbone_*` kwargs, native LoRA via `peft_config`, UNetDecoder,
  `expand_temporal_dimension` + Flatten/Unflatten temporal transforms for augmentation.
- Literature: LoRA ≈ full FT and transfers better; frozen probe much worse; UNet baselines are strong.
  No published Prithvi work in the Andes found.
- **Pending:** basins shapefile, label source decision.

## Template
### YYYY-MM-DD — <tag>
- **Data / split:** (synthetic | spatial blocks | holdout site X), cache built from commit …
- **Best:** val_macro_f1 = … (commit …)
- **What worked:**
- **What didn't:**
- **Ecological sanity check:** (confusions that matter for Noah-MP, e.g. páramo ↔ puna, cloud forest ↔ premontane)
- **Next / questions:**
