# Experiment log

Human-readable summary of experiments (manual and autoresearch). Newest first.
Per-run numbers live in `autoresearch/results.tsv`; keep this file to conclusions and decisions.

## 2026-11-18 — exp004_ec_sites_multi_fm (setup, Colab)
- EXP003 protocol unchanged (same patches, PCA, k, scoring) for six frozen encoders: Prithvi-EO-2.0 300M/600M-TL,
  TerraMind v1 small/base, DOFA base/large (optional: Satlas Swin-B Landsat, SSL4EO-L ResNet-50).
  `bridge_lc.models.embeddings` puts every encoder on the same 14×14 patch grid per 224-px tile (single-image
  encoders: seasons encoded separately and averaged; Prithvi 600M's 14-px patches and Swin/ResNet stages
  average-pooled). TerraMind uses its own S2-L2A statistics, the rest the HLS/Prithvi statistics.
- Verified locally (MPS): Prithvi 300M embeddings identical to EXP003 and all scores reproduced; full pipeline
  (embed → analyze → figures) run with Prithvi 300M + TerraMind small. All eight encoders build with random weights.
- Run on Colab: `colab/exp004_multi_fm_smoke.ipynb` (inputs zipped by `colab/pack_inputs.sh`, kept on Drive).
- **Pending:** the Colab run with all six models; results to be logged here.

## 2026-11-17 — exp003_ec_sites_smoke (Antisana + Zhurucay, unsupervised)
- **Data:** two rectangles drawn by Manuel (`Organized_Database/Ecuador/rectangles_landuse`; bbox of each shape).
  HLS 2023 dry (Jun–Sep) + wet (Dec 2022–Mar 2023) medians, 30 m, cached per granule (LP DAAC reads fail
  transiently, so `scripts/exp003_ec_sites_smoke.py` loads one granule at a time with retries). Clear in both
  seasons: Antisana 76 %, Zhurucay 92 %. References clipped: MAATE ecosystems, land cover 2000–2024 (all 8
  editions, labels repaired: several DBFs store UTF-8 as Latin-1), Ramsar, 2025 soil points (Antisana only).
- **Method:** frozen Prithvi-EO-2.0 300M-TL, patch tokens (16 px = 480 m, mean over the 2 frames), layers 11 and
  23 → PCA 50 → k-means k ∈ {8, 12, 16, 24}. Baseline: k-means on patch-mean spectra + NDVI (14 features).
  "Recognised" = class F1 ≥ 0.5 after mapping each cluster to its majority class (pure patches only, ≥ 60 %).
- **Result:** Prithvi recognises **1 of 24** MAATE ecosystems (herbazal del páramo, F1 0.75) and **4–5 of 11**
  land-cover classes (páramo, mosaico agropecuario, bosque nativo, área poblada, glaciar at layer 11).
  The spectral baseline does better: **4 of 24** ecosystems (+ arbustal desértico, herbazal ultrahúmedo subnival,
  bosque montano norte oriental) and 5 of 11 land-cover classes; NMI 0.28 vs 0.15.
- **Caveats:** (1) Prithvi clusters are noisy and one cluster traces cloud-gap edges → filled nodata leaks
  through attention inside each 224-px tile; (2) MAATE ecosystems differ by elevation, slope side and region
  (norte/sur, oriental/occidental), which one year of 6-band spectra cannot see, so the ceiling is low for any
  encoder; (3) 480 m patches cannot resolve small wetlands.
- **Wetlands:** none of the 181 humedal soil points fall in the Ramsar layer (only Ñucanchi Turupamba is in the
  rectangle); MAATE puts 166 of them in "herbazal del páramo". Prithvi k=8 spreads them over 5 clusters,
  mixed with pajonal → no separate wetland signal at 480 m.
- **Next:** mask cloud gaps out of attention (or pick only fully clear tiles), pixel-level features (upsampled
  tokens or decoder) instead of 480 m patches, add DEM, then the labelled fine-tune; AVIRIS/UAVSAR for Antisana.

## 2026-11-12 — exp002_fm_benchmark (setup)
- Eight encoders behind one interface (`bridge_lc.models.terratorch_fm`): Prithvi-EO-2.0 300M/600M-TL, TerraMind v1
  small/base, DOFA base/large, SatlasPretrain Swin-B (Landsat), SSL4EO-L ResNet-50. Same chips, UNet decoder, loss and
  split as EXP001; two protocols (frozen encoder, full fine-tune).
- Verified with terratorch 1.2.11: all eight build and run fit + test on fake chips with random weights.
  Non-Prithvi encoders take each seasonal frame separately (`backbone_use_temporal`, concat pooling).
- LandSegmenter checked (arXiv 2511.08156, weights on Hugging Face): external repo, CUDA only, no licence stated → not
  integrated yet.
- National land-cover / ecosystem maps for CO, EC and PE are now local (see `docs/data/reference_sources.md`).
- Pretrained weights load for the 6 HLS bands for TerraMind small/base, DOFA base and Satlas Swin-B (compared
  against random-init twins; only a minority of tensors coincide, presumably constant-initialised ones — not
  inspected). Not yet checked: SSL4EO-L, DOFA large, Prithvi 600M.
- **Benchmark set fixed by Manuel:** Prithvi, TerraMind, DOFA, OlmoEarth v1.3, THOR, SpectralEarth. The last three
  are not wired: OlmoEarth and THOR pin conflicting torch / TerraTorch versions and need Sentinel-2 chips;
  SpectralEarth needs EnMAP. Design → three input tracks on one chip grid (`docs/science/foundation_models.md`).
- Later the same day: `.venv-thor` and `.venv-olmoearth` created. **THOR** wired into FM_SPECS and smoke-tested
  (accepts the HLS bands; CPU/CUDA only because of float64; `torch.compile` off on macOS). **OlmoEarth** wrapper
  written and tested with v1.2 Tiny/Base on random Sentinel-2-shaped input (CPU and MPS). **v1.3 Base does not load**
  with olmoearth-pretrain 0.1.2.
- **Pending:** real chips (HLS composites + label decision); Sentinel-2 L2A track (OlmoEarth, and native input for
  TerraMind/THOR); OlmoEarth v1.3 via the GitHub package; SpectralEarth (EnMAP coverage check, own repo).

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
