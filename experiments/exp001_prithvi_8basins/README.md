# EXP001: Prithvi-EO-2.0 fine-tuning over 8 Andean basins

**Status:** pipeline built and smoke-tested on fake data. **Waiting on:** the basins shapefile and the choice of label source.

## Questions
1. **Q1:** Does a fine-tuned geospatial foundation model (Prithvi-EO-2.0) map Andean ecosystems better than
   (a) a plain UNet trained from scratch and (b) a Random Forest on spectral, terrain and climate features?
2. **Q2:** Does it **transfer to an unseen basin** (leave-one-basin-out) better than the baselines?
   This is the first test of the "global algorithm" goal.
3. **Q3 (diagnostic):** Which classes and transitions fail? Examples: páramo vs. humid puna, cloud forest vs. premontane
   forest, bofedal vs. grassland. These confusions matter most for Noah-MP.

## Design decisions (and why)
| Decision | Choice | Reason |
|----------|--------|--------|
| Input | HLS v2 S30+L30, 6 bands, **2 seasonal median composites** (dry, wet) | Prithvi's pretraining sensor and bands. Seasonality separates páramo, puna and bofedales, and composites beat single dates in the cloudy Andes |
| Backbone | `prithvi_eo_v2_300_tl` (600M-TL later) | TL variants beat non-TL ones. The 300M model fits a single GPU or a Mac |
| Adaptation | **LoRA r=16** on Q/V + MLP (main), full fine-tuning (comparison) | LoRA matches full fine-tuning (68.0 vs 68.1 mIoU) and transfers better to unseen regions. A frozen probe is far worse (51.6) [arXiv 2504.17397] |
| Decoder | UNet `[512,256,128,64]` | Beats UperNet for Prithvi in the PEFT study. UperNet also crashes on Apple MPS |
| Normalisation | Prithvi pretraining mean/std on reflectance ×10000 | Same scale as our int16 chips |
| Time/location coords | Off (`backbone_coords_encoding: []`) | The generic datamodule does not provide them. Pretraining used coordinate dropout |
| Loss | Cross-entropy, ignore 0, inverse-√frequency class weights | Rare classes (bofedales, glaciers, Polylepis) |
| Optimiser | AdamW lr 1e-4, wd 0.1, ReduceLROnPlateau, early stopping on val macro-F1 | Official TerraTorch / Prithvi multi-temporal examples |
| Split | Spatial blocks (3×3 chips) **within each basin** → 70/15/15. Plus **leave-one-basin-out** (LOBO) | Random pixel splits inflate scores. LOBO tests transfer |

## Runs
| Run | Model | Trainable | Hardware |
|-----|-------|-----------|----------|
| A | UNet, ResNet-50 encoder from scratch, 12 stacked bands | 44 M | Mac (MPS) OK |
| **B** | **Prithvi-EO-2.0 300M-TL + LoRA + UNet** | 51 M (5.5 M LoRA) | Mac (MPS) OK, GPU faster |
| C | Prithvi-EO-2.0 300M-TL full fine-tune + UNet | 349 M | GPU with ≥24 GB |
| RF | Random Forest, pixel features (`bridge_lc.models.baselines`) | — | CPU |

Then LOBO for the best Prithvi run and for A: 8 folds each, one basin held out per fold.

## How to run
```bash
# 0. Put the shapefile in data/external/basins/ and set basins.path / id_field in experiment.yaml.
#    Choose labels (labels.source / labels.path).
# 1. Earthdata login for HLS: ~/.netrc with  machine urs.earthdata.nasa.gov login <user> password <pw>
python scripts/exp001_build_dataset.py --step composites      # slow: downloads and composites HLS per basin
python scripts/exp001_build_dataset.py --step labels
python scripts/exp001_build_dataset.py --step chips           # chips, splits, class_counts.csv, class_weights.txt

# 2. Train and test (from the repo root)
experiments/exp001_prithvi_8basins/run_exp001.sh real a
experiments/exp001_prithvi_8basins/run_exp001.sh real b
experiments/exp001_prithvi_8basins/run_exp001.sh real b <basin_id>   # LOBO fold

# Wiring test without data (about 2 min on a Mac)
experiments/exp001_prithvi_8basins/run_exp001.sh smoke b
```
Logs and checkpoints go to `models_store/exp001/<run>/` (git-ignored). Record conclusions in `../LOG.md`.

## Files
- `experiment.yaml`: data preparation (basins, seasons, chips, split)
- `base.yaml`: shared TerraTorch config. `run_*.yaml`: one per run. `smoke.yaml`: fake-data overlay
- `run_exp001.sh`: fit, then test with the best checkpoint. Injects class weights and LOBO splits
- `../../scripts/exp001_build_dataset.py`: builds the dataset. `../../scripts/make_fake_chips.py`: fake chips

## Open issues and caveats
- **Labels decide everything.** Coarse reference maps (WorldCover) can't separate páramo from puna. See the label options in `experiment.yaml`.
- **Season windows are generic.** The northern Andes (Ecuador/Colombia) are bimodal and the central Andes are unimodal. Adjust per basin if needed.
- **Persistent cloud** (east-slope cloud forest): check the `*_nclear.tif` maps. Pixels with fewer than 2 clear observations become nodata.
- **Terrain:** illumination effects on steep slopes are not corrected yet. Compare against a corrected input later.
- **Metric:** TerraTorch's `val/F1_Score` is macro over all 19 classes. Recompute the final numbers with
  `bridge_lc.evaluation` (classes present only), per basin and per class.
- **Next (EXP002):** 600M-TL; late fusion with DEM and climate; temporal/location coordinates; 3–4 frames.
