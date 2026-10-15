# Pipeline design

## Objectives
1. **Map** Andean and Andes–Amazon transition ecosystems at 10–30 m with a legend that is ecohydrologically meaningful.
2. **Generalise**: the same pipeline (inputs, features, models) must run on any mountain region; the Andean
   test beds are the development and validation ground, not a hard-coded target.
3. **Integrate** with Noah-MP and test whether the new classification improves simulated water/energy fluxes.

## Stages

### 1. Inputs
| Group | Products | Notes |
|-------|----------|-------|
| Optical | HLS v2 (L30/S30), Sentinel-2 L2A | HLS = Prithvi's pretraining input; S2 red-edge helps grassland/wetland separation |
| SAR | Sentinel-1 RTC | Cloud-independent structure/moisture; useful for cloud forest and wetlands |
| Terrain | Copernicus DEM GLO-30 | Elevation, slope, aspect, TPI, TWI; strong prior for altitudinal belts |
| Climate | CHELSA, TerraClimate, CHIRPS, ERA5-Land | Aridity, seasonality, dry months, cloud frequency |
| Labels | Field plots, MapBiomas (Amazonía + Andean countries), ESA WorldCover, national ecosystem maps | Harmonised via `configs/legend/label_mappings/` |

### 2. Pre-processing
- Cloud/shadow masking (Fmask/SCL), seasonal median composites (dry/wet per site config).
- Topographic illumination correction on steep terrain (evaluate impact; FMs may not need it).
- Co-registration of all layers on the site UTM grid; chipping (224 px for Prithvi).

### 3. Features (for classical baselines and late fusion)
Spectral indices (`features/indices.py`), terrain (`features/terrain.py`), climate (`features/climate.py`).
This mirrors the feature-engineering approach of classical high-mountain ecosystem pipelines (e.g. Becker et al.;
see references) and provides an interpretable baseline.

### 4. Models
| Tier | Model | Strategy |
|------|-------|----------|
| Baseline | Random Forest / XGBoost on pixel features | Reference to beat |
| FM probe | Prithvi-EO-2.0 (300M/600M, temporal+location) frozen + UPerNet | Cheap, tests representation quality |
| FM adapt | Prithvi-EO-2.0 + LoRA | Parameter-efficient fine-tuning |
| FM full | Full fine-tuning | Only if data volume allows |
| Other FMs | LandSegmenter, others | Same interface via `models/registry.py` |
| Fusion | FM embeddings + DEM/climate (late fusion) | Altitude/climate are key Andean priors |

### 5. Evaluation
- **Spatial** block cross-validation (`evaluation/spatial_cv.py`); optionally leave-one-site-out for transferability.
- Macro-F1 (primary), mIoU, per-class F1, confusion between *ecologically adjacent* classes.
- **Transition-zone metrics**: accuracy within buffers around ecotones (treeline, páramo–cloud forest, Andes–Amazon foothills).
- Map-level area estimates with uncertainty (stratified sampling, e.g. Olofsson-style good practices).

### 6. Noah-MP
See [noahmp_integration.md](noahmp_integration.md).

## Autoresearch
Model/feature search on a frozen spatial split is delegated to the agent loop in `autoresearch/`.
Humans own: the legend, the labels, the split, the metric, and the ecological interpretation.
