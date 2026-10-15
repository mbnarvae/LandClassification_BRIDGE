# LandClassification_BRIDGE

**Ecohydrologically meaningful land classification for the tropical Andes and the Andes–Amazon transition, built on geospatial foundation models, designed for Noah‑MP.**

This repository is the land‑classification component of **BRIDGE**. Its goal is a *global‑capable* classification pipeline, developed and validated over Andean test beds (páramo, puna, bofedales, montane cloud forest, yungas and the lowland Amazon edge), whose output can be ingested by the **Noah‑MP** land surface model. The scientific question:

> Does an ecosystem‑aware classification of high‑mountain and transition landscapes improve Noah‑MP's simulation of the water and energy budgets compared with the standard MODIS/IGBP or USGS land‑use maps?

Everything here is open source (Apache‑2.0) and designed to be reproducible.

---

## Pipeline at a glance

```
 ┌──────────────┐   ┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌────────────────┐
 │ 1. Data      │──▶│ 2. Features       │──▶│ 3. Classification    │──▶│ 4. Evaluation    │──▶│ 5. Noah‑MP     │
 │ HLS / S2 /S1 │   │ spectral indices  │   │ baselines (RF/XGB)   │   │ spatial CV       │   │ crosswalk to   │
 │ DEM, climate │   │ terrain (DEM)     │   │ Prithvi‑EO‑2.0       │   │ macro‑F1 / mIoU  │   │ IGBP‑20 + new  │
 │ labels       │   │ climate/ecohydro  │   │ LandSegmenter, …     │   │ transition zones │   │ Andean classes │
 └──────────────┘   └───────────────────┘   │ fine‑tuning (LoRA…)  │   └──────────────────┘   │ → LU_INDEX     │
                                            └──────────────────────┘                          └────────────────┘
                         ▲                              │
                         └──── autoresearch loop ◀──────┘   (agent‑driven experiment search, see autoresearch/)
```

1. **Data** – Harmonized Landsat‑Sentinel (HLS, the native input of Prithvi), Sentinel‑1/2, Copernicus DEM GLO‑30, climate (CHELSA, TerraClimate, ERA5‑Land, CHIRPS), reference labels (field plots, MapBiomas, ESA WorldCover, regional ecosystem maps).
2. **Features** – spectral indices, terrain (elevation, slope, aspect, TPI, TWI), climate & ecohydrological descriptors (aridity, cloud‑immersion proxies, seasonality). Follows the high‑mountain ecosystem pipelines of Becker et al. and related work (see [`docs/science/references.md`](docs/science/references.md)).
3. **Classification** – classical baselines + geospatial foundation models (Prithvi‑EO‑2.0 via TerraTorch, LandSegmenter, others), frozen‑encoder probing and fine‑tuning.
4. **Evaluation** – spatially blocked cross‑validation, per‑class and transition‑zone metrics, transferability between test beds.
5. **Noah‑MP integration** – hierarchical legend → IGBP‑20 crosswalk *and* an extended legend with Andean classes (new `MPTABLE.TBL` parameter sets), exported as `LU_INDEX` for WRF/HRLDAS geo grids.

## Repository layout

```
├── autoresearch/        Agent-driven experiment loop (program.md, prepare.py, train.py)
├── configs/             YAML configs: default, sites (test beds), models, legend + Noah-MP crosswalk
├── data/                Local data (git-ignored) – raw / interim / processed / labels / external
├── docs/                Science background, pipeline, legend, Noah-MP integration, data sources
├── experiments/         Experiment log & results summaries (lightweight, committed)
├── models_store/        Local checkpoints (git-ignored)
├── notebooks/           Exploratory notebooks
├── scripts/             CLI-style helper scripts (downloads, chipping, export)
├── src/bridge_lc/       Python package
│   ├── data/            AOIs, STAC search, chip datasets, label handling
│   ├── features/        Spectral indices, terrain, climate/ecohydrology features
│   ├── models/          Model registry: baselines, Prithvi, LandSegmenter
│   ├── training/        Training loops / fine-tuning utilities
│   ├── evaluation/      Metrics & spatial cross-validation
│   └── noahmp/          Legend crosswalk + LU_INDEX export
├── tests/               Unit tests (pytest)
└── .claude/agents/      Specialized Claude Code sub-agents for this project
```

## Quick start

```bash
# 1. Environment (uv recommended; conda alternative in environment.yml for GDAL-heavy setups)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"            # core + tests
uv pip install -e ".[geo,fm]"         # geospatial stack + foundation models (torch, terratorch)

# 2. Sanity checks
make test
bridge-lc info                         # prints legend, sites and crosswalk summary

# 3. Smoke-test the autoresearch loop on synthetic data (no downloads needed)
python autoresearch/prepare.py --synthetic
python autoresearch/train.py
```

## Test beds (initial)

| ID | Region | Why |
|----|--------|-----|
| `paramo_ec` | Northern Andes páramo (Ecuador) | Páramo–cloud forest ecotone, peat soils, water towers |
| `puna_pe` | Central Andes puna & bofedales (Peru) | Dry puna, high‑Andean wetlands, glacier retreat |
| `yungas_bo` | Bolivian Yungas → Amazon foreland | Steep elevation gradient, montane forest to lowland transition |

Bounding boxes in [`configs/sites/`](configs/sites/) are **placeholders** — refine them to your field sites.

## Autoresearch

`autoresearch/` adapts the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) pattern: an AI agent repeatedly edits `train.py`, runs a fixed‑budget experiment, and keeps only changes that improve a fixed validation metric (spatially held‑out macro‑F1). `prepare.py` (data + evaluation) is frozen so the metric can't be gamed. See [`autoresearch/README.md`](autoresearch/README.md).

## Claude Code agents

Project sub‑agents live in [`.claude/agents/`](.claude/agents/): `geodata-engineer`, `fm-modeler`, `ecohydrology-reviewer`, `noahmp-integrator`, `autoresearch-runner`, `literature-scout`. Project conventions for AI assistants are in [`CLAUDE.md`](CLAUDE.md).

## Citing

See [`CITATION.cff`](CITATION.cff). Please also cite the underlying datasets and models listed in [`docs/science/references.md`](docs/science/references.md).

## License

Code: Apache‑2.0. Data products derived from third‑party sources keep their original licenses (see [`docs/data/data_sources.md`](docs/data/data_sources.md)).
