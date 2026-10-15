# Roadmap

## Phase 0 — Foundations (this scaffold)
- [x] Repo structure, package, configs, CI, agents, autoresearch loop
- [ ] Verify and complete references (Becker et al., LandSegmenter)
- [ ] Refine site extents and seasons with field knowledge

## Phase 1 — Data
- [ ] HLS/S2 seasonal composites per test bed
- [ ] DEM + terrain + climate stacks on the same grid
- [ ] Label harmonisation (field plots + MapBiomas + WorldCover + national maps)
- [ ] Chipping → `data/processed/chips/`

## Phase 2 — Baselines
- [ ] RF/XGBoost with spectral + terrain + climate features (Becker-style pipeline)
- [ ] Spatial CV + leave-one-site-out

## Phase 3 — Foundation models
- [ ] Prithvi-EO-2.0 frozen probe → LoRA → full fine-tune
- [ ] LandSegmenter and other FMs
- [ ] Late fusion with DEM/climate
- [ ] Autoresearch runs per split

## Phase 4 — Noah-MP
- [ ] IGBP crosswalk run (B-IGBP) vs CTRL
- [ ] Extended categories + MPTABLE parameters (B-EXT)
- [ ] Evaluation vs streamflow / ET / soil moisture

## Phase 5 — Generalisation & release
- [ ] Apply to other mountain regions; level1 global product
- [ ] Release maps (Zenodo DOI), model weights (HF hub), paper
