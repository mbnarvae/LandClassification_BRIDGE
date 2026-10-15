# CLAUDE.md — conventions for AI assistants

This repo is the land-classification module of the BRIDGE project (Andes–Amazon ecohydrology, Noah-MP).
Read `README.md` and `docs/science/pipeline.md` first.

## Ground rules
- **Science first.** Any change to the legend (`configs/legend/`) or the Noah-MP crosswalk must be justified
  in `docs/science/legend.md` with an ecohydrological rationale. Run the `ecohydrology-reviewer` agent on such changes.
- **Never fabricate citations.** If a reference is uncertain, add it to `docs/science/references.md` marked `TODO: verify`.
- **Spatial validation only.** Never evaluate with random pixel splits — use spatial blocks (`bridge_lc.evaluation.spatial_cv`).
  Spatial autocorrelation makes random splits wildly optimistic.
- **Data never goes in git.** Raw/processed rasters live in `data/` (git-ignored). Document every new source in
  `docs/data/data_sources.md` with license and access method.
- **Configs over constants.** Site extents, bands, model hyper-parameters belong in `configs/*.yaml`.

## Autoresearch
- In `autoresearch/`, the agent may edit **only `train.py`**. `prepare.py` defines the data split and the metric and is frozen.
- Follow `autoresearch/program.md` exactly when running the loop. Log every run in `autoresearch/results.tsv`.

## Code
- Package: `src/bridge_lc`. Python ≥3.10, type hints, ruff (line length 110).
- Heavy deps (torch, terratorch, rasterio) are optional extras: import them lazily inside functions so core tests run without them.
- Tests: `make test`. Add a test for any new pure function (legend, crosswalk, metrics, indices).

## Common commands
```bash
make install     # uv editable install with dev extras
make test        # pytest
make lint        # ruff
bridge-lc info   # summary of legend / sites / crosswalk
python autoresearch/prepare.py --synthetic && python autoresearch/train.py
```
