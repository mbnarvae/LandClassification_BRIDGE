# data/ (git-ignored)

Local data only — see [docs/data/data_sources.md](../docs/data/data_sources.md) for sources and licences.

- `raw/` untouched downloads · `interim/` masked/composited · `processed/` analysis-ready stacks and `chips/`
- `labels/` harmonised reference labels (BRIDGE legend ids, 0 = nodata) · `external/` third-party maps

Chip format consumed by `autoresearch/prepare.py`: `.npz` with `x (C,H,W)`, `y (H,W)`, `origin (2,)`, `site`, `bands`.
