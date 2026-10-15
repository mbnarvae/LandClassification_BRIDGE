# Contributing

Contributions are welcome — code, ecological expertise, labels and field data alike.

1. Open an issue first (templates: bug, experiment, legend change, feature).
2. Fork / branch from `main` (`feat/…`, `fix/…`, `science/…`, `autoresearch/<tag>`).
3. `make install && pre-commit install`.
4. Keep PRs focused; add tests for new pure functions; run `make lint test`.
5. Legend or crosswalk changes need an ecohydrological rationale in `docs/science/legend.md`.
6. Never commit data, checkpoints or credentials. Register new data sources in `docs/data/data_sources.md`.
7. Report model results on a **spatial** split and state which one.

By contributing you agree your contributions are licensed under Apache-2.0.
