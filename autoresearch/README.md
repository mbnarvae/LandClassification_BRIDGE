# autoresearch

Agent-driven experiment loop, adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
for spatial land classification.

| File | Who edits it | Role |
|------|--------------|------|
| `program.md` | **human** | Instructions for the agent (goal, rules, loop, idea bank) — this is where you "program" the research |
| `prepare.py` | **nobody during a run** | Builds the fixed spatial train/val split, defines `evaluate()` (macro-F1) and `TIME_BUDGET_S` |
| `train.py` | **agent** | Model + features + training; must end with `prepare.evaluate(pred)` |
| `results.tsv` | agent (append) | One row per experiment |
| `cache/`, `run.log` | generated | git-ignored |

## Why these choices
- **Spatial split, frozen.** Random pixel splits inflate accuracy massively in spatially autocorrelated
  landscapes; freezing the split stops the agent from "winning" by leakage.
- **Macro-F1.** Rare but hydrologically critical classes (bofedales, glaciers, cloud forest) count as much as
  extensive lowland forest.
- **Fixed time budget.** Makes runs comparable and lets the agent do ~12 experiments/hour.
- **Held-out site mode** (`--holdout-site`) tests the *global* ambition: does what works in Ecuador transfer to Bolivia?

## Run it

```bash
# smoke test, no data needed
python autoresearch/prepare.py --synthetic
python autoresearch/train.py

# with an agent (Claude Code), from the repo root:
claude "Read autoresearch/program.md and start a new autoresearch run with tag oct06-baseline"
```
Or use the project sub-agent: *"use the autoresearch-runner agent to start a run"*.

When real chips exist in `data/processed/chips/`, rebuild the cache with `python autoresearch/prepare.py`
(spatial blocks) or `--holdout-site <id>`. **Start a new branch/tag whenever the cache changes** — scores
across different splits are not comparable.
