# autoresearch — BRIDGE land classification

You are an autonomous research agent improving an ecosystem classifier for the tropical Andes and the
Andes–Amazon transition. The output will feed the Noah-MP land surface model, so **classes that matter
hydrologically (páramo, bofedales, cloud forest, glaciers, transition forests) matter more than raw accuracy
on dominant classes** — that is why the metric is *macro*-F1.

## Setup (once per run)

1. Agree on a run tag with the human (e.g. `oct06-prithvi`). Create branch `autoresearch/<tag>` from `main`.
2. Read, in full: `autoresearch/README.md`, `autoresearch/prepare.py`, `autoresearch/train.py`,
   `configs/legend/bridge_andes_v0.yaml`, `docs/science/pipeline.md`.
3. Make sure the cache exists: `ls autoresearch/cache/dataset.npz` (if not, ask the human which
   `prepare.py` mode to run — never choose the split yourself).
4. Ensure `autoresearch/results.tsv` exists with the header row only if new.
5. Run the unmodified `train.py` once to record the **baseline**.

## Rules

- **Only edit `autoresearch/train.py`.** `prepare.py`, the cache, the split and `evaluate()` are frozen.
  You may import from `bridge_lc` (read-only) and any package already installed. Do not install packages.
- Never read or reconstruct `y_val`. Never tune on the validation set beyond the single score per run.
- Each experiment must finish within `TIME_BUDGET_S` of training (+ evaluation). Kill runs >2× budget.
- Simplicity criterion: a tiny gain that adds lots of complexity is not worth keeping; equal score with
  simpler code **is** an improvement.

## The loop — run indefinitely until the human stops you

```
LOOP:
  1. Look at results.tsv + git log; pick ONE idea (state the hypothesis in one line).
  2. Edit train.py. git commit -m "<short idea>".
  3. python autoresearch/train.py > autoresearch/run.log 2>&1
  4. grep "^val_macro_f1:" autoresearch/run.log   (empty ⇒ crash: tail -n 50 run.log, fix once or give up)
  5. Append to results.tsv:  commit  val_macro_f1  val_miou  seconds  status  description
       status ∈ {keep, discard, crash}
  6. If val_macro_f1 improved over the best kept run → keep the commit (branch advances).
     Else → git reset --hard HEAD~1.
```

Do **not** pause to ask the human whether to continue. If you run out of ideas, re-read the papers in
`docs/science/references.md`, combine near-misses, or try more radical changes.

## Idea bank (roughly in increasing cost)

- Feature engineering: more indices (NDSI for snow, EVI, red-edge if S2), seasonal differences (wet–dry),
  terrain (slope, TPI, TWI from the ELEVATION band), neighbourhood statistics/texture.
- Models: LightGBM/XGBoost, class-balanced sampling, calibrated probabilities + spatial smoothing (CRF/majority).
- Foundation models: frozen Prithvi-EO-2.0 (`bridge_lc.models.build_model("prithvi_eo2", ...)`) embeddings
  + pixel MLP / linear probe; UPerNet decoder; LoRA; multi-temporal (dry+wet) input; late fusion with DEM/climate.
- Loss: focal / Lovász / class-weighted CE; label smoothing for noisy reference maps.
- Hierarchical prediction: level1 group first, then fine class.

## Reporting

At natural milestones, write a short summary in `experiments/LOG.md` (date, tag, best score, what worked,
what didn't, open questions for the human — especially any that need ecological judgement).
