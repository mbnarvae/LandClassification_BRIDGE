# Colab runs

Smoke tests on a Colab GPU before moving to the server. Code comes from this repo (cloned by the notebook);
data and results live on Google Drive under `MyDrive/BRIDGE_colab/`.

## EXP004 — Antisana + Zhurucay, several foundation models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbnarvae/LandClassification_BRIDGE/blob/colab-exp004/colab/exp004_multi_fm_smoke.ipynb)

EXP003 (HLS 2023 dry + wet → frozen encoder → 480 m patch embeddings → k-means vs MAATE) repeated for
Prithvi-EO-2.0 300M / 600M, TerraMind v1 small / base and DOFA base / large (optional: Satlas Swin-B Landsat,
SSL4EO-L ResNet-50). Config: `experiments/exp004_ec_sites_multi_fm/experiment.yaml`; script:
`scripts/exp004_multi_fm_smoke.py`; encoders: `src/bridge_lc/models/embeddings.py`.

1. **On the Mac** (once): `bash colab/pack_inputs.sh` → `data/colab/exp003_inputs.zip` (EXP003 HLS composites +
   clipped MAATE references for both sites + EXP003 scores).
2. **Upload** the zip to Google Drive → `MyDrive/BRIDGE_colab/exp003_inputs.zip`.
3. **Open the notebook** with the badge, *Runtime → Change runtime type → T4 GPU*, *Run all*.
   Drive asks for permission once.
4. **Results** appear in `MyDrive/BRIDGE_colab/exp004/` (`summary.csv`, `scores.csv`, `figures/`, embeddings).

Notes
- Every step skips what is already on Drive: if the session drops, run all again.
- The notebook first checks that Prithvi 300M reproduces the EXP003 scores (verified locally on MPS: identical
  embeddings and scores).
- Not included: THOR and OlmoEarth (separate environments, Sentinel-2 input), SpectralEarth (EnMAP).
- Same thing without Colab: `python scripts/exp004_multi_fm_smoke.py all` (CUDA, MPS or CPU).
