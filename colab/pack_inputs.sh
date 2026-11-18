#!/usr/bin/env bash
# Pack the EXP003 inputs that the Colab notebook needs (run on your Mac, from the repo root):
#   bash colab/pack_inputs.sh
# → data/colab/exp003_inputs.zip  (HLS 2023 composites + clipped MAATE references for both sites,
#   plus the EXP003 scores for the sanity check). Upload the zip to Google Drive, folder MyDrive/BRIDGE_colab/.
set -euo pipefail
SRC=data/processed/exp003
OUT=data/colab/exp003_inputs.zip
mkdir -p data/colab
rm -f "$OUT"
for f in "$SRC"/ec_*/hls_2023.tif "$SRC"/ec_*/references.gpkg "$SRC"/scores.csv; do
  [[ -f $f ]] || { echo "missing $f — run: python scripts/exp003_ec_sites_smoke.py refs hls" >&2; exit 1; }
done
(cd "$SRC" && zip -r -9 "$OLDPWD/$OUT" ec_*/hls_2023.tif ec_*/references.gpkg scores.csv)
ls -lh "$OUT"
echo "Upload $OUT to Google Drive → MyDrive/BRIDGE_colab/exp003_inputs.zip"
