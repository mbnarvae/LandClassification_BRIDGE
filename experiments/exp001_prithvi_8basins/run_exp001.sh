#!/usr/bin/env bash
# EXP001 runner (run from the repo root).
#   experiments/exp001_prithvi_8basins/run_exp001.sh real  b            # main split, run B
#   experiments/exp001_prithvi_8basins/run_exp001.sh real  b basin_03   # leave-one-basin-out: hold out basin_03
#   experiments/exp001_prithvi_8basins/run_exp001.sh smoke b            # wiring test on fake chips
set -euo pipefail
MODE=${1:?mode: real|smoke}; RUN=${2:?run: a|b|c}; LOBO=${3:-}
DIR=experiments/exp001_prithvi_8basins
RUN_CFG=$(ls $DIR/run_${RUN}_*.yaml)
ARGS=(-c $DIR/base.yaml -c "$RUN_CFG")
DATA=data/processed/exp001

if [[ $MODE == smoke ]]; then
  [[ -d data/processed/fake_exp001 ]] || python scripts/make_fake_chips.py
  ARGS+=(-c $DIR/smoke.yaml)
else
  if [[ -f $DATA/class_weights.txt ]]; then
    ARGS+=(--model.init_args.class_weights "[$(cat $DATA/class_weights.txt)]")
  fi
  if [[ -n $LOBO ]]; then
    S=$DATA/splits/lobo/$LOBO
    NAME=$(basename "$RUN_CFG" .yaml)_lobo_$LOBO
    ARGS+=(--data.init_args.train_split $S/train.txt --data.init_args.val_split $S/val.txt
           --data.init_args.test_split $S/test.txt --trainer.logger.init_args.name "$NAME")
  fi
fi

terratorch fit "${ARGS[@]}"
# Evaluate the best checkpoint on the held-out test split.
ROOT=$( [[ $MODE == smoke ]] && echo models_store/exp001_smoke || echo models_store/exp001 )
CKPT=$(ls -t $(find "$ROOT" -name "best-*.ckpt" -o -name "last.ckpt" | head -50) 2>/dev/null | head -1)
terratorch test "${ARGS[@]}" --ckpt_path "$CKPT"
