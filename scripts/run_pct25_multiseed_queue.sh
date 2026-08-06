#!/usr/bin/env bash
# pct25 multiseed finetune for one seed (base+psmp), then optional val50 eval.
# Env: SEED=43 GPU=4 DO_EVAL=1
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
PAPER="$ROOT/nar_paper"
SEED="${SEED:-43}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
DO_EVAL="${DO_EVAL:-1}"
MARK="$PAPER/data/pipeline_markers"
LOG="$PAPER/data/multiseed_pct25_seed${SEED}.log"

mkdir -p "$MARK"
: >> "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "pct25 multiseed SEED=$SEED GPU=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" SEED="$SEED" PCTS=pct25 INITS="base psmp" \
  bash "$PAPER/scripts/run_multiseed_lowdata_finetune.sh" 2>&1 | tee -a "$LOG"

if [[ "$DO_EVAL" == "1" ]]; then
  log "eval pct25 seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU" SEED="$SEED" PCT=pct25 INITS="base psmp" \
    bash "$PAPER/scripts/eval_multiseed_val50.sh" 2>&1 | tee -a "$LOG"
  touch "$MARK/multiseed_eval_pct25_seed${SEED}.ok"
fi

touch "$MARK/multiseed_finetune_pct25_seed${SEED}.ok"
log "DONE pct25 multiseed seed=$SEED"
