#!/usr/bin/env bash
# CPU scoring + optional GPU val50 PEFT; then summary tables.
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
PAPER="$ROOT/nar_paper"
SCR="$PAPER/scripts"
PY="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
MARK="$PAPER/data/pipeline_markers"
GPU_PEFT="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$MARK"
log() { echo "[$(date -Is)] $*"; }

log "1/3 score Boltz Temporal20 (CPU)"
"$PY" "$SCR/score_boltz_temporal20.py" | tee "$PAPER/data/boost_boltz_score.log"

if [[ ! -f "$MARK/boost_peft_val50.ok" ]]; then
  log "2/3 PEFT val50 five-sample infer+score (GPU $GPU_PEFT) — may take 1-3h"
  nohup env CUDA_VISIBLE_DEVICES="$GPU_PEFT" bash "$SCR/run_peft_pairformer_val50.sh" \
    > "$ROOT/train/runs/rnp_real_lowdata_peft/peft_val50.nohup" 2>&1 &
  echo "peft_val50_pid=$!"
else
  log "2/3 PEFT val50 already done"
fi

log "3/3 build summary tables (partial if PEFT pending)"
"$PY" "$SCR/build_boost_summary_tables.py" | tee "$PAPER/data/boost_tables.log"

log "wrapup launched; re-run build_boost_summary_tables.py after PEFT finishes"
