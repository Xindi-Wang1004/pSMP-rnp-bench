#!/usr/bin/env bash
# Third PEFT recipe through frozen harness: pairformer-only pct10 from base
# (drops diffusion_module vs production recipe). Boltz has no public train CLI.
set -eo pipefail

FUSAI="/home/wangxindi/RNA_Protein/fusai"
PD="${PROTENIX_DATA:-$FUSAI/../protenix_data}"
DATA_DIR="$FUSAI/train/data/rnp_real/lowdata/pct10"
RUN_DIR="${FUSAI}/train/runs/rnp_real_lowdata_peft/pct10_base_pairformer_only"
INIT_CKPT="$PD/checkpoint/protenix_base_default_v1.0.0.pt"
GPU="${CUDA_VISIBLE_DEVICES:-7}"
MAX_STEPS="${MAX_STEPS:-400}"
TAG="pct10_base_pairformer_only"

mkdir -p "$RUN_DIR" "$FUSAI/nar_paper/data/pipeline_markers"
source /home/wangxindi/miniconda3/etc/profile.d/conda.sh
conda activate protenix
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PROTENIX_DATA="$PD"
export CUDA_VISIBLE_DEVICES="$GPU"

if compgen -G "$RUN_DIR"/*/checkpoints/*_ema_0.999.pt > /dev/null; then
  echo "already done"
  date -Is > "$FUSAI/nar_paper/data/pipeline_markers/boost_peft_pairformer.ok"
  exit 0
fi

# Delete diffusion_module line entirely (do NOT comment — breaks bash line continuations)
PATCHED="$FUSAI/train/.run_finetune_pairformer_only.sh"
grep -v 'finetune_params_with_substring "diffusion_module"' \
  "$FUSAI/train/run_finetune_rnp_real.sh" > "$PATCHED"
chmod +x "$PATCHED"
# sanity: pairformer still present; diffusion gone
grep -q 'pairformer' "$PATCHED"
! grep -q 'diffusion_module' "$PATCHED"

LOG="$RUN_DIR/finetune.log"
echo "[$(date -Is)] pairformer-only FT gpu=$GPU steps=$MAX_STEPS" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" \
  PROTENIX_DATA="$PD" \
  INIT_CKPT="$INIT_CKPT" \
  DATA_DIR="$DATA_DIR" \
  RUN_DIR="$RUN_DIR" \
  RUN_NAME="pct10_from_base_pairformer_only" \
  TAG="$TAG" \
  MAX_STEPS="$MAX_STEPS" \
  bash "$PATCHED" 2>&1 | tee -a "$LOG"

rm -f "$PATCHED"
date -Is > "$FUSAI/nar_paper/data/pipeline_markers/boost_peft_pairformer.ok"
echo "DONE pairformer-only"
