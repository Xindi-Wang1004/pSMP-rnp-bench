#!/usr/bin/env bash
# Multi-seed low-data finetune (init seed sweep) on frozen pct splits.
# Does NOT rebuild subset seed-42 splits; only changes training --seed.
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
PD="${PROTENIX_DATA:-$ROOT/../protenix_data}"
LOWDATA="$ROOT/train/data/rnp_real/lowdata"
RUN_ROOT="${RUN_ROOT:-$ROOT/train/runs/rnp_real_lowdata_multiseed}"
BASE_CKPT="${PD}/checkpoint/protenix_base_default_v1.0.0.pt"
PSMP_CKPT="${PD}/checkpoint/protenix_base_psmp_pretrain_v1.pt"
GPU="${CUDA_VISIBLE_DEVICES:-4}"
SEED="${SEED:-43}"
# default: pct10 only (fastest confirmatory slice); override e.g. PCTS="pct10 pct25"
PCTS="${PCTS:-pct10}"
INITS="${INITS:-base psmp}"
LOG="$RUN_ROOT/multiseed_seed${SEED}.log"

mkdir -p "$RUN_ROOT"
: >> "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

steps_for() {
  case "$1" in
    pct10) echo 400 ;;
    pct25) echo 600 ;;
    pct50) echo 1000 ;;
    pct100) echo 1500 ;;
    *) echo 400 ;;
  esac
}

source /home/wangxindi/miniconda3/etc/profile.d/conda.sh
conda activate protenix
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"

log "multiseed finetune SEED=$SEED GPU=$GPU PCTS=$PCTS INITS=$INITS"

for PCT in $PCTS; do
  DATA_DIR="$LOWDATA/$PCT"
  STEPS="$(steps_for "$PCT")"
  [[ -d "$DATA_DIR" ]] || { log "missing $DATA_DIR"; exit 1; }
  for INIT in $INITS; do
    if [[ "$INIT" == "base" ]]; then CKPT="$BASE_CKPT"; else CKPT="$PSMP_CKPT"; fi
    TAG="${PCT}_${INIT}_seed${SEED}"
    RUN_DIR="$RUN_ROOT/$TAG"
    mkdir -p "$RUN_DIR"
    if compgen -G "$RUN_DIR"/*/checkpoints/*_ema_0.999.pt > /dev/null; then
      log "skip done $TAG"
      continue
    fi
    log "=== finetune $TAG steps=$STEPS ==="
    # Keep patched script under train/ so dirname-based ROOT resolves correctly
    # (writing to /tmp makes ROOT=/ and PROTENIX_DATA=//../protenix_data).
    PATCHED="$ROOT/train/.run_finetune_seed${SEED}_${TAG}.sh"
    sed "s/--seed 42/--seed ${SEED}/" "$ROOT/train/run_finetune_rnp_real.sh" > "$PATCHED"
    chmod +x "$PATCHED"
    CUDA_VISIBLE_DEVICES="$GPU" \
      PROTENIX_DATA="$PD" \
      INIT_CKPT="$CKPT" \
      DATA_DIR="$DATA_DIR" \
      RUN_DIR="$RUN_DIR" \
      RUN_NAME="${PCT}_from_${INIT}_seed${SEED}" \
      TAG="$TAG" \
      MAX_STEPS="$STEPS" \
      bash "$PATCHED" 2>&1 | tee -a "$LOG"
    rm -f "$PATCHED"
  done
done

log "DONE multiseed seed=$SEED"
