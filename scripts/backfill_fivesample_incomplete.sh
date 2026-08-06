#!/usr/bin/env bash
# Retry five-sample cases that failed / are incomplete (n_cif < NSAMPLE).
# Does NOT advance past tags blindly — only fills gaps under MAX_TOKENS.
#
# Env:
#   TAG=pct50_psmp          # required unless TAGS="pct50_psmp pct50_base ..."
#   TAGS="pct50_psmp ..."   # optional multi-tag
#   GPU / CUDA_VISIBLE_DEVICES
#   NSAMPLE=5 MAX_TOKENS=4000
#   CLEAN_PARTIAL=1         # remove incomplete pred dirs before retry
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
PAPER="$ROOT/nar_paper"
RUN_ROOT="$ROOT/train/runs/rnp_real_lowdata"
EVAL_WORK="$RUN_ROOT/eval_work_5seed_val50"
CASES="$ROOT/train/data/rnp_real/val_protenix_inputs/cases.tsv"
MANIFEST="$RUN_ROOT/sweep_manifest.json"
PD="${PROTENIX_DATA:-$ROOT/../protenix_data}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-101}"
NSAMPLE="${NSAMPLE:-5}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
CLEAN_PARTIAL="${CLEAN_PARTIAL:-1}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
PROTENIX="${PROTENIX:-$HOME/miniconda3/envs/protenix/bin/protenix}"
LOG="$PAPER/data/fivesample_incomplete_backfill.log"
MARK="$PAPER/data/pipeline_markers"

if [[ -n "${TAGS:-}" ]]; then
  read -r -a TAG_ARR <<< "$TAGS"
elif [[ -n "${TAG:-}" ]]; then
  TAG_ARR=("$TAG")
else
  TAG_ARR=(pct50_psmp pct50_base pct100_base pct100_psmp pct25_psmp pct25_base pct10_psmp pct10_base)
fi

mkdir -p "$EVAL_WORK" "$MARK" "$PAPER/data"
: >> "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

resolve_ckpt() {
  local tag="$1"
  "$PYTHON" - <<PY
import json
m=json.load(open("$MANIFEST"))
for row in m.get("runs", []):
    if row.get("tag")=="$tag":
        print(row["ckpt"]); break
PY
}

setup_eval_env() {
  local ckpt="$1" env_dir="$2"
  rm -rf "$env_dir"
  mkdir -p "$env_dir/checkpoint"
  ln -sf "$PD/common" "$env_dir/common"
  ln -sf "$PD/rna_msa" "$env_dir/rna_msa"
  ln -sf "$(readlink -f "$ckpt")" "$env_dir/checkpoint/protenix_base_default_v1.0.0.pt"
}

missing_names_for_tag() {
  local tag="$1"
  "$PYTHON" - <<PY
import csv
from pathlib import Path
cases=list(csv.DictReader(open("$CASES"), delimiter="\t"))
pred=Path("$EVAL_WORK")/f"pred_$tag"
for c in cases:
    tok=float(c.get("num_tokens") or 0)
    if tok > float("$MAX_TOKENS"):
        continue
    name=c["name"]
    n=len(list(pred.glob(f"**/{name}_sample_*.cif"))) if pred.is_dir() else 0
    if n < int("$NSAMPLE"):
        print(name)
PY
}

for tag in "${TAG_ARR[@]}"; do
  mapfile -t NAMES < <(missing_names_for_tag "$tag")
  if [[ ${#NAMES[@]} -eq 0 ]]; then
    log "OK complete $tag (no gaps under MAX_TOKENS=$MAX_TOKENS)"
    touch "$MARK/fivesample_backfill_${tag}.ok"
    continue
  fi
  ckpt=$(resolve_ckpt "$tag")
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    log "MISSING ckpt $tag"; exit 1
  fi
  env_dir="$EVAL_WORK/eval_env_${tag}"
  pred_root="$EVAL_WORK/pred_${tag}"
  setup_eval_env "$ckpt" "$env_dir"
  export PROTENIX_ROOT_DIR="$env_dir"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
  mkdir -p "$pred_root"
  log "=== backfill $tag n=${#NAMES[@]} GPU=$GPU NSAMPLE=$NSAMPLE ckpt=$ckpt ==="

  for name in "${NAMES[@]}"; do
    jf="$ROOT/train/data/rnp_real/val_protenix_inputs/${name}.json"
    od="$pred_root/$name"
    [[ -f "$jf" ]] || { log "SKIP no json $name"; continue; }
    n_cif=$(find "$od" -name "${name}_sample_*.cif" 2>/dev/null | wc -l | tr -d ' ' || true)
    if [[ "${n_cif:-0}" -ge "$NSAMPLE" ]]; then
      continue
    fi
    if [[ "$CLEAN_PARTIAL" == "1" && -d "$od" && "${n_cif:-0}" -lt "$NSAMPLE" ]]; then
      rm -rf "$od"
    fi
    mkdir -p "$od"
    log "retry $tag $name (had ${n_cif:-0})"
    if ! "$PROTENIX" pred -i "$jf" -o "$od" -s "$SEED" -n protenix_base_default_v1.0.0 \
      --use_msa false --use_template false --use_rna_msa false \
      --use_default_params true \
      --triatt_kernel torch --trimul_kernel torch \
      -e "$NSAMPLE" >> "$LOG" 2>&1; then
      log "WARN fail $tag $name"
      continue
    fi
  done

  # re-check
  mapfile -t LEFT < <(missing_names_for_tag "$tag")
  if [[ ${#LEFT[@]} -eq 0 ]]; then
    touch "$MARK/fivesample_backfill_${tag}.ok"
    log "DONE backfill complete $tag"
  else
    log "DONE backfill partial $tag still_missing=${#LEFT[@]} (${LEFT[*]})"
    # still write a soft ok so queue can move on after one pass; force re-run by deleting marker
    touch "$MARK/fivesample_backfill_${tag}.partial"
  fi
done

"$PYTHON" "$PAPER/scripts/list_fivesample_incomplete.py" --max-tokens "$MAX_TOKENS" --n-sample "$NSAMPLE" >> "$LOG" 2>&1 || true
log "backfill pass finished tags=${TAG_ARR[*]}"
