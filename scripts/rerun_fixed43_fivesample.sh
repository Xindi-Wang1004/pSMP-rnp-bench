#!/usr/bin/env bash
# Fixed43 five-sample inference @ pct100 base/psmp (Table-2 protocol: seed 101, -e 5).
# Reuses eval_work_5seed_val50 pred trees; skips cases that already have 5 CIFs.
# Env: GPU TAG MAX_TOKENS=4000
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
RUN_ROOT="$ROOT/train/runs/rnp_real_lowdata"
EVAL_WORK="$RUN_ROOT/eval_work_5seed_val50"
CASES="$ROOT/nar_paper/data/cases_fixed43.tsv"
MANIFEST="$RUN_ROOT/sweep_manifest.json"
PD="${PROTENIX_DATA:-$ROOT/../protenix_data}"
TAG="${TAG:?set TAG=pct100_base|pct100_psmp}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-101}"
NSAMPLE="${NSAMPLE:-5}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
LOG="$ROOT/nar_paper/data/fixed43_fivesample_${TAG}.log"
SKIP_LOG="$ROOT/nar_paper/data/fixed43_fivesample_skipped.tsv"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
PROTENIX="${PROTENIX:-$HOME/miniconda3/envs/protenix/bin/protenix}"

mkdir -p "$EVAL_WORK" "$ROOT/nar_paper/data"
: >> "$LOG"
[[ -f "$SKIP_LOG" ]] || echo -e "tag\tname\tnum_tokens\treason" > "$SKIP_LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

setup_eval_env() {
  local ckpt="$1" env_dir="$2"
  rm -rf "$env_dir"
  mkdir -p "$env_dir/checkpoint"
  ln -sf "$PD/common" "$env_dir/common"
  ln -sf "$PD/rna_msa" "$env_dir/rna_msa"
  ln -sf "$(readlink -f "$ckpt")" "$env_dir/checkpoint/protenix_base_default_v1.0.0.pt"
}

resolve_ckpt() {
  local tag="$1"
  if [[ -f "$MANIFEST" ]]; then
    local p
    p=$("$PYTHON" - <<PY
import json
m=json.load(open("$MANIFEST"))
rows = m["runs"] if isinstance(m, dict) and "runs" in m else (m if isinstance(m, list) else [])
for row in rows:
    if row.get("tag")=="$tag" or row.get("name")=="$tag":
        print(row.get("ckpt") or row.get("checkpoint") or "")
        break
PY
)
    if [[ -n "$p" && -f "$p" ]]; then echo "$p"; return; fi
  fi
  find "$RUN_ROOT/$tag" -type f -name '*ema*.pt' 2>/dev/null | head -1
}

n_cif_for() {
  local od="$1" name="$2"
  # nested or flat
  local n
  n=$(find "$od" -name "${name}_sample_*.cif" 2>/dev/null | wc -l | tr -d ' ')
  echo "${n:-0}"
}

ckpt=$(resolve_ckpt "$TAG" || true)
if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
  log "MISSING ckpt for $TAG"
  exit 1
fi
env_dir="$EVAL_WORK/eval_env_${TAG}"
pred_root="$EVAL_WORK/pred_${TAG}"
setup_eval_env "$ckpt" "$env_dir"
export PROTENIX_ROOT_DIR="$env_dir"
export CUDA_VISIBLE_DEVICES="$GPU"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
mkdir -p "$pred_root"
log "=== Fixed43 five-sample $TAG ckpt=$ckpt GPU=$GPU MAX_TOKENS=$MAX_TOKENS ==="

while IFS=$'\t' read -r name json_path native_cif pdb_id p_chain r_chain protein_len rna_len num_tokens _rest || [[ -n "${name:-}" ]]; do
  [[ "$name" == "name" ]] && continue
  [[ -n "$name" ]] || continue
  jf="$ROOT/train/data/rnp_real/val_protenix_inputs/${name}.json"
  od="$pred_root/$name"
  mkdir -p "$od"
  n_cif=$(n_cif_for "$od" "$name")
  if [[ "${n_cif:-0}" -ge "$NSAMPLE" ]]; then
    log "skip complete $TAG $name n_cif=$n_cif"
    continue
  fi
  [[ -f "$jf" ]] || { log "WARN missing json $name"; continue; }
  if [[ -n "${num_tokens:-}" && "${num_tokens}" =~ ^[0-9.]+$ ]]; then
    if awk -v t="$num_tokens" -v m="$MAX_TOKENS" 'BEGIN{exit !(t+0>m)}'; then
      log "SKIP large $TAG $name tokens=$num_tokens"
      echo -e "$TAG\t$name\t$num_tokens\ttoo_large" >> "$SKIP_LOG"
      continue
    fi
  fi
  log "infer Fixed43 five-sample $TAG: $name tokens=$num_tokens (have $n_cif)"
  "$PROTENIX" pred -i "$jf" -o "$od" -s "$SEED" -n protenix_base_default_v1.0.0 \
    --use_msa false --use_template false --use_rna_msa false \
    --use_default_params true \
    --triatt_kernel torch --trimul_kernel torch \
    -e "$NSAMPLE" >> "$LOG" 2>&1 || log "WARN failed $TAG $name"
done < "$CASES"

log "DONE Fixed43 five-sample $TAG"
echo "log: $LOG"
