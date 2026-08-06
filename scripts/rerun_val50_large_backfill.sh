#!/usr/bin/env bash
# Backfill five-sample (or single-sample) preds for cases skipped as too_large.
# Reads SKIP_LOG from fivesample run; only processes listed names.
# Env: GPU=7 TAG=pct10_psmp NSAMPLE=1 MAX_TOKENS=8000
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
RUN_ROOT="$ROOT/train/runs/rnp_real_lowdata"
EVAL_WORK="$RUN_ROOT/eval_work_5seed_val50"
CASES="$ROOT/train/data/rnp_real/val_protenix_inputs/cases.tsv"
MANIFEST="$RUN_ROOT/sweep_manifest.json"
PD="${PROTENIX_DATA:-$ROOT/../protenix_data}"
SKIP_LOG="${SKIP_LOG:-$ROOT/nar_paper/data/val50_fivesample_skipped.tsv}"
TAG="${TAG:-pct10_psmp}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-101}"
NSAMPLE="${NSAMPLE:-1}"
MAX_TOKENS="${MAX_TOKENS:-8000}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
PROTENIX="${PROTENIX:-$HOME/miniconda3/envs/protenix/bin/protenix}"
LOG="$ROOT/nar_paper/data/val50_large_backfill.log"
MARK="$ROOT/nar_paper/data/pipeline_markers"

mkdir -p "$EVAL_WORK" "$MARK"
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

# unique names from skip log for this tag (or any tag if TAG_ANY=1)
mapfile -t NAMES < <(awk -F'\t' -v t="$TAG" 'NR>1 && ($1==t || ENVIRON["TAG_ANY"]=="1"){print $2}' "$SKIP_LOG" 2>/dev/null | sort -u)
if [[ ${#NAMES[@]} -eq 0 ]]; then
  # fallback: cases with num_tokens > 4000 from cases.tsv
  mapfile -t NAMES < <(awk -F'\t' 'NR>1 && $9+0>4000 && $9+0<='"$MAX_TOKENS"'{print $1}' "$CASES")
fi

ckpt=$(resolve_ckpt "$TAG")
[[ -f "$ckpt" ]] || { log "missing ckpt $TAG"; exit 1; }
env_dir="$EVAL_WORK/eval_env_${TAG}"
pred_root="$EVAL_WORK/pred_${TAG}"
setup_eval_env "$ckpt" "$env_dir"
export PROTENIX_ROOT_DIR="$env_dir"
export CUDA_VISIBLE_DEVICES="$GPU"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
mkdir -p "$pred_root"

log "large backfill TAG=$TAG GPU=$GPU n=${#NAMES[@]} NSAMPLE=$NSAMPLE MAX_TOKENS=$MAX_TOKENS"
declare -A TOK
while IFS=$'\t' read -r name _ _ _ _ _ _ _ num_tokens _; do
  [[ "$name" == "name" ]] && continue
  TOK["$name"]="$num_tokens"
done < "$CASES"

for name in "${NAMES[@]}"; do
  [[ -n "$name" ]] || continue
  nt="${TOK[$name]:-0}"
  if awk -v t="$nt" -v m="$MAX_TOKENS" 'BEGIN{exit !(t+0>m)}'; then
    log "SKIP still-too-large $name tokens=$nt"
    continue
  fi
  jf="$ROOT/train/data/rnp_real/val_protenix_inputs/${name}.json"
  od="$pred_root/$name"
  n_cif=$(find "$od" -name "${name}_sample_*.cif" 2>/dev/null | wc -l | tr -d ' ' || true)
  if [[ "${n_cif:-0}" -ge "$NSAMPLE" ]]; then
    continue
  fi
  [[ -f "$jf" ]] || continue
  mkdir -p "$od"
  log "infer $TAG $name tokens=$nt"
  "$PROTENIX" pred -i "$jf" -o "$od" -s "$SEED" -n protenix_base_default_v1.0.0 \
    --use_msa false --use_template false --use_rna_msa false \
    --use_default_params true \
    --triatt_kernel torch --trimul_kernel torch \
    -e "$NSAMPLE" >> "$LOG" 2>&1 || log "WARN fail $name"
done

touch "$MARK/large_backfill_${TAG}.ok"
log "DONE large backfill $TAG"
