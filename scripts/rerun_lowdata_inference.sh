#!/usr/bin/env bash
# Re-run low-data val inference (pred CIFs deleted) then extended precision/F1.
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
RUN_ROOT="$ROOT/train/runs/rnp_real_lowdata"
EVAL_WORK="$RUN_ROOT/eval_work"
CASES="$ROOT/train/data/rnp_real/val_protenix_inputs/cases.tsv"
PD="${PROTENIX_DATA:-$ROOT/../protenix_data}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-101}"
LOG="$ROOT/nar_paper/data/lowdata_reinfer.log"
PYTHON=~/miniconda3/envs/protenix/bin/python3
PROTENIX=~/miniconda3/envs/protenix/bin/protenix
SCRIPT="$ROOT/nar_paper/scripts/compute_extended_metrics.py"

mkdir -p "$ROOT/nar_paper/data"
: >> "$LOG"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

setup_eval_env() {
  local ckpt="$1" env_dir="$2"
  rm -rf "$env_dir"
  mkdir -p "$env_dir/checkpoint"
  ln -sf "$PD/common" "$env_dir/common"
  ln -sf "$PD/rna_msa" "$env_dir/rna_msa"
  ln -sf "$(readlink -f "$ckpt")" "$env_dir/checkpoint/protenix_base_default_v1.0.0.pt"
}

infer_tag() {
  local tag="$1" ckpt="$2"
  local env_dir="$EVAL_WORK/eval_env_${tag}"
  local pred_root="$EVAL_WORK/pred_${tag}"
  setup_eval_env "$ckpt" "$env_dir"
  export PROTENIX_ROOT_DIR="$env_dir"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"

  mkdir -p "$pred_root"
  while IFS=$'\t' read -r name _ _ _ _ _ _ _ _ || [[ -n "${name:-}" ]]; do
    [[ "$name" == "name" ]] && continue
    [[ -n "$name" ]] || continue
    local jf="$ROOT/train/data/rnp_real/val_protenix_inputs/${name}.json"
    local od="$pred_root/$name"
    local pred_done="$od/$name/seed_${SEED}/predictions/${name}_sample_0.cif"
    [[ -f "$jf" ]] || continue
    [[ -f "$pred_done" ]] && continue
    mkdir -p "$od"
    log "infer $tag: $name"
    "$PROTENIX" pred -i "$jf" -o "$od" -s "$SEED" -n protenix_base_default_v1.0.0 \
      --use_msa false --use_template false --use_rna_msa false \
      --use_default_params true \
      --triatt_kernel torch --trimul_kernel torch \
      -e 1 >> "$LOG" 2>&1
  done < "$CASES"
}

manifest="$RUN_ROOT/sweep_manifest.json"

ext_is_valid() {
  local f="$1"
  "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('n_ok',0)>0 and d.get('mean_contact_precision_5A') is not None else 1)" "$f" 2>/dev/null
}

while read -r tag ckpt || [[ -n "${tag:-}" ]]; do
  [[ -n "$tag" ]] || continue
  in_json="$EVAL_WORK/interface_${tag}.json"
  out_json="$EVAL_WORK/interface_${tag}_ext.json"
  pred_root="$EVAL_WORK/pred_${tag}"
  mkdir -p "$pred_root"
  n_cif=$(find "$pred_root" -name "*_sample_0.cif" 2>/dev/null | wc -l || true)
  n_cif=${n_cif// /}
  if [[ "${n_cif:-0}" -ge 40 ]] && ext_is_valid "$out_json"; then
    log "skip $tag (preds=$n_cif, ext valid)"
    continue
  fi
  log "=== infer $tag (existing preds=$n_cif) ==="
  infer_tag "$tag" "$ckpt"
  rm -f "$out_json"
  log "extended metrics $tag"
  "$PYTHON" "$SCRIPT" --interface-json "$in_json" --cases-tsv "$CASES" --out-json "$out_json" >> "$LOG" 2>&1
done < <(python3 - <<'PY' "$manifest"
import json, sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text())["runs"]
for r in rows:
    print(r["tag"], r["ckpt"])
PY
) || true

"$PYTHON" "$ROOT/nar_paper/scripts/build_extended_tables.py" >> "$LOG" 2>&1
log "lowdata re-inference + extended metrics done"
