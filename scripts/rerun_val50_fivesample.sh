#!/usr/bin/env bash
# Val50 five-sample re-inference for all low-data checkpoints (seed 101, -e 5).
# After preds exist, score with compute_extended_metrics.py then build_paired_intersection_table_5seed.py
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
RUN_ROOT="$ROOT/train/runs/rnp_real_lowdata"
EVAL_WORK="$RUN_ROOT/eval_work_5seed_val50"
CASES="$ROOT/train/data/rnp_real/val_protenix_inputs/cases.tsv"
MANIFEST="$RUN_ROOT/sweep_manifest.json"
PD="${PROTENIX_DATA:-$ROOT/../protenix_data}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-101}"
NSAMPLE="${NSAMPLE:-5}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
LOG="$ROOT/nar_paper/data/val50_fivesample_reinfer.log"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
PROTENIX="${PROTENIX:-$HOME/miniconda3/envs/protenix/bin/protenix}"
METRICS="$ROOT/nar_paper/scripts/compute_extended_metrics.py"
SKIP_LOG="$ROOT/nar_paper/data/val50_fivesample_skipped.tsv"

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

# Resolve checkpoint path from sweep_manifest or conventional layout
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
  # fallback: largest/newest .pt under run dir
  local d="$RUN_ROOT/$tag"
  find "$d" -type f -name '*.pt' 2>/dev/null | head -1
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
  while IFS=$'\t' read -r name json_path native_cif pdb_id p_chain r_chain protein_len rna_len num_tokens _rest || [[ -n "${name:-}" ]]; do
    [[ "$name" == "name" ]] && continue
    [[ -n "$name" ]] || continue
    local jf="$ROOT/train/data/rnp_real/val_protenix_inputs/${name}.json"
    local od="$pred_root/$name"
    mkdir -p "$od"
    # skip if all five samples present
    local n_cif
    n_cif=$(find "$od" -name "${name}_sample_*.cif" 2>/dev/null | wc -l | tr -d ' ' || true)
    if [[ "${n_cif:-0}" -ge "$NSAMPLE" ]]; then
      continue
    fi
    [[ -f "$jf" ]] || continue
    # skip huge complexes that hang / OOM (same budget as contact export)
    if [[ -n "${num_tokens:-}" && "${num_tokens}" =~ ^[0-9.]+$ ]]; then
      if awk -v t="$num_tokens" -v m="$MAX_TOKENS" 'BEGIN{exit !(t+0>m)}'; then
        log "SKIP large $tag $name tokens=$num_tokens"
        echo -e "$tag\t$name\t$num_tokens\ttoo_large" >> "$SKIP_LOG"
        continue
      fi
    fi
    log "infer five-sample $tag: $name"
    "$PROTENIX" pred -i "$jf" -o "$od" -s "$SEED" -n protenix_base_default_v1.0.0 \
      --use_msa false --use_template false --use_rna_msa false \
      --use_default_params true \
      --triatt_kernel torch --trimul_kernel torch \
      -e "$NSAMPLE" >> "$LOG" 2>&1 || log "WARN failed $tag $name"
  done < "$CASES"
}

TAGS=(pct10_base pct10_psmp pct25_base pct25_psmp pct50_base pct50_psmp pct100_base pct100_psmp)

for tag in "${TAGS[@]}"; do
  ckpt=$(resolve_ckpt "$tag" || true)
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    log "MISSING ckpt for $tag"
    continue
  fi
  log "=== five-sample $tag ckpt=$ckpt ==="
  infer_tag "$tag" "$ckpt"
  out_json="$EVAL_WORK/interface_5seed_${tag}_val50_ext.json"
  # Prefer existing paper pipeline that ranks by ipTM if available; else score sample_0 as fallback marker
  if [[ -f "$METRICS" ]]; then
    # interface json may need a prior interface_*.json; if compute script supports pred_root, use it
    if "$PYTHON" "$METRICS" --help 2>&1 | grep -q pred_root; then
      "$PYTHON" "$METRICS" --pred_root "$EVAL_WORK/pred_${tag}" --cases "$CASES" --out "$out_json" \
        --best_of_iptm --n_sample "$NSAMPLE" --seed "$SEED" >> "$LOG" 2>&1 || log "metrics help path failed $tag"
    fi
  fi
done

log "DONE five-sample launch pass. Next: select best-of-five by ipTM and run build_paired_intersection on 5seed JSONs."
echo "log: $LOG"
