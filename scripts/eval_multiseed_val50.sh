#!/usr/bin/env bash
# Evaluate a finished multiseed pct* checkpoint on val50 (single sample, seed 101).
# Env: SEED=43 GPU=4 PCT=pct10  (INITS default: base psmp)
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
PAPER="$ROOT/nar_paper"
PD="${PROTENIX_DATA:-$ROOT/../protenix_data}"
RUN_ROOT="${RUN_ROOT:-$ROOT/train/runs/rnp_real_lowdata_multiseed}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/eval_work_val50}"
CASES="$ROOT/train/data/rnp_real/val_protenix_inputs/cases.tsv"
SEED="${SEED:-43}"
PCT="${PCT:-pct10}"
INITS="${INITS:-base psmp}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
PRED_SEED="${PRED_SEED:-101}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
PROTENIX="${PROTENIX:-$HOME/miniconda3/envs/protenix/bin/protenix}"
LOG="$PAPER/data/multiseed_eval_seed${SEED}.log"
MARK="$PAPER/data/pipeline_markers"

mkdir -p "$EVAL_ROOT" "$MARK" "$(dirname "$LOG")"
: >> "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

latest_ema() {
  local tag="$1"
  local d="$RUN_ROOT/$tag"
  # prefer newest ema
  find "$d" -type f -name '*_ema_0.999.pt' 2>/dev/null | sort | tail -1
}

setup_env() {
  local ckpt="$1" env_dir="$2"
  rm -rf "$env_dir"
  mkdir -p "$env_dir/checkpoint"
  ln -sf "$PD/common" "$env_dir/common"
  ln -sf "$PD/rna_msa" "$env_dir/rna_msa"
  ln -sf "$(readlink -f "$ckpt")" "$env_dir/checkpoint/protenix_base_default_v1.0.0.pt"
}

infer_one() {
  local tag="$1" ckpt="$2"
  local env_dir="$EVAL_ROOT/eval_env_${tag}"
  local pred_root="$EVAL_ROOT/pred_${tag}"
  setup_env "$ckpt" "$env_dir"
  export PROTENIX_ROOT_DIR="$env_dir"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
  mkdir -p "$pred_root"
  while IFS=$'\t' read -r name json_path native_cif pdb_id p_chain r_chain protein_len rna_len num_tokens _rest || [[ -n "${name:-}" ]]; do
    [[ "$name" == "name" ]] && continue
    [[ -n "$name" ]] || continue
    local jf="$ROOT/train/data/rnp_real/val_protenix_inputs/${name}.json"
    local od="$pred_root/$name"
    local pred_done="$od/$name/seed_${PRED_SEED}/predictions/${name}_sample_0.cif"
    [[ -f "$jf" ]] || continue
    [[ -f "$pred_done" ]] && continue
    if [[ -n "${num_tokens:-}" && "${num_tokens}" =~ ^[0-9.]+$ ]]; then
      if awk -v t="$num_tokens" -v m="$MAX_TOKENS" 'BEGIN{exit !(t+0>m)}'; then
        log "SKIP large $tag $name tokens=$num_tokens"
        continue
      fi
    fi
    mkdir -p "$od"
    log "infer $tag $name"
    "$PROTENIX" pred -i "$jf" -o "$od" -s "$PRED_SEED" -n protenix_base_default_v1.0.0 \
      --use_msa false --use_template false --use_rna_msa false \
      --use_default_params true \
      --triatt_kernel torch --trimul_kernel torch \
      -e 1 >> "$LOG" 2>&1 || log "WARN fail $tag $name"
  done < "$CASES"
}

score_tag() {
  local tag="$1"
  local pred_root="$EVAL_ROOT/pred_${tag}"
  local out_json="$EVAL_ROOT/interface_${tag}_val50_ext.json"
  log "score $tag"
  "$PYTHON" - <<PY
import csv, json, sys
from pathlib import Path
sys.path.insert(0, "$ROOT/train")
sys.path.insert(0, "$PAPER/scripts")
from compute_extended_metrics import eval_case_extended

cases = list(csv.DictReader(open("$CASES"), delimiter="\t"))
pred_root = Path("$pred_root")
results = []
n_ok = 0
for c in cases:
    name = c["name"]
    hits = list(pred_root.glob(f"**/{name}_sample_0.cif"))
    if not hits:
        results.append({"name": name, "error": "missing_prediction_no_cif"})
        continue
    cif = str(hits[0])
    try:
        m = eval_case_extended(c["native_cif_gz"], cif, c["pdb_id"], c["protein_chain_id"], c["rna_chain_id"])
        m = {**m, "name": name, "pred_cif": cif}
        if "contact_f1_5A" in m:
            n_ok += 1
        results.append(m)
    except Exception as e:
        results.append({"name": name, "error": str(e), "pred_cif": cif})

def mean(k):
    xs = [r[k] for r in results if k in r and r[k] is not None]
    return sum(xs)/len(xs) if xs else None

out = {
    "tag": "$tag",
    "n_ok": n_ok,
    "n_cases": len(cases),
    "mean_contact_f1_5A": mean("contact_f1_5A"),
    "mean_contact_recall_5A": mean("contact_recall_5A"),
    "mean_contact_precision_5A": mean("contact_precision_5A"),
    "mean_interface_lddt": mean("interface_lddt"),
    "results": results,
}
Path("$out_json").write_text(json.dumps(out, indent=2) + "\n")
print(f"written $out_json n_ok={n_ok} f1={out['mean_contact_f1_5A']}")
PY
}

log "multiseed eval SEED=$SEED PCT=$PCT GPU=$GPU INITS=$INITS"
for INIT in $INITS; do
  TAG="${PCT}_${INIT}_seed${SEED}"
  CKPT="$(latest_ema "$TAG" || true)"
  if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    log "MISSING ema for $TAG — abort (finetune not finished)"
    exit 2
  fi
  log "=== $TAG ckpt=$CKPT ==="
  infer_one "$TAG" "$CKPT"
  score_tag "$TAG"
done

"$PYTHON" "$PAPER/scripts/build_multiseed_summary.py" >> "$LOG" 2>&1 || log "WARN summary rebuild failed"
touch "$MARK/multiseed_eval_${PCT}_seed${SEED}.ok"
log "DONE multiseed eval seed=$SEED pct=$PCT"
