#!/usr/bin/env bash
# Five-sample val50 inference + score for pairformer-only pct10 checkpoint (harness walk-through).
set -eo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
PAPER="$ROOT/nar_paper"
RUN_ROOT="$ROOT/train/runs/rnp_real_lowdata_peft"
EVAL_WORK="$RUN_ROOT/eval_work_5seed_val50"
CASES="$ROOT/train/data/rnp_real/val_protenix_inputs/cases.tsv"
PD="${PROTENIX_DATA:-$ROOT/../protenix_data}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-101}"
NSAMPLE="${NSAMPLE:-5}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
TAG="pct10_pairformer_only"
LOG="$RUN_ROOT/${TAG}_val50.log"

CKPT=$(find "$RUN_ROOT/pct10_base_pairformer_only" -name '*_ema_0.999.pt' | sort | tail -1)
[[ -n "$CKPT" && -f "$CKPT" ]] || { echo "missing EMA ckpt under pct10_base_pairformer_only"; exit 1; }

source /home/wangxindi/miniconda3/etc/profile.d/conda.sh
conda activate protenix
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="$HOME/miniconda3/envs/protenix/bin/python3"
PROTENIX="$HOME/miniconda3/envs/protenix/bin/protenix"

mkdir -p "$EVAL_WORK" "$RUN_ROOT"
: >> "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

env_dir="$EVAL_WORK/eval_env_${TAG}"
pred_root="$EVAL_WORK/pred_${TAG}"
rm -rf "$env_dir"
mkdir -p "$env_dir/checkpoint" "$pred_root"
ln -sf "$PD/common" "$env_dir/common"
ln -sf "$PD/rna_msa" "$env_dir/rna_msa"
ln -sf "$(readlink -f "$CKPT")" "$env_dir/checkpoint/protenix_base_default_v1.0.0.pt"
export PROTENIX_ROOT_DIR="$env_dir"
export CUDA_VISIBLE_DEVICES="$GPU"

log "infer $TAG ckpt=$CKPT gpu=$GPU n_sample=$NSAMPLE"
while IFS=$'\t' read -r name json_path rest || [[ -n "${name:-}" ]]; do
  [[ "$name" == "name" ]] && continue
  [[ -z "$name" ]] && continue
  jf="$ROOT/train/data/rnp_real/val_protenix_inputs/${name}.json"
  od="$pred_root/$name"
  mkdir -p "$od"
  n_cif=$(find "$od" -name "${name}_sample_*.cif" 2>/dev/null | wc -l | tr -d ' ')
  [[ "${n_cif:-0}" -ge "$NSAMPLE" ]] && continue
  tok=$(echo "$rest" | awk -F'\t' '{print $(NF-1)}')
  if [[ -n "${tok:-}" && "$tok" -gt "$MAX_TOKENS" ]]; then
    log "skip large $name tokens=$tok"
    continue
  fi
  [[ -f "$jf" ]] || continue
  log "  $name"
  protenix pred -i "$jf" -o "$od" -s "$SEED" -n protenix_base_default_v1.0.0 \
    --use_msa false --use_template false --use_rna_msa false \
    --use_default_params true --triatt_kernel torch --trimul_kernel torch \
    -e "$NSAMPLE" >> "$LOG" 2>&1 || log "WARN infer failed $name"
done < "$CASES"

log "score $TAG"
"$PYTHON" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "$ROOT/train")
sys.path.insert(0, "$PAPER/scripts")
from score_5seed_val50 import load_cases, find_samples
from compute_extended_metrics import eval_case_extended

EVAL = Path("$EVAL_WORK")
pred_root = EVAL / "pred_$TAG"
cases = load_cases()
results = []
n_ok = 0
for c in cases:
    name = c["name"]
    samples = find_samples(pred_root, name)
    if not samples:
        results.append({"name": name, "error": "missing_prediction_no_cif"})
        continue
    samples.sort(key=lambda x: x[0], reverse=True)
    best_iptm, best_cif = samples[0]
    try:
        m = eval_case_extended(
            c["native_cif_gz"], str(best_cif), c["pdb_id"],
            c["protein_chain_id"], c["rna_chain_id"],
        )
    except Exception as e:
        results.append({"name": name, "error": str(e)})
        continue
    m = {**m, "name": name, "pred_cif": str(best_cif), "best_iptm": best_iptm,
         "n_samples_found": len(samples)}
    if m.get("contact_f1_5A") is not None:
        n_ok += 1
    results.append(m)

f1s = [r["contact_f1_5A"] for r in results if r.get("contact_f1_5A") is not None]
fa = []
for r in results:
    fa.append(float(r["contact_f1_5A"]) if r.get("contact_f1_5A") is not None else 0.0)
payload = {
    "tag": "$TAG",
    "protocol": "five-sample_best_of_iptm_seed101_pairformer_only_pct10",
    "ckpt": "$CKPT",
    "pred_dir": str(pred_root),
    "n_cases": len(cases),
    "n_ok": n_ok,
    "mean_contact_f1_5A": (sum(f1s)/len(f1s)) if f1s else None,
    "FA_mean_contact_f1_5A": sum(fa)/len(fa) if fa else None,
    "results": results,
}
out = EVAL / "interface_5seed_${TAG}_val50_ext.json"
out.write_text(json.dumps(payload, indent=2)+"\n")
print(f"wrote {out} n_ok={n_ok} FA_F1={payload['FA_mean_contact_f1_5A']:.4f}")
PY

date -Is > "$PAPER/data/pipeline_markers/boost_peft_val50.ok"
log "DONE $TAG val50"
