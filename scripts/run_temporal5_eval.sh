#!/usr/bin/env bash
# Temporal5 inference: base@10% vs pSMP@10% on recommended shortlist.
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# When run from fusai/nar_paper/scripts, ROOT is fusai/nar_paper; FUSAI is parent if present
if [[ -d "${ROOT}/../train" ]]; then
  FUSAI="$(cd "${ROOT}/.." && pwd)"
elif [[ -d "${ROOT}/train" ]]; then
  FUSAI="$ROOT"
else
  FUSAI="/home/wangxindi/RNA_Protein/fusai"
fi
cd "$FUSAI"

source /home/wangxindi/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate protenix 2>/dev/null || true

export PYTHONPATH="${FUSAI}:${PYTHONPATH:-}"
export PROTENIX_DATA="${PROTENIX_DATA:-$FUSAI/../protenix_data}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"

OUT="${FUSAI}/train/runs/temporal5"
INPUTS="${OUT}/inputs"
PD="${PROTENIX_DATA}"
SEED="${SEED:-101}"
GPU_BASE="${GPU_BASE:-7}"
GPU_PSMP="${GPU_PSMP:-3}"

CKPT_BASE="${FUSAI}/train/runs/rnp_real_lowdata/pct10_base/pct10_from_base_20260702_021526/checkpoints/399_ema_0.999.pt"
CKPT_PSMP="${FUSAI}/train/runs/rnp_real_lowdata/pct10_psmp/pct10_from_psmp_20260702_030131/checkpoints/399_ema_0.999.pt"

mkdir -p "$OUT" "$INPUTS"
LOG="${OUT}/pipeline.log"
: > "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# 1) prepare inputs
log "prepare temporal inputs"
python3 "${FUSAI}/nar_paper/scripts/prepare_temporal_inputs.py" \
  --pdb-ids 10yz 9zy0 9o7t 13fn 9ype \
  --out-dir "$INPUTS" \
  --prefer-pairs "10yz:A:B" \
  2>&1 | tee -a "$LOG"

CASES="${INPUTS}/cases_temporal5.tsv"
[[ -f "$CASES" ]] || { log "missing $CASES"; exit 1; }

infer_tag() {
  local tag="$1" ckpt="$2" gpu="$3"
  local work="${OUT}/${tag}"
  local env_dir="${work}/eval_env"
  local pred_dir="${work}/pred"
  mkdir -p "$work" "$pred_dir"
  rm -rf "$env_dir"
  mkdir -p "$env_dir/checkpoint"
  ln -sfn "$PD/common" "$env_dir/common"
  ln -sfn "$PD/rna_msa" "$env_dir/rna_msa"
  ln -sfn "$(readlink -f "$ckpt")" "$env_dir/checkpoint/protenix_base_default_v1.0.0.pt"
  export PROTENIX_ROOT_DIR="$env_dir"
  export CUDA_VISIBLE_DEVICES="$gpu"

  log "infer $tag on GPU $gpu"
  while IFS=$'\t' read -r name json_path rest; do
    [[ "$name" == "name" ]] && continue
    [[ -z "$name" ]] && continue
    local jf="$json_path"
    local od="${pred_dir}/${name}"
    mkdir -p "$od"
    if compgen -G "${od}/${name}/seed_${SEED}/predictions/${name}_sample_*.cif" > /dev/null; then
      log "skip existing $tag $name"
      continue
    fi
    log "  $tag $name"
    protenix pred -i "$jf" -o "$od" -s "$SEED" -n protenix_base_default_v1.0.0 \
      --use_msa false --use_template false --use_rna_msa false \
      --use_default_params true \
      --triatt_kernel torch --trimul_kernel torch \
      -e 1 >> "${work}/infer.log" 2>&1 || log "WARN infer failed $tag $name"
  done < "$CASES"
}

# 2) infer base + pSMP in parallel
infer_tag base_pct10 "$CKPT_BASE" "$GPU_BASE" &
pid_base=$!
infer_tag psmp_pct10 "$CKPT_PSMP" "$GPU_PSMP" &
pid_psmp=$!
wait "$pid_base" || true
wait "$pid_psmp" || true
log "inference finished"

# 3) interface eval
log "evaluate interfaces"
python3 - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${FUSAI}/train")
from psmp.eval_interface import eval_case, find_best_pred_cif
import pandas as pd

cases = pd.read_csv("${CASES}", sep="\t")
out_root = Path("${OUT}")
summary = {}
for tag in ("base_pct10", "psmp_pct10"):
    pred_root = out_root / tag / "pred"
    rows = []
    for _, c in cases.iterrows():
        name = c["name"]
        pred = find_best_pred_cif(pred_root / name, name)
        if pred is None:
            rows.append({"name": name, "error": "pred_missing"})
            continue
        try:
            m = eval_case(
                str(c["native_cif_gz"]),
                str(pred),
                str(c["pdb_id"]).lower(),
                str(c["protein_chain_id"]),
                str(c["rna_chain_id"]),
            )
            rows.append({"name": name, "pred_cif": str(pred), **m})
        except Exception as e:
            rows.append({"name": name, "error": str(e)})
    ok = [r for r in rows if "contact_recall_5A" in r]
    mean = lambda k: (sum(r[k] for r in ok) / len(ok)) if ok else float("nan")
    summary[tag] = {
        "n_ok": len(ok),
        "n_cases": len(rows),
        "mean_interface_lddt": mean("interface_lddt"),
        "mean_contact_recall_5A": mean("contact_recall_5A"),
        "results": rows,
    }
    (out_root / f"interface_{tag}.json").write_text(json.dumps(summary[tag], indent=2) + "\\n")
    print(tag, "n_ok=", len(ok),
          "lddt=", round(summary[tag]["mean_interface_lddt"], 3) if ok else None,
          "recall=", round(summary[tag]["mean_contact_recall_5A"], 3) if ok else None)

# extended precision/F1 via nar_paper script if available
(out_root / "interface_summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
print("wrote", out_root / "interface_summary.json")
PY

# 4) extended metrics
for tag in base_pct10 psmp_pct10; do
  python3 "${FUSAI}/nar_paper/scripts/compute_extended_metrics.py" \
    --interface-json "${OUT}/interface_${tag}.json" \
    --cases-tsv "$CASES" \
    --out-json "${OUT}/interface_${tag}_ext.json" \
    2>&1 | tee -a "$LOG" || true
done

log "temporal5 pipeline done"
ls -la "$OUT" | tee -a "$LOG"
