#!/usr/bin/env bash
# Temporal20 confirmatory inference: base@10% vs pSMP@10% (parallel GPUs).
# Freeze list first via freeze_temporal20.py — do not change recipe based on this set.
set -eo pipefail

FUSAI="/home/wangxindi/RNA_Protein/fusai"
PAPER="$FUSAI/nar_paper"
source /home/wangxindi/miniconda3/etc/profile.d/conda.sh
conda activate protenix

export PYTHONPATH="${FUSAI}:${PYTHONPATH:-}"
export PROTENIX_DATA="${PROTENIX_DATA:-$FUSAI/../protenix_data}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"

OUT="${FUSAI}/train/runs/temporal20"
INPUTS="${OUT}/inputs"
PD="$PROTENIX_DATA"
SEED="${SEED:-101}"
GPU_BASE="${GPU_BASE:-0}"
GPU_PSMP="${GPU_PSMP:-1}"
NSAMPLE="${NSAMPLE:-1}"

CKPT_BASE="${FUSAI}/train/runs/rnp_real_lowdata/pct10_base/pct10_from_base_20260702_021526/checkpoints/399_ema_0.999.pt"
CKPT_PSMP="${FUSAI}/train/runs/rnp_real_lowdata/pct10_psmp/pct10_from_psmp_20260702_030131/checkpoints/399_ema_0.999.pt"

mkdir -p "$OUT" "$INPUTS" "$PAPER/data/pipeline_markers"
LOG="${OUT}/pipeline.log"
: >> "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# 1) freeze shortlist
log "freeze Temporal20"
python3 "$PAPER/scripts/freeze_temporal20.py" 2>&1 | tee -a "$LOG"
IDS=$(python3 -c "import json; print(' '.join(json.load(open('$PAPER/data/temporal20_freeze.json'))['pdb_ids']))")
[[ -n "$IDS" ]] || { log "empty freeze"; exit 1; }

# 2) prepare inputs
log "prepare inputs: $IDS"
python3 "$PAPER/scripts/prepare_temporal_inputs.py" \
  --pdb-ids $IDS \
  --out-dir "$INPUTS" \
  2>&1 | tee -a "$LOG"

CASES="${INPUTS}/cases_temporal5.tsv"
# prepare script may write cases_temporal5.tsv historically; also accept cases_temporal.tsv
if [[ ! -f "$CASES" ]]; then
  CASES="${INPUTS}/cases_temporal.tsv"
fi
if [[ ! -f "$CASES" ]]; then
  CASES=$(ls "$INPUTS"/cases_*.tsv | head -1)
fi
[[ -f "$CASES" ]] || { log "missing cases tsv in $INPUTS"; ls -la "$INPUTS"; exit 1; }
cp -f "$CASES" "${OUT}/cases_temporal20.tsv"
CASES="${OUT}/cases_temporal20.tsv"
log "cases=$(wc -l < "$CASES") file=$CASES"

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

  log "infer $tag on GPU $gpu n_sample=$NSAMPLE"
  while IFS=$'\t' read -r name json_path rest || [[ -n "${name:-}" ]]; do
    [[ "$name" == "name" ]] && continue
    [[ -z "$name" ]] && continue
    local od="${pred_dir}/${name}"
    mkdir -p "$od"
    if compgen -G "${od}/${name}/seed_${SEED}/predictions/${name}_sample_*.cif" > /dev/null; then
      log "skip existing $tag $name"
      continue
    fi
    log "  $tag $name"
    protenix pred -i "$json_path" -o "$od" -s "$SEED" -n protenix_base_default_v1.0.0 \
      --use_msa false --use_template false --use_rna_msa false \
      --use_default_params true \
      --triatt_kernel torch --trimul_kernel torch \
      -e "$NSAMPLE" >> "${work}/infer.log" 2>&1 || log "WARN infer failed $tag $name"
  done < "$CASES"
  log "infer done $tag"
}

log "launch parallel infer base@$GPU_BASE psmp@$GPU_PSMP"
infer_tag base_pct10 "$CKPT_BASE" "$GPU_BASE" &
pid_base=$!
infer_tag psmp_pct10 "$CKPT_PSMP" "$GPU_PSMP" &
pid_psmp=$!
wait "$pid_base" || true
wait "$pid_psmp" || true
log "inference finished"

# 3) interface eval (reuse temporal5 evaluator pattern)
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
    def mean(k):
        return (sum(r[k] for r in ok) / len(ok)) if ok else float("nan")
    # failure-aware over all prepared cases
    fa_recall = []
    for r in rows:
        if "contact_recall_5A" in r:
            fa_recall.append(float(r["contact_recall_5A"]))
        else:
            fa_recall.append(0.0)
    summary[tag] = {
        "n_ok": len(ok),
        "n_cases": len(rows),
        "mean_interface_lddt_ok": mean("interface_lddt"),
        "mean_contact_recall_5A_ok": mean("contact_recall_5A"),
        "FA_mean_contact_recall_5A": sum(fa_recall) / len(fa_recall) if fa_recall else float("nan"),
        "results": rows,
    }
    (out_root / f"interface_{tag}.json").write_text(json.dumps(summary[tag], indent=2) + "\n")
    print(tag, "n_ok", len(ok), "FA_recall", summary[tag]["FA_mean_contact_recall_5A"])

(out_root / "interface_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("wrote", out_root / "interface_summary.json")
PY

for tag in base_pct10 psmp_pct10; do
  python3 "$PAPER/scripts/compute_extended_metrics.py" \
    --interface-json "${OUT}/interface_${tag}.json" \
    --cases-tsv "$CASES" \
    --out-json "${OUT}/interface_${tag}_ext.json" \
    2>&1 | tee -a "$LOG" || true
done

# FA F1 summary if ext present
python3 - <<'PY'
import json
from pathlib import Path
out = Path("/home/wangxindi/RNA_Protein/fusai/train/runs/temporal20")
rows = []
for tag in ("base_pct10", "psmp_pct10"):
    p = out / f"interface_{tag}_ext.json"
    if not p.is_file():
        continue
    d = json.loads(p.read_text())
    res = d.get("results") or d.get("per_case") or []
    if isinstance(d, list):
        res = d
    f1s = []
    for r in res:
        if isinstance(r, dict) and r.get("contact_f1_5A") is not None:
            f1s.append(float(r["contact_f1_5A"]))
        else:
            f1s.append(0.0)
    # if results nested
    if not f1s and isinstance(d, dict) and "results" in d:
        for r in d["results"]:
            f1s.append(float(r["contact_f1_5A"]) if r.get("contact_f1_5A") is not None else 0.0)
    fa = sum(f1s)/len(f1s) if f1s else float("nan")
    rows.append({"tag": tag, "n": len(f1s), "FA_F1": fa})
    print(tag, "FA_F1", fa, "n", len(f1s))
(out / "fa_f1_summary.json").write_text(json.dumps(rows, indent=2)+"\n")
PY

date -Is > "$PAPER/data/pipeline_markers/boost_temporal20.ok"
log "Temporal20 pipeline done"
