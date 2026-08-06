#!/usr/bin/env bash
# Boltz-2 on Temporal20 (yaml via competition build_alt_model_inputs; fixed GPU).
# Boltz has no public train CLI — this is cross-architecture confirmatory under FA contract.
set -uo pipefail

FUSAI="/home/wangxindi/RNA_Protein/fusai"
COMP="/home/wangxindi/RNA_Protein/competition"
PAPER="$FUSAI/nar_paper"
CASES="${CASES:-$FUSAI/train/runs/temporal20/cases_temporal20.tsv}"
TAG="${TAG:-boltz2_temporal20}"
RUN_DIR="${RUN_DIR:-$FUSAI/train/runs/boltz2_baseline_eval/${TAG}}"
PRED_ROOT="${RUN_DIR}/pred_boltz"
YAML_DIR="${RUN_DIR}/yaml"
LOG="${RUN_DIR}/pipeline.log"
BOLTZ_CACHE="${BOLTZ_CACHE:-$COMP/tools/boltz_cache}"
BOLTZ_BIN="${BOLTZ_BIN:-/home/wangxindi/miniconda3/envs/boltz/bin/boltz}"
PYTHON_SYS="${PYTHON_SYS:-/usr/bin/python3}"
PYTHON_BOLTZ="${PYTHON_BOLTZ:-/home/wangxindi/miniconda3/envs/boltz/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-3600}"
DIFFUSION_SAMPLES="${DIFFUSION_SAMPLES:-1}"
RECYCLING_STEPS="${RECYCLING_STEPS:-3}"

mkdir -p "$RUN_DIR" "$PRED_ROOT" "$YAML_DIR" "$BOLTZ_CACHE" "$PAPER/data/pipeline_markers"
: >> "$LOG"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

export PATH="/home/wangxindi/miniconda3/envs/boltz/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"

log "wait for Temporal20 cases"
for i in $(seq 1 180); do
  [[ -f "$CASES" ]] && break
  sleep 20
done
[[ -f "$CASES" ]] || { log "missing $CASES"; exit 1; }

mapfile -t CASE_NAMES < <(awk -F'\t' 'NR>1 && $1!="" {print $1}' "$CASES")
log "n_cases=${#CASE_NAMES[@]} gpu=$GPU"

# json dir = dirname of first json_path column
JSON_ROOT=$(awk -F'\t' 'NR==2 {print $2}' "$CASES" | xargs dirname)
log "json_root=$JSON_ROOT"

for name in "${CASE_NAMES[@]}"; do
  jf="${JSON_ROOT}/${name}.json"
  [[ -f "$jf" ]] || { log "skip missing $jf"; continue; }
  out_case="${PRED_ROOT}/${name}"
  [[ -f "${out_case}/done.marker" ]] && { log "skip done $name"; continue; }
  mkdir -p "$out_case"
  yaml="${YAML_DIR}/${name}.yaml"
  boltz_out="${RUN_DIR}/boltz_raw/${name}"
  case_log="${RUN_DIR}/boltz_${name}.log"

  log "[$name] prepare yaml"
  "$PYTHON_SYS" - <<PY >> "$LOG" 2>&1
import sys
from pathlib import Path
sys.path.insert(0, "${COMP}/scripts")
from build_alt_model_inputs import load_case, write_boltz_yaml
p = Path("${yaml}")
write_boltz_yaml(load_case(Path("${jf}")), p)
out = []
in_protein = False
for line in p.read_text().splitlines():
    if line.strip().startswith("- protein:"):
        in_protein = True
    elif line.strip().startswith("- "):
        in_protein = False
    out.append(line)
    if in_protein and line.strip().startswith("sequence:"):
        out.append("      msa: empty")
        in_protein = False
p.write_text("\n".join(out) + "\n")
print("wrote", p)
PY

  rm -rf "$boltz_out"
  mkdir -p "$boltz_out"
  : > "$case_log"
  set +e
  timeout --signal=TERM --kill-after=60 "$CASE_TIMEOUT_SEC" \
    stdbuf -oL -eL "$BOLTZ_BIN" predict "$yaml" \
      --out_dir "$boltz_out" \
      --cache "$BOLTZ_CACHE" \
      --accelerator gpu \
      --devices 1 \
      --diffusion_samples "$DIFFUSION_SAMPLES" \
      --recycling_steps "$RECYCLING_STEPS" \
      < /dev/null >> "$case_log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "[$name] FAILED rc=$rc"
    echo "$rc" > "${out_case}/fail.rc"
    tail -20 "$case_log" | tee -a "$LOG" || true
    continue
  fi

  set +e
  "$PYTHON_BOLTZ" - <<PY >> "$LOG" 2>&1
from pathlib import Path
import re, gemmi
raw = Path("${boltz_out}")
case_out = Path("${out_case}")
name = "${name}"
cifs = sorted(raw.rglob("*.cif"))
if not cifs:
    raise SystemExit(f"no boltz cif for {name}")
pref = [p for p in cifs if ("rank" in p.name.lower() or "model_0" in p.name or "pred" in p.name.lower())]
best = pref[0] if pref else cifs[0]
target = case_out / f"{name}_sample_0.cif"
st = gemmi.read_structure(str(best.resolve()))
prot_i = rna_i = 0
maps = []
for model in st:
    for chain in model:
        old = chain.name
        if re.fullmatch(r"P\\d+", old):
            prot_i += 1
            chain.name = "A" if prot_i == 1 else chr(ord("A") + prot_i - 1)
            maps.append(f"{old}->{chain.name}")
        elif re.fullmatch(r"R\\d+", old):
            rna_i += 1
            chain.name = "B" if rna_i == 1 else chr(ord("B") + rna_i - 1)
            maps.append(f"{old}->{chain.name}")
if target.exists() or target.is_symlink():
    target.unlink()
st.make_mmcif_document().write_file(str(target))
(case_out / "done.marker").write_text("ok\n")
(case_out / "boltz_cif_source.txt").write_text(str(best) + "\n")
(case_out / "chain_rename.txt").write_text(", ".join(maps) + "\n")
print(f"selected {best} rename={maps}")
PY
  set -e
  log "[$name] done"
done

date -Is > "$PAPER/data/pipeline_markers/boost_boltz_temporal20.ok"
log "boltz Temporal20 predict finished — score with compute_extended_metrics later"
