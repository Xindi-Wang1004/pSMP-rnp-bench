#!/usr/bin/env bash
# Idle-GPU auto queue for remaining NAR review experiments.
# Safe / idempotent: marker files under data/pipeline_markers/
#
# Start once on server:
#   nohup bash scripts/idle_gpu_auto_queue.sh > data/idle_gpu_auto_queue.nohup 2>&1 &
#
# Priority (GPU):
#   1) multiseed val50 eval for pct10 seeds 43/44/45
#   2) five-sample incomplete backfill (failed/missing small cases)
#   3) pct25 multiseed finetune+eval seeds 43/44/45
#   4) large-case single-sample backfill (tokens 4k–8k)
#
# Always (CPU, each loop):
#   inventory incomplete five-sample; cluster-disjoint / multiseed / external FA
set -euo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
PAPER="$ROOT/nar_paper"
DATA="$PAPER/data"
MARK="$DATA/pipeline_markers"
LOG="$DATA/idle_gpu_auto_queue.log"
PY="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
MULTI_ROOT="$ROOT/train/runs/rnp_real_lowdata_multiseed"
POLL_SEC="${POLL_SEC:-120}"
FREE_MIB="${FREE_MIB:-800}"
RESERVE_GPUS="${RESERVE_GPUS:-}"
ENABLE_PCT25_MULTI="${ENABLE_PCT25_MULTI:-1}"
ENABLE_FIVESAMPLE_BACKFILL="${ENABLE_FIVESAMPLE_BACKFILL:-1}"
ENABLE_LARGE_BACKFILL="${ENABLE_LARGE_BACKFILL:-1}"
MAX_LOOPS="${MAX_LOOPS:-720}"

mkdir -p "$MARK" "$DATA"
exec 9>"$DATA/idle_gpu_auto_queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another idle_gpu_auto_queue is running; exit"
  exit 0
fi

exec >>"$LOG" 2>&1
# shellcheck source=/dev/null
source "$PAPER/scripts/lib_idle_gpu.sh"

log() { echo "[$(date -Is)] $*"; }

ema_ready() {
  local tag="$1"
  compgen -G "$MULTI_ROOT/$tag"/*/checkpoints/*_ema_0.999.pt > /dev/null
}

multiseed_train_done() {
  local seed="$1" pct="${2:-pct10}"
  ema_ready "${pct}_base_seed${seed}" && ema_ready "${pct}_psmp_seed${seed}"
}

launch_bg() {
  local gpu="$1"
  shift
  local logfile="$1"
  shift
  local runmark=""
  if [[ "${1:-}" == RUNMARK=* ]]; then
    runmark="${1#RUNMARK=}"
    shift
  fi
  log "LAUNCH gpu=$gpu cmd=$* log=$logfile"
  (
    [[ -n "$runmark" ]] && echo $$ >"$runmark"
    env CUDA_VISIBLE_DEVICES="$gpu" "$@"
    ec=$?
    [[ -n "$runmark" ]] && rm -f "$runmark"
    exit "$ec"
  ) >"$logfile" 2>&1 < /dev/null &
  local pid=$!
  [[ -n "$runmark" ]] && echo "$pid" >"$runmark"
  echo "$pid"
}

is_alive_mark() {
  local mark="$1"
  [[ -f "$mark" ]] || return 1
  local pid
  pid=$(cat "$mark" 2>/dev/null || true)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

tag_needs_fivesample_backfill() {
  local tag="$1"
  "$PY" - <<PY
import csv
from pathlib import Path
cases=list(csv.DictReader(open("$ROOT/train/data/rnp_real/val_protenix_inputs/cases.tsv"), delimiter="\t"))
pred=Path("$ROOT/train/runs/rnp_real_lowdata/eval_work_5seed_val50")/f"pred_$tag"
n=0
for c in cases:
    if float(c.get("num_tokens") or 0) > 4000: continue
    name=c["name"]
    have=len(list(pred.glob(f"**/{name}_sample_*.cif"))) if pred.is_dir() else 0
    if have < 5: n += 1
print(n)
PY
}

run_cpu_maintenance() {
  "$PY" "$PAPER/scripts/list_fivesample_incomplete.py" >/dev/null 2>&1 || true

  if [[ ! -f "$MARK/cluster_disjoint_scored.ok" ]] || [[ "${FORCE_CPU:-0}" == "1" ]]; then
    log "CPU: score cluster-disjoint subset"
    if "$PY" "$PAPER/scripts/score_cluster_disjoint_subset.py"; then
      touch "$MARK/cluster_disjoint_scored.ok"
    else
      log "WARN cluster-disjoint score failed"
    fi
  fi
  if [[ ! -f "$MARK/external_fa.ok" ]]; then
    "$PY" "$PAPER/scripts/build_external_failure_aware.py" && touch "$MARK/external_fa.ok" || true
  fi
  if [[ ! -f "$MARK/cluster_list.ok" ]]; then
    "$PY" "$PAPER/scripts/build_cluster_disjoint_candidates.py" && touch "$MARK/cluster_list.ok" || true
  fi
  if compgen -G "$MARK/multiseed_eval_*.ok" > /dev/null; then
    "$PY" "$PAPER/scripts/build_multiseed_summary.py" || true
  fi
}

multiseed_eval_busy() {
  local seed="$1" pct="${2:-pct10}"
  # argv contains pred path or SEED= in env wrapper cmdline
  pgrep -af "eval_work_val50/pred_${pct}_.*_seed${seed}" >/dev/null 2>&1 \
    && return 0
  pgrep -af "eval_multiseed_val50.sh" >/dev/null 2>&1 || return 1
  # check environ of eval bash processes for SEED=
  local pid
  for pid in $(pgrep -f 'eval_multiseed_val50.sh' || true); do
    if tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | grep -qx "SEED=${seed}" \
      && tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | grep -qx "PCT=${pct}"; then
      return 0
    fi
  done
  return 1
}

try_launch_multiseed_eval() {
  local seed gpu runmark n=0
  for seed in 43 44 45; do
    [[ -f "$MARK/multiseed_eval_pct10_seed${seed}.ok" ]] && continue
    runmark="$MARK/multiseed_eval_pct10_seed${seed}.running"
    if is_alive_mark "$runmark" || multiseed_eval_busy "$seed" pct10; then
      # refresh mark from live pid if missing
      if ! is_alive_mark "$runmark"; then
        local pid
        pid=$(pgrep -f 'eval_multiseed_val50.sh' | while read -r p; do
          tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null | grep -qx "SEED=${seed}" && echo "$p" && break
        done || true)
        [[ -n "${pid:-}" ]] && echo "$pid" >"$runmark"
      fi
      continue
    fi
    rm -f "$runmark"
    multiseed_train_done "$seed" pct10 || { log "wait pct10 train seed=$seed"; continue; }
    gpu=$(pick_idle_gpu "$RESERVE_GPUS" || true)
    [[ -n "${gpu:-}" ]] || break
    echo "claiming" >"$runmark"
    RESERVE_GPUS="${RESERVE_GPUS} ${gpu}"
    launch_bg "$gpu" "$DATA/multiseed_eval_seed${seed}.nohup" \
      "RUNMARK=$runmark" \
      env SEED="$seed" PCT=pct10 INITS="base psmp" \
      bash "$PAPER/scripts/eval_multiseed_val50.sh" >/dev/null
    n=$((n + 1))
  done
  [[ "$n" -gt 0 ]]
}

try_launch_fivesample_backfill() {
  [[ "$ENABLE_FIVESAMPLE_BACKFILL" == "1" ]] || return 1
  local tag gpu runmark need
  # prioritize known hole tags first
  for tag in pct50_psmp pct50_base pct100_psmp pct100_base pct25_psmp pct25_base pct10_psmp pct10_base; do
    [[ -f "$MARK/fivesample_backfill_${tag}.ok" ]] && continue
    runmark="$MARK/fivesample_backfill_${tag}.running"
    if is_alive_mark "$runmark"; then continue; fi
    rm -f "$runmark"
    # avoid colliding with the main five-sample job on the same tag
    if pgrep -af 'rerun_val50_fivesample.sh' >/dev/null 2>&1; then
      if grep -q "five-sample ${tag} " "$DATA/val50_fivesample_reinfer.log" 2>/dev/null \
         && ! grep -q "DONE five-sample launch pass" "$DATA/val50_fivesample_reinfer.log" 2>/dev/null; then
        # if main job's latest tag header is this tag, defer
        local latest
        latest=$(grep -E '=== five-sample ' "$DATA/val50_fivesample_reinfer.log" | tail -1 || true)
        if [[ "$latest" == *" $tag "* ]]; then
          log "defer backfill $tag (main five-sample currently on this tag)"
          continue
        fi
      fi
    fi
    need=$(tag_needs_fivesample_backfill "$tag" || echo 0)
    need=${need//[^0-9]/}
    if [[ "${need:-0}" -eq 0 ]]; then
      touch "$MARK/fivesample_backfill_${tag}.ok"
      continue
    fi
    gpu=$(pick_idle_gpu "$RESERVE_GPUS" || true)
    [[ -n "${gpu:-}" ]] || return 1
    echo "claiming" >"$runmark"
    RESERVE_GPUS="${RESERVE_GPUS} ${gpu}"
    log "backfill needed $tag n_missing=$need"
    launch_bg "$gpu" "$DATA/fivesample_backfill_${tag}.nohup" \
      "RUNMARK=$runmark" \
      env TAG="$tag" NSAMPLE=5 MAX_TOKENS=4000 CLEAN_PARTIAL=1 \
      bash "$PAPER/scripts/backfill_fivesample_incomplete.sh" >/dev/null
    return 0
  done
  return 1
}

try_launch_pct25_multiseed() {
  [[ "$ENABLE_PCT25_MULTI" == "1" ]] || return 1
  local seed gpu runmark n=0
  for seed in 43 44 45; do
    [[ -f "$MARK/multiseed_finetune_pct25_seed${seed}.ok" ]] && continue
    runmark="$MARK/multiseed_finetune_pct25_seed${seed}.running"
    if is_alive_mark "$runmark"; then continue; fi
    rm -f "$runmark"
    # prefer after pct10 train done for that seed (already true for 43/44/45)
    multiseed_train_done "$seed" pct10 || continue
    if ema_ready "pct25_base_seed${seed}" && ema_ready "pct25_psmp_seed${seed}"; then
      # train done; ensure eval
      if [[ ! -f "$MARK/multiseed_eval_pct25_seed${seed}.ok" ]]; then
        gpu=$(pick_idle_gpu "$RESERVE_GPUS" || true)
        [[ -n "${gpu:-}" ]] || break
        echo "claiming" >"$runmark"
        RESERVE_GPUS="${RESERVE_GPUS} ${gpu}"
        launch_bg "$gpu" "$DATA/multiseed_pct25_eval_seed${seed}.nohup" \
          "RUNMARK=$runmark" \
          env SEED="$seed" PCT=pct25 INITS="base psmp" \
          bash "$PAPER/scripts/eval_multiseed_val50.sh" >/dev/null
        # eval script writes multiseed_eval_pct25 marker? it writes pct25 in tag — touch both on wrapper
        n=$((n + 1))
        continue
      fi
      touch "$MARK/multiseed_finetune_pct25_seed${seed}.ok"
      continue
    fi
    gpu=$(pick_idle_gpu "$RESERVE_GPUS" || true)
    [[ -n "${gpu:-}" ]] || break
    echo "claiming" >"$runmark"
    RESERVE_GPUS="${RESERVE_GPUS} ${gpu}"
    launch_bg "$gpu" "$DATA/multiseed_pct25_seed${seed}.nohup" \
      "RUNMARK=$runmark" \
      env SEED="$seed" DO_EVAL=1 \
      bash "$PAPER/scripts/run_pct25_multiseed_queue.sh" >/dev/null
    n=$((n + 1))
  done
  [[ "$n" -gt 0 ]]
}

try_launch_large_backfill() {
  [[ "$ENABLE_LARGE_BACKFILL" == "1" ]] || return 1
  local tag gpu runmark
  for tag in pct10_psmp pct10_base pct100_psmp pct100_base pct50_psmp pct50_base pct25_psmp pct25_base; do
    [[ -f "$MARK/large_backfill_${tag}.ok" ]] && continue
    runmark="$MARK/large_backfill_${tag}.running"
    if is_alive_mark "$runmark"; then continue; fi
    rm -f "$runmark"
    if [[ ! -f "$MARK/fivesample_infer.ok" ]] && ! grep -q 'DONE five-sample launch pass' "$DATA/val50_fivesample_reinfer.log" 2>/dev/null; then
      log "defer large backfill until five-sample small pass finishes"
      return 1
    fi
    gpu=$(pick_idle_gpu "$RESERVE_GPUS" || true)
    [[ -n "${gpu:-}" ]] || return 1
    echo "claiming" >"$runmark"
    RESERVE_GPUS="${RESERVE_GPUS} ${gpu}"
    launch_bg "$gpu" "$DATA/large_backfill_${tag}.nohup" \
      "RUNMARK=$runmark" \
      env TAG="$tag" NSAMPLE=1 MAX_TOKENS=8000 \
      bash "$PAPER/scripts/rerun_val50_large_backfill.sh" >/dev/null
    return 0
  done
  return 1
}

all_done() {
  [[ -f "$MARK/multiseed_eval_pct10_seed43.ok" ]] \
    && [[ -f "$MARK/multiseed_eval_pct10_seed44.ok" ]] \
    && [[ -f "$MARK/multiseed_eval_pct10_seed45.ok" ]] || return 1
  if [[ "$ENABLE_FIVESAMPLE_BACKFILL" == "1" ]]; then
    [[ -f "$MARK/fivesample_backfill_pct50_psmp.ok" ]] || return 1
  fi
  if [[ "$ENABLE_PCT25_MULTI" == "1" ]]; then
    [[ -f "$MARK/multiseed_finetune_pct25_seed43.ok" ]] || return 1
  fi
  if [[ "$ENABLE_LARGE_BACKFILL" == "1" ]]; then
    [[ -f "$MARK/large_backfill_pct10_psmp.ok" ]] || return 1
  fi
  return 0
}

log "idle GPU queue start pid=$$ FREE_MIB=$FREE_MIB ENABLE_PCT25_MULTI=$ENABLE_PCT25_MULTI ENABLE_FIVESAMPLE_BACKFILL=$ENABLE_FIVESAMPLE_BACKFILL ENABLE_LARGE_BACKFILL=$ENABLE_LARGE_BACKFILL RESERVE_GPUS=$RESERVE_GPUS"

for ((i=1; i<=MAX_LOOPS; i++)); do
  run_cpu_maintenance
  try_launch_multiseed_eval || true
  try_launch_fivesample_backfill || true
  try_launch_pct25_multiseed || true
  try_launch_large_backfill || true

  idle=$(idle_gpus | tr '\n' ' ' || true)
  log "loop $i idle_gpus=[${idle}] markers=$(ls "$MARK"/*.ok 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

  if all_done; then
    log "IDLE_GPU_QUEUE_COMPLETE"
    exit 0
  fi
  sleep "$POLL_SEC"
done

log "idle GPU queue timeout"
exit 1
