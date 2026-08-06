#!/usr/bin/env bash
# Auto-chain next Stage-2 steps when prerequisites finish.
# Safe to re-run: uses marker files under nar_paper/data/pipeline_markers/
set -euo pipefail

ROOT="/home/wangxindi/RNA_Protein/fusai"
PAPER="$ROOT/nar_paper"
DATA="$PAPER/data"
MARK="$DATA/pipeline_markers"
LOG="$DATA/pipeline_auto_chain.log"
PY="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
EVAL5="$ROOT/train/runs/rnp_real_lowdata/eval_work_5seed_val50"

mkdir -p "$MARK" "$DATA"
exec >>"$LOG" 2>&1

log() { echo "[$(date -Is)] $*"; }

contacts_done() {
  # export script finished (process gone) AND done line present, OR marker set
  [[ -f "$MARK/contacts_knn_scored.ok" ]] && return 0
  if pgrep -f 'export_native_contacts_server.sh' >/dev/null 2>&1; then
    return 1
  fi
  grep -q 'done ok=' "$DATA/export_contacts.nohup" 2>/dev/null
}

fivesample_infer_done() {
  [[ -f "$MARK/fivesample_infer.ok" ]] && return 0
  if pgrep -f 'rerun_val50_fivesample.sh' >/dev/null 2>&1; then
    return 1
  fi
  grep -q 'DONE five-sample launch pass' "$DATA/val50_fivesample_reinfer.log" 2>/dev/null
}

run_knn() {
  if [[ -f "$MARK/contacts_knn_scored.ok" ]]; then
    log "k-NN already scored"
    return 0
  fi
  log "=== run k-NN contact baseline ==="
  # ensure local mirror path used by script
  mkdir -p "$PAPER/data/stage12_raw/contacts"
  # prefer server contacts dir
  if [[ -d "$DATA/contacts" ]]; then
    rsync -a "$DATA/contacts/" "$PAPER/data/stage12_raw/contacts/" || cp -a "$DATA/contacts/." "$PAPER/data/stage12_raw/contacts/"
  fi
  # also keep audit/cases available
  mkdir -p "$PAPER/data/stage12_raw/audit" "$PAPER/data/stage12_raw/splits"
  [[ -f "$PAPER/data/stage12_raw/audit/TableS11_partner_sequence_audit.tsv" ]] || \
    cp -a "$ROOT/nar_paper_audit/out/TableS11_partner_sequence_audit.tsv" "$PAPER/data/stage12_raw/audit/" 2>/dev/null || true
  [[ -f "$PAPER/data/stage12_raw/splits/cases.tsv" ]] || \
    cp -a "$ROOT/train/data/rnp_real/val_protenix_inputs/cases.tsv" "$PAPER/data/stage12_raw/splits/"

  "$PY" "$PAPER/scripts/build_knn_contact_baseline.py" || log "WARN knn failed"
  touch "$MARK/contacts_knn_scored.ok"
  log "k-NN marker written"
}

run_score_5seed() {
  if [[ -f "$MARK/fivesample_scored.ok" ]]; then
    log "five-sample already scored"
    return 0
  fi
  log "=== score five-sample best-of-iptm ==="
  "$PY" "$PAPER/scripts/score_5seed_val50.py" || log "WARN score_5seed failed"
  log "=== build paired intersection 5seed ==="
  "$PY" "$PAPER/scripts/build_paired_intersection_5seed_val50.py" || log "WARN paired 5seed failed"
  log "=== failure-aware five-sample ==="
  "$PY" "$PAPER/scripts/build_failure_aware_fivesample.py" || log "WARN FA 5seed failed"
  touch "$MARK/fivesample_scored.ok"
  touch "$MARK/fivesample_infer.ok"
  log "five-sample scoring markers written"
  log "PIPELINE_CHAIN_COMPLETE"
}

log "watcher start pid=$$"

# If contacts already done at start, score immediately
if contacts_done; then
  run_knn
fi

# Main poll loop
for i in $(seq 1 720); do  # up to ~24h at 2 min
  if contacts_done; then
    run_knn
  else
    log "waiting contacts export... ($(ls "$DATA/contacts" 2>/dev/null | wc -l) json)"
  fi

  if fivesample_infer_done; then
    run_score_5seed
  else
    ok=$(grep -c succeeded "$DATA/val50_fivesample_reinfer.log" 2>/dev/null || echo 0)
    log "waiting five-sample infer... succeeded_lines=$ok"
  fi

  if [[ -f "$MARK/contacts_knn_scored.ok" && -f "$MARK/fivesample_scored.ok" ]]; then
    log "all chained steps done; exiting watcher"
    exit 0
  fi
  sleep 120
done

log "watcher timeout after 24h"
exit 1
