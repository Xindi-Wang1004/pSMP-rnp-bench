#!/usr/bin/env bash
# Compute precision/F1 on existing interface eval JSONs (no re-inference).
set -eo pipefail

FUSAI=/home/wangxindi/RNA_Protein/fusai
RUNS=$FUSAI/train/runs
VAL=$FUSAI/train/data/rnp_real/val_protenix_inputs
SCRIPT=$FUSAI/nar_paper/scripts/compute_extended_metrics.py
PT=$RUNS/paper_tables
LOW=$RUNS/rnp_real_lowdata/eval_work
LOG=$FUSAI/nar_paper/data/extended_eval.log
PYTHON=~/miniconda3/envs/protenix/bin/python3

mkdir -p "$PT" "$FUSAI/nar_paper/data" "$FUSAI/nar_paper/tables"

run_one() {
  local in="$1" cases="$2" out="$3"
  if [[ -f "$out" ]]; then
    echo "skip (exists): $out" | tee -a "$LOG"
    return 0
  fi
  echo "==> $out" | tee -a "$LOG"
  "$PYTHON" "$SCRIPT" --interface-json "$in" --cases-tsv "$cases" --out-json "$out" 2>&1 | tee -a "$LOG"
}

# 5-seed test10 / dev40
run_one "$PT/interface_5seed_base_pct10_test10.json" "$VAL/cases_test10.tsv" "$PT/interface_5seed_base_pct10_test10_ext.json"
run_one "$PT/interface_5seed_allmix_pct10_test10.json" "$VAL/cases_test10.tsv" "$PT/interface_5seed_allmix_pct10_test10_ext.json"
run_one "$PT/interface_5seed_base_pct10_dev40.json" "$VAL/cases_dev40.tsv" "$PT/interface_5seed_base_pct10_dev40_ext.json"
run_one "$PT/interface_5seed_allmix_pct10_dev40.json" "$VAL/cases_dev40.tsv" "$PT/interface_5seed_allmix_pct10_dev40_ext.json"

# val43 low-data curve
for pct in 10 25 50 100; do
  run_one "$LOW/interface_pct${pct}_base.json" "$VAL/cases.tsv" "$LOW/interface_pct${pct}_base_ext.json"
  run_one "$LOW/interface_pct${pct}_psmp.json" "$VAL/cases.tsv" "$LOW/interface_pct${pct}_psmp_ext.json"
done

# Chai-1 test10
run_one "$RUNS/chai1_baseline_eval/chai1_test10/interface_chai1_test10.json" "$VAL/cases_test10.tsv" \
  "$RUNS/chai1_baseline_eval/chai1_test10/interface_chai1_test10_ext.json"

# Aggregate summary for paper tables
"$PYTHON" "$FUSAI/nar_paper/scripts/build_extended_tables.py"

# Statistics + figures
"$PYTHON" "$FUSAI/nar_paper/scripts/paper_statistics.py"

echo "Done."
