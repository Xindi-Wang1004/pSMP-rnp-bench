#!/usr/bin/env bash
# Launch all three boost tracks on idle GPUs 0,1,6,7.
set -eo pipefail
FUSAI=/home/wangxindi/RNA_Protein/fusai
PAPER=$FUSAI/nar_paper
SCR=$PAPER/scripts
MARK=$PAPER/data/pipeline_markers
mkdir -p "$MARK" "$PAPER/data" "$FUSAI/train/runs"

# Ensure scripts present (copied by deploy)
chmod +x "$SCR"/freeze_temporal20.py \
  "$SCR"/score_cluster_disjoint_fivesample.py \
  "$SCR"/run_temporal20_eval.sh \
  "$SCR"/run_boltz_temporal20.sh \
  "$SCR"/run_peft_pairformer_only_pct10.sh

# Track B CPU: homology five-sample rescore (immediate)
nohup python3 "$SCR/score_cluster_disjoint_fivesample.py" \
  > "$PAPER/data/boost_homology_fa5.log" 2>&1 &
echo "homology_pid=$!"

# Track A: Temporal20 on GPU 0+1
nohup env GPU_BASE=0 GPU_PSMP=1 NSAMPLE=1 bash "$SCR/run_temporal20_eval.sh" \
  > "$FUSAI/train/runs/temporal20.nohup" 2>&1 &
echo "temporal20_pid=$!"

# Track C1: Boltz Temporal20 on GPU 6 (waits for cases tsv)
nohup env CUDA_VISIBLE_DEVICES=6 bash "$SCR/run_boltz_temporal20.sh" \
  > "$FUSAI/train/runs/boltz2_temporal20.nohup" 2>&1 &
echo "boltz_pid=$!"

# Track C2: pairformer-only PEFT on GPU 7
nohup env CUDA_VISIBLE_DEVICES=7 bash "$SCR/run_peft_pairformer_only_pct10.sh" \
  > "$FUSAI/train/runs/peft_pairformer.nohup" 2>&1 &
echo "peft_pid=$!"

echo "launched. monitor:"
echo "  tail -f $FUSAI/train/runs/temporal20.nohup"
echo "  tail -f $FUSAI/train/runs/boltz2_temporal20.nohup"
echo "  tail -f $FUSAI/train/runs/peft_pairformer.nohup"
echo "  tail -f $PAPER/data/boost_homology_fa5.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
