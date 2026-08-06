#!/usr/bin/env bash
# Helpers: discover idle GPUs (low memory + no compute apps).
# Usage: source this file, then call idle_gpus / pick_idle_gpu

FREE_MIB="${FREE_MIB:-800}"

# Print space-separated idle GPU indices (sorted).
idle_gpus() {
  local free_mib="${1:-$FREE_MIB}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  # memory.used below threshold AND no running compute apps on that index
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v thr="$free_mib" '{
        gsub(/ /,"",$1); gsub(/ /,"",$2);
        if (($2+0) < thr) print $1
      }' \
    | while read -r idx; do
        apps=$(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)
        apps=${apps:-0}
        if [[ "$apps" -eq 0 ]]; then
          echo "$idx"
        fi
      done
}

# Echo one idle GPU index, or empty if none. Optional exclude list: "1 2"
pick_idle_gpu() {
  local exclude="${1:-}"
  local g
  for g in $(idle_gpus); do
    if [[ " $exclude " == *" $g "* ]]; then
      continue
    fi
    echo "$g"
    return 0
  done
  return 1
}
