#!/usr/bin/env bash
set -euo pipefail

# Request an interactive GPU allocation on VT ARC TinkerCliffs.
#
# Useful commands after allocation starts:
#   echo "$SLURM_JOBID"
#   hostname
#   nvidia-smi
#   squeue -u "$USER"
#   squeue -j "$SLURM_JOBID" -o "%.18i %.2t %.10M %.10L %R"

ALLOCATION_ID="${ALLOCATION_ID:-llms-lab}"
PARTITION="${PARTITION:-a100_normal_q}"
QOS="${QOS:-tc_a100_normal_short}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_HOURS="${NUM_HOURS:-1}"
NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-16}"

interact -A "$ALLOCATION_ID" \
  --partition "$PARTITION" \
  --qos "$QOS" \
  --cpus-per-task "$NUM_CPUS_PER_TASK" \
  --time="${NUM_HOURS}:00:00" \
  --gres="gpu:${NUM_GPUS}" \
  --verbose
