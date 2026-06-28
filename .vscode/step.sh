#!/usr/bin/env bash

# VT ARC quick reference for EvoAgent.
# This file is intentionally safe: no tokens, no private keys, no secrets.

export VT_PID="${VT_PID:-stephenallstar24}"
export LOGIN_HOSTNAME="${LOGIN_HOSTNAME:-tinkercliffs1}"

# Login:
# ssh -i ~/.ssh/arc "$VT_PID@$LOGIN_HOSTNAME.arc.vt.edu"

# Recommended remote workspace:
# cd /home/$VT_PID/temp/evoagent

# tmux:
# tmux new -s evoagent
# tmux a -t evoagent
# Detach with: Ctrl-b d

export ALLOCATION_ID="${ALLOCATION_ID:-llms-lab}"
export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_HOURS="${NUM_HOURS:-1}"
export NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-16}"
export PARTITION="${PARTITION:-a100_normal_q}"
export QOS="${QOS:-tc_a100_normal_short}"

# Interactive A100 request:
# interact -A "$ALLOCATION_ID" \
#   --partition "$PARTITION" \
#   --qos "$QOS" \
#   --cpus-per-task "$NUM_CPUS_PER_TASK" \
#   --time="${NUM_HOURS}:00:00" \
#   --gres="gpu:${NUM_GPUS}" \
#   --verbose

# Monitoring:
# squeue
# squeue -j "$SLURM_JOBID" -o "%.18i %.2t %.10M %.10L %R"
# echo "$SLURM_JOBID"
# hostname
# nvidia-smi

# Cancel a job:
# scancel <jobid>

# Environment setup after entering a GPU node:
# scripts/arc_setup_env.sh

# Local graders:
# scripts/arc_run_local_graders.sh

# Proof helpers on ARC, after confirming the GPU env and HF token:
# export HF_TOKEN=...
# cd assignment03
# python3 arc_proofs.py sandbox
# python3 arc_proofs.py smoke
# python3 arc_proofs.py evolution
# python3 arc_proofs.py all
