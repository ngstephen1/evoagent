#!/usr/bin/env bash
set -euo pipefail

# Prepare a Python environment for EvoAgent on a VT ARC compute node.
#
# Run this after you have an interactive GPU allocation:
#   scripts/arc_interact_a100.sh
#   cd /home/<YOUR_VT_PID>/temp/evoagent
#   scripts/arc_setup_env.sh
#
# This installs the assignment's lightweight requirements. It does not blindly
# install the heavy GPU stack. QwenInference requires SGLang, Transformers, and
# a CUDA-compatible PyTorch build; install those only after confirming the ARC
# module/runtime combination you plan to use.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-evoagent}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/evoagent}"

cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
  # Module names can vary by ARC software stack. These attempts are safe no-ops
  # if Miniforge3 is not available in the active environment.
  module --silent load Miniforge3 2>/dev/null || \
    module --silent load miniforge3 2>/dev/null || true
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
  fi
  conda activate "$ENV_NAME"
else
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

python3 --version
python3 -m pip install --upgrade pip
python3 -m pip install -r assignment03/requirements.txt

cat <<'MSG'

GPU runtime note:
  assignment03/src/model.py imports SGLang and Transformers at model-load time.
  The assignment requirements intentionally avoid installing the full CUDA stack
  because the original Modal image provides it.

  Before running long inference jobs on ARC, confirm a compatible PyTorch +
  SGLang runtime for the allocated GPU/CUDA environment. If your ARC software
  stack does not already provide SGLang, install it intentionally, for example:

    python3 -m pip install "transformers>=4.51.0"
    python3 -m pip install "sglang[all]>=0.5.13"

  This may install large packages. Do it only on ARC, not in the repository.
MSG

python3 - <<'PY'
try:
    import torch
    print("torch.cuda.is_available()", torch.cuda.is_available())
    print("torch.cuda.device_count()", torch.cuda.device_count())
except Exception as exc:
    print("torch check failed:", exc)
PY

python3 - <<'PY'
try:
    import sglang
    print("sglang import OK:", getattr(sglang, "__version__", "version unknown"))
except Exception as exc:
    print("sglang import failed:", exc)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found. Are you on a GPU compute node?"
fi
