#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/assignment03"

python3 graders/grade_stage1_executor.py
python3 graders/grade_stage2_reflector.py
python3 graders/grade_stage3_proposer.py
python3 graders/grade_stage4_harness.py || {
  echo
  echo "Stage 4 code checks may pass while the final proof check fails until evolution_proof.json exists."
  exit 1
}

cat <<'MSG'

Optional proof-dependent checks:
  python3 graders/grade_stage0.py
  PYTHONPATH=. python3 graders/grade_smoke_proof.py

Those checks require generated proof JSON files:
  sandbox_proof.json
  smoke_proof.json
  evolution_proof.json
MSG
