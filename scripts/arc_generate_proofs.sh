#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for ARC proof generation.
# Run this inside a VT ARC GPU allocation after environment setup.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/assignment03"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Export it in your shell before running GPU proof generation:"
  echo "  export HF_TOKEN=hf_your_token"
  exit 1
fi

python3 arc_proofs.py all "$@"

cat <<'MSG'

Proof generation finished. Suggested verification commands:
  python3 graders/grade_stage0.py
  PYTHONPATH=. python3 graders/grade_smoke_proof.py
  python3 graders/grade_stage4_harness.py
MSG
