#!/usr/bin/env bash
set -euo pipefail

# Login helper for VT ARC TinkerCliffs.
#
# Usage:
#   export VT_PID=your_vt_pid
#   scripts/arc_login.sh
#
# This script contains no secrets. It expects your private key at ~/.ssh/arc.

LOGIN_HOST="${LOGIN_HOST:-tinkercliffs1.arc.vt.edu}"
PRIVATE_KEY="${PRIVATE_KEY:-$HOME/.ssh/arc}"

if [[ -z "${VT_PID:-}" ]]; then
  echo "ERROR: VT_PID is not set."
  echo "Example: export VT_PID=your_vt_pid"
  exit 1
fi

if [[ ! -f "$PRIVATE_KEY" ]]; then
  echo "ERROR: SSH private key not found: $PRIVATE_KEY"
  exit 1
fi

exec ssh -i "$PRIVATE_KEY" "${VT_PID}@${LOGIN_HOST}"
