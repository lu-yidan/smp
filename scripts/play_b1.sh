#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=src:scripts:.
export OMP_NUM_THREADS=4
unset MUJOCO_GL
exec .venv/bin/python -u scripts/play_balanced.py "$@"
