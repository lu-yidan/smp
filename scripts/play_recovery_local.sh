#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd /home/luyd/workspace/smp-flat93
export PYTHONPATH=src:scripts:.
export OMP_NUM_THREADS=4
# Let the native viewer use the desktop's OpenGL context.
unset MUJOCO_GL
exec .venv/bin/python "$script_dir/play_recovery_checkpoint.py" "$@"
