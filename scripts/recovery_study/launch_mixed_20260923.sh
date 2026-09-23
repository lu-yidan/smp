#!/usr/bin/env bash
set -euo pipefail
cd /root/workplace/smp-recovery-study
.venv/bin/python - <<'PY'
import json
from pathlib import Path
for i in range(3):
 p=Path(f'outputs/preflight_mixed/m{i}_v8_4096')
 assert (p/'completed.json').is_file(), f'Preflight incomplete: {p}'
 assert not (p/'failed.json').exists()
 assert json.loads((p/'partial_reset_pass.json').read_text())['pass']
for name in ('R2','A6'):
 assert (Path('outputs/initial_mixed_v2')/name/'summary.json').is_file(),f'Missing frozen evaluation: {name}'
PY
runroot=logs/rsl_rl/mixed_recovery/formal_20260923_v2
mkdir -p "$runroot"
for entry in '0 M0_R2_mix' '1 M1_R2_mix' '2 M2_A6_mix'; do
 read -r gpu arm <<< "$entry"
 test ! -e "$runroot/$arm"
 nohup env PYTHONPATH=src:scripts:. CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MUJOCO_GL=egl WANDB_MODE=online WANDB_RUN_GROUP=mixed_recovery_20260923_v2 .venv/bin/python -u scripts/train_mixed_recovery.py --arm "$arm" --out "$runroot/$arm" --num-envs 4096 --updates 10000 > "$runroot/$arm.stdout.log" 2>&1 < /dev/null &
 echo "$!" > "$runroot/$arm.pid"
 echo "$arm gpu=$gpu pid=$!"
done
