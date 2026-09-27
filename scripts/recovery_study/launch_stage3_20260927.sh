#!/usr/bin/env bash
set -euo pipefail
cd /root/workplace/smp-stage3
.venv/bin/python - <<'PY'
import json
from pathlib import Path
arms=['S3-C2','S3-C3','S3-G0','S3-P0','S3-C0','S3-C1','S3-PG0']
for arm in arms:
 p=Path('outputs/preflight_stage3')/(arm+'_4096')
 assert (p/'completed.json').is_file(),f'Incomplete preflight: {arm}'
 assert not (p/'failed.json').exists(),f'Failed preflight: {arm}'
 for name in ('partial_reset_pass','reward_contract_pass'):
  assert json.loads((p/(name+'.json')).read_text())['pass']
 a=json.loads((p/'launch.json').read_text())
 assert a['num_envs']==4096 and a['actor_exact'] and a['actor_noise']
 assert a['checkpoint_sha256']=='8b4889f80c6b7cc675f6e9b1e98f2d4a1886a15e372f86070f63c329ba9ca5d1'
assert Path('outputs/initial_stage3_R2_final/summary.json').is_file(),'Initial R2 evaluation incomplete'
PY
runroot=logs/rsl_rl/stage3_recovery/formal_20260927_v1
mkdir -p "$runroot"
for entry in '0 S3-C2' '1 S3-C3' '2 S3-G0' '3 S3-P0' '4 S3-C0' '5 S3-C1' '6 S3-PG0'; do
 read -r gpu arm <<< "$entry"
 test ! -e "$runroot/$arm"
 nohup env PYTHONPATH=src:scripts:. CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MUJOCO_GL=egl WANDB_MODE=online WANDB_RUN_GROUP=stage3_R2_20260927_v1 .venv/bin/python -u scripts/train_stage3_recovery.py --arm "$arm" --out "$runroot/$arm" --num-envs 4096 --updates 10000 --seed 20260927 --eval-gpu 7 > "$runroot/$arm.stdout.log" 2>&1 < /dev/null &
 echo "$!" > "$runroot/$arm.pid"
 echo "$arm gpu=$gpu pid=$!"
done
