#!/usr/bin/env bash
set -euo pipefail
cd /mnt/workspace/user/luyidan/smp-a6-paper
exec 9>/tmp/smp-a6-paper-dsw1.lock
flock -n 9 || { echo 'An A6 controller is already active'; exit 1; }
export PYTHONPATH=src:scripts:.:tests
export MUJOCO_GL=egl OMP_NUM_THREADS=4
export WANDB_MODE=online WANDB_ENTITY=tabletennis
export WANDB_RUN_GROUP=a6_alpha_convergence_20260930_seed20261013
export NETRC=/root/workspace/smp-a6-tools/private/wandb.netrc
arms=(AC0_a005 AC1_a020 AC2_a050 AC3_a100 AC4_close AC5_anchor AC6_close_anchor)
pre=outputs/preflight_a6_alpha/20260930_v1
run=logs/rsl_rl/a6_alpha/formal_20260930_seed20261013
mkdir -p "$pre" "$run"
for arm in "${arms[@]}"; do
  test ! -e "$pre/$arm" && test ! -e "$run/$arm"
done
.venv/bin/python -m unittest discover -s tests -p 'test_a6*py' -v
pids=()
for gpu in "${!arms[@]}"; do
  arm=${arms[$gpu]}
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/train_a6_alpha_convergence.py --arm "$arm" --out "$pre/$arm" --preflight --num-envs 4096 --updates 200 > "$pre/$arm.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$pre/$arm.pid"
  echo "PREFLIGHT $arm GPU$gpu pid=$!"
done
# Exercise final evaluation size, renderer, and all newly saved metrics.
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u scripts/train_a6_alpha_convergence.py --arm AC6_close_anchor --out "$pre/eval2048" --eval --num-envs 2048 --steps 1000 --video > "$pre/eval2048.log" 2>&1 &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0 || { echo 'Preflight failed; no formal run started'; exit 1; }
.venv/bin/python - <<'PY'
import json
import numpy as np
from pathlib import Path
from smp.rl.tasks.getup.a6_alpha_convergence import ARMS
root=Path('outputs/preflight_a6_alpha/20260930_v1')
rows=[]
for arm in ARMS:
 p=root/arm
 assert not (p/'failed.json').exists()
 assert json.loads((p/'completed.json').read_text())['iteration']>=199
 assert json.loads((p/'reward_contract.json').read_text())['pass']
 assert json.loads((p/'partial_reset_pass.json').read_text())['pass']
 a=json.loads((p/'launch.json').read_text())
 assert a['num_envs']==4096 and a['updates']==200 and a['actor_noise']
 assert a['counts']['scene']==[2048,1024,1024,0]
 assert a['counts']['kind_low_middle_late']==[2868,820,408]
 assert a['counts']['natural_procedural']==[3380,716]
 rows.append(a)
for key in ('actor_sha','critic_sha','initial_qpos_sha','assets','code_sha256'):
 assert all(a[key]==rows[0][key] for a in rows),key
p=root/'eval2048'
s=json.loads((p/'summary.json').read_text())
assert s['guided_plate']['n']==512 and s['free_plate']['n']==512
a=np.load(p/'initial.npz')
for scene in (1,2):
 for direction in range(4):
  ix=(a['scene']==scene)&(a['direction']==direction)
  assert ix.sum()==128
  assert (a['source'][ix]==1).sum()==32
v=np.load(p/'per_trial.npz')
assert v['post_stand_max_offset'].shape==(2048,)
assert np.isfinite(v['peaks']).all()
assert all((p/(name+'_20s.mp4')).stat().st_size>10000 for name in ('flat','guided_plate','free_plate'))
print('Seven 4096-env preflights and 2048-env video evaluation passed; starting formal runs')
PY
for gpu in "${!arms[@]}"; do
  arm=${arms[$gpu]}
  nohup env CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/train_a6_alpha_convergence.py --arm "$arm" --out "$run/$arm" --num-envs 4096 --updates 10000 --seed 20261013 --eval-gpu 7 > "$run/$arm.stdout.log" 2>&1 < /dev/null 9>&- &
  printf '%s\n' "$!" > "$run/$arm.pid"
  echo "FORMAL $arm GPU$gpu pid=$!"
done
