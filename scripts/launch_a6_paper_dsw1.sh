#!/usr/bin/env bash
set -euo pipefail
cd /mnt/workspace/user/luyidan/smp-a6-paper
exec 9>/tmp/smp-a6-paper-dsw1.lock
flock -n 9 || { echo 'An A6 controller is already active'; exit 1; }
export PYTHONPATH=src:scripts:.
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export WANDB_MODE=online
export WANDB_ENTITY=tabletennis
export WANDB_RUN_GROUP=a6_paper_20260929_seed20261013
export NETRC=/root/workspace/smp-a6-tools/private/wandb.netrc
arms=(A6_full A6_no_bundle A6_no_geometry A6_no_alpha A6_no_hand A6_no_Q A6_no_L)
pre=outputs/preflight_a6_paper/formal_size_v1
run=logs/rsl_rl/a6_paper/formal_20260929_seed20261013
mkdir -p "$pre" "$run"
for arm in "${arms[@]}"; do
  test ! -e "$pre/$arm" && test ! -e "$run/$arm"
done
.venv/bin/python -m unittest discover -s tests -p test_a6_paper_config.py -v
pids=()
for gpu in "${!arms[@]}"; do
  arm=${arms[$gpu]}
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/train_a6_paper_ablation.py --arm "$arm" --out "$pre/$arm" --preflight --num-envs 4096 --updates 200 > "$pre/$arm.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$pre/$arm.pid"
  echo "PREFLIGHT $arm GPU$gpu pid=$!"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0 || { echo 'Preflight failed; no formal run started'; exit 1; }
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from smp.rl.tasks.getup.a6_paper_ablation import ARMS
rows=[]
for arm in ARMS:
 p=Path('outputs/preflight_a6_paper/formal_size_v1')/arm
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
print('All seven preflights completed and matched')
PY
for suite in nominal upper130; do
  mass=1.; test "$suite" != upper130 || mass=1.3
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u scripts/train_a6_paper_ablation.py --arm A6_full --out "outputs/a6_paper_R2_baseline/$suite" --eval --num-envs 192 --steps 1000 --stress-upper "$mass" > "$pre/R2_$suite.log" 2>&1
done
echo 'Baseline evaluation passed; starting seven formal runs from original R2'
for gpu in "${!arms[@]}"; do
  arm=${arms[$gpu]}
  nohup env CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/train_a6_paper_ablation.py --arm "$arm" --out "$run/$arm" --num-envs 4096 --updates 10000 --seed 20261013 --eval-gpu 7 > "$run/$arm.stdout.log" 2>&1 < /dev/null 9>&- &
  printf '%s\n' "$!" > "$run/$arm.pid"
  echo "FORMAL $arm GPU$gpu pid=$!"
done
