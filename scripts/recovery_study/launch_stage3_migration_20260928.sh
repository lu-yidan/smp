#!/usr/bin/env bash
# Recover the core comparison on the replacement host. No changes to rewards/reset.
set -euo pipefail
root=${STAGE3_WORKSPACE:-/root/workplace/smp-stage3}
cd "$root"
exec 9> /tmp/smp-stage3-migration-20260928.lock
flock -n 9 || { echo 'Migration controller already running'; exit 1; }
export PYTHONPATH=src:scripts:.
export OMP_NUM_THREADS=4
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB_RUN_GROUP=stage3_R2_20260928_replacement_v1
export NETRC=/root/workplace/stage3-tools/private/wandb.netrc
checkpoint=outputs/clutter_eval_v1/checkpoints/R2_9000.pt
pre=outputs/preflight_stage3_migration_20260928
baseline=outputs/initial_stage3_R2_migration_20260928
runroot=logs/rsl_rl/stage3_recovery/formal_20260928_replacement_v1
mkdir -p "$pre" "$runroot"
for path in "$pre/S3-C2" "$pre/S3-G0" "$baseline" "$runroot/S3-C2" "$runroot/S3-G0"; do
  test ! -e "$path" || { echo "Refusing to overwrite $path"; exit 1; }
done
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('outputs/clutter_eval_v1/checkpoints/R2_9000.pt')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='8b4889f80c6b7cc675f6e9b1e98f2d4a1886a15e372f86070f63c329ba9ca5d1'
for split in ('train','validation'):
 p=Path('outputs/stage3_bank_v1')/(split+'.npz')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads(p.with_suffix('.json').read_text())['sha256']
PY

# GPU2 belongs to an existing job. GPU5 is reserved for serialized evaluation.
# Preflight uses the actual 4096-env workload, then its trained weights are discarded.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/train_stage3_recovery.py --arm S3-C2 --checkpoint "$checkpoint" --out "$pre/S3-C2" --num-envs 4096 --updates 200 --preflight --seed 20260927 --eval-gpu 5 > "$pre/S3-C2.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/train_stage3_recovery.py --arm S3-G0 --checkpoint "$checkpoint" --out "$pre/S3-G0" --num-envs 4096 --updates 200 --preflight --seed 20260927 --eval-gpu 5 > "$pre/S3-G0.log" 2>&1 &
p1=$!
printf '%s\n' "$p0" > "$pre/S3-C2.pid"
printf '%s\n' "$p1" > "$pre/S3-G0.pid"
echo "Preflight started: C2=$p0 GPU0; G0=$p1 GPU1"
status=0
wait "$p0" || status=1
wait "$p1" || status=1
test "$status" -eq 0 || { echo 'Preflight failed; formal training will not start'; exit 1; }
.venv/bin/python - <<'PY'
import json
from pathlib import Path
rows=[]
for arm in ('S3-C2','S3-G0'):
 p=Path('outputs/preflight_stage3_migration_20260928')/arm
 assert not (p/'failed.json').exists()
 assert json.loads((p/'completed.json').read_text())['iteration']>=199
 for name in ('partial_reset_pass','reward_contract_pass'):
  assert json.loads((p/(name+'.json')).read_text())['pass']
 a=json.loads((p/'launch.json').read_text())
 assert a['num_envs']==4096 and a['updates']==200 and a['actor_exact'] and a['actor_noise']
 rows.append(a)
for key in ('actor_sha','critic_sha','initial_qpos_sha','bank_sha256','code_sha256'):
 assert rows[0][key]==rows[1][key],key
print('Both 200-update preflights passed; matched initialization verified')
PY
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -u scripts/train_stage3_recovery.py --arm S3-C2 --checkpoint "$checkpoint" --out "$baseline" --eval --num-envs 224 --steps 1000 --seed 20260927 --eval-gpu 5 > "$pre/initial_R2_eval.log" 2>&1
test -f "$baseline/summary.json"
echo 'Initial R2 evaluation completed; starting formal runs from original R2'
for entry in '0 S3-C2' '1 S3-G0'; do
  read -r gpu arm <<< "$entry"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/train_stage3_recovery.py --arm "$arm" --checkpoint "$checkpoint" --out "$runroot/$arm" --num-envs 4096 --updates 10000 --seed 20260927 --eval-gpu 5 > "$runroot/$arm.stdout.log" 2>&1 < /dev/null 9>&- &
  printf '%s\n' "$!" > "$runroot/$arm.pid"
  echo "Started $arm on GPU$gpu pid=$!"
done
