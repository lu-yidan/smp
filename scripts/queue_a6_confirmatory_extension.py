"""Extend the existing three-seed study by two complete paired seeds on GPUs 1/2."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('C20_full', 'C20_no_geometry', 'C20_no_Q', 'C20_no_L')
SEEDS = (20261024, 20261025)
GPUS = (1, 2)
RUN = ROOT / 'logs/rsl_rl/a6_confirmatory/extension_20261003_20k'
PREFLIGHT = ROOT / 'outputs/preflight_a6_confirmatory/20261001_v1'

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def main():
    os.chdir(ROOT)
    lock = open('/tmp/smp-a6-confirmatory-extension-controller.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launches = []
    for arm in ARMS:
        p = PREFLIGHT / arm
        assert not (p/'failed.json').exists(), arm
        assert json.loads((p/'completed.json').read_text())['iteration'] == 31
        assert json.loads((p/'reward_contract.json').read_text())['pass']
        assert json.loads((p/'partial_reset_pass.json').read_text())['pass']
        launches.append(json.loads((p/'launch.json').read_text()))
    for key in ('actor_sha', 'critic_sha', 'initial_qpos_sha', 'assets', 'counts'):
        assert all(v[key] == launches[0][key] for v in launches), key
    assert launches[0]['counts']['scene'] == [2048, 1024, 1024, 0]
    assert launches[0]['counts']['kind_low_middle_late'] == [2868, 820, 408]
    RUN.mkdir(parents=True, exist_ok=False)
    jobs = [dict(arm=arm, seed=seed, state='queued') for seed in SEEDS for arm in ARMS]
    write(RUN/'registration.json', dict(arms=ARMS, seeds=SEEDS, updates=20000,
        num_envs=4096, actor_parent='R2_9000', alpha=1., save_interval=500,
        training_gpus=GPUS, evaluation_gpu=7,
        extension_note='Two additional complete paired seeds requested after inspecting initial results; report all five seeds, not only favorable runs.',
        original_run='formal_20261001_20k',
        aggregate_seeds=[20261021,20261022,20261023,20261024,20261025],
        primary_checkpoint=19999, secondary_checkpoint=10000,
        success_metric='SR10 within the unchanged 20s evaluation horizon',
        evaluation_updates=[5000,10000,15000,19999], eval_seed=20261013,
        evaluation_note='10k/final: 2048 rollouts per suite; nominal and upper130; fixed validation, not held-out.',
        training_branch='codex/a6-alpha1-confirmatory-20k',
        source_script_sha256=hashlib.sha256((ROOT/'scripts/train_a6_confirmatory.py').read_bytes()).hexdigest()))
    pending = list(jobs); active = {}
    while pending or active:
        for gpu in list(active):
            process, job = active[gpu]
            status = process.poll()
            if status is None:
                continue
            folder = Path(job['out'])
            success = status == 0 and (folder/'completed.json').exists()
            if success:
                success = json.loads((folder/'completed.json').read_text())['iteration'] == 19999
            job.update(state='completed' if success else 'failed', exit_code=status)
            del active[gpu]
            print(job['state'], job['arm'], job['seed'], flush=True)
        for gpu in GPUS:
            if gpu in active or not pending:
                continue
            # Respect jobs started by other users/controllers on these devices.
            used = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            if int(used.strip()) > 512:
                continue
            job = pending.pop(0)
            out = RUN / f"seed{job['seed']}" / job['arm']; out.parent.mkdir(exist_ok=True)
            args = ['.venv/bin/python', '-u', 'scripts/train_a6_confirmatory.py',
                '--arm', job['arm'], '--seed', str(job['seed']), '--out', str(out),
                '--num-envs', '4096', '--updates', '20000', '--eval-gpu', '7', '--eval-seed', '20261013']
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu), 'PYTHONPATH':'src:scripts:.:tests',
                'MUJOCO_GL':'egl', 'OMP_NUM_THREADS':'4', 'WANDB_MODE':'online',
                'WANDB_ENTITY':'tabletennis', 'WANDB_RUN_GROUP':'a6_alpha1_confirmatory_extension_20261003',
                'NETRC':'/root/workspace/smp-a6-tools/private/wandb.netrc'}
            with out.with_suffix('.stdout.log').open('w') as stream:
                process = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
            job.update(state='running', gpu=gpu, pid=process.pid, out=str(out), args=args)
            active[gpu] = process, job
            print('START', job['arm'], job['seed'], 'GPU', gpu, 'PID', process.pid, flush=True)
        write(RUN/'queue_status.json', dict(jobs=jobs, controller_pid=os.getpid(), updated_unix=time.time()))
        if pending or active:
            time.sleep(20)
    print('QUEUE_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
