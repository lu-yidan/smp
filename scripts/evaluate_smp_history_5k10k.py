"""Evaluate the fixed three-seed, two-arm, two-budget historical SMP inventory."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import argparse


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='4,5');a=p.parse_args()
    root=Path('outputs/history_5k10k_20261006');root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for seed in (20261021,20261022,20261023):
        for epoch in (5000,10000):
            for arm in ('C20_full','C20_no_geometry'):
                checkpoint=Path(f'logs/rsl_rl/a6_confirmatory/formal_20261001_20k/seed{seed}/{arm}/model_{epoch}.pt')
                assert checkpoint.is_file(),checkpoint
                jobs.append(dict(seed=seed,epoch=epoch,arm=arm,checkpoint=str(checkpoint),
                    sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    plan=dict(evaluation_seed=20261013,num_envs=2048,steps=1150,case='mixed',jobs=jobs,
              note='All three seeds at both budgets; not an outcome-selected subset.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    gpus=a.gpus.split(',');assert len(gpus)==2
    def worker(index):
        for job in jobs[index::2]:
            name=f"seed{job['seed']}_{job['arm']}_{job['epoch']}";out=root/name
            cmd=[sys.executable,'-u','scripts/evaluate_free_robustness.py','--family','confirmatory',
                '--case','mixed','--eval','--arm',job['arm'],'--checkpoint',job['checkpoint'],
                '--out',str(out),'--num-envs','2048','--steps','1150','--seed','20261013']
            with (root/(name+'.log')).open('w') as log:
                print('START',gpus[index],name,flush=True)
                rc=subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[index],
                    'PYTHONPATH':'src:scripts:.','MUJOCO_GL':'egl','OMP_NUM_THREADS':'4',
                    'OPENBLAS_NUM_THREADS':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
            (root/(name+'.status.json')).write_text(json.dumps(dict(returncode=rc,command=cmd)))
            if rc:raise RuntimeError(f'{name} failed with {rc}')
            print('END',name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,range(2)))
    (root/'completed.json').write_text(json.dumps(dict(completed=len(jobs)))+'\n')


if __name__=='__main__':main()
