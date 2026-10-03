"""Run all six 4096-world preflights, then stop; never starts formal training."""
import json, os, subprocess
from pathlib import Path
from queue_a6_clear_settle import ARMS, ROOT, PREFLIGHT

def main():
    os.chdir(ROOT)
    PREFLIGHT.mkdir(parents=True, exist_ok=False)
    jobs=[]
    for gpu,arm in enumerate(ARMS):
        out=PREFLIGHT/arm
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'src:scripts:.:tests',
             'MUJOCO_GL':'egl','OMP_NUM_THREADS':'4','WANDB_MODE':'offline'}
        args=['.venv/bin/python','-u','scripts/train_a6_clear_settle.py','--arm',arm,
              '--seed','20261026','--out',str(out),'--preflight','--num-envs','4096','--updates','32']
        log=out.with_suffix('.log').open('x')
        p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((arm,p,log))
        print('START',gpu,arm,p.pid,flush=True)
    results={}
    for arm,p,log in jobs:
        results[arm]=p.wait();log.close()
        print('FINISH',arm,results[arm],flush=True)
    (PREFLIGHT/'controller_results.json').write_text(json.dumps(results,indent=2)+'\n')
    assert all(v==0 for v in results.values()),results

if __name__=='__main__':main()
