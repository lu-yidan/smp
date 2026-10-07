"""Launch only the six authorized CR arms; never stop unrelated jobs."""
import json, os, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'logs/crawl_first_20261007'
if out.exists():raise RuntimeError(f'Refusing duplicate launch: {out}')
for arm in ('CR0','CR1','CR2','CR3','CR4','CR5'):
 check=root/'logs/crawl_preflight_v3'/arm/'completed.json'
 assert check.exists(),f'Missing preflight: {check}'
out.mkdir(parents=True)
processes=[]
for i,gpu in enumerate((0,3,4,5,6,7)):
 arm=f'CR{i}'
 cmd=[sys.executable,'-u','scripts/train_crawl_first.py','--arm',arm,'--out',str(out/arm),'--num-envs','4096','--updates','10000','--eval-gpu',str(gpu)]
 with (out/f'{arm}.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'MUJOCO_EGL_DEVICE_ID':str(gpu),'PYTHONPATH':'src:scripts:.'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 processes.append(dict(arm=arm,gpu=gpu,pid=proc.pid,command=cmd))
(out/'processes.json').write_text(json.dumps(processes,indent=2)+'\n')
print(json.dumps(processes,indent=2))
