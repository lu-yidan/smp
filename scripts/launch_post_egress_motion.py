"""Preflight each arm, then queue it behind the matching old EC GPU job."""
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--tag',required=True);P.add_argument('--worker',type=int);a=P.parse_args()
root=Path.cwd();folder=root/'logs/rsl_rl/post_egress_motion'/a.tag
source=Path('/root/workplace/smp-r2-v33-path/logs/rsl_rl/v33_path_ablation/formal_20260920_r2_v33path_v1/A6/model_9999.pt')
ref=Path('/root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt')
old=root.parent/'smp-a6-egress/logs/rsl_rl/egress_convergence/formal_20260921_a6_egress_v1/launches.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
if a.worker is None:
 folder.mkdir(parents=True,exist_ok=False)
 records=[]
 for gpu in range(4):
  arm=f'ES{gpu}';env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MUJOCO_GL='egl',WANDB_PROJECT='smp',WANDB_RUN_GROUP='a6-post-egress-motion',WANDB_RUN_ID=f'{a.tag}-{arm}'.lower())
  with (folder/f'{arm}_worker.log').open('w') as f:
   proc=subprocess.Popen([str(root/'.venv/bin/python'),'-u',__file__,'--tag',a.tag,'--worker',str(gpu)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  records.append({'arm':arm,'gpu':gpu,'worker_pid':proc.pid})
 write(folder/'workers.json',records);print(json.dumps(records));raise SystemExit
arm=f'ES{a.worker}';state=folder/f'{arm}_state.json'
def status(phase,**kwargs):write(state,{'arm':arm,'phase':phase,'time':time.time(),**kwargs})
def run(kind,extra,checkpoint=source,n=192):
 out=folder/'checks'/arm/kind;out.parent.mkdir(parents=True,exist_ok=True)
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_egress_convergence.py','--arm',arm,'--checkpoint',str(checkpoint),'--reference',str(ref),'--out',str(out),'--num-envs',str(n),*extra]
 status(kind)
 with (out.parent/f'{kind}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 return out
try:
 subprocess.run([str(root/'.venv/bin/python'),'scripts/check_post_egress_motion.py'],check=True)
 pre=run('preflight',['--preflight','--updates','4'],n=4096)
 manifest=json.loads((pre/'launch.json').read_text())
 assert manifest['actor_exact'] and manifest['fresh_critic'] and manifest['fresh_optimizer'] and manifest['actor_noise'] and not manifest['low_smp_termination']
 assert manifest['sha256']==sha(source)
 for f in ('completed.json','actual_dynamics_check.json','partial_reset_pass.json','fixed_roof_check.json'):assert (pre/f).exists()
 cal=run('calibration',['--calibrate'])
 values=json.loads((cal/'calibration.json').read_text())['weighted_term_integrals']
 assert all(__import__('math').isfinite(x) for x in values.values())
 ev=run('reload',['--eval','--steps','1000'],checkpoint=pre/'final.pt')
 result=json.loads((ev/'summary.json').read_text())
 assert sum(result[k]['post_egress_escaped_n'] for k in ('fixed_ceiling','free_plate'))>0
 assert all(v['post_egress_outward_reward_mean']==0 for v in result.values())
 old_pid=next(x['pid'] for x in json.loads(old.read_text()) if x['gpu']==a.worker)
 status('queued',waiting_for_old_pid=old_pid,calibration=values)
 while True:
  proc=Path(f'/proc/{old_pid}/cmdline')
  running=proc.exists() and b'train_egress_convergence.py' in proc.read_bytes()
  used=int(subprocess.check_output(['nvidia-smi',f'--id={a.worker}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if not running and used<2000:break
  time.sleep(30)
 out=folder/arm
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_egress_convergence.py','--arm',arm,'--checkpoint',str(source),'--reference',str(ref),'--out',str(out),'--num-envs','4096','--updates','10000']
 with (folder/f'{arm}.log').open('w') as f:
  proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
  status('training',pid=proc.pid,command=cmd,source_sha256=sha(source))
  rc=proc.wait()
  if rc:raise RuntimeError(f'training exited {rc}')
 status('completed')
except BaseException as e:
 status('failed',error=repr(e));traceback.print_exc();raise
