import os,json,subprocess,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--envs',type=int,default=128);p.add_argument('--formal',action='store_true');p.add_argument('--eval',action='store_true');p.add_argument('--video',action='store_true');a=p.parse_args();r=Path.cwd()
ref='/root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt'
if a.formal:
 assert a.envs==4096
 assert (r/'outputs/reload_upper130/summary.json').exists()
 hashes=[]
 for arm in ['B0_reset','B1_dynamics']:
  pre=r/'outputs/preflight4096_v2'/arm
  assert (pre/'completed.json').exists() and (pre/'actual_dynamics_check.json').exists() and (pre/'partial_reset_pass.json').exists()
  assert (r/'outputs/zero_eval_v1'/arm/'summary.json').exists()
  meta=json.load(open(pre/'launch.json'));hashes.append(meta['common_critic_sha'])
 assert len(set(hashes))==1
 for previous in r.glob('logs/rsl_rl/balanced/*/launches.json'):
  for record in json.load(open(previous)):
   proc=Path('/proc')/str(record['pid'])/'cmdline'
   assert not (proc.exists() and b'train_balanced_dynamics.py' in proc.read_bytes()),'Existing balanced training still active'
out=r/('logs/rsl_rl/balanced' if a.formal else 'outputs')/a.tag;out.mkdir(parents=True,exist_ok=False);records=[]
for arm,gpu in [('B0_reset','3'),('B1_dynamics','4')]:
 cmd=[str(r/'.venv/bin/python'),'-u','scripts/train_balanced_dynamics.py','--arm',arm,'--checkpoint',ref,'--reference',ref,'--out',str(out/arm),'--num-envs',str(a.envs),'--updates',str(10000 if a.formal else 4)]
 if a.eval:cmd+=['--eval']
 elif not a.formal:cmd+=['--preflight']
 if a.video:cmd+=['--video']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MUJOCO_GL='egl',WANDB_PROJECT='smp',WANDB_RUN_GROUP='ft12k-balanced-plate-dynamics',WANDB_RUN_ID=f'{a.tag}-{arm}'.lower())
 with open(out/f'{arm}.log','w') as f:proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 records.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'out':str(out/arm),'command':cmd})
(out/'launches.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
