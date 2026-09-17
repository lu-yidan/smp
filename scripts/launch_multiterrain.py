"""Launch bounded preflights, nominal evaluation, or preflight-gated formal runs."""
import argparse,datetime,json,os,subprocess,hashlib,fcntl
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','eval','formal'],required=True);p.add_argument('--tag',required=True);p.add_argument('--arms',nargs='+',default=['T_P5','T_FT12k','T_L4']);p.add_argument('--gpus',nargs='+',default=['0','1','2']);p.add_argument('--preflight-root',type=Path);p.add_argument('--steps',type=int,default=1000);p.add_argument('--video',action='store_true');p.add_argument('--envs',type=int);a=p.parse_args()
root=Path.cwd();assert len(a.arms)==len(a.gpus)
ref=Path('/root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt')
sources={'T_P5':Path('/root/workplace/smp-prior-replay-ft/logs/rsl_rl/prior_replay_transfer/formal_20260917_113521/P5_all_low/model_2500.pt'),'T_FT12k':ref,'T_L4':Path('/root/workplace/smp-v33-reward/logs/rsl_rl/fixed_low/formal_20260913_112049/L4/model_9999.pt')}
banksha=hashlib.sha256((root/'outputs/multiterrain_bank/train.npz').read_bytes()).hexdigest()
if a.mode=='formal':
 lock_path=root/'outputs/formal_multiterrain.lock';lock_path.parent.mkdir(exist_ok=True)
 lock=open(lock_path,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for previous in root.glob('logs/rsl_rl/multiterrain/*/launches.json'):
  for entry in json.load(open(previous)):
   proc=Path(f'/proc/{entry["pid"]}/cmdline')
   if entry['arm'] in a.arms and proc.exists() and b'scripts/train_multiterrain.py' in proc.read_bytes():raise RuntimeError(f'Already running {entry["arm"]}: {entry["pid"]}')
 manifest=json.load(open(root/'outputs/multiterrain_bank/manifest.json'));assert manifest.get('status')!='PROTOTYPE_ONLY_NOT_FOR_FORMAL'
 assert a.preflight_root
 expected={v['name']:v['sha256'] for v in json.load(open(root/'configs/p5_multiterrain_v1.json'))['initialization_arms']}
 for arm in a.arms:
  assert hashlib.sha256(sources[arm].read_bytes()).hexdigest()==expected[arm]
  assert (root/'outputs/zero_valid_v2'/arm/'summary.json').exists()
  assert (root/'outputs/reload_checks_v1'/arm/'summary.json').exists()
 hashes={json.load(open(a.preflight_root/arm/'launch.json'))['common_critic_sha'] for arm in a.arms}
 assert len(hashes)==1,hashes
 for arm in a.arms:
  d=a.preflight_root/arm;assert (d/'completed.json').exists();m=json.load(open(d/'launch.json'));assert m['bank_sha']==banksha
folder=root/('logs/rsl_rl/multiterrain' if a.mode=='formal' else 'outputs')/a.tag;folder.mkdir(parents=True,exist_ok=False);records=[]
for arm,gpu in zip(a.arms,a.gpus):
 out=folder/arm;cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_multiterrain.py','--arm',arm,'--checkpoint',str(sources[arm]),'--reference',str(ref),'--out',str(out),'--num-envs',str(a.envs or (4096 if a.mode=='formal' else 128 if a.mode=='preflight' else 256))]
 if a.mode=='preflight':cmd+=['--preflight','--updates','4']
 elif a.mode=='eval':
  cmd+=['--eval','--steps',str(a.steps)]
  if a.video:cmd+=['--video']
 else:cmd+=['--updates','10000']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MUJOCO_GL='egl',WANDB_PROJECT='smp',WANDB_RUN_GROUP='p5-multiterrain-initialization',WANDB_RUN_ID=f'{a.tag}-{arm}'.lower())
 with open(folder/f'{arm}.log','w') as f:proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 records.append({'arm':arm,'pid':proc.pid,'gpu':gpu,'out':str(out),'command':cmd,'source_sha256':hashlib.sha256(sources[arm].read_bytes()).hexdigest()})
(folder/'launches.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
