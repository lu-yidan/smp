import argparse,os,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['preflight','calibrate','eval','formal'],default='preflight');p.add_argument('--envs',type=int,default=128);a=p.parse_args();r=Path.cwd()
ref='/root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt'
l4='/root/workplace/smp-v33-reward/logs/rsl_rl/fixed_low/formal_20260913_112049/L4/model_9999.pt'
arms=['D0','D1','D2','D3','D4','D5'];gpus=['1','2','3','4','6','7']
if a.mode=='formal':
 assert a.envs==4096
 critics=set()
 code_sha=hashlib.sha256(b''.join((r/f).read_bytes() for f in ['scripts/train_d_series.py','src/smp/rl/tasks/getup/multiterrain.py','src/smp/rl/tasks/getup/balanced_dynamics.py','src/smp/rl/tasks/getup/d_recovery.py'])).hexdigest()
 for arm in arms:
  pre=r/'outputs/preflight4096_final'/arm
  assert (pre/'completed.json').exists() and (pre/'partial_reset_pass.json').exists() and (pre/'actual_dynamics_check.json').exists()
  manifest=json.load(open(pre/'launch.json'));critics.add(manifest['common_critic_sha'])
  assert manifest['code_sha256']==code_sha and manifest['actor_noise'] and manifest['actor_exact'] and manifest['fresh_critic'] and manifest['fresh_optimizer']
  assert manifest['mid_bank_sha']==hashlib.sha256((r/'datasets/d_mid/train.npz').read_bytes()).hexdigest()
  assert manifest['sha256']==hashlib.sha256(Path(ref).read_bytes()).hexdigest()
  assert (r/'outputs/reload_eval_final'/arm/'summary.json').exists()
  cal=json.load(open(r/'outputs/calibration_final'/arm/'calibration.json'))
  assert cal['max_episode_credit']<=2.00001
  assert cal['progress_integral_mean']>0
 assert len(critics)==1
 for prev in r.glob('logs/rsl_rl/d_series/*/launches.json'):
  for rec in json.load(open(prev)):
   f=Path(f'/proc/{rec["pid"]}/cmdline');assert not(f.exists() and b'train_d_series.py' in f.read_bytes()),'already active'
out=r/('logs/rsl_rl/d_series' if a.mode=='formal' else 'outputs')/a.tag;out.mkdir(parents=True,exist_ok=False);records=[]
for arm,gpu in zip(arms,gpus):
 checkpoint=ref
 if a.mode=='eval':checkpoint=str(r/'outputs/preflight4096_final'/arm/'final.pt')
 cmd=[str(r/'.venv/bin/python'),'-u','scripts/train_d_series.py','--arm',arm,'--checkpoint',checkpoint,'--reference',ref,'--out',str(out/arm),'--num-envs',str(a.envs),'--updates',str(10000 if a.mode=='formal' else 4)]
 if a.mode=='preflight':cmd+=['--preflight']
 if a.mode=='calibrate':cmd+=['--calibrate']
 if a.mode=='eval':cmd+=['--eval','--steps','1000']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MUJOCO_GL='egl',WANDB_PROJECT='smp',WANDB_RUN_GROUP='d-series-recovery-reward-reset',WANDB_RUN_ID=f'{a.tag}-{arm}'.lower())
 with open(out/f'{arm}.log','w') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 records.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'out':str(out/arm),'checkpoint':checkpoint,'checkpoint_sha':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'cmd':cmd})
(out/'launches.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
