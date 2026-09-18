import argparse,os,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['preflight','calibrate','eval','formal'],default='preflight');p.add_argument('--envs',type=int,default=128);a=p.parse_args();r=Path.cwd()
ref='/root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt'
l4='/root/workplace/smp-v33-reward/logs/rsl_rl/fixed_low/formal_20260913_112049/L4/model_9999.pt'
arms=['FT_R0','FT_R1','FT_R2'];gpus=['1','5','7']
if a.mode=='formal':
 assert a.envs==4096
 critics=set()
 code_sha=hashlib.sha256(b''.join((r/f).read_bytes() for f in ['scripts/train_ft_prone_speed.py','src/smp/rl/tasks/getup/multiterrain.py','src/smp/rl/tasks/getup/balanced_dynamics.py','src/smp/rl/tasks/getup/ft_prone_speed.py','src/smp/rl/tasks/getup/v33_reward_transfer.py'])).hexdigest()
 for arm in arms:
  pre=r/'outputs/preflight4096_final'/arm
  assert (pre/'completed.json').exists() and (pre/'partial_reset_pass.json').exists() and (pre/'actual_dynamics_check.json').exists()
  manifest=json.load(open(pre/'launch.json'));critics.add(manifest['common_critic_sha'])
  assert manifest['code_sha256']==code_sha and manifest['actor_noise'] and manifest['actor_exact'] and manifest['fresh_critic'] and manifest['fresh_optimizer']
  assert manifest['sha256']==hashlib.sha256(Path(l4).read_bytes()).hexdigest()
  assert (r/'outputs/reload_eval_final'/arm/'summary.json').exists()
  cal=json.load(open(r/'outputs/calibration_final'/arm/'calibration.json'))
  assert 'task_smp_product' in cal['weighted_term_integrals']
 assert len(critics)==1
 for prev in r.glob('logs/rsl_rl/ft_prone_speed/*/launches.json'):
  for rec in json.load(open(prev)):
   f=Path(f'/proc/{rec["pid"]}/cmdline');assert not(f.exists() and b'train_ft_prone_speed.py' in f.read_bytes()),'already active'
out=r/('logs/rsl_rl/ft_prone_speed' if a.mode=='formal' else 'outputs')/a.tag;out.mkdir(parents=True,exist_ok=False);records=[]
for arm,gpu in zip(arms,gpus):
 checkpoint=l4
 if a.mode=='eval':checkpoint=str(r/'outputs/preflight4096_final'/arm/'final.pt')
 cmd=[str(r/'.venv/bin/python'),'-u','scripts/train_ft_prone_speed.py','--arm',arm,'--checkpoint',checkpoint,'--reference',ref,'--out',str(out/arm),'--num-envs',str(a.envs),'--updates',str(10000 if a.mode=='formal' else 4)]
 if a.mode=='preflight':cmd+=['--preflight']
 if a.mode=='calibrate':cmd+=['--calibrate']
 if a.mode=='eval':cmd+=['--eval','--steps','1000']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MUJOCO_GL='egl',WANDB_PROJECT='smp',WANDB_RUN_GROUP='l4-ft-prone-speed',WANDB_RUN_ID=f'{a.tag}-{arm}'.lower())
 with open(out/f'{arm}.log','w') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 records.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'out':str(out/arm),'checkpoint':checkpoint,'checkpoint_sha':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'cmd':cmd})
(out/'launches.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
