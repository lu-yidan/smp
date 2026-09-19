"""Launch the eight matched R2-transfer arms. Formal launch requires preflight evidence."""
import argparse,os,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['preflight','calibrate','eval','formal'],required=True);p.add_argument('--envs',type=int,default=4096);a=p.parse_args();root=Path.cwd()
checkpoint='/root/workplace/smp-ft-speed/logs/rsl_rl/ft_prone_speed/formal_20260918_ft_speed_v1/FT_R2/model_9000.pt'
reference='/root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt'
arms=['F0','F1','F2','F3','T0','T1','T2','T3']
if a.mode=='formal':
 assert a.envs==4096
 files=['scripts/train_r2_ablation.py','src/smp/rl/tasks/getup/multiterrain.py','src/smp/rl/tasks/getup/balanced_dynamics.py','src/smp/rl/tasks/getup/ft_prone_speed.py','src/smp/rl/tasks/getup/v33_reward_transfer.py','src/smp/rl/tasks/getup/r2_ablation.py']
 code_sha=hashlib.sha256(b''.join((root/f).read_bytes() for f in files)).hexdigest()
 critics=set()
 for arm in arms:
  pre=root/'outputs/preflight4096_final'/arm
  for name in ['completed.json','actual_dynamics_check.json','partial_reset_pass.json']:assert (pre/name).exists(),(arm,name)
  m=json.loads((pre/'launch.json').read_text());critics.add(m['common_critic_sha']);assert m['code_sha256']==code_sha;assert m['actor_exact'] and m['fresh_critic'] and m['actor_noise'] and m['low_smp_termination']==False
  assert (root/'outputs/reload_eval_final'/arm/'summary.json').exists()
  assert (root/'outputs/calibration_final'/arm/'calibration.json').exists()
 assert len(critics)==1
 for prev in root.glob('logs/rsl_rl/r2_ablation/*/launches.json'):
  for item in json.loads(prev.read_text()):
   f=Path(f'/proc/{item["pid"]}/cmdline');assert not(f.exists() and b'train_r2_ablation.py' in f.read_bytes())
folder=root/('logs/rsl_rl/r2_ablation' if a.mode=='formal' else 'outputs')/a.tag;folder.mkdir(parents=True,exist_ok=False);records=[]
for gpu,arm in enumerate(arms):
 source=str(root/'outputs/preflight4096_final'/arm/'final.pt') if a.mode=='eval' else checkpoint
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_r2_ablation.py','--arm',arm,'--checkpoint',source,'--reference',reference,'--out',str(folder/arm),'--num-envs',str(a.envs),'--updates',str(10000 if a.mode=='formal' else 4)]
 if a.mode=='preflight':cmd+=['--preflight']
 if a.mode=='calibrate':cmd+=['--calibrate']
 if a.mode=='eval':cmd+=['--eval','--steps','1000']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MUJOCO_GL='egl',WANDB_PROJECT='smp',WANDB_RUN_GROUP='r2-flat-plate-ablation',WANDB_RUN_ID=f'{a.tag}-{arm}'.lower())
 with (folder/(arm+'.log')).open('w') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 records.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'command':cmd,'checkpoint':source,'checkpoint_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest()})
(folder/'launches.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
