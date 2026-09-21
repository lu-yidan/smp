"""Launch the eight matched A6 egress-convergence arms. Formal launch requires preflight evidence."""
import argparse,os,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['preflight','calibrate','eval','formal'],required=True);p.add_argument('--envs',type=int,default=4096);p.add_argument('--video-arm',action='append',default=[]);a=p.parse_args();root=Path.cwd()
checkpoint='/root/workplace/smp-r2-v33-path/logs/rsl_rl/v33_path_ablation/formal_20260920_r2_v33path_v1/A6/model_9999.pt'
reference='/root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt'
arms=['EC'+str(i) for i in range(8)]
if a.mode=='formal':
 assert a.envs==4096
 assert (root/'outputs/egress_reward_checks.json').exists()
 files=['scripts/train_egress_convergence.py','src/smp/rl/tasks/getup/multiterrain.py','src/smp/rl/tasks/getup/balanced_dynamics.py','src/smp/rl/tasks/getup/ft_prone_speed.py','src/smp/rl/tasks/getup/v33_reward_transfer.py','src/smp/rl/tasks/getup/r2_ablation.py','src/smp/rl/tasks/getup/v33_path_ablation.py','src/smp/rl/tasks/getup/egress_convergence.py','src/smp/rl/tasks/getup/multiterrain_geometry.py']
 code_sha=hashlib.sha256(b''.join((root/f).read_bytes() for f in files)).hexdigest()
 critics=set()
 for arm in arms:
  pre=root/'outputs/egress_preflight4096'/arm
  for name in ['completed.json','actual_dynamics_check.json','partial_reset_pass.json','fixed_roof_check.json']:assert (pre/name).exists(),(arm,name)
  m=json.loads((pre/'launch.json').read_text());critics.add(m['common_critic_sha']);assert m['code_sha256']==code_sha;assert m['actor_exact'] and m['fresh_critic'] and m['actor_noise'] and m['low_smp_termination']==False
  assert m['sha256']==hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
  assert m['bank_sha']==hashlib.sha256((root/'outputs/ceiling_bank/train.npz').read_bytes()).hexdigest()
  ev=json.loads((root/'outputs/egress_reload_eval'/arm/'summary.json').read_text())
  assert sum(ev[k]['post_egress_escaped_n'] for k in ('fixed_ceiling','free_plate'))>0,'vacuous egress validation'
  if arm!='EC0':assert all(v['post_egress_outward_reward_mean']==0 for v in ev.values())
  cal=json.loads((root/'outputs/egress_calibration'/arm/'calibration.json').read_text())
  assert all(__import__('math').isfinite(v) for v in cal['weighted_term_integrals'].values())
 assert len(critics)==1
 for prev in root.glob('logs/rsl_rl/egress_convergence/*/launches.json'):
  for item in json.loads(prev.read_text()):
   f=Path(f'/proc/{item["pid"]}/cmdline');assert not(f.exists() and b'train_egress_convergence.py' in f.read_bytes())
folder=root/('logs/rsl_rl/egress_convergence' if a.mode=='formal' else 'outputs')/a.tag;folder.mkdir(parents=True,exist_ok=False);records=[]
for gpu,arm in enumerate(arms):
 source=str(root/'outputs/egress_preflight4096'/arm/'final.pt') if a.mode=='eval' else checkpoint
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_egress_convergence.py','--arm',arm,'--checkpoint',source,'--reference',reference,'--out',str(folder/arm),'--num-envs',str(a.envs),'--updates',str(10000 if a.mode=='formal' else 4)]
 if a.mode=='preflight':cmd+=['--preflight']
 if a.mode=='calibrate':cmd+=['--calibrate']
 if a.mode=='eval':cmd+=['--eval','--steps','1000']
 if a.mode=='eval' and arm in a.video_arm:cmd+=['--video']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MUJOCO_GL='egl',WANDB_PROJECT='smp',WANDB_RUN_GROUP='a6-egress-convergence',WANDB_RUN_ID=f'{a.tag}-{arm}'.lower())
 with (folder/(arm+'.log')).open('w') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 records.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'command':cmd,'checkpoint':source,'checkpoint_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest()})
(folder/'launches.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
