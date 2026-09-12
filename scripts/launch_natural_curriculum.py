"""Launch exactly two paired curriculum runs, with disjoint validation GPUs."""
import argparse,datetime,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--num-envs',type=int,default=128);p.add_argument('--initial-lr',type=float,default=1e-4);a=p.parse_args()
root=Path.cwd();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');tag=('preflight_' if a.preflight else 'formal_')+stamp
control=root/'run_control'/'natural_curriculum'/tag;control.mkdir(parents=True)
records=[]
for i,arm in enumerate(['N0','N1']):
 logdir=root/'logs/rsl_rl/natural_curriculum'/tag/arm
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_natural_curriculum.py','--arm',arm,'--bank-dir',str(root/'datasets/reset_banks/natural_curriculum_v1'),'--log-dir',str(logdir),'--pair-dir',str(control/'pair'),'--eval-workspace','/root/workplace/smp-flat93','--eval-gpu',str(i+2),'--initial-lr',str(a.initial_lr)]
 if a.preflight:cmd+=['--preflight','--num-envs',str(a.num_envs),'--iterations','4']
 if arm=='N1':cmd+=['--b1-checkpoint','/root/workplace/smp-master-repro/logs/rsl_rl/scratch93_termination/B1_seed20260911_10000/model_9999.pt']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4',MUJOCO_GL='egl')
 if not a.preflight:env.update(WANDB_PROJECT='smp',WANDB_ENTITY='tabletennis',WANDB_RUN_ID=f'natural-curriculum-{arm.lower()}-{stamp}',WANDB_NAME=f'{arm}-natural-no-gsi-{stamp}',WANDB_RUN_GROUP='natural-reset-curriculum-v1')
 with open(control/f'{arm}.log','w') as log:proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 records.append({'arm':arm,'pid':proc.pid,'gpu':i,'eval_gpu':i+2,'log_dir':str(logdir),'console':str(control/f'{arm}.log'),'command':cmd,'wandb_id':env.get('WANDB_RUN_ID')})
(control/'launches.json').write_text(json.dumps(records,indent=2));print(control);print(json.dumps(records,indent=2))
