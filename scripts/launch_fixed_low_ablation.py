"""Launch fixed-quota ablations on GPU4/5; checkpoint evaluation reuses each GPU."""
import argparse,datetime,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');a=p.parse_args()
root=Path.cwd();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');tag=('preflight_' if a.preflight else 'formal_')+stamp
control=root/'run_control/fixed_low'/tag;control.mkdir(parents=True)
records=[]
for arm,gpu in [('L0',4),('L1',5)]:
 logdir=root/'logs/rsl_rl/fixed_low'/tag/arm
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_fixed_low_ablation.py','--arm',arm,'--bank-dir',str(root/'datasets/reset_banks/natural_curriculum_v1'),'--log-dir',str(logdir),'--eval-workspace','/root/workplace/smp-flat93','--eval-gpu',str(gpu)]
 if a.preflight:cmd+=['--preflight','--iterations','4']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MUJOCO_GL='egl')
 if not a.preflight:env.update(WANDB_PROJECT='smp',WANDB_ENTITY='tabletennis',WANDB_RUN_ID=f'fixed-low-{arm.lower()}-{stamp}',WANDB_NAME=f'{arm}-fixed-low-{stamp}',WANDB_RUN_GROUP='fixed-low-smp-ablation')
 with open(control/f'{arm}.log','w') as log:proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 records.append({'arm':arm,'pid':proc.pid,'gpu':gpu,'eval_gpu':gpu,'log_dir':str(logdir),'console':str(control/f'{arm}.log'),'command':cmd,'wandb_id':env.get('WANDB_RUN_ID')})
(control/'launches.json').write_text(json.dumps(records,indent=2));print(control);print(json.dumps(records,indent=2))
