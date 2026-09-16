import argparse,datetime,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['probe','preflight','formal'],required=True);a=p.parse_args()
root=Path.cwd();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');tag=a.mode+'_'+stamp
control=root/'run_control/f2_task_tradeoffs'/tag;control.mkdir(parents=True);rows=[]
for i,arm in enumerate(['C3'] if a.mode=='probe' else ['C0','C1','C2','C3']):
 gpu=5 if a.mode=='probe' else i;log=root/'logs/rsl_rl/f2_task_tradeoffs'/tag/arm
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_f2_task_tradeoffs.py','--arm',arm,'--checkpoint',str(root/'logs/rsl_rl/r1_quality/formal_20260915_235447/F2/model_24999.pt'),'--log-dir',str(log),'--eval-gpu',str(gpu)]
 if a.mode=='probe':cmd+=['--probe','--num-envs','512']
 if a.mode=='preflight':cmd+=['--preflight','--additional','4']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4');wid=None
 if a.mode=='formal':
  wid=f'f2-task-speed5k-{arm.lower()}-{stamp}';env.update(WANDB_PROJECT='smp',WANDB_ENTITY='tabletennis',WANDB_RUN_ID=wid,WANDB_RUN_GROUP='f2-task-speed-finetune-5k')
 with open(control/f'{arm}.log','w') as f:proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 rows.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'log_dir':str(log),'wandb_id':wid,'command':cmd})
(control/'launches.json').write_text(json.dumps(rows,indent=2));print(control);print(json.dumps(rows,indent=2))
