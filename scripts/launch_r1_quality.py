import argparse,datetime,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['probe','preflight','formal'],required=True);a=p.parse_args()
root=Path.cwd();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');tag=a.mode+'_'+stamp
control=root/'run_control/r1_quality'/tag;control.mkdir(parents=True);rows=[]
for i,arm in enumerate(['F4'] if a.mode=='probe' else ['F0','F1','F2','F3','F4']):
 gpu=5 if a.mode=='probe' else i;log=root/'logs/rsl_rl/r1_quality'/tag/arm
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_r1_quality.py','--arm',arm,'--checkpoint',str(root/'logs/rsl_rl/scratch_tradeoffs/formal_20260914_234350/R1/model_19999.pt'),'--log-dir',str(log),'--eval-gpu',str(gpu)]
 if a.mode=='probe':cmd+=['--probe','--num-envs','512']
 if a.mode=='preflight':cmd+=['--preflight','--additional','4']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4');wid=None
 if a.mode=='formal':
  wid=f'r1-quality5k-{arm.lower()}-{stamp}';env.update(WANDB_PROJECT='smp',WANDB_ENTITY='tabletennis',WANDB_RUN_ID=wid,WANDB_RUN_GROUP='r1-quality-finetune-5k')
 with open(control/f'{arm}.log','w') as f:proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 rows.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'log_dir':str(log),'wandb_id':wid,'command':cmd})
(control/'launches.json').write_text(json.dumps(rows,indent=2));print(control);print(json.dumps(rows,indent=2))
