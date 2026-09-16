import argparse,datetime,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['probe','preflight','formal'],required=True);a=p.parse_args()
root=Path.cwd();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');tag=a.mode+'_'+stamp
control=root/'run_control/v33_reward_transfer'/tag;control.mkdir(parents=True);rows=[]
for i,mode in enumerate(['FT'] if a.mode=='probe' else ['FT','Scratch']):
 gpu=4+i;log=root/'logs/rsl_rl/v33_reward_transfer'/tag/mode
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_v33_reward_transfer.py','--mode',mode,'--log-dir',str(log),'--eval-gpu',str(gpu)]
 if a.mode=='probe':cmd+=['--probe','--num-envs','512']
 if a.mode=='preflight':cmd+=['--preflight','--updates','4']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4');wid=None
 if a.mode=='formal':
  wid=f'v33-reward-{mode.lower()}-{stamp}';env.update(WANDB_PROJECT='smp',WANDB_ENTITY='tabletennis',WANDB_RUN_ID=wid,WANDB_RUN_GROUP='v33-reward-transfer')
 with open(control/f'{mode}.log','w') as f:proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 rows.append({'mode':mode,'gpu':gpu,'pid':proc.pid,'log_dir':str(log),'wandb_id':wid,'command':cmd})
(control/'launches.json').write_text(json.dumps(rows,indent=2));print(control);print(json.dumps(rows,indent=2))
