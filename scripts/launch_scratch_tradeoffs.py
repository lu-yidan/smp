import argparse,datetime,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','formal','probe'],required=True);a=p.parse_args()
root=Path.cwd();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');tag=a.mode+'_'+stamp
control=root/'run_control/scratch_tradeoffs'/tag;control.mkdir(parents=True);rows=[]
for gpu,arm in enumerate(['R7'] if a.mode=='probe' else ['R'+str(i) for i in range(8)]):
    log=root/'logs/rsl_rl/scratch_tradeoffs'/tag/arm
    cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_scratch_tradeoffs.py','--arm',arm,'--log-dir',str(log),'--eval-gpu',str(gpu)]
    if a.mode=='preflight':cmd+=['--preflight','--iterations','4']
    if a.mode=='probe':cmd+=['--probe','--num-envs','512']
    env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
    wid=None
    if a.mode=='formal':
        wid=f'l4-scratch20k-{arm.lower()}-{stamp}';env.update(WANDB_PROJECT='smp',WANDB_ENTITY='tabletennis',WANDB_RUN_ID=wid,WANDB_RUN_GROUP='l4-scratch-tradeoffs-20k')
    with open(control/f'{arm}.log','w') as f:proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    rows.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'command':cmd,'log_dir':str(log),'wandb_id':wid})
(control/'launches.json').write_text(json.dumps(rows,indent=2));print(control);print(json.dumps(rows,indent=2))
