import argparse,datetime,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['calibrate','preflight','formal'],required=True);p.add_argument('--arms',nargs='+');p.add_argument('--gpu-offset',type=int,default=0);a=p.parse_args();root=Path.cwd();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');tag=a.mode+'_'+stamp
control=root/'run_control/recovery_quality'/tag;control.mkdir(parents=True);records=[]
arms=['L3','L4','L5'] if a.mode=='calibrate' else ['L3','L4','L5','L4-Q1','L4-Q2']
for gpu,arm in enumerate(arms):
 if a.arms and arm not in a.arms:continue
 gpu+=a.gpu_offset
 source=arm.split('-')[0];ck=root/'logs/rsl_rl/fixed_low/formal_20260913_112049'/source/'model_9999.pt';reference=root/'datasets/recovery_references'/f'{source}_initial_reference.pt';logdir=root/'logs/rsl_rl/recovery_quality'/tag/arm
 cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_recovery_continuation.py','--arm',arm,'--checkpoint',str(ck),'--reference',str(reference),'--log-dir',str(logdir),'--eval-gpu',str(gpu)]
 if a.mode=='calibrate':cmd+=['--calibrate']
 if a.mode=='preflight':cmd+=['--preflight','--additional','4']
 env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
 if a.mode=='formal':env.update(WANDB_PROJECT='smp',WANDB_ENTITY='tabletennis',WANDB_RUN_ID=f'recovery-quality-{arm.lower()}-{stamp}',WANDB_NAME=f'{arm}-10k-to12k-{stamp}',WANDB_RUN_GROUP='recovery-quality-continuation')
 with open(control/f'{arm}.log','w') as f:proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 records.append({'arm':arm,'pid':proc.pid,'gpu':gpu,'log_dir':str(logdir),'console':str(control/f'{arm}.log'),'command':cmd,'wandb_id':env.get('WANDB_RUN_ID')})
(control/'launches.json').write_text(json.dumps(records,indent=2));print(control);print(json.dumps(records,indent=2))
