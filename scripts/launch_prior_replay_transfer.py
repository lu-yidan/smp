"""Launch seven independent cohorts, never touching existing processes."""
import argparse,datetime,json,os,subprocess,hashlib
from pathlib import Path
from smp.rl.tasks.getup.prior_replay_transfer import ARMS
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','formal'],required=True);p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3,4,5,6]);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--updates',type=int,default=10000);p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));a=p.parse_args();assert len(a.gpus)==len(a.arms)
root=Path.cwd();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');tag=a.mode+'_'+stamp;control=root/'run_control/prior_replay_transfer'/tag;control.mkdir(parents=True);rows=[]
if a.mode=='formal':
    evidence=root/'outputs/prior_replay_preflight_verified.json'
    report=json.loads(evidence.read_text());assert report['passed'] and report['source_sha256']=='8f05543b644b0a1d11246e85778f99220460ef2a744417ea940296eed768c768'
    for name,digest in report['code_hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for gpu,arm in zip(a.gpus,a.arms):
    log=root/'logs/rsl_rl/prior_replay_transfer'/tag/arm
    cmd=[str(root/'.venv/bin/python'),'-u','scripts/train_prior_replay_transfer.py','--arm',arm,'--checkpoint',str(a.checkpoint.resolve()),'--log-dir',str(log),'--num-envs','4096','--updates',str(8 if a.mode=='preflight' else a.updates)]
    if a.mode=='preflight':cmd+=['--preflight']
    env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4');wid=None
    if a.mode=='formal':
        wid=f'prior-replay-ft12k-{arm.lower()}-{stamp}';env.update(WANDB_PROJECT='smp',WANDB_ENTITY='tabletennis',WANDB_RUN_ID=wid,WANDB_RUN_GROUP='ft12k-prior-replay')
    with open(control/(arm+'.log'),'w') as f:proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    rows.append({'arm':arm,'gpu':gpu,'pid':proc.pid,'log_dir':str(log),'stdout':str(control/(arm+'.log')),'wandb_id':wid,'command':cmd})
(control/'launches.json').write_text(json.dumps(rows,indent=2));print(control);print(json.dumps(rows,indent=2))
