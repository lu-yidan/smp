"""L0/L1 from scratch: identical fixed low-state quotas, low-SMP termination on/off."""
import argparse, hashlib, json, os, random, subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml
from train_natural_curriculum import build_config as natural_config, CourseWrapper, atomic_json, summarize_validation
from smp.rl.tasks.getup.natural_curriculum import STAGES,DIRECTIONS
from smp.rl.tasks.getup.fixed_low_reset import reset_fixed_low,fixed_low_smp,recorded_task_smp_product,quota_counts


def build_config(arm,bank_path,num_envs=4096,seed=20260912):
  assert arm in ('L0','L1','L2','L3','L4','L5')
  cfg,agent=natural_config(bank_path,num_envs,seed,initial_lr=1e-4)
  cfg.events['gsi_reset']=EventTermCfg(func=reset_fixed_low,mode='reset',params={'bank_path':str(Path(bank_path).resolve())})
  cfg.terminations['smp_too_low'].func=fixed_low_smp
  cfg.terminations['smp_too_low'].params['low_enabled']=arm not in ('L1','L3','L4','L5')
  if arm=='L2':cfg.terminations['smp_too_low'].params['low_threshold']=.005
  if arm in ('L3','L4','L5'):
    cfg.events['gsi_reset'].params.update(low_fraction=.2,late_fraction=.4,procedural_bank_path=str(Path(bank_path).resolve().parent.parent/'procedural_low_v1/train.npz'),procedural_fraction={'L3':0.,'L4':.05,'L5':.1}[arm])
  cfg.rewards['task_smp_product'].func=recorded_task_smp_product
  return cfg,agent


class FixedWrapper(CourseWrapper):
  def step(self,actions):
    env=self.unwrapped
    if not hasattr(self,'counts'):self.counts=torch.zeros(7,5,device=env.device,dtype=torch.float64)
    obs,reward,done,extras=super().step(actions)
    logs=extras['log']
    for key in list(logs):
      if key.startswith('Curriculum/'):
        value=logs.pop(key)
        if key!='Curriculum/stage':logs['Quota/'+key[len('Curriculum/'):]]=value
    masks=[env._course_type==i for i in range(3)]+[(env._course_type==2)&(env._course_direction==i) for i in range(4)]
    for name,mask in zip(STAGES+DIRECTIONS,masks):
      extras['log'][f'Quota/{name}/step_fraction']=mask.float().mean()
      for key,value in [('task_score',env._fixed_task_score),('smp_reward_factor',env._fixed_smp_score),('reward_product',env._fixed_product),('raw_smp_score',torch.exp(-6*env._smp_raw_err)),('would_low',env._fixed_would_low.float()),('current_age_s',env.episode_length_buf.float()*env.step_dt)]:
        extras['log'][f'LowGuidance/{name}/{key}']=value[mask].mean().detach()
    for source in (0,1):
      extras['log'][f'Quota/source_{source}/step_fraction']=(env._fixed_source==source).float().mean()
    assert torch.bincount(env._fixed_group,minlength=6).tolist()==list(env._fixed_quota_counts)
    return obs,reward,done,extras


class FixedRunner(MjlabOnPolicyRunner):
  active=False
  def save(self,path,infos=None):
    if self.active and self.current_learning_iteration==0 and Path(path).name=='model_0.pt':return
    super().save(path,infos={**(infos or {}),'fixed_quota_counts':self.env.unwrapped._fixed_quota_counts})
    if not self.active:return
    a=self.args;iteration=self.current_learning_iteration
    output=a.log_dir/'validation'/f'{iteration}.json';output.parent.mkdir(exist_ok=True)
    bank=np.load(a.bank_dir/'validation.npz')
    cmd=[str(a.eval_workspace/'.venv/bin/python'),'-u','scripts/audit_recovery_distribution.py','--policy-family','master','--checkpoint',str(Path(path).resolve()),'--num-envs',str(len(bank['qpos'])),'--steps','1000','--eval-bank',str(a.bank_dir/'validation.npz'),'--output',str(output)]
    if not a.preflight and (iteration%1000==0 or iteration==a.iterations-1):cmd+=['--video',str(output.with_name(f'{iteration}_20s.mp4'))]
    with open(output.with_suffix('.log'),'w') as log:
      subprocess.run(cmd,cwd=a.eval_workspace,env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=a.eval_gpu,MUJOCO_GL='egl',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=summarize_validation(json.loads(output.read_text()),bank)
    atomic_json(a.log_dir/'progress.json',{'iteration':iteration,'quota_counts':self.env.unwrapped._fixed_quota_counts,'validation':summary})
    for name,values in summary.items():
      for key,value in values.items():
        if self.logger.writer is not None:self.logger.writer.add_scalar(f'Validation/{name}/{key}',value,iteration)
    print('VALIDATION_COMPLETE',iteration,json.dumps(summary),flush=True)
    if a.arm in ('L3','L4','L5'):
      proc_path=a.bank_dir.parent/'procedural_low_v1/validation.npz'
      proc_bank=np.load(proc_path);proc_output=output.with_name(f'{iteration}_procedural.json')
      proc_cmd=[str(a.eval_workspace/'.venv/bin/python'),'-u','scripts/audit_recovery_distribution.py','--policy-family','master','--checkpoint',str(Path(path).resolve()),'--num-envs',str(len(proc_bank['qpos'])),'--steps','1000','--eval-bank',str(proc_path),'--output',str(proc_output)]
      if not a.preflight and (iteration%1000==0 or iteration==a.iterations-1):proc_cmd+=['--video',str(proc_output.with_suffix('.mp4'))]
      with open(proc_output.with_suffix('.log'),'w') as log:
        subprocess.run(proc_cmd,cwd=a.eval_workspace,env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=a.eval_gpu,MUJOCO_GL='egl',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
      values=json.loads(proc_output.read_text())['per_env'];proc_summary={}
      for label in DIRECTIONS:
        mask=proc_bank['labels']==label
        proc_summary[label]={'n':int(mask.sum()),'stable_1s':float(np.array(values['ever_stood_1s'])[mask].mean()),'stable_10s':float(np.array(values['success_10s'])[mask].mean())}
        for key,value in proc_summary[label].items():
          if self.logger.writer is not None:self.logger.writer.add_scalar(f'ProceduralValidation/{label}/{key}',value,iteration)
      atomic_json(a.log_dir/'procedural_progress.json',{'iteration':iteration,'validation':proc_summary})
      print('PROCEDURAL_VALIDATION_COMPLETE',iteration,json.dumps(proc_summary),flush=True)



def main():
  p=argparse.ArgumentParser();p.add_argument('--arm',choices=['L0','L1','L2','L3','L4','L5'],required=True);p.add_argument('--bank-dir',type=Path,required=True);p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--eval-workspace',type=Path,required=True);p.add_argument('--eval-gpu',required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=10000);p.add_argument('--seed',type=int,default=20260912);p.add_argument('--preflight',action='store_true');a=p.parse_args()
  for key in ('bank_dir','log_dir','eval_workspace'):setattr(a,key,getattr(a,key).resolve())
  cfg,agent=build_config(a.arm,a.bank_dir/'train.npz',a.num_envs,a.seed)
  random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
  agent.max_iterations=a.iterations;agent.save_interval=500;agent.logger='tensorboard' if a.preflight else 'wandb';agent.upload_model=False;agent.run_name=f'{a.arm}_fixed_low_seed{a.seed}'
  a.log_dir.mkdir(parents=True,exist_ok=False)
  metadata={'arm':a.arm,'from_scratch':True,'checkpoint_loaded':False,'seed':a.seed,'num_envs':a.num_envs,'iterations':a.iterations,'initial_learning_rate':1e-4,'low_smp_termination':a.arm not in ('L1','L3','L4','L5'),'low_smp_threshold':.005 if a.arm=='L2' else .02,'other_smp_threshold':.02,'grace_steps':5,'quota_counts_late_middle_supine_prone_left_right':quota_counts(a.num_envs,low_fraction=.2,late_fraction=.4) if a.arm in ('L3','L4','L5') else quota_counts(a.num_envs),'quota_scope':'fixed env IDs throughout training; no adaptive curriculum','gsi_pool_size':0,'prior':'f2s2','episode_seconds':10,'save_interval':500,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'procedural_fraction':cfg.events['gsi_reset'].params.get('procedural_fraction',0.),'procedural_bank_sha256':hashlib.sha256(Path(cfg.events['gsi_reset'].params['procedural_bank_path']).read_bytes()).hexdigest() if cfg.events['gsi_reset'].params.get('procedural_bank_path') else None,'train_bank_sha256':hashlib.sha256((a.bank_dir/'train.npz').read_bytes()).hexdigest(),'validation_bank_sha256':hashlib.sha256((a.bank_dir/'validation.npz').read_bytes()).hexdigest()}
  atomic_json(a.log_dir/'launch.json',metadata);dump_yaml(a.log_dir/'params/env.yaml',asdict(cfg));dump_yaml(a.log_dir/'params/agent.yaml',asdict(agent))
  env=None
  try:
    env=ManagerBasedRlEnv(cfg,device='cuda:0');wrapper=FixedWrapper(env,clip_actions=agent.clip_actions);runner=FixedRunner(wrapper,asdict(agent),str(a.log_dir),'cuda:0');runner.args=a
    obs,_=env.reset();assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
    assert env._smp_gsi_pool.shape[0]==0 and env.sim.data.qvel.abs().max()<1e-6
    for key in ('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','ee_pos_w','joint_pos','joint_vel'):
      v=getattr(env._smp_buffer,key);assert torch.allclose(v,v[:,:1].expand_as(v)),key
    assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
    runner.save(str(a.log_dir/'initial.pt'))
    atomic_json(a.log_dir/'reset_verification.json',{'qpos_sha256':hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),'qvel_max':float(env.sim.data.qvel.abs().max()),'group_sha256':hashlib.sha256(env._fixed_group.cpu().numpy().tobytes()).hexdigest(),'quota_counts':torch.bincount(env._fixed_group,minlength=6).cpu().tolist(),'source_by_direction':[[int(((env._fixed_group==g)&(env._fixed_source==s)).sum()) for s in (0,1)] for g in range(6)],'actor_dim':93,'critic_dim':960,'gsi_pool_size':0})
    print('CONFIG_VERIFIED',json.dumps(metadata),flush=True);runner.active=True
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    assert torch.bincount(env._fixed_group,minlength=6).tolist()==list(env._fixed_quota_counts)
    stats={name:{key:float(v) for key,v in zip(('steps','ended','stood','low_ended','ended_age_steps'),row)} for name,row in zip(STAGES+DIRECTIONS,wrapper.counts.cpu())}
    atomic_json(a.log_dir/'completed.json',{'iteration':runner.current_learning_iteration,'counts':stats,'current_episode_s_by_low_direction':{name:float(env.episode_length_buf[env._fixed_group==i+2].float().mean()*env.step_dt) for i,name in enumerate(DIRECTIONS)}})
  except BaseException as e:
    atomic_json(a.log_dir/'failed.json',{'error':repr(e)});raise
  finally:
    if env is not None:env.close()

if __name__=='__main__':main()
