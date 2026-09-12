"""Paired N0 scratch / N1 B1 fine-tune, no GSI reset, synchronized curriculum."""
import argparse, hashlib, json, os, random, subprocess, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml
from train_termination_ablation import build_config as baseline_config, AblationWrapper
from smp.rl.tasks.getup.natural_curriculum import reset_natural_curriculum, STAGES, DIRECTIONS, MIXES


def build_config(bank_path,num_envs=4096,seed=20260912):
  cfg,agent=baseline_config('B1',num_envs,seed)
  cfg.events['init_smp_state'].params['gsi_buffer_size']=0
  cfg.events.pop('gsi_refresh')
  cfg.events['gsi_reset']=EventTermCfg(func=reset_natural_curriculum,mode='reset',params={'bank_path':str(Path(bank_path).resolve())})
  return cfg,agent


def atomic_json(path,payload):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2));temp.replace(path)


def summarize_validation(report,bank):
  success=np.array(report['per_env']['ever_stood_1s']);strict=np.array(report['per_env']['success_10s']);out={}
  masks={stage:bank['stages']==stage for stage in STAGES}
  masks.update({label:(bank['stages']=='low')&(bank['labels']==label) for label in DIRECTIONS})
  for label,mask in masks.items():
    assert mask.any(),label
    out[label]={'n':int(mask.sum()),'stable_1s':float(success[mask].mean()),'stable_10s':float(strict[mask].mean())}
  return out


def passes(summary,stage):
  low_threshold,easy_threshold=((.15,.60),(.40,.75))[min(stage,1)]
  return min(summary[x]['stable_1s'] for x in DIRECTIONS)>=low_threshold and min(summary[x]['stable_1s'] for x in ('late','middle'))>=easy_threshold


class CourseWrapper(AblationWrapper):
  def step(self,actions):
    env=self.unwrapped;before=env._course_type.clone();direction=env._course_direction.clone();ages=env.episode_length_buf.clone()+1
    obs,reward,done,extras=super().step(actions);logs=extras['log']
    if not hasattr(self,'counts'):self.counts=torch.zeros(7,5,device=env.device)
    low=env.termination_manager.get_term('smp_too_low');stood=env.termination_manager.get_term('stood_up')
    masks=[before==i for i in range(3)]+[(before==2)&(direction==i) for i in range(4)]
    for i,(name,mask) in enumerate(zip(STAGES+DIRECTIONS,masks)):
      ended=mask&done.bool();self.counts[i]+=torch.stack((mask.sum(),ended.sum(),(ended&stood).sum(),(ended&low).sum(),ages[ended].sum())).float()
      steps,ends,wins,bads,age_sum=self.counts[i];den=ends.clamp_min(1)
      for key,value in [('steps',steps),('ended',ends),('stood_fraction',wins/den),('low_end_fraction',bads/den),('mean_episode_s',age_sum/den*env.step_dt)]:logs[f'Curriculum/{name}/{key}_cumulative']=value.clone()
    logs['Curriculum/stage']=env._course_stage
    for i,name in enumerate(STAGES):
      logs['Curriculum/'+name+'/draw_fraction']=env._course_draws[i]/env._course_draws.sum().clamp_min(1)
      logs['Curriculum/'+name+'/step_fraction']=(before==i).float().mean()
    return obs,reward,done,extras


class CourseRunner(MjlabOnPolicyRunner):
  active=False
  def save(self,path,infos=None):
    if self.active and self.current_learning_iteration==0 and Path(path).name=='model_0.pt':return
    env=self.env.unwrapped
    super().save(path,infos={**(infos or {}),'course_stage':getattr(env,'_course_stage',0)})
    if not self.active:return
    a=self.args;iteration=self.current_learning_iteration
    report_path=a.log_dir/'validation'/f'{iteration}.json';report_path.parent.mkdir(exist_ok=True)
    bank=np.load(a.bank_dir/'validation.npz')
    cmd=[str(a.eval_workspace/'.venv/bin/python'),'-u','scripts/audit_recovery_distribution.py','--policy-family','master','--checkpoint',str(Path(path).resolve()),'--steps','1000','--num-envs',str(len(bank['qpos'])),'--eval-bank',str((a.bank_dir/'validation.npz').resolve()),'--output',str(report_path.resolve())]
    if a.preflight or iteration%1000==0 or iteration==a.iterations-1:
      cmd+=['--video',str((a.log_dir/'validation'/f'{iteration}_20s.mp4').resolve())]
    with open(report_path.with_suffix('.log'),'w') as log:
      subprocess.run(cmd,cwd=a.eval_workspace,env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=a.eval_gpu,MUJOCO_GL='egl',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=summarize_validation(json.loads(report_path.read_text()),bank)
    stage=env._course_stage
    result={'arm':a.arm,'iteration':iteration,'stage':stage,'summary':summary,'passes':passes(summary,stage),'checkpoint':str(path)}
    atomic_json(a.pair_dir/f'{a.arm}_{iteration}.json',result)
    for name,metrics in summary.items():
      for key,value in metrics.items():
        if self.logger.writer is not None:self.logger.writer.add_scalar(f'Validation/{name}/{key}',value,iteration)
    print('VALIDATION_COMPLETE',json.dumps(result),flush=True)
    if a.preflight:return
    peer='N1' if a.arm=='N0' else 'N0';peer_path=a.pair_dir/f'{peer}_{iteration}.json'
    # A shared checkpoint barrier keeps the actual curriculum identical by update.
    while not peer_path.exists():
      if (a.pair_dir/f'{peer}_failed.json').exists():raise RuntimeError(f'{peer} failed; preserving checkpoints and stopping paired training')
      time.sleep(3)
    other=json.loads(peer_path.read_text());assert other['stage']==stage
    both=result['passes'] and other['passes']
    self.pass_streak=getattr(self,'pass_streak',0)+1 if both else 0
    if self.pass_streak>=2 and stage<2:
      env._course_stage=stage+1;self.pass_streak=0
      print('CURRICULUM_ADVANCED',env._course_stage,MIXES[env._course_stage],flush=True)
    atomic_json(a.log_dir/'progress.json',{'iteration':iteration,'next_stage':env._course_stage,'pair_pass_streak':self.pass_streak})


def main():
  p=argparse.ArgumentParser();p.add_argument('--arm',choices=['N0','N1'],required=True);p.add_argument('--bank-dir',type=Path,required=True);p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--pair-dir',type=Path,required=True);p.add_argument('--b1-checkpoint',type=Path);p.add_argument('--eval-workspace',type=Path,required=True);p.add_argument('--eval-gpu',default='2');p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=10000);p.add_argument('--seed',type=int,default=20260912);p.add_argument('--preflight',action='store_true');a=p.parse_args()
  a.bank_dir=a.bank_dir.resolve();a.log_dir=a.log_dir.resolve();a.pair_dir=a.pair_dir.resolve();a.eval_workspace=a.eval_workspace.resolve()
  assert (a.arm=='N1')==(a.b1_checkpoint is not None)
  cfg,agent=build_config(a.bank_dir/'train.npz',a.num_envs,a.seed)
  random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
  agent.max_iterations=a.iterations;agent.save_interval=500;agent.logger='tensorboard' if a.preflight else 'wandb';agent.upload_model=False;agent.run_name=f'{a.arm}_natural_curriculum_seed{a.seed}'
  a.log_dir.mkdir(parents=True,exist_ok=False);a.pair_dir.mkdir(parents=True,exist_ok=True)
  metadata={'arm':a.arm,'initialization':'random actor and critic' if a.arm=='N0' else 'B1 actor and critic including observation normalizers; fresh optimizer, learning rate and iteration','b1_checkpoint':str(a.b1_checkpoint),'prior':'f2s2 frozen','gsi_reset':False,'seed':a.seed,'num_envs':a.num_envs,'iterations':a.iterations,'save_interval':500,'stage_mixes_late_middle_low':MIXES,'low_direction_weights':[.25]*4,'zero_velocity_static_history':True,'episode_seconds':10,'terminations':list(cfg.terminations),'validation_bank_sha256':hashlib.sha256((a.bank_dir/'validation.npz').read_bytes()).hexdigest(),'train_bank_sha256':hashlib.sha256((a.bank_dir/'train.npz').read_bytes()).hexdigest(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
  if a.b1_checkpoint:metadata['b1_sha256']=hashlib.sha256(a.b1_checkpoint.read_bytes()).hexdigest()
  atomic_json(a.log_dir/'launch.json',metadata);dump_yaml(a.log_dir/'params/env.yaml',asdict(cfg));dump_yaml(a.log_dir/'params/agent.yaml',asdict(agent))
  env=None
  try:
    env=ManagerBasedRlEnv(cfg,device='cuda:0');runner=CourseRunner(CourseWrapper(env,clip_actions=agent.clip_actions),asdict(agent),str(a.log_dir),'cuda:0');runner.args=a
    obs,_=env.reset();assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
    assert env._smp_gsi_pool.shape[0]==0 and env.sim.data.qvel.abs().max()<1e-6
    buf=env._smp_buffer
    for key in ('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','ee_pos_w','joint_pos','joint_vel'):
      value=getattr(buf,key)
      assert torch.allclose(value,value[:,:1].expand_as(value)),key
    assert buf.root_lin_vel_w.abs().max()<1e-6 and buf.root_ang_vel_w.abs().max()<1e-6 and buf.joint_vel.abs().max()<1e-6
    assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
    runner.save(str(a.log_dir/'random_initial.pt'))
    if a.b1_checkpoint:
      runner.load(str(a.b1_checkpoint),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})
      env.common_step_counter=0;runner.current_learning_iteration=0
    runner.save(str(a.log_dir/'initial.pt'))
    atomic_json(a.log_dir/'reset_verification.json',{'qpos_sha256':hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),'qvel_max':float(env.sim.data.qvel.abs().max()),'stage_draws':env._course_draws.cpu().tolist(),'gsi_pool_size':0,'actor_dim':93,'critic_dim':960})
    print('CONFIG_VERIFIED',metadata,flush=True);runner.active=True
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    atomic_json(a.log_dir/'completed.json',{'iteration':runner.current_learning_iteration})
  except BaseException as e:
    atomic_json(a.pair_dir/f'{a.arm}_failed.json',{'error':repr(e)});raise
  finally:
    if env is not None:env.close()

if __name__=='__main__':main()
