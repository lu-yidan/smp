"""Soft recovery costs; engineering thresholds, not certified hardware limits."""
import torch
from smp.rl.tasks.getup.master_deployment_contract import CONTROL,JOINT_NAMES

def speed_reference(names):
  # mjlab G1 motor reference speeds: 7520-14=32, 7520-22=20,
  # 5020=37, 4010=22 rad/s; coupled ankle/waist nominal 1:1.
  return [20. if ('knee' in n or 'hip_roll' in n) else 32. if ('hip_pitch' in n or 'hip_yaw' in n or n=='waist_yaw_joint') else 22. if ('wrist_pitch' in n or 'wrist_yaw' in n) else 37. for n in names]

def cost_vectors(tau,dq,limits,speeds):
  effort=((tau.abs()/limits-.7).clamp_min(0)/.3).square().mean(-1)
  speed=(dq.abs()/(.5*speeds)-1).clamp_min(0).square().clamp_max(25).mean(-1)
  return effort,speed

def sample_quality(env):
  robot=env.scene['robot']
  if not hasattr(env,'_quality_accum'):
    env._quality_limits=torch.tensor([CONTROL['tau_limit'][JOINT_NAMES.index(n)] for n in robot.joint_names],device=env.device)
    env._quality_speeds=torch.tensor(speed_reference(robot.joint_names),device=env.device)
    env._quality_accum=torch.zeros(env.num_envs,2,device=env.device)
    env._quality_peaks=torch.zeros(env.num_envs,2,device=env.device)
    env._quality_tick=0;env._quality_cached_step=-1
    env._quality_prev_target=torch.zeros_like(robot.data.joint_pos)
    env._quality_costs=torch.zeros(env.num_envs,3,device=env.device)
  if env._quality_tick%env.cfg.decimation==0:
    env._quality_accum.zero_();env._quality_peaks.zero_()
  tau=robot.data.qfrc_actuator;dq=robot.data.joint_vel
  e,v=cost_vectors(tau,dq,env._quality_limits,env._quality_speeds)
  env._quality_accum+=torch.stack([e,v],-1)
  torch.maximum(env._quality_peaks,torch.stack([tau.abs().amax(-1),dq.abs().amax(-1)],-1),out=env._quality_peaks)
  env._quality_tick+=1
  return v

def quality_cost(env,index):
  if env._quality_cached_step!=env.common_step_counter:
    env._quality_cached_step=env.common_step_counter
    target=env.action_manager.get_term('joint_pos')._processed_actions
    slew=((target-env._quality_prev_target)/.1).square().clamp_max(25).mean(-1)
    slew=torch.where(env.episode_length_buf<=1,0.,slew)
    env._quality_prev_target.copy_(target)
    env._quality_costs[:,:2]=env._quality_accum/env.cfg.decimation
    env._quality_costs[:,2]=slew
  # Fixed warmup avoids a sudden objective jump after loading the critic.
  start=getattr(env,'_quality_start_counter',0)
  ramp=min(1.,max(0.,(env.common_step_counter-start)/(24*250)))
  return env._quality_costs[:,index]*ramp
