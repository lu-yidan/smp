"""D-series finite-credit recovery progress, staged costs, and intermediate resets."""
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import balanced_dynamics as bd,multiterrain as mt,v33_reward_transfer as v
from smp.rl.tasks.getup.natural_low_reset import prime_static_history

def init(env):
 if hasattr(env,'_d_best'):return
 n=env.num_envs;dev=env.device
 for name in ['best','credit','progress','escape_bonus','stall_time','first_progress','elapsed']:
  setattr(env,'_d_'+name,torch.zeros(n,device=dev))
 env._d_window=torch.zeros(10,n,device=dev);env._d_cursor=0
 env._d_paid=torch.zeros(n,dtype=torch.bool,device=dev);env._d_mid=torch.zeros(n,dtype=torch.long,device=dev)
 env._d_tick=-1
 env._d_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+43191)
 b=np.load('datasets/d_mid/train.npz');env._d_mid_bank=torch.tensor(b['qpos'],device=dev,dtype=torch.float32)
 env._d_mid_pools=[torch.tensor(np.where(b['kind']==k)[0],device=dev) for k in (0,1)]

def gate(x,lo,hi):return ((x-lo)/(hi-lo)).clamp(0,1)

def potential(env):
 r=env.scene['robot'];z=mt.height(env);u=(-r.data.projected_gravity_b[:,2]).clamp(0,1)
 feet=mt.ground_force(env,'quality_feet')[...,2].abs().sum(-1)
 hands=env.scene['d_support'].data.force[...,2].abs().sum(-1)+env.scene['d_support_terrain'].data.force[...,2].abs().sum(-1)
 supported=(feet+hands>20).float()
 # Physical contact is a gate, not a repeatable standalone reward. No benefit from height alone.
 score=gate(z,.20,.85)*gate(u,.10,.80)*supported
 trapped=((env._mt_scene==1)|(env._mt_scene==2))&~env._mt_escaped
 return torch.where(trapped,torch.zeros_like(score),score)

@requires_model_fields('body_mass','body_inertia','geom_size',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,middle=False,**kwargs):
 bd.reset(env,env_ids,**kwargs);v.init(env);init(env)
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 if middle:
  # Equal removal in each direction and source keeps remaining low quotas balanced.
  for direction in range(4):
   for source in range(2):
    pool=torch.where((env._mt_scene==0)&(env._mt_direction==direction)&(env._mt_source==source))[0]
    count=round(len(pool)*.4);chosen=pool[:count];env._d_mid[chosen[::2]]=1;env._d_mid[chosen[1::2]]=2
  for k in (0,1):
   target=ids[env._d_mid[ids]==k+1]
   if not len(target):continue
   pool=env._d_mid_pools[k];idx=pool[torch.randint(len(pool),(len(target),),generator=env._d_rng,device=env.device)]
   q=env._d_mid_bank[idx];r=env.scene['robot'];state=r.data.default_root_state[target].clone();state[:,:3]=q[:,:3]+env.scene.env_origins[target];state[:,3:7]=q[:,3:7];state[:,7:]=0
   r.write_root_state_to_sim(state,env_ids=target);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=target)
  env.sim.forward();prime_static_history(env,ids)
 env._d_best[ids]=0;env._d_window[:,ids]=0;env._d_credit[ids]=0;env._d_progress[ids]=0;env._d_paid[ids]=False;env._d_escape_bonus[ids]=0;env._d_stall_time[ids]=0;env._d_first_progress[ids]=-1;env._d_elapsed[ids]=0
 # Baseline geometric potential prevents credit merely for being reset halfway up.
 r=env.scene['robot'];z=mt.height(env);u=(-r.data.projected_gravity_b[:,2]).clamp(0,1)
 env._d_best[ids]=(gate(z,.20,.85)*gate(u,.10,.80))[ids]

def update(env):
 if env._d_tick==env.common_step_counter:return
 env._d_tick=env.common_step_counter
 x=potential(env);env._d_window[env._d_cursor]=x;env._d_cursor=(env._d_cursor+1)%10
 confirmed=env._d_window.amin(0)
 delta=(confirmed-env._d_best).clamp_min(0);env._d_best=torch.maximum(env._d_best,confirmed)
 env._d_progress=2*delta/env.step_dt;env._d_credit+=2*delta
 env._d_elapsed+=env.step_dt
 first=(env._d_first_progress<0)&(env._d_credit>.1);env._d_first_progress[first]=env._d_elapsed[first]
 valid=env._mt_escaped&env._mt_ever_contact&~env._mt_invalid
 bonus=valid&~env._d_paid;env._d_paid|=valid;env._d_escape_bonus=bonus.float()/env.step_dt
 r=env.scene['robot'];quiet=r.data.joint_vel.square().mean(-1).sqrt()<.1
 stalled=(mt.height(env)<.65)&quiet
 env._d_stall_time=torch.where(stalled,env._d_stall_time+env.step_dt,0.)

def progress(env):update(env);return env._d_progress

def escape_bonus(env):update(env);return env._d_escape_bonus

def cost(env,index):
 raw=v.cost(env,index);r=env.scene['robot'];u=(-r.data.projected_gravity_b[:,2]).clamp(0,1)
 # Resume full cost as supported crouch is reached; no hard phase switch.
 load=mt.ground_force(env,'quality_feet')[...,2].abs().sum(-1)
 readiness=gate(mt.height(env),.55,.85)*gate(u,.45,.80)*gate(load,20.,80.)
 return raw*(.5+.5*readiness)

def metric(env):update(env);return env._d_credit
