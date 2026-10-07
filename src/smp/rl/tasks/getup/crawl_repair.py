"""Repair crawl reward credit and preserve post-clear recovery."""
import torch
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import crawl_first as c, multiterrain as mt, v33_reward_transfer as v

@requires_model_fields('body_mass','body_inertia','geom_size','geom_aabb','geom_rbound','geom_friction',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,bank_path='outputs/crawl_bank_v1/train.npz',evaluation=False):
 c.reset(env,env_ids,arm='CR1',bank_path=bank_path,evaluation=evaluation)
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 if not hasattr(env,'_cf_stand_anchor'):
  env._cf_stand_anchor=torch.zeros(env.num_envs,2,device=env.device);env._cf_stood=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
  env._cf_stand_hold=torch.zeros(env.num_envs,device=env.device);env._cf_bonus=torch.zeros(env.num_envs,device=env.device);env._cf_good=env._cf_stood.clone();env._cf_tick=-1
 env._cf_stood[ids]=False;env._cf_stand_hold[ids]=0;env._cf_bonus[ids]=0;env._cf_good[ids]=False
 if len(ids)==env.num_envs:env._cf_tick=-1

def update(env,env_ids=None):
 if not hasattr(env,'_cf_stood') or env._cf_tick==env.common_step_counter:return
 env._cf_tick=env.common_step_counter;before=env._tb_done.clone();c.update(env,env_ids)
 # One transition per episode, with timestep compensation in reward.
 env._cf_bonus=(env._tb_done&~before&(env._cr_kind!=2)).float()
 r=env.scene['robot'];z=mt.height(env);u=-r.data.projected_gravity_b[:,2];feet=mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)
 good=env._cr_clear&(z>=1.15)&(u>=.90)&(feet>20)
 env._cf_good=good;env._cf_stand_hold=torch.where(good,env._cf_stand_hold+env.step_dt,0)
 first=(env._cf_stand_hold>=1.)&~env._cf_stood
 env._cf_stand_anchor[first]=r.data.root_link_pos_w[first,:2];env._cf_stood|=first

def task(env,task_terms,**kwargs):
 update(env)
 product=mt.recorded_task_smp_product(env,task_terms,**kwargs)
 # Keep SMP, but ensure the restored rise component retains 30% strength at S=0.
 rise=sum(w*func(env,**params) for func,w,params in task_terms[:4])
 extra=.30*(1-env._fixed_smp_score.clamp(0,1))*rise
 env._cf_rise_floor=extra.detach();env._plate_product=product+extra
 return product+extra

def completion(env,index=2):
 update(env);return env._cf_bonus/env.step_dt

def stand_reward(env):
 update(env);return env._cf_good.float()

def position_cost(env):
 update(env);r=env.scene['robot'];z=mt.height(env);u=-r.data.projected_gravity_b[:,2]
 gate=((z-.85)/.3).clamp(0,1)*((u-.7)/.23).clamp(0,1)
 dist=(r.data.root_link_pos_w[:,:2]-env._cf_stand_anchor).norm(dim=-1)
 settled=env._cf_stood*gate*c.displacement_cost(dist)
 # Mild pre-standing boundary only for genuinely escaped table trials.
 escape_dist=(r.data.root_link_pos_w[:,:2]-env._tb_anchor).norm(dim=-1)
 pre=env._tb_done&~env._cf_stood&(env._cr_kind!=2)
 return settled+.25*pre*((escape_dist-.6).clamp_min(0)/.4).square().clamp_max(2)
