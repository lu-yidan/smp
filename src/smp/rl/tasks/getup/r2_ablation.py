"""R2 transfer: standing convergence, stalled load, finite planar escape credit."""
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import balanced_dynamics as bd,multiterrain as mt,ft_prone_speed as ft,v33_reward_transfer as v
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
from smp.rl.tasks.getup.natural_curriculum import weights_for
from smp.rl.tasks.getup.master_deployment_contract import CONTROL

ARMS=('F0','F1','F2','F3','T0','T1','T2','T3')
S_ARMS=('F1','F3','T1','T2','T3')
L_ARMS=('F2','F3','T2','T3')
NAMES=('stand_pose','stand_feet','stand_body','stand_slew','stalled_load','escape_credit')
WEIGHTS=(.08,-.03,-.02,-.005,-.20,1.)

def gate(x,lo,hi):return ((x-lo)/(hi-lo)).clamp(0,1)

def initialize(env):
 v.init(env)
 if hasattr(env,'_ra_timer'):return
 r=env.scene['robot'];n=env.num_envs;dev=env.device
 env._ra_timer=torch.zeros_like(r.data.joint_pos)
 env._ra_motion=torch.zeros(25,n,31,device=dev);env._ra_motion_cursor=0
 env._ra_no_progress=torch.zeros(n,device=dev,dtype=torch.bool)
 env._ra_load_acc=torch.zeros(n,device=dev);env._ra_subtick=0
 env._ra_hold=torch.zeros(n,device=dev);env._ra_gate=torch.zeros(n,device=dev)
 env._ra_values=torch.zeros(n,6,device=dev);env._ra_tick=-1
 env._ra_prev_target=torch.zeros_like(r.data.joint_pos)
 env._ra_window=torch.zeros(10,n,device=dev);env._ra_cursor=0
 env._ra_best=torch.zeros(n,device=dev);env._ra_credit=torch.zeros(n,device=dev)
 env._ra_default=torch.tensor(CONTROL['default_joint_pos'],device=dev)
 env._ra_pose_weight=torch.tensor([.25 if any(k in name for k in ('hip','knee','ankle')) else 1. for name in r.joint_names],device=dev)
 env._ra_kind=torch.zeros(n,dtype=torch.long,device=dev) # 0 low, 1 middle, 2 late
 env._ra_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+62019)
 env._ra_taupeak=torch.zeros_like(r.data.joint_pos);env._ra_speedpeak=env._ra_taupeak.clone();env._ra_powerpeak=env._ra_taupeak.clone()
 env._ra_hightime=env._ra_taupeak.clone();env._ra_longest=env._ra_taupeak.clone()

def planar_score(env):
 pos,ext=mt.robot_bounds(env);rows=torch.arange(env.num_envs,device=env.device)
 gid=torch.stack(env._mt_plate_ids)[(env._mt_scene==2).long()]
 pp=env.sim.data.geom_xpos[rows,gid];rot=env.sim.data.geom_xmat[rows,gid];size=env.sim.model.geom_size[rows,gid]
 pe=torch.einsum('nij,nj->ni',rot.abs(),size)
 sep=((pos[:,:,:2]-pp[:,None,:2]).abs()-ext[:,:,:2]-pe[:,None,:2]).amax(-1)
 return -(-sep).clamp_min(0).sum(-1) # no above-board exception: lifting alone earns nothing

@requires_model_fields('body_mass','body_inertia','geom_size',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,arm='F0',evaluation=False,bank_path='outputs/multiterrain_bank/train.npz',**kwargs):
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 if arm.startswith('F') and not evaluation:
  ft.reset(env,ids,arm='FT_R2',evaluation=False,bank_path=bank_path,**kwargs)
 else:
  bd.reset(env,ids,bank_path=bank_path,evaluation=evaluation,multiterrain=True,**kwargs)
 initialize(env);env._ft_relaxed=True
 if arm.startswith('T') and not evaluation:
  if not hasattr(env,'_ra_natural'):
   b=np.load('datasets/reset_banks/natural_curriculum_v1/train.npz');env._ra_natural=torch.tensor(b['qpos'],device=env.device,dtype=torch.float32)
   env._ra_natpools=[];w=weights_for(b,0)
   for kind,stage in [(1,'middle'),(2,'late')]:
    pool=np.flatnonzero(b['stages']==stage);env._ra_natpools.append((kind,torch.tensor(pool,device=env.device),torch.tensor(w[pool]/w[pool].sum(),device=env.device,dtype=torch.float32)))
   for direction in range(4):
    chosen=torch.where((env._mt_scene==0)&(env._mt_direction==direction))[0]
    low=round(len(chosen)*.4);late=round(len(chosen)*.2)
    env._ra_kind[chosen[low:len(chosen)-late]]=1;env._ra_kind[chosen[len(chosen)-late:]]=2
    env._mt_source[chosen]=0;env._mt_source[chosen[:round(low*.25)]]=1
  # New source quotas are already used by mt.reset on subsequent calls; on first
  # call redraw the flat low subset so source labels match physical states.
  lowids=ids[(env._mt_scene[ids]==0)&(env._ra_kind[ids]==0)]
  if len(lowids):mt.reset(env,lowids,bank_path)
  for kind,pool,weights in env._ra_natpools:
   target=ids[env._ra_kind[ids]==kind]
   if not len(target):continue
   ix=pool[torch.multinomial(weights,len(target),replacement=True,generator=env._ra_rng)]
   q=env._ra_natural[ix];r=env.scene['robot'];state=r.data.default_root_state[target].clone();state[:,:3]=q[:,:3]+env.scene.env_origins[target];state[:,3:7]=q[:,3:7];state[:,7:]=0
   r.write_root_state_to_sim(state,env_ids=target);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=target)
  env._fixed_group=torch.where(env._ra_kind==1,1,torch.where(env._ra_kind==2,0,env._mt_direction+2))
  env.sim.forward();prime_static_history(env,ids)
 elif arm.startswith('F') and not evaluation:
  env._ra_kind=torch.where(env._ft_group==0,2,torch.where(env._ft_group==1,1,0))
 for name in ('timer','load_acc','hold','gate','values','prev_target','credit','taupeak','speedpeak','powerpeak','hightime','longest'):
  getattr(env,'_ra_'+name)[ids]=0
 env._ra_window[:,ids]=0;env._ra_best[ids]=0
 r=env.scene['robot'];motion=torch.cat([r.data.joint_pos,mt.height(env)[:,None],-r.data.projected_gravity_b[:,2,None]],-1)
 env._ra_motion[:,ids]=motion[ids][None];env._ra_no_progress[ids]=False
 score=planar_score(env)
 if not hasattr(env,'_ra_initial'):env._ra_initial=torch.zeros_like(score)
 env._ra_initial[ids]=(-score[ids]).clamp_min(.01)

# Shared pure tensor helper, also calibrated on measured R0/R2 trajectories.
def stalled_update(timer,tau_ratio,dq,dt):
 high=tau_ratio.abs()>.7
 timer=torch.where(high,timer+dt,torch.zeros_like(timer))
 cost=gate(tau_ratio.abs(),.7,.95).square()*gate(timer,.3,1.)*gate(.3-dq.abs(),0,.3)
 return timer,cost

def sample_substep(env):
 initialize(env)
 if env._ra_subtick%env.cfg.decimation==0:env._ra_load_acc.zero_()
 r=env.scene['robot'];tau=r.data.qfrc_actuator;dq=r.data.joint_vel
 env._ra_timer,c=stalled_update(env._ra_timer,tau/env._r_limits,dq,env.physics_dt)
 # Reward must remain sensitive to a single stalled ankle; mean over all29 dilutes it.
 env._ra_load_acc+=c.amax(-1)
 env._ra_taupeak=torch.maximum(env._ra_taupeak,tau.abs());env._ra_speedpeak=torch.maximum(env._ra_speedpeak,dq.abs());env._ra_powerpeak=torch.maximum(env._ra_powerpeak,(tau*dq).abs())
 env._ra_hightime+=(tau.abs()>.7*env._r_limits)*env.physics_dt;env._ra_longest=torch.maximum(env._ra_longest,env._ra_timer)
 env._ra_subtick+=1
 return c.amax(-1)

def update(env):
 initialize(env)
 if env._ra_tick==env.common_step_counter:return
 env._ra_tick=env.common_step_counter;r=env.scene['robot'];z=mt.height(env);u=(-r.data.projected_gravity_b[:,2]).clamp(0,1)
 feet=mt.ground_force(env,'quality_feet')[...,2].abs();other=mt.ground_force(env,'quality_other')[...,2].abs().sum(-1)
 trapped=((env._mt_scene==1)|(env._mt_scene==2))&~env._mt_escaped
 ready=(z>.95)&(u>.85)&(feet.amin(-1)>15)&(other<20)&~trapped
 env._ra_hold=torch.where(ready,env._ra_hold+env.step_dt,0.)
 g=gate(env._ra_hold,.2,.5)*gate(z,.95,1.1)*gate(u,.85,.95)*gate(feet.amin(-1),15,40)
 env._ra_gate=g
 motion=torch.cat([r.data.joint_pos,z[:,None],u[:,None]],-1)
 env._ra_motion[env._ra_motion_cursor]=motion;env._ra_motion_cursor=(env._ra_motion_cursor+1)%25
 span=env._ra_motion.amax(0)-env._ra_motion.amin(0)
 env._ra_no_progress=(span[:,:29].amax(-1)<.10)&(span[:,29]<.04)&(span[:,30]<.08)
 error=((r.data.joint_pos-env._ra_default)/.35).square()*env._ra_pose_weight
 pose=1/(1+error.mean(-1))
 foot_speed=r.data.body_link_lin_vel_w[:,env._r_feet,:].square().sum(-1).mean(-1)
 body=r.data.root_link_lin_vel_w[:,:2].square().sum(-1)/.15**2+r.data.root_link_ang_vel_w.square().sum(-1)/.5**2
 target=env.action_manager.get_term('joint_pos')._processed_actions
 slew=((target-env._ra_prev_target)/.1).square().mean(-1).clamp_max(10)
 slew=torch.where(env.episode_length_buf<=1,0.,slew);env._ra_prev_target=target.clone()
 active=(env._mt_scene==1)|(env._mt_scene==2)
 progress=(1+planar_score(env)/env._ra_initial).clamp(0,1)
 progress=torch.where(active&env._mt_ever_contact&~env._mt_invalid,progress,0.)
 env._ra_window[env._ra_cursor]=progress;env._ra_cursor=(env._ra_cursor+1)%10
 confirmed=env._ra_window.amin(0);delta=(confirmed-env._ra_best).clamp_min(0);env._ra_best=torch.maximum(env._ra_best,confirmed);env._ra_credit+=.5*delta
 env._ra_values.copy_(torch.stack([g*pose,g*(foot_speed/.1**2).clamp_max(10),g*body.clamp_max(10),g*slew,env._ra_load_acc/env.cfg.decimation*env._ra_no_progress,.5*delta/env.step_dt],-1))

def reward(env,index):update(env);return env._ra_values[:,index]
def metric(env):update(env);return env._ra_gate
