"""V33-inspired ordered recovery, adapted to the deployed 93D model.
This is reward transfer, not a reproduction of the legacy V7/board experiment.
"""
import torch
from smp.rl.tasks.getup import scratch_tradeoffs as b, r1_quality as f

TASK_NAMES=('stage_pose','head_velocity','height','upright','feet_quiet','base_quiet','angular_quiet','joint_quiet','action_quiet')
TASK_WEIGHTS=(.22,.18,.10,.15,.08,.07,.07,.06,.07)
COST_NAMES=('action_rate','action_acc','joint_acc','torque','joint_speed','joint_power','head_overspeed','sustained_effort')
COST_WEIGHTS=(-.0015,-.0012,-5e-8,-1e-6,-.02,-2e-6,-1.,-.05)

def init(env):
    b.init_buffers(env);f.init(env)
    if hasattr(env,'_v_stage'):return
    n=env.num_envs;d=env.device;r=env.scene['robot']
    env._v_stage=torch.zeros(n,dtype=torch.long,device=d)
    env._v_hold=torch.zeros(n,dtype=torch.long,device=d)
    env._v_task=torch.zeros(n,9,device=d);env._v_cost=torch.zeros(n,8,device=d)
    env._v_accum=torch.zeros(n,6,device=d)
    env._v_last_action=torch.zeros_like(r.data.joint_pos);env._v_last_delta=torch.zeros_like(r.data.joint_pos)
    env._v_cache=-1;env._v_tick=0
    env._v_start_counter=env.common_step_counter
    env._v_ramp_updates=getattr(env.cfg,'v33_ramp_updates',500)

def ramp(env):
    return min(1.,max(0.,(env.common_step_counter-env._v_start_counter)/24/env._v_ramp_updates))

def reset(env,env_ids=None):
    init(env)
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    r=env.scene['robot'];z=r.data.site_pos_w[env_ids,env._r_head,2]-env.scene.env_origins[env_ids,2]
    if hasattr(env,'_mt_bank'):
        from smp.rl.tasks.getup.multiterrain import height
        z=height(env)[env_ids]
    u=(-r.data.projected_gravity_b[env_ids,2]).clamp(0,1)
    # Start from the current pose: late resets need not crouch down to unlock standing.
    s=torch.zeros_like(env_ids)
    s=torch.where((z>=.55)&(u>=.55),1,s)
    s=torch.where((z>=.78)&(u>=.72),2,s)
    s=torch.where((z>=1.08)&(u>=.85),3,s)
    env._v_stage[env_ids]=s;env._v_hold[env_ids]=0
    env._v_accum[env_ids]=0;env._f_effort_ms[env_ids]=0
    env._v_last_action[env_ids]=0;env._v_last_delta[env_ids]=0
    # Keep the pre-reset reward cache for this control step; other environments
    # must not advance their stage twice when just one environment resets.

def sample_substep(env):
    init(env)
    if env._v_tick%env.cfg.decimation==0:env._v_accum.zero_()
    r=env.scene['robot'];s=env._v_stage;tau=r.data.qfrc_actuator;dq=r.data.joint_vel
    env._f_effort_ms.copy_(f.sustained_update(env._f_effort_ms,tau/env._r_limits,env.physics_dt))
    limits=dq.new_tensor((6.,5.,4.,3.5))[s,None]
    powers=dq.new_tensor((140.,110.,90.,75.))[s,None]
    vz=r.data.site_lin_vel_w[:,env._r_head,2]
    env._v_accum+=torch.stack([
        r.data.joint_acc.square().sum(-1)*dq.new_tensor((.35,.65,1.,1.))[s],
        tau.square().sum(-1)*dq.new_tensor((.5,.75,1.,1.))[s],
        (dq.abs()-limits).clamp_min(0).square().sum(-1),
        ((tau*dq).abs()-powers).clamp_min(0).square().mean(-1),
        (vz.abs()-.2).clamp_min(0).square(),f.effort_cost(env._f_effort_ms)],-1)
    env._v_tick+=1
    return vz.abs()

def components(env):
    init(env)
    if env._v_cache==env.common_step_counter:return env._v_task
    env._v_cache=env.common_step_counter
    r=env.scene['robot'];z=r.data.site_pos_w[:,env._r_head,2]-env.scene.env_origins[:,2]
    if hasattr(env,'_mt_bank'):
        from smp.rl.tasks.getup.multiterrain import height
        z=height(env)
    vz=r.data.site_lin_vel_w[:,env._r_head,2];u=(-r.data.projected_gravity_b[:,2]).clamp(0,1)
    knees=r.data.joint_pos[:,env._r_knees];s=env._v_stage
    fallen=(z<.65)&(u<.45);s[fallen]=0;env._v_hold[fallen]=0
    # Contact-supported holds; reset labels do not determine eligibility.
    load=ground_force(env,'quality_feet')[...,2].abs().amin(-1)
    ready=((s==0)&(z>=.55)&(u>=.55)&(knees.amin(-1)>=.8)&(vz.abs()<=.16))|((s==1)&(z>=.78)&(u>=.72)&(knees.amin(-1)>=.6)&(vz.abs()<=.18))|((s==2)&(z>=1.08)&(u>=.85)&(knees.abs().amax(-1)<.8)&(load>20)&(vz.abs()<=.12))
    env._v_hold=torch.where(ready,env._v_hold+1,0)
    advance=env._v_hold>=torch.where(s==2,25,10)
    s.copy_(torch.where(advance,(s+1).clamp_max(3),s));env._v_hold[advance]=0
    height=z.new_tensor((.62,.86,1.15,1.15))[s];upr=z.new_tensor((.60,.76,.93,.93))[s]
    klo=z.new_tensor((.8,.6,0.,0.))[s,None];khi=z.new_tensor((1.8,1.6,.65,.65))[s,None]
    kerr=((klo-knees).clamp_min(0).square()+(knees-khi).clamp_min(0).square()).mean(-1)
    pose=torch.exp(-8*(height-z).clamp_min(0).square()-6*(upr-u).clamp_min(0).square()-5*kerr)
    target=z.new_tensor((.06,.08,.10,0.))[s]*((height-z)/.2).clamp(0,1)
    limit=z.new_tensor((.16,.18,.18,.12))[s]
    vel=torch.exp(-45*(vz-target).square()-140*(vz.abs()-limit).clamp_min(0).square())
    gate=b.smooth_gate(z,.85,1.15)*b.smooth_gate(u,.7,.93)
    foot=r.data.body_link_lin_vel_w[:,env._r_feet,:].square().sum(-1).mean(-1)
    delta=env.action_manager.action-env._v_last_action
    acc=delta-env._v_last_delta;fresh=env.episode_length_buf<=1
    delta=torch.where(fresh[:,None],0.,delta);acc=torch.where(fresh[:,None],0.,acc)
    env._v_task.copy_(torch.stack([pose,vel,torch.exp(-2*(1.15-z).clamp_min(0).square()),u.square(),
        1-gate+gate*torch.exp(-20*foot),1-gate+gate*torch.exp(-8*r.data.root_link_lin_vel_w.square().sum(-1)),
        torch.exp(-.8*r.data.root_link_ang_vel_w.square().sum(-1)),torch.exp(-.04*r.data.joint_vel.square().mean(-1)),torch.exp(-4*delta.square().mean(-1))],-1))
    env._v_cost[:,:2]=torch.stack([delta.square().sum(-1)*z.new_tensor((.5,.75,1.,1.))[s],acc.square().sum(-1)*z.new_tensor((.3,.6,1.,1.))[s]],-1)
    env._v_cost[:,2:]=env._v_accum/env.cfg.decimation
    env._v_last_action.copy_(env.action_manager.action);env._v_last_delta.copy_(delta)
    return env._v_task

def task(env,index):return components(env)[:,index]
def cost(env,index):
    components(env)
    return env._v_cost[:,index]*ramp(env)
def metric(env):return components(env)[:,0]


def ground_force(env,name):
    force=env.scene[name].data.force
    if hasattr(env,'_mt_bank'):force=force+env.scene[name+'_terrain'].data.force
    return force
