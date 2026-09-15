"""R1 continuation: posture, sustained effort, and late-recovery motion costs.
All diagnostic buffers are per-environment; episode reset clears effort memory.
The reference is a soft engineering target, not a tracked get-up trajectory.
"""
import math
import torch
from smp.rl.tasks.getup import scratch_tradeoffs as base
from smp.rl.tasks.getup.master_deployment_contract import CONTROL, JOINT_NAMES

NAMES=('natural_pose','sustained_effort','near_stand_speed','near_stand_slew')
WEIGHTS={'F0':(0.,0.,0.,0.),'F1':(.1,0.,0.,0.),'F2':(0.,-.05,0.,0.),'F3':(0.,0.,-.025,-.001),'F4':(.05,-.025,-.0125,-.0005)}

def reference_values(names):
    q=dict(zip(JOINT_NAMES,CONTROL['default_joint_pos']))
    for side in ('left','right'):
        q[f'{side}_hip_pitch_joint']=-.15
        q[f'{side}_knee_joint']=.30
        q[f'{side}_ankle_pitch_joint']=-.15
    return [q[n] for n in names]

def ramp(env):
    return min(1.,max(0.,(env.common_step_counter-env._f_start_counter)/24/500))

def sustained_update(previous,normalized_tau,dt):
    return previous+(1-math.exp(-dt/.5))*(normalized_tau.square()-previous)

def effort_cost(mean_square):
    return ((mean_square.clamp_min(0).sqrt()-.45).clamp_min(0)/.55).square().topk(3,dim=-1).values.mean(-1)

def init(env):
    base.init_buffers(env)
    if hasattr(env,'_f_effort_ms'):return
    r=env.scene['robot'];dev=env.device
    env._f_effort_ms=torch.zeros_like(r.data.joint_pos)
    env._f_speed_accum=torch.zeros(env.num_envs,device=dev)
    env._f_previous_target=torch.zeros_like(r.data.joint_pos)
    env._f_values=torch.zeros(env.num_envs,4,device=dev)
    env._f_cache_step=-1;env._f_tick=0
    env._f_reference=torch.tensor(reference_values(r.joint_names),device=dev)
    env._f_tolerances=torch.tensor([.35 if 'shoulder' in n else .4 if 'elbow' in n else .2 if 'waist' in n else .25 for n in r.joint_names],device=dev)
    env._f_regions=[[i for i,n in enumerate(r.joint_names) if any(k in n for k in keys)] for keys in [('shoulder','elbow','wrist'),('waist',),('hip','knee','ankle')]]
    env._f_gate=torch.zeros(env.num_envs,device=dev)
    env._f_pose_error=torch.zeros(env.num_envs,device=dev)

def sample_substep(env):
    init(env)
    if env._f_tick%env.cfg.decimation==0:
        env._f_effort_ms[env.episode_length_buf==0]=0
        env._f_speed_accum.zero_()
    r=env.scene['robot']
    env._f_effort_ms.copy_(sustained_update(env._f_effort_ms,r.data.qfrc_actuator/env._r_limits,env.physics_dt))
    speed=(r.data.joint_vel.abs()/(.2*env._r_speeds)-1).clamp_min(0).square().clamp_max(25).topk(3,dim=-1).values.mean(-1)
    env._f_speed_accum+=speed;env._f_tick+=1
    return effort_cost(env._f_effort_ms)

def cache(env):
    init(env)
    if env._f_cache_step==env.common_step_counter:return env._f_values
    env._f_cache_step=env.common_step_counter
    r=env.scene['robot'];z=r.data.site_pos_w[:,env._r_head,2]-env.scene.env_origins[:,2]
    upright=(-r.data.projected_gravity_b[:,2]).clamp(0,1)
    gate=base.smooth_gate(z,1.,1.15)*base.smooth_gate(upright,.8,.93)
    errors=((r.data.joint_pos-env._f_reference)/env._f_tolerances).square()
    pose=sum(w*errors[:,ids].mean(-1) for w,ids in zip((1.,1.,.5),env._f_regions))
    tilt=(r.data.projected_gravity_b[:,:2]/.15).square().sum(-1)
    feet=r.data.body_link_pos_w[:,env._r_feet,:];width=(feet[:,0,:2]-feet[:,1,:2]).norm(dim=-1)
    load=env.scene['quality_feet'].data.force[...,2].abs().amin(-1)
    other=env.scene['quality_other'].data.force[...,2].abs().sum(-1)
    support=base.smooth_gate(load,5.,20.)*base.smooth_gate(width,.08,.12)*(1-base.smooth_gate(width,.45,.55))
    target=env.action_manager.get_term('joint_pos')._processed_actions
    slew=((target-env._f_previous_target)/.1).square().clamp_max(25).mean(-1)
    slew=torch.where(env.episode_length_buf<=1,0.,slew);env._f_previous_target.copy_(target)
    env._f_values[:,0]=gate*support/(1+pose+tilt)/(1+other/80)
    env._f_values[:,1]=effort_cost(env._f_effort_ms)
    env._f_values[:,2]=gate*env._f_speed_accum/env.cfg.decimation
    env._f_values[:,3]=gate*slew
    env._f_gate=gate;env._f_pose_error=pose
    return env._f_values

def metric(env):
    return cache(env)[:,0]

def reward(env,index):
    return cache(env)[:,index]*ramp(env)
