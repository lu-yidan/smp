"""C0--C3: replace height-only success and cover early recovery overspeed.
F2 sustained-effort cost remains fully active; only the new terms ramp.
"""
import torch
from smp.rl.tasks.getup import r1_quality as f
from smp.rl.tasks.getup import scratch_tradeoffs as b
from smp.rl.tasks.getup.mdp.rewards import upward_velocity

ARMS=('C0','C1','C2','C3')
SPEED_WEIGHT=-.01

def ramp(env):
    return min(1.,max(0.,(env.common_step_counter-env._c_start_counter)/24/500))

def init(env):
    f.init(env)
    if hasattr(env,'_c_accum'):return
    env._c_accum=torch.zeros(env.num_envs,2,device=env.device)
    env._c_tick=0;env._c_cache_step=-1
    env._c_cost=torch.zeros(env.num_envs,device=env.device)
    env._c_support=torch.zeros_like(env._c_cost);env._c_quality=torch.zeros_like(env._c_cost)
    env._c_gate=torch.zeros_like(env._c_cost);env._c_new_up=torch.zeros_like(env._c_cost)
    env._c_old_up=torch.zeros_like(env._c_cost)

def shaped_upward(vz):
    # Preserve original low-state zero-velocity value exp(-6.25), unlike R3.
    return torch.exp(-100*(.25-vz).clamp_min(0).square())

def blend_upward(height,vz,quality):
    gate=b.smooth_gate(height,.85,1.15)
    return (1-gate)*shaped_upward(vz)+gate*quality

def speed_cost(dq,speeds,near_gate):
    limit=(.5-.3*near_gate[:,None])*speeds
    return (dq.abs()/limit-1).clamp_min(0).square().clamp_max(25).topk(3,dim=-1).values.mean(-1)

def sample_substep(env):
    init(env)
    if env._c_tick%env.cfg.decimation==0:env._c_accum.zero_()
    r=env.scene['robot'];z=r.data.site_pos_w[:,env._r_head,2]-env.scene.env_origins[:,2]
    upright=-r.data.projected_gravity_b[:,2]
    near=b.smooth_gate(z,1.,1.15)*b.smooth_gate(upright,.8,.93)
    cost=speed_cost(r.data.joint_vel,env._r_speeds,near)
    env._c_accum[:,0]+=cost
    env._c_accum[:,1]+=cost*(near==0)
    env._c_tick+=1
    return cost

def cache(env):
    init(env)
    if env._c_cache_step==env.common_step_counter:return env._c_quality
    env._c_cache_step=env.common_step_counter;f.cache(env)
    r=env.scene['robot'];z=r.data.site_pos_w[:,env._r_head,2]-env.scene.env_origins[:,2]
    upright=-r.data.projected_gravity_b[:,2]
    feet=r.data.body_link_pos_w[:,env._r_feet,:];width=(feet[:,0,:2]-feet[:,1,:2]).norm(dim=-1)
    load=env.scene['quality_feet'].data.force[...,2].abs().amin(-1)
    other=env.scene['quality_other'].data.force[...,2].abs().sum(-1)
    support=b.smooth_gate(upright,.8,.93)*b.smooth_gate(load,5.,20.)
    support*=b.smooth_gate(width,.08,.12)*(1-b.smooth_gate(width,.45,.55))/(1+other/80)
    tilt=(r.data.projected_gravity_b[:,:2]/.15).square().sum(-1)
    pose=1/(1+env._f_pose_error+tilt)
    base=r.data.root_link_lin_vel_w.norm(dim=-1);angular=r.data.root_link_ang_vel_w.norm(dim=-1)
    jrms=r.data.joint_vel.square().mean(-1).sqrt();fspeed=r.data.body_link_lin_vel_w[:,env._r_feet,:].norm(dim=-1).amax(-1)
    quiet=1/(1+(base/.25).square()+(angular/.5).square()+(jrms/2.).square()+(fspeed/.2).square())
    # Half the score rewards valid bilateral support, half graded posture/quietness.
    # Avoid a near-zero standing score under training exploration/noise.
    env._c_quality=support*(.5+.25*pose+.25*quiet)
    env._c_support=support;env._c_gate=b.smooth_gate(z,.85,1.15)
    env._c_new_up=blend_upward(z,r.data.site_lin_vel_w[:,env._r_head,2],env._c_quality)
    env._c_old_up=upward_velocity(env,head_height_threshold=.9)
    env._c_cost=env._c_accum[:,0]/env.cfg.decimation
    return env._c_quality

def quality_upward(env,**kwargs):
    cache(env)
    return env._c_old_up+ramp(env)*(env._c_new_up-env._c_old_up)

def early_speed_reward(env):
    cache(env)
    return env._c_cost*ramp(env)
