"""Measured reward modules for the R0--R7 from-scratch recovery ablation."""
import torch
from smp.rl.tasks.getup.master_deployment_contract import CONTROL, JOINT_NAMES
from smp.rl.tasks.getup.recovery_quality import speed_reference
from smp.rl.tasks.getup.mdp.rewards import upward_velocity

# Fixed for all arms. Learn the initial recovery objective before full quality costs.
def ramp(env):
    return min(1., max(0., (env.common_step_counter / 24 - 500) / 2000))

def no_stand_termination(env, **kwargs):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

def smooth_gate(value, low, high):
    t=((value-low)/(high-low)).clamp(0,1)
    return t*t*(3-2*t)

def velocity_band(v):
    # Broad positive progress band; no unlimited reward for throwing the head up.
    return torch.exp(-100*(.10-v).clamp_min(0).square()-25*(v-.30).clamp_min(0).square())

def slow_upward(env, **kwargs):
    original=upward_velocity(env, **kwargs)
    robot=env.scene['robot'];head=robot.find_sites(['head'],preserve_order=True)[0][0]
    z=robot.data.site_pos_w[:,head,2]-env.scene.env_origins[:,2]
    band=torch.where(z<kwargs.get('head_height_threshold',.9),velocity_band(robot.data.site_lin_vel_w[:,head,2]),torch.ones_like(z))
    return original+ramp(env)*(band-original)

def load_costs(tau,dq,limits,speeds):
    effort=((tau.abs()/limits-.7).clamp_min(0)/.3).square().clamp_max(25)
    speed=(dq.abs()/(.5*speeds)-1).clamp_min(0).square().clamp_max(25)
    # Most-loaded three joints, not all-joint mean that dilutes a single spike.
    return effort.topk(3,dim=-1).values.mean(-1),speed.topk(3,dim=-1).values.mean(-1)

def init_buffers(env):
    if hasattr(env,'_r_accum'): return
    robot=env.scene['robot'];device=env.device;n=env.num_envs
    env._r_limits=torch.tensor([CONTROL['tau_limit'][JOINT_NAMES.index(x)] for x in robot.joint_names],device=device)
    env._r_speeds=torch.tensor(speed_reference(robot.joint_names),device=device)
    env._r_head=robot.find_sites(['head'],preserve_order=True)[0][0]
    env._r_feet=robot.find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)[0]
    env._r_knees=robot.find_joints(['left_knee_joint','right_knee_joint'],preserve_order=True)[0]
    env._r_head_geom=[i for i,name in enumerate(env.scene['quality_other'].primary_names) if 'head' in name]
    assert len(env._r_head_geom)==1
    env._r_accum=torch.zeros(n,3,device=device);env._r_peaks=torch.zeros(n,4,device=device)
    env._r_costs=torch.zeros(n,4,device=device);env._r_prev_target=torch.zeros_like(robot.data.joint_pos)
    env._r_tick=0;env._r_cache_step=-1
    env._r_quiet=torch.zeros(n,device=device);env._r_stable=torch.zeros(n,dtype=torch.bool,device=device)
    env._r_hold=torch.zeros(n,device=device)

def sample_substep(env):
    init_buffers(env)
    if env._r_tick%env.cfg.decimation==0:env._r_accum.zero_();env._r_peaks.zero_()
    robot=env.scene['robot'];tau=robot.data.qfrc_actuator;dq=robot.data.joint_vel
    effort,speed=load_costs(tau,dq,env._r_limits,env._r_speeds)
    force=env.scene['quality_other'].data.force[:,env._r_head_geom,:].norm(dim=-1).amax(-1)
    vz=robot.data.site_lin_vel_w[:,env._r_head,2].abs()
    # Low static force <=150N is not penalized. High dynamic contact costs more.
    contact=((force-150).clamp_min(0)/850).square().clamp_max(25)*(.25+.75*(vz/.3).clamp(0,1))
    contact*= (env.episode_length_buf.float()*env.step_dt/.1).clamp(0,1)
    env._r_accum+=torch.stack([effort,speed,contact],-1)
    peaks=torch.stack([tau.abs().amax(-1),dq.abs().amax(-1),(tau*dq).abs().amax(-1),force],-1)
    torch.maximum(env._r_peaks,peaks,out=env._r_peaks);env._r_tick+=1
    return speed

def cache_control(env):
    init_buffers(env)
    if env._r_cache_step==env.common_step_counter:return env._r_quiet
    env._r_cache_step=env.common_step_counter
    robot=env.scene['robot'];target=env.action_manager.get_term('joint_pos')._processed_actions
    slew=((target-env._r_prev_target)/.1).square().clamp_max(25).mean(-1)
    fresh=env.episode_length_buf<=1;slew=torch.where(fresh,0.,slew)
    env._r_prev_target.copy_(target)
    env._r_costs[:,:2]=env._r_accum[:,:2]/env.cfg.decimation
    env._r_costs[:,2]=slew;env._r_costs[:,3]=env._r_accum[:,2]/env.cfg.decimation
    z=robot.data.site_pos_w[:,env._r_head,2]-env.scene.env_origins[:,2]
    upright=(-robot.data.projected_gravity_b[:,2]).clamp(0,1)
    feet=robot.data.body_link_pos_w[:,env._r_feet,:]
    width=(feet[:,0,:2]-feet[:,1,:2]).norm(dim=-1)
    foot_speed=robot.data.body_link_lin_vel_w[:,env._r_feet,:].norm(dim=-1).amax(-1)
    load=env.scene['quality_feet'].data.force[...,2].abs().amin(-1)
    other=env.scene['quality_other'].data.force[...,2].abs().sum(-1)
    base=robot.data.root_link_lin_vel_w.norm(dim=-1);angular=robot.data.root_link_ang_vel_w.norm(dim=-1)
    joint_rms=robot.data.joint_vel.square().mean(-1).sqrt()
    knee=robot.data.joint_pos[:,env._r_knees].abs().amax(-1)
    gate=smooth_gate(z,1.,1.2)*smooth_gate(upright,.8,.95)*smooth_gate(load,5.,20.)
    gate*=smooth_gate(width,.08,.12)*(1-smooth_gate(width,.45,.55))
    env._r_quiet=gate*torch.exp(-(base/.15).square()-(angular/.3).square()-(joint_rms/.5).square()-(foot_speed/.1).square()-4*(knee-.65).clamp_min(0).square()-other/40)
    env._r_stable=(z>=1.15)&(upright>=.93)&(knee<.8)&(base<.15)&(angular<.3)&(joint_rms<.5)&(foot_speed<.1)&(load>20)&(other<20)&(width>=.12)&(width<=.45)
    env._r_hold=torch.where(fresh,0.,env._r_hold)
    env._r_hold=torch.where(env._r_stable,env._r_hold+env.step_dt,0.)
    return env._r_quiet

def quiet_reward(env):
    return cache_control(env)*ramp(env)

def safety_cost(env,index):
    cache_control(env)
    return env._r_costs[:,index]*ramp(env)
