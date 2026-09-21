"""A6 matched ablations: latch egress credit, bound drift, relax ascent."""
import torch

ARMS=tuple(f'EC{i}' for i in range(8))

def latch_step(seen,safe,anchor,xy,age,dt):
    first=safe&~seen
    return seen|safe,torch.where(first[:,None],xy,anchor),torch.where(safe,age+dt,0.),first

def progress_mask(arm,seen):
    return torch.ones_like(seen) if arm=='EC0' else ~seen

def speed_multiplier(arm,seen,safe_age):
    factor={'EC3':1.5,'EC4':2.,'EC7':1.5}.get(arm,1.)
    return 1+(factor-1)*seen*(safe_age/.5).clamp(0,1)

def task_multiplier(arm,active,safe,safe_age):
    released=(safe_age/.5).clamp(0,1) if arm=='EC5' else safe.float()
    return torch.where(active,.05+.95*released,1.)

def anchor_cost(offset,age,enabled):
    return enabled*(age/.5).clamp(0,1)*((offset-.30).clamp_min(0)/.30).square().clamp_max(4)

def initialize(env):
    if hasattr(env,'_ec_seen'):return
    n=env.num_envs;dev=env.device
    env._ec_seen=torch.zeros(n,dtype=torch.bool,device=dev)
    env._ec_anchor=torch.zeros(n,2,device=dev)
    env._ec_previous_xy=env._ec_anchor.clone()
    for name in ('safe_age','offset','max_offset','path','yaw','time_after','first_escape_time','anchor_cost'):
        setattr(env,'_ec_'+name,torch.zeros(n,device=dev))
    env._ec_tick=-1

def reset(env,ids):
    initialize(env)
    env._ec_seen[ids]=False
    xy=env.scene['robot'].data.root_link_pos_w[ids,:2]
    env._ec_anchor[ids]=xy;env._ec_previous_xy[ids]=xy
    for name in ('safe_age','offset','max_offset','path','yaw','time_after','first_escape_time','anchor_cost'):
        getattr(env,'_ec_'+name)[ids]=0

def update(env):
    initialize(env)
    if env._ec_tick==env.common_step_counter:return
    env._ec_tick=env.common_step_counter
    r=env.scene['robot'];xy=r.data.root_link_pos_w[:,:2]
    safe=(env._mt_scene>0)&env._mt_escaped&~env._mt_invalid
    was_seen=env._ec_seen.clone()
    seen,anchor,age,first=latch_step(env._ec_seen,safe,env._ec_anchor,xy,env._ec_safe_age,env.step_dt)
    env._ec_seen.copy_(seen);env._ec_anchor.copy_(anchor);env._ec_safe_age.copy_(age)
    env._ec_first_escape_time[first]=env.episode_length_buf[first]*env.step_dt
    env._ec_time_after+=seen*env.step_dt
    env._ec_offset.copy_((xy-anchor).norm(dim=-1)*seen)
    env._ec_max_offset.copy_(torch.maximum(env._ec_max_offset,env._ec_offset))
    env._ec_path+=(xy-env._ec_previous_xy).norm(dim=-1)*was_seen
    env._ec_previous_xy.copy_(xy)
    # World-z angular velocity integral, distinct from total 3D angular motion.
    env._ec_yaw+=r.data.root_link_ang_vel_w[:,2].abs()*was_seen*env.step_dt

def scale(env):
    if not hasattr(env,'_ec_seen'):return 1.
    return speed_multiplier(env._ce_arm,env._ec_seen,env._ec_safe_age)

def reward(env):
    # Frozen world-space anchor. Suppress attraction if the board moved over it.
    rows=torch.arange(env.num_envs,device=env.device)
    gid=torch.stack(env._mt_plate_ids)[(env._mt_scene==2).long()]
    pp=env.sim.data.geom_xpos[rows,gid];rot=env.sim.data.geom_xmat[rows,gid]
    pe=torch.einsum('nij,nj->ni',rot.abs(),env.sim.model.geom_size[rows,gid])
    anchor_clear=((env._ec_anchor-pp[:,:2]).abs()-pe[:,:2]).amax(-1)>.10
    enabled=env._ec_seen&env._mt_escaped&~env._mt_invalid&anchor_clear
    env._ec_anchor_cost.copy_(anchor_cost(env._ec_offset,env._ec_safe_age,enabled))
    return env._ec_anchor_cost
