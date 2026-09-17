"""Balanced low recovery with passive plates and edge terrain; actor stays 93D."""
from pathlib import Path
import numpy as np
import torch
import mujoco
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup.multiterrain_geometry import quotas,support_height,STRATA
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
from smp.rl.tasks.getup.fixed_low_reset import recorded_task_smp_product
from smp.rl.tasks.getup import v33_reward_transfer as v

def height(env):
    r=env.scene['robot'];pos=r.data.site_pos_w[:,env._r_head]-env.scene.env_origins
    feet=r.data.body_link_pos_w[:,env._r_feet]-env.scene.env_origins[:,None]
    if getattr(env,'_bd_flat_support',False):return pos[:,2]
    # Highest ground directly under either foot is conservative for split-level standing.
    support=support_height(feet[...,:2]).amax(-1)
    return pos[:,2]-support

def _init(env,bank_path):
    if hasattr(env,'_mt_bank'):return
    n=env.num_envs;dev=env.device
    b=np.load(bank_path);env._mt_bank=torch.as_tensor(b['qpos'],device=dev,dtype=torch.float32)
    scene,direction,st=quotas(n)
    env._mt_scene=torch.tensor(scene,device=dev);env._mt_direction=torch.tensor(direction,device=dev);env._mt_stratum=torch.tensor(st,device=dev)
    env._mt_source=torch.zeros(n,dtype=torch.long,device=dev)
    for s in range(8):
        for dr in range(4):
            ids=torch.where((env._mt_stratum==s)&(env._mt_direction==dr))[0]
            env._mt_source[ids[:round(len(ids)*.25)]]=1
    env._mt_pools={(s,d,k):torch.as_tensor(np.flatnonzero((b['stratum']==s)&(b['direction']==d)&(b['source']==k)),device=dev) for s in range(8) for d in range(4) for k in range(2)}
    assert all(len(p) for p in env._mt_pools.values())
    env._mt_rng=torch.Generator(device=dev).manual_seed(env.cfg.seed+91019)
    env._fixed_group=env._mt_direction+2;env._fixed_source=env._mt_source
    env._mt_ever_contact=torch.zeros(n,dtype=torch.bool,device=dev)
    env._mt_clear_hold=torch.zeros(n,dtype=torch.long,device=dev)
    env._mt_escaped=torch.zeros(n,dtype=torch.bool,device=dev)
    env._mt_invalid=torch.zeros(n,dtype=torch.bool,device=dev)
    for name in ('best','progress','clearance','force','initial_overlap','best_distance','separation_progress'):
        setattr(env,'_mt_'+name,torch.zeros(n,device=dev))
    env._mt_tick=-1
    ids,_=env.scene['robot'].find_geoms(COLLISION_PATTERN)
    env._mt_geoms=env.scene['robot'].indexing.geom_ids[torch.tensor(ids,device=dev)].long()
    env._mt_plate_ids=[]
    for entity,name in [('escape_obstacle','escape_plate_geom'),('free_obstacle','plate_geom')]:
        e=env.scene[entity];idx,_=e.find_geoms([name]);env._mt_plate_ids.append(e.indexing.geom_ids[idx[0]].long())

def robot_bounds(env):
    ids=env._mt_geoms;data=env.sim.data
    pos=data.geom_xpos[:,ids];mat=data.geom_xmat[:,ids];size=env.sim.model.geom_size[:,ids];typ=env.sim.model.geom_type[ids]
    ext=torch.einsum('ngij,ngj->ngi',mat.abs(),size)
    ext=torch.where((typ==int(mujoco.mjtGeom.mjGEOM_SPHERE))[None,:,None],size[:,:,0,None],ext)
    ext=torch.where((typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE))[None,:,None],size[:,:,0,None]+size[:,:,1,None]*mat[:,:,:,2].abs(),ext)
    return pos,ext

def plate_geometry(env):
    pos,ext=robot_bounds(env);which=(env._mt_scene==2).long();rows=torch.arange(env.num_envs,device=env.device)
    gid=torch.stack(env._mt_plate_ids)[which];pp=env.sim.data.geom_xpos[rows,gid];rot=env.sim.data.geom_xmat[rows,gid];size=env.sim.model.geom_size[rows,gid]
    # Conservative world projection of rotating plate. False clearance is avoided
    # at tilted corners; contact-free + complete shadow clearance required.
    pe=torch.einsum('nij,nj->ni',rot.abs(),size)
    delta=(pos[:,:,:2]-pp[:,None,:2]).abs()-ext[:,:,:2]-pe[:,None,:2]
    sep=delta.amax(-1)
    # Body parts above the entire board do not remain trapped underneath it.
    above=(pos[:,:,2]-ext[:,:,2])>pp[:,None,2]+pe[:,None,2]+.025
    sep=torch.where(above,torch.full_like(sep,1.),sep)
    return -(-sep).clamp_min(0).sum(-1)+.5*sep.amin(-1).clamp(0,.04),sep.amin(-1)

@requires_model_fields('body_mass','body_inertia','geom_size',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,bank_path='outputs/multiterrain_bank/train.npz'):
    _init(env,bank_path)
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    indices=torch.empty(len(ids),dtype=torch.long,device=env.device)
    for (s,d,k),pool in env._mt_pools.items():
        mask=(env._mt_stratum[ids]==s)&(env._mt_direction[ids]==d)&(env._mt_source[ids]==k);n=int(mask.sum())
        if n:indices[mask]=pool[torch.randint(len(pool),(n,),generator=env._mt_rng,device=env.device)]
    q=env._mt_bank[indices];r=env.scene['robot'];state=r.data.default_root_state[ids].clone()
    state[:,:3]=q[:,:3]+env.scene.env_origins[ids];state[:,3:7]=q[:,3:7];state[:,7:]=0
    r.write_root_state_to_sim(state,env_ids=ids);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=ids)
    # Hide both obstacles before forward and establish final robot geometry.
    park=state[:,:7].clone();park[:,:3]=env.scene.env_origins[ids]+park.new_tensor([20.,20.,.1]);park[:,3:]=park.new_tensor([1.,0,0,0])
    guided=env.scene['escape_obstacle'];free=env.scene['free_obstacle']
    guided.write_mocap_pose_to_sim(park,env_ids=ids);guided.write_joint_state_to_sim(torch.zeros((len(ids),1),device=env.device),torch.zeros((len(ids),1),device=env.device),env_ids=ids)
    fs=free.data.default_root_state[ids].clone();fs[:,:7]=park;fs[:,0]+=2.;fs[:,7:]=0;free.write_root_state_to_sim(fs,env_ids=ids)
    env.sim.forward();pos,ext=robot_bounds(env)
    # Axis-aligned initial board; varied body yaw is already in the screened bank.
    board=park.clone();board[:,:2]=r.data.root_link_pos_w[ids,:2]
    covered=((pos[ids,:,:2]-board[:,None,:2]).abs()<ext[ids,:,:2]+board.new_tensor([.45,.32])).all(-1)
    top=torch.where(covered,pos[ids,:,2]+ext[ids,:,2],-torch.inf).amax(-1)
    board[:,2]=top+.035+.002
    for scene,e in [(1,guided),(2,free)]:
        choose=env._mt_scene[ids]==scene;ei=ids[choose]
        if not len(ei):continue
        if scene==1:e.write_mocap_pose_to_sim(board[choose],env_ids=ei)
        else:
            fs=e.data.default_root_state[ei].clone();fs[:,:7]=board[choose];fs[:,7:]=0;e.write_root_state_to_sim(fs,env_ids=ei)
        bid=e.indexing.body_ids[-1].long();progress=min(env.common_step_counter/100000,1.)
        mass=4+torch.rand(len(ei),generator=env._mt_rng,device=env.device)*(2+6*progress)
        base=env.sim.get_default_field('body_mass')[bid];inertia=env.sim.get_default_field('body_inertia')[bid]
        env.sim.model.body_mass[ei,bid]=mass;env.sim.model.body_inertia[ei,bid]=inertia*mass[:,None]/base
    env.sim.forward();score,_=plate_geometry(env)
    env._mt_best[ids]=score[ids];env._mt_initial_overlap[ids]=(-score[ids]).clamp_min(.01)
    env._mt_best_distance[ids]=0;env._mt_separation_progress[ids]=0;env._mt_force[ids]=0
    env._mt_ever_contact[ids]=False;env._mt_escaped[ids]=False;env._mt_invalid[ids]=False;env._mt_clear_hold[ids]=0;env._mt_progress[ids]=0
    prime_static_history(env,ids)

def update(env,env_ids=None):
    if not hasattr(env,'_mt_bank') or env._mt_tick==env.common_step_counter:return
    env._mt_tick=env.common_step_counter
    active=(env._mt_scene==1)|(env._mt_scene==2);which=(env._mt_scene==2)
    contacts=[];forces=[];depths=[]
    for name in ('guided_contact','free_contact'):
        s=env.scene[name].data;contacts.append((s.found>0).any(-1));forces.append(s.force.norm(dim=-1).amax(-1));depths.append(s.dist.amin(-1))
    contact=torch.where(which,contacts[1],contacts[0]);force=torch.where(which,forces[1],forces[0]);depth=torch.where(which,depths[1],depths[0])
    env._mt_force=force;env._mt_ever_contact|=contact&active
    score,clearance=plate_geometry(env);env._mt_clearance=clearance
    rows=torch.arange(env.num_envs,device=env.device);gid=torch.stack(env._mt_plate_ids)[which.long()]
    distance=(env.scene['robot'].data.root_link_pos_w[:,:2]-env.sim.data.geom_xpos[rows,gid,:2]).norm(dim=-1)
    env._mt_separation_progress=torch.where(active&env._mt_ever_contact,((distance-env._mt_best_distance)/.025).clamp(0,1),0.)
    env._mt_best_distance=torch.maximum(env._mt_best_distance,distance)
    progress=(score-env._mt_best).clamp_min(0);env._mt_best=torch.maximum(env._mt_best,score)
    env._mt_progress=torch.where(active&env._mt_ever_contact,(progress/.025).clamp_max(1.5),0.)
    clear=active&env._mt_ever_contact&~contact&(clearance>=.025)
    env._mt_clear_hold=torch.where(clear,env._mt_clear_hold+1,0)
    env._mt_escaped=env._mt_clear_hold>=15
    env._mt_invalid=active&((depth<-.02)|(force>1500)|((env.episode_length_buf>25)&~env._mt_ever_contact))

def task(env,task_terms,**kwargs):
    result=recorded_task_smp_product(env,task_terms,**kwargs)
    constrained=((env._mt_scene==1)|(env._mt_scene==2))&~env._mt_escaped
    env._plate_product=result*torch.where(constrained,.05,1.)
    return env._plate_product

def escape_reward(env,index):
    active=(env._mt_scene==1)|(env._mt_scene==2)
    if index==0:return env._mt_progress
    if index==1:return active.float()*(1+env._mt_best/env._mt_initial_overlap).clamp(0,1)
    if index==2:return env._mt_escaped.float()
    if index==4:return env._mt_separation_progress
    return active.float()*((env._mt_force-300).clamp_min(0)/300).square()

def invalid(env):return env._mt_invalid


def ground_force(env,name):
    force=env.scene[name].data.force
    if hasattr(env,'_mt_bank'):force=force+env.scene[name+'_terrain'].data.force
    return force
