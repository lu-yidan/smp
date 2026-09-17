"""FT12k prior/reset/replay ablations. Replay stores complete paired physical states."""
from __future__ import annotations
import numpy as np
import torch
import mujoco
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from smp.rl.tasks.getup import plate_transfer as p, v33_reward_transfer as v
from smp.rl.tasks.getup.fixed_low_reset import reset_fixed_low, fixed_low_smp
from smp.rl.tasks.getup.natural_low_reset import prime_static_history

ARMS=('P0_control','P1_v7','P2_v7_ws4','P3_no_smp_term','P4_replay','P5_all_low','P6_combined','P7_transition')
LOW_ARMS=('P5_all_low','P6_combined','P7_transition')
REPLAY_ARMS=('P4_replay','P6_combined','P7_transition')
BUFFER_FIELDS=('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','ee_pos_w','joint_pos','joint_vel')


def init(env):
    if hasattr(env,'_pr_replayed'): return
    n=env.num_envs;d=env.device
    env._pr_replayed=torch.zeros(n,dtype=torch.bool,device=d)
    env._pr_best=torch.full((n,),-1.,device=d)
    env._pr_stagnant=torch.zeros(n,dtype=torch.long,device=d)
    env._pr_reset_total=0;env._pr_replay_total=0;env._pr_rejected=0
    env._pr_counts=[0]*5;env._pr_cursor=[0]*5;env._pr_store={}
    env._pr_rng=torch.Generator(device=d).manual_seed(env.cfg.seed+907331)
    env._pr_cpu=mujoco.MjData(env.sim.mj_model)


def direction(gravity):
    # Body +x is prone, +y is left-side-down for this deployment model.
    x,y=gravity[:,0],gravity[:,1]
    return torch.where(x.abs()>=y.abs(),torch.where(x>0,1,0),torch.where(y>0,2,3))


def configure_low(env,ids,kwargs):
    half=env.num_envs//2
    if not getattr(env,'_pr_low_configured',False):
        assert half%4==0
        each=half//4
        env._fixed_group[:half]=torch.arange(4,device=env.device).repeat_interleave(each)+2
        env._fixed_group[half:]=3
        env._fixed_source.zero_()
        for group in range(2,6):
            members=torch.where((env._fixed_group==group)&(~env._plate_cohort))[0]
            env._fixed_source[members[:len(members)//4]]=1
        env._course_type.fill_(2)
        env._fixed_quota_counts=tuple(torch.bincount(env._fixed_group,minlength=6).tolist())
        # Small eval batches may have rounded the original procedural quota to zero.
        # Materialize the newly requested procedural pools before changing source quotas.
        natural_count=len(np.load(kwargs['bank_path'],allow_pickle=False)['qpos'])
        for group in range(2,6):
            if not any(g==group and source==1 for g,source,_,_ in env._fixed_pools):
                rows=torch.where((env._course_stages==2)&(env._course_labels==group-2)&(torch.arange(len(env._course_bank),device=env.device)>=natural_count))[0]
                assert len(rows)>0
                env._fixed_pools.append((group,1,rows,torch.full((len(rows),),1./len(rows),device=env.device)))
        env._pr_low_configured=True
    flat=ids[~env._plate_cohort[ids]]
    if len(flat):reset_fixed_low(env,flat,**kwargs)


@requires_model_fields('body_mass','body_inertia',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,arm='P0_control',**kwargs):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    p.reset(env,ids,arm='E3_guided',**kwargs)
    init(env)
    if arm in LOW_ARMS:configure_low(env,ids,kwargs)
    env._pr_replayed[ids]=False
    env._pr_best[ids]=-1.;env._pr_stagnant[ids]=0
    env._pr_reset_total+=len(ids)
    if arm in REPLAY_ARMS and not getattr(env,'_pr_disable_replay',False):restore(env,ids)
    env.sim.forward()
    prime_static_history(env,ids[~env._pr_replayed[ids]])


def termination(env,**kwargs):
    # A formerly middle/late assigned env may replay an actual low state.
    return fixed_low_smp(env,**kwargs)&~env._pr_replayed


def snapshot(env,ids):
    r=env.scene['robot'];o=env.scene['escape_obstacle'];origin=env.scene.env_origins[ids]
    root=torch.cat((r.data.root_link_pos_w[ids]-origin,r.data.root_link_quat_w[ids],r.data.root_link_lin_vel_w[ids],r.data.root_link_ang_vel_w[ids]),-1)
    bid=o.indexing.body_ids[-1].long();mid=o.indexing.mocap_id
    mpos=env.sim.data.mocap_pos[ids,mid].reshape(len(ids),3)-origin
    mquat=env.sim.data.mocap_quat[ids,mid].reshape(len(ids),4)
    data={'root':root,'joint_pos':r.data.joint_pos[ids],'joint_vel':r.data.joint_vel[ids],
          'mocap':torch.cat((mpos,mquat),-1),'plate_q':o.data.joint_pos[ids],'plate_dq':o.data.joint_vel[ids],
          'mass':env.sim.model.body_mass[ids,bid],'inertia':env.sim.model.body_inertia[ids,bid]}
    data['raw_qpos']=env.sim.data.qpos[ids].clone();data['raw_qpos'][:,:3]-=origin
    data['raw_qvel']=env.sim.data.qvel[ids].clone()
    for name,value in vars(env).items():
        if name.startswith('_escape_') and isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==env.num_envs:
            data[name]=value[ids].clone()
    for name in ('_escape_start_robot_xy','_escape_start_obstacle_xy'):
        data[name]-=origin[:,:2]
    for name in BUFFER_FIELDS:data['history_'+name]=getattr(env._smp_buffer,name)[ids]
    return {k:t.detach().clone() for k,t in data.items()}


def valid_cpu(env,ids):
    # Independent forward detects ground/self/plate intersections; no reward-score filter.
    q=env.sim.data.qpos[ids].cpu().numpy();dq=env.sim.data.qvel[ids].cpu().numpy()
    pos=env.sim.data.mocap_pos[ids].cpu().numpy();quat=env.sim.data.mocap_quat[ids].cpu().numpy()
    good=[];cpu=env._pr_cpu
    for j in range(len(ids)):
        if not (np.isfinite(q[j]).all() and np.isfinite(dq[j]).all()):continue
        cpu.qpos[:]=q[j];cpu.qvel[:]=dq[j];cpu.mocap_pos[:]=pos[j];cpu.mocap_quat[:]=quat[j]
        mujoco.mj_forward(env.sim.mj_model,cpu)
        if all(c.dist>=-.005 for c in cpu.contact):good.append(j)
    return ids[torch.tensor(good,device=env.device,dtype=torch.long)]


def insert(env,ids):
    ids=valid_cpu(env,ids)
    if not len(ids):return 0
    data=snapshot(env,ids)
    buckets=torch.where(env._plate_active[ids],4,direction(env.scene['robot'].data.projected_gravity_b[ids]))
    capacity=2048
    for key,value in data.items():
        if key not in env._pr_store:env._pr_store[key]=torch.zeros((5,capacity,*value.shape[1:]),dtype=value.dtype,device=env.device)
    for bucket in range(5):
        mask=torch.where(buckets==bucket)[0];n=len(mask)
        if not n:continue
        slots=(torch.arange(n,device=env.device)+env._pr_cursor[bucket])%capacity
        for key,value in data.items():env._pr_store[key][bucket,slots]=value[mask]
        env._pr_cursor[bucket]=(env._pr_cursor[bucket]+n)%capacity
        env._pr_counts[bucket]=min(capacity,env._pr_counts[bucket]+n)
    return len(ids)


def record(env):
    init(env)
    if getattr(env,'_pr_disable_replay',False):return torch.zeros(env.num_envs,device=env.device)
    r=env.scene['robot'];z=r.data.site_pos_w[:,env._r_head,2]-env.scene.env_origins[:,2]
    upright=(-r.data.projected_gravity_b[:,2]).clamp(0,1)
    progress=z+.25*upright+.1*(env._escape_phase==3)
    improved=progress>env._pr_best+.02
    env._pr_best=torch.maximum(env._pr_best,progress)
    env._pr_stagnant=torch.where(improved,0,env._pr_stagnant+1)
    if env.common_step_counter%25==0:
        limits=r.data.joint_pos_limits
        finite=torch.isfinite(env.sim.data.qpos).all(-1)&torch.isfinite(env.sim.data.qvel).all(-1)
        eligible=(env._pr_stagnant>=75)&(env.episode_length_buf>=75)&(z<.65)&(z>.08)&finite
        eligible&=(env._escape_phase!=4)&(~env.reset_buf.bool())&(r.data.joint_vel.abs().amax(-1)<20)
        eligible&=((r.data.joint_pos>=limits[...,0]-.01)&(r.data.joint_pos<=limits[...,1]+.01)).all(-1)
        ids=torch.where(eligible)[0]
        if len(ids):
            ids=ids[torch.randperm(len(ids),device=env.device,generator=env._pr_rng)[:32]]
            accepted=insert(env,ids);env._pr_rejected+=len(ids)-accepted
            env._pr_stagnant[ids]=0
    return env._pr_replayed.float()


def restore(env,ids,probability=.20,minimum=128):
    selected=ids[torch.rand(len(ids),device=env.device,generator=env._pr_rng)<probability]
    if not len(selected) or not env._pr_store:return
    groups=env._fixed_group[selected]
    # Existing mid/late reset slots may revisit any failed low direction.
    choices=torch.randint(0,4,(len(selected),),device=env.device,generator=env._pr_rng)
    buckets=torch.where(env._plate_active[selected],4,torch.where(groups>=2,groups-2,choices))
    r=env.scene['robot'];o=env.scene['escape_obstacle'];bid=o.indexing.body_ids[-1].long()
    for bucket in range(5):
        target=selected[buckets==bucket]
        if not len(target) or env._pr_counts[bucket]<minimum:continue
        slots=torch.randint(env._pr_counts[bucket],(len(target),),device=env.device,generator=env._pr_rng)
        data={k:t[bucket,slots].clone() for k,t in env._pr_store.items()};origin=env.scene.env_origins[target]
        root=data['root'];root[:,:3]+=origin;r.write_root_state_to_sim(root,env_ids=target)
        r.write_joint_state_to_sim(data['joint_pos'],data['joint_vel'],env_ids=target)
        m=data['mocap'];m[:,:3]+=origin;o.write_mocap_pose_to_sim(m,env_ids=target)
        o.write_joint_state_to_sim(data['plate_q'],data['plate_dq'],env_ids=target)
        # Derived body poses can lag one physics substep during metrics: restore exact qpos/qvel.
        raw=data['raw_qpos'];raw[:,:3]+=origin
        env.sim.data.qpos[target]=raw;env.sim.data.qvel[target]=data['raw_qvel']
        env.sim.model.body_mass[target,bid]=data['mass'];env.sim.model.body_inertia[target,bid]=data['inertia']
        for k,t in data.items():
            if k.startswith('_escape_'):
                if k in ('_escape_start_robot_xy','_escape_start_obstacle_xy'):t+=origin[:,:2]
                getattr(env,k)[target]=t
            elif k.startswith('history_'):getattr(env._smp_buffer,k[8:])[target]=t
        env._escape_sensor_grace[target]=1
        for k in ('_escape_coverage_delta','_escape_clearance_delta','_escape_separation_delta'):getattr(env,k)[target]=0
        env._pr_replayed[target]=True;env._pr_replay_total+=len(target)
