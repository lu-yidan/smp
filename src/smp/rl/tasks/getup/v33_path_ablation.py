"""R2 three-scene V33-path ablation. G restores supported low planar progress.
For a rotating free board, use its conservative world XY projection; no
above-board exemption. All arms retain the same escape success criterion.
"""
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from smp.rl.tasks.getup import r2_ablation as ra, multiterrain as mt
ARMS=tuple('C'+str(i) for i in range(8))
G_ARMS=ARMS
Q_ARMS=('C3',)
L_ARMS=('C4',)

def footprint(pos,ext,pp,pe):
    overlap=ext[...,:2]+pe[:,None,:2]-(pos[...,:2]-pp[:,None,:2]).abs()
    covered=(overlap>0).all(-1)
    score=torch.where(covered,overlap.clamp_min(0).amin(-1),0.).sum(-1)
    clearance=torch.where(covered,0.,(-overlap).clamp_min(0).norm(dim=-1)).amin(-1)
    return covered.sum(-1),score,clearance

def geometry(env):
    pos,ext=mt.robot_bounds(env);rows=torch.arange(env.num_envs,device=env.device)
    gid=torch.stack(env._mt_plate_ids)[(env._mt_scene==2).long()]
    pp=env.sim.data.geom_xpos[rows,gid];rot=env.sim.data.geom_xmat[rows,gid];size=env.sim.model.geom_size[rows,gid]
    pe=torch.einsum('nij,nj->ni',rot.abs(),size)
    return footprint(pos,ext,pp,pe)

def quiet_gate(z,u):
    # Continuous posture-only gate. No contact requirement or resettable timer.
    return ra.gate(z,.85,1.15)*ra.gate(u,.70,.93)

def path_progress(coverage_delta,clearance_delta,z,support,active):
    return active*(z<=.90)*support*((coverage_delta/.025).clamp(0,1)+.5*(clearance_delta/.02).clamp(0,1))

def joint_stall_gate(span):
    # Per-joint, smooth no-progress test; unrelated joints cannot suppress it.
    return (1-span/.10).clamp(0,1)

@requires_model_fields('body_mass','body_inertia','geom_size','geom_aabb','geom_rbound',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,arm='C2',**kwargs):
    env._ce_arm=arm;env._ce_evaluation=kwargs.get('evaluation',False)
    ra.reset(env,env_ids,arm='T0',**kwargs)
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    env._pa_arm=arm;env._bd_flat_support=True
    if not hasattr(env,'_pa_best_score'):
        for name in ('best_score','best_clearance','initial_count','progress','clearance_score','gate','load_acc','load','support','escape_time'):
            setattr(env,'_pa_'+name,torch.zeros(env.num_envs,device=env.device))
        env._pa_tick=-1
        env._pa_joint_acc=torch.zeros_like(env.scene['robot'].data.joint_pos)
    count,score,clearance=geometry(env)
    env._pa_best_score[ids]=score[ids];env._pa_best_clearance[ids]=clearance[ids];env._pa_initial_count[ids]=count[ids].float()
    for name in ('progress','clearance_score','gate','load_acc','load','support','escape_time'):getattr(env,'_pa_'+name)[ids]=0
    env._pa_joint_acc[ids]=0

def sample_substep(env):
    c=ra.sample_substep(env)
    if not hasattr(env,'_pa_joint_acc'):return c
    if (env._ra_subtick-1)%env.cfg.decimation==0:env._pa_joint_acc.zero_()
    r=env.scene['robot'];ratio=(r.data.qfrc_actuator/env._r_limits).abs()
    cost=ra.gate(ratio,.7,.95).square()*ra.gate(env._ra_timer,.3,1.)*ra.gate(.3-r.data.joint_vel.abs(),0,.3)
    env._pa_joint_acc+=cost
    return c

def update(env):
    ra.update(env)
    if env._pa_tick==env.common_step_counter:return
    env._pa_tick=env.common_step_counter
    count,score,clearance=geometry(env)
    dc=(env._pa_best_score-score).clamp_min(0);dd=(clearance-env._pa_best_clearance).clamp_min(0)
    env._pa_best_score=torch.minimum(env._pa_best_score,score);env._pa_best_clearance=torch.maximum(env._pa_best_clearance,clearance)
    support=(env.scene['path_hands'].data.found>0).float().mean(-1)
    env._pa_support=support
    active=(env._mt_scene>0)&env._mt_ever_contact&~env._mt_escaped&~env._mt_invalid
    env._pa_progress=path_progress(dc,dd,mt.height(env),torch.ones_like(support) if env._pa_arm in ARMS else support,active)
    env._pa_clearance_score=(env._mt_scene>0)*~env._mt_invalid*(.85*(1-count/env._pa_initial_count.clamp_min(1)).clamp(0,1)+.15*(clearance/.04).clamp(0,1))
    env._pa_escape_time=torch.where(env._mt_escaped,env._pa_escape_time+env.step_dt,0.)
    r=env.scene['robot'];env._pa_gate=quiet_gate(mt.height(env),-r.data.projected_gravity_b[:,2])
    span=(env._ra_motion.amax(0)-env._ra_motion.amin(0))[:,:29]
    env._pa_load=(env._pa_joint_acc/env.cfg.decimation*joint_stall_gate(span)).amax(-1)

def reward(env,index):
    update(env)
    if index==0:return env._pa_progress
    if index==1:return env._pa_clearance_score
    if index==2:
        foot=env.scene['robot'].data.body_link_lin_vel_w[:,env._r_feet,:].square().sum(-1).mean(-1)
        return env._pa_gate*(foot/.1**2).clamp_max(10)
    if index==4:
        yaw=env.scene['robot'].data.root_link_ang_vel_w[:,2]
        return (env._mt_scene>0)*ra.gate(env._pa_escape_time,.2,.5)*env._pa_gate*(yaw/.5).square().clamp_max(10)
    return env._pa_load

def metric(env):update(env);return env._pa_gate
