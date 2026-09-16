"""FT12k paired reset/plate/shaping experiment; actor remains 93D."""
from __future__ import annotations
import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from smp.rl.tasks.getup import plate_geometry as g, v33_reward_transfer as v
from smp.rl.tasks.getup.fixed_low_reset import reset_fixed_low, quota_counts, recorded_task_smp_product
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN

ARMS=('E0_flat','E1_prone','E2_plate','E3_guided')

def plate_spec():
    spec=mujoco.MjSpec();b=spec.worldbody.add_body(name='escape_plate')
    b.add_joint(name='escape_plate_slide',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=(0,0,1),limited=True,range=(-1.2,0),damping=60.)
    geom=b.add_geom(name='escape_plate_geom',type=mujoco.mjtGeom.mjGEOM_BOX,size=(.45,.32,.035),mass=8.,friction=(1.2,.01,.001),rgba=(.12,.72,.24,.8))
    geom.priority=1;geom.solref=(.01,1.);geom.solimp=(.98,.995,.001,.5,2.)
    return spec

@requires_model_fields('body_mass','body_inertia',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,arm='E0_flat',**bank_params):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    reset_fixed_low(env,env_ids,**bank_params)
    if not hasattr(env,'_plate_cohort'):
        n=env.num_envs;half=n//2
        env._plate_cohort=torch.arange(n,device=env.device)>=half if arm!='E0_flat' else torch.zeros(n,dtype=torch.bool,device=env.device)
        env._plate_active=env._plate_cohort & (arm in ('E2_plate','E3_guided'))
        if arm!='E0_flat':
            counts=quota_counts(half,.2,.4)
            labels=torch.repeat_interleave(torch.arange(6,device=env.device),torch.tensor(counts,device=env.device))
            rng=torch.Generator(device=env.device).manual_seed(env.cfg.seed+4521)
            env._fixed_group[:half]=labels[torch.randperm(half,device=env.device,generator=rng)]
            env._fixed_group[half:]=3;env._fixed_source.zero_()
            for group in range(2,6):
                ids=torch.where((env._fixed_group==group)&(~env._plate_cohort))[0]
                env._fixed_source[ids[:int(half*.05/4)]]=1
            env._fixed_quota_counts=tuple(torch.bincount(env._fixed_group,minlength=6).tolist())
            env._course_type=torch.where(env._fixed_group<2,env._fixed_group,2)
            reset_fixed_low(env,env_ids,**bank_params)
    ids=env_ids[env._plate_cohort[env_ids]];r=env.scene['robot'];n=len(ids)
    # Pre-screened current-model bank; no legacy arm/root convention assumed.
    if not hasattr(env,'_plate_bank'):
        env._plate_bank=torch.as_tensor(np.load('datasets/reset_banks/plate_prone_v1/train.npz')['qpos'],device=env.device)
        env._plate_rng=torch.Generator(device=env.device).manual_seed(env.cfg.seed+1267)
    if n:
        rows=torch.randint(len(env._plate_bank),(n,),device=env.device,generator=env._plate_rng)
        q=env._plate_bank[rows];state=r.data.default_root_state[ids].clone()
        state[:,:3]=q[:,:3]+env.scene.env_origins[ids];state[:,3:7]=q[:,3:7];state[:,7:]=0
        r.write_root_state_to_sim(state,env_ids=ids);r.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=ids);env.sim.forward()
        env._course_direction[ids]=1
    g.reset_guided_escape_plate_curriculum(env,env_ids,plate_mass_range=(4.,12.),initial_max_mass=6.,mass_curriculum_steps=100000,
        active_mask=env._plate_active,prepare_mask=env._plate_cohort,crawl_ready_prone=False,crawl_arm_noise=0.,
        ground_clearance=.004,surface_gap=.001,align_to_body=True,longitudinal_offset=-.10,
        longitudinal_offset_curriculum=(.18,.04),lateral_offset_curriculum=(.22,.05),overlap_curriculum_steps=100000,
        xy_offset_range=.005,collision_geom_pattern=COLLISION_PATTERN,inactive_xy=(20.,20.))
    env.sim.forward()
    # Re-prime history after all physical pose changes, including root height.
    from smp.rl.tasks.getup.natural_low_reset import prime_static_history
    prime_static_history(env,env_ids)
    if not hasattr(env,'_plate_done_hold'):
        env._plate_done_hold=torch.zeros(env.num_envs,device=env.device)
        env._plate_last_tick=-1
    env._plate_done_hold[env_ids]=0

def update(env,env_ids=None):
    g.update_escape_phase(env,env_ids,geometry_clearance=True,collision_geom_pattern=COLLISION_PATTERN,
        min_planar_clearance=.025,clear_hold_steps=15,max_penetration=.02,max_contact_force=1500.,
        max_wait_steps=12,max_initial_contact_head_height=.4,hand_sensor_name='hand_ground_contact',
        min_hand_support_steps=5,min_hand_supported_progress=.04)

def invalid(env):
    return (env._escape_phase==4)&env._plate_active

def task(env,task_terms,guided=False,**kwargs):
    product=recorded_task_smp_product(env,task_terms,**kwargs)
    if guided:
        constrained=(env._escape_phase==1)|(env._escape_phase==2)
        product=product*torch.where(constrained,.05,1.)
    env._plate_product=product.detach()
    return product
