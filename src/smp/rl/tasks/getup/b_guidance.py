"""Finite-credit local pose-path guidance. No clock-driven trajectory tracking."""
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields,RecomputeLevel
from smp.rl.tasks.getup import balanced_dynamics as bd

def initialize(env):
    if hasattr(env,'_bg_ref'):return
    bank=np.load('datasets/guidance/supine_to_crouch.npz')
    env._bg_ref=torch.tensor(bank['features'],device=env.device)
    env._bg_joints=torch.tensor(bank['joint_indices'],device=env.device)
    env._bg_scale=torch.tensor([.7,.7,.5,.7,.7,.5,.4,.4,.4,1.,1.,1.],device=env.device)
    n=env.num_envs
    env._bg_phase=torch.zeros(n,device=env.device,dtype=torch.long)
    env._bg_best=torch.zeros(n,device=env.device)
    env._bg_credit=torch.zeros(n,device=env.device)
    env._bg_eligible=torch.zeros(n,device=env.device,dtype=torch.bool)
    env._bg_reward=torch.zeros(n,device=env.device);env._bg_error=torch.zeros(n,device=env.device)
    env._bg_tick=-1

def features(env):
    r=env.scene['robot']
    return torch.cat((r.data.joint_pos[:,env._bg_joints],r.data.projected_gravity_b),-1)

def distance(env,x,ref):return ((x-ref)/env._bg_scale).square().mean(-1)

@requires_model_fields('body_mass','body_inertia','geom_size',recompute=RecomputeLevel.set_const)
def reset(env,env_ids=None,guidance=False,**kwargs):
    bd.reset(env,env_ids,**kwargs);initialize(env)
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    x=features(env)[ids];err=distance(env,x[:,None],env._bg_ref[None]);best,phase=err.min(-1)
    eligible=(env._mt_scene[ids]==0)&(env._mt_direction[ids]==0)&(best<.75)&(phase<len(env._bg_ref)*.8)&guidance
    env._bg_eligible[ids]=eligible;env._bg_phase[ids]=phase
    env._bg_best[ids]=phase.float()/(len(env._bg_ref)-1)+.2*torch.exp(-best/.25)
    env._bg_credit[ids]=0;env._bg_reward[ids]=0;env._bg_error[ids]=best

def reward(env):
    if env._bg_tick==env.common_step_counter:return env._bg_reward
    env._bg_tick=env.common_step_counter
    # Only search forward locally; no time target, no global jump to final pose.
    candidates=(env._bg_phase[:,None]+torch.arange(2,device=env.device)[None]).clamp_max(len(env._bg_ref)-1)
    x=features(env);err=distance(env,x[:,None],env._bg_ref[candidates]);e,j=err.min(-1)
    phase=candidates.gather(1,j[:,None]).squeeze(1)
    env._bg_phase=torch.where((e<.25)&env._bg_eligible,phase,env._bg_phase)
    current_error=distance(env,x,env._bg_ref[env._bg_phase]);env._bg_error=current_error
    potential=env._bg_phase.float()/(len(env._bg_ref)-1)+.2*torch.exp(-current_error/.25)
    delta=torch.where(env._bg_eligible,(potential-env._bg_best).clamp_min(0),0.)
    env._bg_best=torch.maximum(env._bg_best,potential);env._bg_credit+=delta
    # Reward manager multiplies step_dt: division makes total raw episode credit <= 1.2.
    env._bg_reward=delta/env.step_dt
    return env._bg_reward
