"""Fixed per-environment phase/direction quotas and isolated low-SMP ablation."""
import numpy as np
import torch
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup.natural_curriculum import STAGES, DIRECTIONS, weights_for
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
from smp.rl.tasks.getup.master_deployment_contract import JOINT_NAMES
from smp.rl.tasks.getup.mdp.terminations import smp_too_low


def quota_counts(num_envs):
  each_low=max(1,int(num_envs*.025))
  late=num_envs//2
  middle=num_envs-late-4*each_low
  if middle<=0:raise ValueError('Too few environments for six nonempty groups')
  return (late,middle,each_low,each_low,each_low,each_low)


def reset_fixed_low(env,env_ids=None,bank_path=''):
  if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
  if not hasattr(env,'_fixed_group'):
    bank=np.load(bank_path,allow_pickle=False)
    env._course_bank=torch.as_tensor(bank['qpos'],device=env.device)
    env._course_labels=torch.tensor([DIRECTIONS.index(x) if x in DIRECTIONS else 4 for x in bank['labels']],device=env.device)
    env._course_stages=torch.tensor([STAGES.index(x) for x in bank['stages']],device=env.device)
    rng=torch.Generator(device=env.device).manual_seed(int(env.cfg.seed)+904173)
    labels=torch.repeat_interleave(torch.arange(6,device=env.device),torch.tensor(quota_counts(env.num_envs),device=env.device))
    env._fixed_group=labels[torch.randperm(env.num_envs,device=env.device,generator=rng)]
    env._fixed_rng=torch.Generator(device=env.device).manual_seed(int(env.cfg.seed)+904193)
    base=weights_for(bank,0);env._fixed_pools=[]
    for group in range(6):
      mask=(bank['stages']==STAGES[group]) if group<2 else ((bank['stages']=='low')&(bank['labels']==DIRECTIONS[group-2]))
      ids=np.flatnonzero(mask);assert len(ids)
      env._fixed_pools.append((torch.as_tensor(ids,device=env.device),torch.as_tensor(base[ids]/base[ids].sum(),device=env.device,dtype=torch.float32)))
    env._course_type=torch.where(env._fixed_group<2,env._fixed_group,2)
    env._course_direction=torch.full_like(env._fixed_group,4)
    env._course_draws=torch.zeros(3,device=env.device,dtype=torch.long)
    env._course_stage=0
  indices=torch.empty(len(env_ids),device=env.device,dtype=torch.long)
  for group,(pool,weight) in enumerate(env._fixed_pools):
    mask=env._fixed_group[env_ids]==group;n=int(mask.sum())
    if n:indices[mask]=pool[torch.multinomial(weight,n,replacement=True,generator=env._fixed_rng)]
  q=env._course_bank[indices];robot=env.scene['robot'];assert tuple(robot.joint_names)==JOINT_NAMES
  root=robot.data.default_root_state[env_ids].clone();root[:,:3]=q[:,:3]+env.scene.env_origins[env_ids];root[:,3:7]=q[:,3:7];root[:,7:]=0
  robot.write_root_state_to_sim(root,env_ids=env_ids)
  robot.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=env_ids)
  env.sim.forward();prime_static_history(env,env_ids)
  env._course_direction[env_ids]=env._course_labels[indices]
  assert torch.equal(env._course_type[env_ids],env._course_stages[indices])
  env._course_draws+=torch.bincount(env._course_stages[indices],minlength=3)


def fixed_low_smp(env,threshold=.02,ws=6.,grace_steps=5,low_enabled=True,low_threshold=None):
  ordinary=smp_too_low(env,threshold=threshold,ws=ws,grace_steps=grace_steps)
  env._fixed_would_low=ordinary.clone()
  if not low_enabled:return ordinary & (env._fixed_group<2)
  if low_threshold is None:return ordinary
  relaxed=smp_too_low(env,threshold=low_threshold,ws=ws,grace_steps=grace_steps)
  return torch.where(env._fixed_group<2,ordinary,relaxed)


def recorded_task_smp_product(env,task_terms,**kwargs):
  # Return the original reward unchanged; only record its components pre-reset.
  result=task_smp_product(env,task_terms,**kwargs)
  task=sum(weight*func(env,**params) for func,weight,params in task_terms)
  env._fixed_task_score=task.detach()
  env._fixed_smp_score=(result/task.clamp_min(1e-12)).detach()
  env._fixed_product=result.detach()
  return result
