"""Fixed per-environment phase/direction quotas and isolated low-SMP ablation."""
import numpy as np
import torch
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup.natural_curriculum import STAGES, DIRECTIONS, weights_for
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
from smp.rl.tasks.getup.master_deployment_contract import JOINT_NAMES
from smp.rl.tasks.getup.mdp.terminations import smp_too_low


def quota_counts(num_envs,low_fraction=.1,late_fraction=.5):
  if not (0 < low_fraction < 1 and 0 < late_fraction < 1 and low_fraction+late_fraction < 1):
    raise ValueError("Invalid fixed reset fractions")
  each_low=max(1,int(num_envs*low_fraction/4))
  late=int(num_envs*late_fraction)
  middle=num_envs-late-4*each_low
  if middle<=0:raise ValueError('Too few environments for six nonempty groups')
  return (late,middle,each_low,each_low,each_low,each_low)


def reset_fixed_low(env,env_ids=None,bank_path='',low_fraction=.1,late_fraction=.5,procedural_bank_path='',procedural_fraction=0.):
  if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
  if not hasattr(env,'_fixed_group'):
    bank=np.load(bank_path,allow_pickle=False)
    qpos=bank['qpos'];labels_np=bank['labels'];stages_np=bank['stages'];source_np=np.zeros(len(qpos),dtype=np.int64)
    base=weights_for(bank,0)
    if procedural_bank_path:
      proc=np.load(procedural_bank_path,allow_pickle=False)
      assert set(proc['labels'])==set(DIRECTIONS) and np.all(proc['stages']=='low')
      qpos=np.concatenate([qpos,proc['qpos']]);labels_np=np.concatenate([labels_np,proc['labels']]);stages_np=np.concatenate([stages_np,proc['stages']])
      source_np=np.concatenate([source_np,np.ones(len(proc['qpos']),dtype=np.int64)]);base=np.concatenate([base,np.ones(len(proc['qpos']))])
    if procedural_fraction and not procedural_bank_path:raise ValueError('Missing procedural bank')
    if not 0<=procedural_fraction<=low_fraction:raise ValueError('Procedural fraction must fit low quota')
    env._course_bank=torch.as_tensor(qpos,device=env.device)
    env._course_labels=torch.tensor([DIRECTIONS.index(x) if x in DIRECTIONS else 4 for x in labels_np],device=env.device)
    env._course_stages=torch.tensor([STAGES.index(x) for x in stages_np],device=env.device)
    env._fixed_quota_counts=quota_counts(env.num_envs,low_fraction,late_fraction)
    rng=torch.Generator(device=env.device).manual_seed(int(env.cfg.seed)+904173)
    labels=torch.repeat_interleave(torch.arange(6,device=env.device),torch.tensor(env._fixed_quota_counts,device=env.device))
    env._fixed_group=labels[torch.randperm(env.num_envs,device=env.device,generator=rng)]
    env._fixed_rng=torch.Generator(device=env.device).manual_seed(int(env.cfg.seed)+904193)
    env._fixed_source=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    each_proc=int(env.num_envs*procedural_fraction/4)
    for group in range(2,6):
      ids=torch.where(env._fixed_group==group)[0]
      assert each_proc<=len(ids)
      env._fixed_source[ids[:each_proc]]=1
    env._fixed_pools=[]
    for group in range(6):
      for source in (0,1):
        if not ((env._fixed_group==group)&(env._fixed_source==source)).any():continue
        mask=(stages_np==STAGES[group]) if group<2 else ((stages_np=='low')&(labels_np==DIRECTIONS[group-2]))
        ids=np.flatnonzero(mask&(source_np==source));assert len(ids)
        env._fixed_pools.append((group,source,torch.as_tensor(ids,device=env.device),torch.as_tensor(base[ids]/base[ids].sum(),device=env.device,dtype=torch.float32)))
    env._course_type=torch.where(env._fixed_group<2,env._fixed_group,2)
    env._course_direction=torch.full_like(env._fixed_group,4)
    env._course_draws=torch.zeros(3,device=env.device,dtype=torch.long)
    env._course_stage=0
  indices=torch.empty(len(env_ids),device=env.device,dtype=torch.long)
  for group,source,pool,weight in env._fixed_pools:
    mask=(env._fixed_group[env_ids]==group)&(env._fixed_source[env_ids]==source);n=int(mask.sum())
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
