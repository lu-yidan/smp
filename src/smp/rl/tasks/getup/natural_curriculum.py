"""Static, deployment-FK-consistent natural reset curriculum without GSI."""
import numpy as np
import torch
from smp.rl.tasks.getup.natural_low_reset import prime_static_history
from smp.rl.tasks.getup.master_deployment_contract import JOINT_NAMES
STAGES=('late','middle','low')
DIRECTIONS=('supine','prone','left_side_down','right_side_down')
MIXES=((.5,.4,.1),(.4,.4,.2),(.2,.4,.4))


def weights_for(bank, stage):
  weights=np.zeros(len(bank['qpos']),dtype=np.float64)
  groups=np.array([x.replace('__mirror','') for x in bank['clips']])
  for name,mass in zip(STAGES,MIXES[stage]):
    ids=np.flatnonzero(bank['stages']==name);assert len(ids)
    labels=DIRECTIONS if name=='low' else (None,)
    for label in labels:
      rows=ids if label is None else ids[bank['labels'][ids]==label];assert len(rows),(name,label)
      clips=sorted(set(groups[rows]))
      for clip in clips:
        subset=rows[groups[rows]==clip]
        weights[subset]=mass/len(labels)/len(clips)/len(subset)
  assert np.isclose(weights.sum(),1.)
  return weights


def reset_natural_curriculum(env,env_ids=None,bank_path=''):
  if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
  if not hasattr(env,'_course_bank'):
    bank=np.load(bank_path,allow_pickle=False)
    env._course_bank=torch.as_tensor(bank['qpos'],device=env.device)
    env._course_weights=[torch.as_tensor(weights_for(bank,i),device=env.device,dtype=torch.float32) for i in range(len(MIXES))]
    env._course_labels=torch.tensor([DIRECTIONS.index(x) if x in DIRECTIONS else 4 for x in bank['labels']],device=env.device)
    env._course_stages=torch.tensor([STAGES.index(x) for x in bank['stages']],device=env.device)
    env._course_type=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    env._course_direction=torch.zeros_like(env._course_type)
    env._course_draws=torch.zeros(3,device=env.device,dtype=torch.long)
    env._course_stage=0
    env._course_rng=torch.Generator(device=env.device).manual_seed(int(env.cfg.seed)+903127)
  idx=torch.multinomial(env._course_weights[env._course_stage],len(env_ids),replacement=True,generator=env._course_rng)
  q=env._course_bank[idx];robot=env.scene['robot'];assert tuple(robot.joint_names)==JOINT_NAMES
  root=robot.data.default_root_state[env_ids].clone();root[:,:3]=q[:,:3]+env.scene.env_origins[env_ids];root[:,3:7]=q[:,3:7];root[:,7:]=0
  robot.write_root_state_to_sim(root,env_ids=env_ids)
  robot.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=env_ids)
  env.sim.forward();prime_static_history(env,env_ids)
  env._course_type[env_ids]=env._course_stages[idx];env._course_direction[env_ids]=env._course_labels[idx]
  env._course_draws+=torch.bincount(env._course_stages[idx],minlength=3)
