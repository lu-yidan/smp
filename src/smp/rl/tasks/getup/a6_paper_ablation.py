"""Matched reward interventions on frozen A6 (1b6d7e6), not later S3 code."""
import torch
from smp.rl.tasks.getup import multiterrain as mt

ARMS = ('A6_full', 'A6_no_bundle', 'A6_no_geometry', 'A6_no_alpha',
        'A6_no_hand', 'A6_no_Q', 'A6_no_L')
GEOMETRY_TERMS = ('plate_geometry_progress', 'plate_clearance', 'plate_separation')


def ungated_task(env, task_terms, **kwargs):
  result = mt.recorded_task_smp_product(env, task_terms, **kwargs)
  env._plate_product = result
  return result


def apply_intervention(cfg, arm):
  """Keep the full A6 reset and all common costs; change only named rewards."""
  assert arm in ARMS
  if arm in ('A6_no_bundle', 'A6_no_geometry'):
    for name in GEOMETRY_TERMS:
      cfg.rewards[name].weight = 0.
  if arm in ('A6_no_bundle', 'A6_no_alpha'):
    cfg.rewards['task_smp_product'].func = ungated_task
  if arm == 'A6_no_hand':
    # Frozen pa.reset(A3) has identical reset/RNG and only removes the hand gate
    # in pa.update. Q/L remain from build_config(A6), unlike historical A3.
    cfg.events['gsi_reset'].params['arm'] = 'A3'
  if arm == 'A6_no_Q':
    cfg.rewards['path_quiet_feet'].weight = 0.
  if arm == 'A6_no_L':
    cfg.rewards['path_joint_stall'].weight = 0.


def init_eval_loads(env):
  r = env.scene['robot']
  env._ap_alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  env._ap_peaks = torch.zeros((*r.data.joint_pos.shape, 3), device=env.device)
  env._ap_high_time = torch.zeros_like(r.data.joint_pos)
  env._ap_high_run = torch.zeros_like(r.data.joint_pos)
  env._ap_high_longest = torch.zeros_like(r.data.joint_pos)
  env._ap_stall_time = torch.zeros_like(r.data.joint_pos)


def sample_eval_loads(env):
  """Persistent first-trial loads; never cleared by an automatic reset."""
  if not hasattr(env, '_ap_alive'):
    return torch.zeros(env.num_envs, device=env.device)
  r = env.scene['robot']
  tau, dq = r.data.qfrc_actuator.abs(), r.data.joint_vel.abs()
  p = torch.stack((tau, dq, tau * dq), -1)
  live = env._ap_alive[:, None]
  env._ap_peaks = torch.maximum(env._ap_peaks, torch.where(live[..., None], p, 0.))
  high = (tau / env._r_limits > .7) & live
  dt = env.cfg.sim.mujoco.timestep
  env._ap_high_time += high * dt
  env._ap_high_run = torch.where(high, env._ap_high_run + dt, 0.)
  env._ap_high_longest = torch.maximum(env._ap_high_longest, env._ap_high_run)
  env._ap_stall_time += (high & (dq < .3)) * dt
  return env._ap_stall_time.amax(-1)
