"""Episode-level mixtures for the from-scratch termination ablation.

Only episode scheduling changes; GSI, rewards, observations and dynamics stay
identical. The private generator avoids consuming the policy/GSI RNG stream.
"""

import math

import torch

from smp.rl.tasks.getup.mdp.terminations import smp_too_low, stood_up


def sample_episode_mode(env, env_ids=None, long_probability: float = 0.2):
  if not hasattr(env, "_ablation_long"):
    env._ablation_long = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._ablation_rng = torch.Generator(device=env.device)
    env._ablation_rng.manual_seed(int(env.cfg.seed) + 710003)
    env._ablation_draws = torch.zeros(2, dtype=torch.long, device=env.device)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  chosen = torch.rand(n, device=env.device, generator=env._ablation_rng) < long_probability
  env._ablation_long[env_ids] = chosen
  env._ablation_draws[1] += chosen.sum()
  env._ablation_draws[0] += n - chosen.sum()


def mixed_time_out(env, short_seconds: float = 5.0, long_seconds: float = 20.0):
  limits = torch.where(
    env._ablation_long,
    math.ceil(long_seconds / env.step_dt),
    math.ceil(short_seconds / env.step_dt),
  )
  return env.episode_length_buf >= limits


def mixed_stood_up(env, **kwargs):
  return stood_up(env, **kwargs) & ~env._ablation_long


def mixed_smp_too_low(env, **kwargs):
  return smp_too_low(env, **kwargs) & ~env._ablation_long
