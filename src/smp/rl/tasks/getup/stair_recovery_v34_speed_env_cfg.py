"""Finish4 stair task with the deployed V34 balanced4 speed objectives.

The reward formulas and limits match v34_escape_recovery.py. Head height uses
the local stair support surface instead of the flat/plate environment origin.
All other Finish4 settings, including task weights and PPO, are preserved.
"""

import torch

from smp.rl.tasks.getup.mdp.rewards import _head_height, _recovery_stage
from smp.rl.tasks.getup.stair_recovery_finish_env_cfg import (
  stair_recovery_finish_env_cfg,
)


def v34_head_velocity(env):
  robot = env.scene["robot"]
  head = robot.find_sites(["head"])[0][0]
  height = _head_height(env, relative_to_env_origin=True)
  velocity = robot.data.site_lin_vel_w[:, head, 2]
  stage = _recovery_stage(env)
  target = height.new_tensor((0.62, 0.86, 1.15, 1.15))[stage]
  desired = height.new_tensor((0.20, 0.30, 0.40, 0.0))[stage]
  desired = desired * ((target - height) / 0.20).clamp(0, 1)
  limit = height.new_tensor((0.45, 0.50, 0.55, 0.20))[stage]
  return torch.exp(
    -20 * (velocity - desired).square()
    - 40 * (velocity.abs() - limit).clamp_min(0).square()
  )


def v34_head_overspeed(env):
  robot = env.scene["robot"]
  head = robot.find_sites(["head"])[0][0]
  stage = _recovery_stage(env)
  velocity = robot.data.site_lin_vel_w[:, head, 2]
  limit = velocity.new_tensor((0.45, 0.50, 0.55, 0.20))[stage]
  return (velocity.abs() - limit).clamp_min(0).square()


def stair_recovery_v34_speed_env_cfg(play=False):
  cfg = stair_recovery_finish_env_cfg(play)
  task = cfg.rewards["task_smp_product"]
  task.params["task_terms"] = tuple(
    (v34_head_velocity, weight, {})
    if fn.__name__ == "staged_head_velocity_profile"
    else (fn, weight, params)
    for fn, weight, params in task.params["task_terms"]
  )
  cfg.rewards["head_vertical_overspeed"].func = v34_head_overspeed
  cfg.rewards["head_vertical_overspeed"].params = {}
  cfg.rewards["joint_speed_excess"].params["speed_limits"] = (6.0, 5.0, 4.0, 3.5)
  cfg.rewards["joint_power_excess"].params["power_limits"] = (140.0, 110.0, 90.0, 75.0)
  return cfg
