"""Safety-only terminations for FIRM expert training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def unsafe_velocity(
  env: ManagerBasedRlEnv,
  joint_speed_limit: float,
  root_linear_speed_limit: float,
  root_angular_speed_limit: float,
  entity_name: str = "robot",
) -> torch.Tensor:
  robot = env.scene[entity_name]
  unsafe_joint = (robot.data.joint_vel.abs() > joint_speed_limit).any(dim=-1)
  unsafe_root_linear = (
    robot.data.root_link_lin_vel_w.norm(dim=-1) > root_linear_speed_limit
  )
  unsafe_root_angular = (
    robot.data.root_link_ang_vel_w.norm(dim=-1) > root_angular_speed_limit
  )
  return unsafe_joint | unsafe_root_linear | unsafe_root_angular
