"""Tracking and safety rewards for the sparse-keyframe expert."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from smp.rl.tasks.firm.mdp.commands import SparseKeyframeCommand

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def keyframe_joint_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(SparseKeyframeCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.robot_joint_pos - command.joint_pos).mean(dim=-1)
  return torch.exp(-error / std**2)


def keyframe_joint_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(SparseKeyframeCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.robot_joint_vel - command.joint_vel).mean(dim=-1)
  return torch.exp(-error / std**2)


def joint_velocity_limit(
  env: ManagerBasedRlEnv,
  velocity_limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize only velocity magnitude above the configured safe limit."""
  robot = env.scene[asset_cfg.name]
  excess = (
    robot.data.joint_vel[:, asset_cfg.joint_ids].abs() - velocity_limit
  ).clamp_min(0.0)
  return excess.sum(dim=-1)
