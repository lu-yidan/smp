"""Paper observation terms for a sparse-keyframe augmentation expert."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from smp.rl.tasks.firm.mdp.commands import SparseKeyframeCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def joint_pos(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  return env.scene[entity_name].data.joint_pos


def joint_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  return env.scene[entity_name].data.joint_vel


def keyframe_joint_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(SparseKeyframeCommand, env.command_manager.get_term(command_name))
  return command.robot_joint_pos - command.joint_pos


def motion_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(SparseKeyframeCommand, env.command_manager.get_term(command_name))
  denominator = max(command.motion.time_step_total - 1, 1)
  return (command.time_steps.float() / denominator).unsqueeze(-1)
