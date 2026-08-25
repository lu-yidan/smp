"""Deployable observations for get-up policies."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["zero_base_linear_velocity"]


def zero_base_linear_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return zeros for the unavailable real-robot base linear velocity.

  Keeping the three dimensions preserves strict compatibility with existing
  96-dimensional actor checkpoints while removing this privileged signal.
  """
  return torch.zeros((env.num_envs, 3), device=env.device)
