"""Deployment-conditioned V3.3 escape policy with no base velocity input."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg

from smp.rl.tasks.getup.escape_v33_env_cfg import (
  g1_getup_escape_plate_v33_smp_env_cfg,
)


def zero_base_linear_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return the real-robot observation contract for unavailable base velocity."""
  return torch.zeros((env.num_envs, 3), device=env.device)


def g1_getup_escape_plate_v33_deploy_smp_env_cfg(play: bool = False):
  """Build V3.3-Deploy with zero-conditioned actor and asymmetric critic.

  The actor remains 96-dimensional for strict checkpoint compatibility, but
  dimensions 0:3 are deterministic zeros in both training and play.  The critic
  retains the original simulator base velocity during training.
  """
  cfg = g1_getup_escape_plate_v33_smp_env_cfg(play=play)
  cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
    func=zero_base_linear_velocity
  )
  return cfg
