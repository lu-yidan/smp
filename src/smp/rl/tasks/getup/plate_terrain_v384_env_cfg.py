"""V3.8.4 symmetry-regularized unified plate and terrain recovery."""

from __future__ import annotations

from smp.rl.tasks.getup.plate_terrain_v382_env_cfg import (
  g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg,
)


def g1_getup_plate_terrain_v384_symmetric_deploy_smp_env_cfg(
  play: bool = False,
):
  """Reuse the audited V3.8.2 H4 environment with symmetric PPO learning.

  The symmetry transform is configured in the runner, so play and deployment
  use an ordinary single actor with the unchanged 384-dimensional input.
  """
  return g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg(play=play)


__all__ = ["g1_getup_plate_terrain_v384_symmetric_deploy_smp_env_cfg"]
