"""V3.6.2 anti-collapse terrain specialization and deployable ablation."""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.terrain_v361_env_cfg import (
  g1_getup_terrain_v361_smp_env_cfg,
)

TERRAIN_REPLAY_FRACTIONS = (0.50, 0.30, 0.15, 0.05)
SMP_FLOOR = 0.15


def g1_getup_terrain_v362_smp_env_cfg(play: bool = False):
  """Build V3.6.2 with an SMP floor and non-collapsing replay cohorts."""
  cfg = g1_getup_terrain_v361_smp_env_cfg(play=play)
  cfg.rewards["task_smp_product"].params["smp_floor"] = SMP_FLOOR
  if not play:
    cfg.curriculum["terrain_levels"].params["minimum_level_fractions"] = (
      TERRAIN_REPLAY_FRACTIONS
    )
  return cfg


def g1_getup_terrain_v362_deploy_smp_env_cfg(play: bool = False):
  """Build V3.6.2 with zero-conditioned actor and asymmetric critic.

  The actor keeps its 96-dimensional shape, but dimensions 0:3 are always
  zero. The critic retains simulator base linear velocity during training.
  """
  cfg = g1_getup_terrain_v362_smp_env_cfg(play=play)
  cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
    func=mdp.zero_base_linear_velocity
  )
  return cfg


__all__ = [
  "SMP_FLOOR",
  "TERRAIN_REPLAY_FRACTIONS",
  "g1_getup_terrain_v362_deploy_smp_env_cfg",
  "g1_getup_terrain_v362_smp_env_cfg",
]
