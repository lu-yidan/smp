"""V3.6.3 conservative terrain adaptation with strict standing progress."""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.terrain_v36_env_cfg import (
  g1_getup_terrain_v36_smp_env_cfg,
)

TERRAIN_REPLAY_FRACTIONS = (0.70, 0.30, 0.0, 0.0)
SMP_FLOOR = 0.03


def g1_getup_terrain_v363_smp_env_cfg(play: bool = False):
  """Build V3.6.3 without the V3.6.1 stage-latch curriculum shortcut."""
  cfg = g1_getup_terrain_v36_smp_env_cfg(play=play)
  cfg.rewards["task_smp_product"].params["smp_floor"] = SMP_FLOOR
  if not play:
    curriculum = cfg.curriculum["terrain_levels"].params
    curriculum["stand_hold_steps"] = 25
    curriculum["accept_completed_recovery_stage"] = False
    curriculum["minimum_level_fractions"] = TERRAIN_REPLAY_FRACTIONS
  return cfg


def g1_getup_terrain_v363_deploy_smp_env_cfg(play: bool = False):
  """Build V3.6.3 with a zero-conditioned actor and asymmetric critic."""
  cfg = g1_getup_terrain_v363_smp_env_cfg(play=play)
  cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
    func=mdp.zero_base_linear_velocity
  )
  return cfg


__all__ = [
  "SMP_FLOOR",
  "TERRAIN_REPLAY_FRACTIONS",
  "g1_getup_terrain_v363_deploy_smp_env_cfg",
  "g1_getup_terrain_v363_smp_env_cfg",
]
