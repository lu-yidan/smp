"""V3.8.2 paired actor-history ablation with pose-balanced plate replay."""

from __future__ import annotations

from smp.rl.tasks.getup.plate_terrain_v381_env_cfg import (
  g1_getup_plate_terrain_v381_deploy_smp_env_cfg,
)

ACTOR_HISTORY_LENGTH_H4 = 4
ACTOR_HISTORY_LENGTH_H10 = 10

# Tuple positions correspond to robust reset types 1..4:
# prone, supine, left side, right side. The old prone/supine plate cohort is
# preserved instead of being diluted when side-lying plate episodes are added.
PLATE_PROBABILITY_BY_RESET_TYPE = (0.90, 0.90, 0.65, 0.65)


def _g1_getup_plate_terrain_v382_deploy_smp_env_cfg(history_length: int, play: bool):
  cfg = g1_getup_plate_terrain_v381_deploy_smp_env_cfg(play=play)
  if not play:
    plate_reset = cfg.events["reset_escape_obstacle"].params
    # The scalar is only a fallback for reset types not represented in the
    # tuple. Eligibility still restricts plate episodes to reset types 1..4.
    plate_reset["obstacle_probability"] = 1.0
    plate_reset["obstacle_probability_by_reset_type"] = PLATE_PROBABILITY_BY_RESET_TYPE
  cfg.observations["actor"].history_length = history_length
  cfg.observations["actor"].flatten_history_dim = True
  return cfg


def g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg(play: bool = False):
  """Build the four-frame control arm of the V3.8.2 ablation."""
  return _g1_getup_plate_terrain_v382_deploy_smp_env_cfg(ACTOR_HISTORY_LENGTH_H4, play)


def g1_getup_plate_terrain_v382_h10_deploy_smp_env_cfg(play: bool = False):
  """Build the ten-frame experimental arm of the V3.8.2 ablation."""
  return _g1_getup_plate_terrain_v382_deploy_smp_env_cfg(ACTOR_HISTORY_LENGTH_H10, play)


__all__ = [
  "ACTOR_HISTORY_LENGTH_H4",
  "ACTOR_HISTORY_LENGTH_H10",
  "PLATE_PROBABILITY_BY_RESET_TYPE",
  "g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v382_h10_deploy_smp_env_cfg",
]
