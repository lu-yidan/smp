"""V3.8.3 short right-side replay for deployable plate/terrain recovery."""

from __future__ import annotations

from smp.rl.tasks.getup.plate_terrain_v382_env_cfg import (
  g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg,
)

# Pose order: prone, supine, left side, right side. Preserve the two original
# plate skills, keep left-side replay, and double the right-side reset share.
RIGHT_REPLAY_POSE_WEIGHTS = (2.0, 1.5, 1.0, 2.0)

# Conditional plate probability for reset types 1..4. The right-side cohort is
# deliberately saturated for this short diagnostic fine-tune. Plate placement
# remains restricted to flat and stairs-center by the inherited task.
RIGHT_REPLAY_PLATE_PROBABILITIES = (0.95, 0.95, 0.65, 1.0)


def g1_getup_plate_terrain_v383_right_deploy_smp_env_cfg(
  play: bool = False,
):
  """Build the 4-frame V3.8.3 right-side recovery diagnostic.

  The actor interface is unchanged: 4 x 96 deployment-safe proprioceptive
  observations. No reset label, plate state, terrain label, contact truth, or
  true base linear velocity is exposed to the actor.
  """
  cfg = g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg(play=play)
  if not play:
    cfg.events["mixed_fall_reset"].params["mode_weights"] = RIGHT_REPLAY_POSE_WEIGHTS
    plate_reset = cfg.events["reset_escape_obstacle"].params
    plate_reset["obstacle_probability"] = 1.0
    plate_reset["obstacle_probability_by_reset_type"] = RIGHT_REPLAY_PLATE_PROBABILITIES
  return cfg


__all__ = [
  "RIGHT_REPLAY_PLATE_PROBABILITIES",
  "RIGHT_REPLAY_POSE_WEIGHTS",
  "g1_getup_plate_terrain_v383_right_deploy_smp_env_cfg",
]
