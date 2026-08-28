"""V3.8.7 clean-flat foundation for deployable policy-scratch recovery."""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.plate_terrain_v386_scratch_env_cfg import (
  g1_getup_plate_terrain_v386_scratch_s1_deploy_smp_env_cfg,
)

SCRATCH_S0_TERRAIN_PROPORTIONS = (1.0, 0.0, 0.0, 0.0)
SCRATCH_S0_PLATE_PROBABILITY_BY_RESET_TYPE = (0.0, 0.0, 0.0, 0.0)


def g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg(
  play: bool = False,
):
  """Train the four-pose clean recovery foundation before adding constraints.

  This retains the final unified task's 384-dimensional actor and 960-dimensional
  critic interfaces. Only the training distribution is simplified: flat terrain,
  no plate, and procedural prone/supine/left/right resets.
  """
  cfg = g1_getup_plate_terrain_v386_scratch_s1_deploy_smp_env_cfg(play=play)
  if play:
    return cfg

  plate_reset = cfg.events["reset_escape_obstacle"].params
  plate_reset["obstacle_probability"] = 0.0
  plate_reset["obstacle_probability_by_reset_type"] = (
    SCRATCH_S0_PLATE_PROBABILITY_BY_RESET_TYPE
  )

  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 1.0,
      "mode_weights": (1.0, 1.0, 1.0, 1.0),
    }
  )

  generator = cfg.scene.terrain.terrain_generator
  if generator is None:
    raise RuntimeError("V3.8.7 scratch S0 requires generated terrain")
  names = ("flat", "slope", "stairs", "rough")
  for name, proportion in zip(names, SCRATCH_S0_TERRAIN_PROPORTIONS, strict=True):
    generator.sub_terrains[name].proportion = proportion
  curriculum = cfg.curriculum["terrain_levels"].params
  curriculum["minimum_level_fractions"] = (1.0, 0.0, 0.0, 0.0)
  curriculum["maximum_terrain_level"] = 0

  # S1 learned to remain on its hands because support/initiation rewards were
  # available without ever completing recovery. Make ordered stage and stable
  # standing progress dominant while retaining small route-shaping signals.
  cfg.rewards["prone_support_route"].weight = 0.03
  cfg.rewards["recovery_initiation"].weight = 0.06
  cfg.rewards["scratch_recovery_stage"] = RewardTermCfg(
    func=mdp.recovery_stage_metric,
    weight=0.60,
  )
  cfg.rewards["scratch_stable_stand"] = RewardTermCfg(
    func=mdp.stable_stand_metric,
    weight=1.20,
    params={
      "head_height": 1.10,
      "min_upright": 0.85,
      "max_linear_speed": 0.50,
      "max_angular_speed": 1.0,
      "relative_to_env_origin": True,
    },
  )
  cfg.episode_length_s = 8.0
  return cfg


__all__ = [
  "SCRATCH_S0_PLATE_PROBABILITY_BY_RESET_TYPE",
  "SCRATCH_S0_TERRAIN_PROPORTIONS",
  "g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg",
]
