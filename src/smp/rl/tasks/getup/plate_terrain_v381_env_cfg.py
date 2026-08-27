"""V3.8.1 ability-balanced unified plate and terrain recovery task."""

from __future__ import annotations

from smp.rl.tasks.getup.plate_terrain_v38_env_cfg import (
  g1_getup_plate_terrain_v38_deploy_smp_env_cfg,
)

# V3.8 sampled the plate only for prone/supine. Enabling all four reset poses
# would double its plate cohort at the old conditional probability, so V3.8.1
# lowers that probability while preserving roughly one third plate episodes.
TRAIN_PLATE_PROBABILITY = 0.55
PLATE_RESET_TYPES = (1, 2, 3, 4)

# Pose order: prone, supine, left side, right side. Prone receives extra replay
# because it is the shared weak mode in plate escape and stair recovery.
ABILITY_BALANCED_POSE_WEIGHTS = (2.0, 1.5, 1.0, 1.0)

# Terrain order is fixed by V3.7: flat, slope, stairs, rough. The distribution
# retains a flat safety anchor while increasing slope exposure and preserving
# substantial stair coverage.
ABILITY_BALANCED_TERRAIN_PROPORTIONS = (0.25, 0.25, 0.35, 0.15)

# Keep 45% of each non-flat family at L1. This is a deterministic replay floor,
# not privileged actor input; the actor still observes proprioception only.
ABILITY_BALANCED_LEVEL_FRACTIONS = (0.55, 0.45, 0.0, 0.0)


def g1_getup_plate_terrain_v381_deploy_smp_env_cfg(play: bool = False):
  """Build V3.8.1 without changing the deployable observation interface.

  Training uses fixed cohort floors to prevent a high aggregate reward from
  silently dropping plate escape, prone stairs, or L1 slope recovery. Play
  keeps the requested pose/terrain controls and allows the plate on side poses.
  """
  cfg = g1_getup_plate_terrain_v38_deploy_smp_env_cfg(play=play)

  plate_reset = cfg.events["reset_escape_obstacle"].params
  plate_reset["eligible_reset_types"] = PLATE_RESET_TYPES
  # A side-lying shoulder produces contact a few simulation steps later than a
  # broad prone/supine torso. Preserve the strict 0.24 s limit for the old
  # skills while allowing side contacts up to 0.40 s.
  cfg.events["update_escape_phase"].params["max_wait_steps_by_reset_type"] = (
    12,
    12,
    20,
    20,
  )
  if not play:
    plate_reset["obstacle_probability"] = TRAIN_PLATE_PROBABILITY
    cfg.events["mixed_fall_reset"].params["mode_weights"] = (
      ABILITY_BALANCED_POSE_WEIGHTS
    )

    generator = cfg.scene.terrain.terrain_generator
    if generator is None:
      raise RuntimeError("V3.8.1 training requires generated terrain")
    names = ("flat", "slope", "stairs", "rough")
    if tuple(generator.sub_terrains) != names:
      raise RuntimeError("V3.8.1 requires the audited V3.7 terrain order")
    for name, proportion in zip(
      names, ABILITY_BALANCED_TERRAIN_PROPORTIONS, strict=True
    ):
      generator.sub_terrains[name].proportion = proportion

    curriculum = cfg.curriculum["terrain_levels"].params
    curriculum["minimum_level_fractions"] = ABILITY_BALANCED_LEVEL_FRACTIONS
    curriculum["maximum_terrain_level"] = 1

  return cfg


__all__ = [
  "ABILITY_BALANCED_LEVEL_FRACTIONS",
  "ABILITY_BALANCED_POSE_WEIGHTS",
  "ABILITY_BALANCED_TERRAIN_PROPORTIONS",
  "PLATE_RESET_TYPES",
  "TRAIN_PLATE_PROBABILITY",
  "g1_getup_plate_terrain_v381_deploy_smp_env_cfg",
]
