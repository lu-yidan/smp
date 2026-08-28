"""V3.8.6 stage-one curriculum for deployable policy-scratch recovery."""

from __future__ import annotations

from smp.rl.tasks.getup.plate_terrain_v382_env_cfg import (
  g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg,
)

# Preserve one unified task from the first scratch update, but make most samples
# flat and keep every non-flat family at its audited L0 difficulty.
SCRATCH_S1_TERRAIN_PROPORTIONS = (0.75, 0.08, 0.12, 0.05)
SCRATCH_S1_LEVEL_FRACTIONS = (1.0, 0.0, 0.0, 0.0)

# Pose order: prone, supine, left side, right side. A random actor needs a large
# clean cohort before it can exploit the plate progress rewards. The inherited
# V3.3 overlap/mass curriculum still begins with an exposed edge and 4--6 kg.
SCRATCH_S1_PLATE_PROBABILITY_BY_RESET_TYPE = (0.35, 0.35, 0.25, 0.25)


def g1_getup_plate_terrain_v386_scratch_s1_deploy_smp_env_cfg(
  play: bool = False,
):
  """Build scratch S1 with unchanged deployable actor observations.

  The actor remains 4 x 96 proprioceptive observations. Terrain identity,
  height maps, reset labels, contact truth, obstacle state, and simulator base
  linear velocity are absent. Later stages can therefore resume this actor
  without checkpoint surgery or a deployment interface change.
  """
  cfg = g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg(play=play)
  if play:
    return cfg

  cfg.events["reset_escape_obstacle"].params["obstacle_probability_by_reset_type"] = (
    SCRATCH_S1_PLATE_PROBABILITY_BY_RESET_TYPE
  )

  generator = cfg.scene.terrain.terrain_generator
  if generator is None:
    raise RuntimeError("V3.8.6 scratch S1 requires generated terrain")
  names = ("flat", "slope", "stairs", "rough")
  if tuple(generator.sub_terrains) != names:
    raise RuntimeError("V3.8.6 scratch S1 requires the audited terrain order")
  for name, proportion in zip(names, SCRATCH_S1_TERRAIN_PROPORTIONS, strict=True):
    generator.sub_terrains[name].proportion = proportion

  curriculum = cfg.curriculum["terrain_levels"].params
  curriculum["minimum_level_fractions"] = SCRATCH_S1_LEVEL_FRACTIONS
  curriculum["maximum_terrain_level"] = 0
  return cfg


__all__ = [
  "SCRATCH_S1_LEVEL_FRACTIONS",
  "SCRATCH_S1_PLATE_PROBABILITY_BY_RESET_TYPE",
  "SCRATCH_S1_TERRAIN_PROPORTIONS",
  "g1_getup_plate_terrain_v386_scratch_s1_deploy_smp_env_cfg",
]
