from __future__ import annotations

import math
import unittest

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.ral_progression_env_cfg import (
  ACTOR_TERMS,
  PLATE_POSE_WEIGHTS,
  PLATE_PROBABILITY_BY_RESET,
  SCRATCH_ARM_BUILDERS,
  STAIR_EDGE_WEIGHTS,
  TERRAIN_LEVEL_WEIGHTS,
  TERRAIN_PROPORTIONS,
  g1_scratch_ral_plate_env_cfg,
  g1_scratch_ral_terrain_env_cfg,
)


class RalProgressionCfgTest(unittest.TestCase):
  def test_progression_actor_is_one_frame_93d_contract(self) -> None:
    for arm in SCRATCH_ARM_BUILDERS:
      for builder in (
        g1_scratch_ral_terrain_env_cfg,
        g1_scratch_ral_plate_env_cfg,
      ):
        with self.subTest(arm=arm, builder=builder.__name__):
          cfg = builder(arm, play=False)
          actor = cfg.observations["actor"]
          self.assertEqual(tuple(actor.terms), ACTOR_TERMS)
          self.assertNotIn("base_lin_vel", actor.terms)
          self.assertIsNone(actor.history_length)

  def test_terrain_distribution_and_reset_order_are_frozen(self) -> None:
    cfg = g1_scratch_ral_terrain_env_cfg("a6", play=False)
    generator = cfg.scene.terrain.terrain_generator
    self.assertIsNotNone(generator)
    assert generator is not None
    self.assertEqual(
      tuple(generator.sub_terrains), ("flat", "slope", "stairs", "rough")
    )
    for actual, expected in zip(
      (x.proportion for x in generator.sub_terrains.values()),
      TERRAIN_PROPORTIONS,
      strict=True,
    ):
      self.assertAlmostEqual(actual, expected)
    self.assertEqual(
      cfg.events["sample_weighted_terrain_levels"].params["level_weights"],
      TERRAIN_LEVEL_WEIGHTS,
    )
    self.assertEqual(
      cfg.events["sample_terrain_edge_reset"].params["cohort_weights"],
      STAIR_EDGE_WEIGHTS,
    )
    names = list(cfg.events)
    self.assertLess(
      names.index("sample_weighted_terrain_levels"), names.index("gsi_reset")
    )
    self.assertLess(
      names.index("mixed_fall_reset"), names.index("sample_terrain_edge_reset")
    )
    self.assertLess(
      names.index("sample_terrain_edge_reset"),
      names.index("ground_procedural_fall_on_terrain"),
    )
    self.assertNotIn("push_robot", cfg.events)

  def test_plate_joint_distribution_matches_half_split(self) -> None:
    pose = [weight / sum(PLATE_POSE_WEIGHTS) for weight in PLATE_POSE_WEIGHTS]
    pinned = [pose[index] * PLATE_PROBABILITY_BY_RESET[index] for index in range(4)]
    unpinned = [pose[index] - pinned[index] for index in range(4)]
    self.assertAlmostEqual(sum(pinned), 0.5)
    for actual, expected in zip(pinned, (0.25, 0.25, 0.0, 0.0), strict=True):
      self.assertAlmostEqual(actual, expected)
    for actual in unpinned:
      self.assertAlmostEqual(actual, 0.125)

    cfg = g1_scratch_ral_plate_env_cfg("a6", play=False)
    reset = cfg.events["reset_escape_obstacle"]
    self.assertIs(reset.func, mdp.reset_stratified_guided_escape_plate)
    self.assertEqual(reset.params["plate_masses"], (4.0, 8.0, 12.0))
    self.assertEqual(reset.params["mass_weights"], (0.25, 0.50, 0.25))
    self.assertEqual(reset.params["friction_range"], (0.4, 1.2))
    self.assertEqual(
      reset.params["obstacle_probability_by_reset_type"],
      PLATE_PROBABILITY_BY_RESET,
    )
    self.assertEqual(reset.params["longitudinal_offset_curriculum"], (0.12, 0.12))
    self.assertEqual(reset.params["lateral_offset_curriculum"], (0.12, 0.12))
    self.assertNotIn("push_robot", cfg.events)

  def test_plate_preserves_selected_arm_bridge(self) -> None:
    for arm, floor in (("a0", 0.0), ("a6", 0.10)):
      with self.subTest(arm=arm):
        cfg = g1_scratch_ral_plate_env_cfg(arm, play=False)
        reward = cfg.rewards["task_smp_product"]
        self.assertIs(reward.func, mdp.escape_gated_procedural_bridge_task_smp_product)
        self.assertTrue(math.isclose(reward.params["procedural_smp_floor"], floor))


if __name__ == "__main__":
  unittest.main()
