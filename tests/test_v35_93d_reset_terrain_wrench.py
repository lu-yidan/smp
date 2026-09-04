from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from mjlab.tasks.registry import load_runner_cls

from smp.rl.tasks.getup.escape_v35_93d_reset_stability_env_cfg import (
  g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg,
  g1_getup_escape_plate_v35_93d_reset_stability_terrain_smp_env_cfg,
  g1_getup_escape_plate_v35_93d_reset_stability_terrain_wrench_smp_env_cfg,
  g1_getup_escape_plate_v35_93d_reset_stability_wrench_smp_env_cfg,
  g1_getup_v36_93d_safe_mixed_smp_env_cfg,
  g1_getup_v36_93d_safe_mixed_wrench_smp_env_cfg,
  g1_getup_v36_93d_safe_stairs_smp_env_cfg,
  g1_getup_v36_93d_safe_stairs_wrench_smp_env_cfg,
)
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from launch_smp_v35_93d_factorial import (  # noqa: E402
  _PROTOCOL_SHA256,
  _validate_protocol,
)
from launch_smp_v36_93d_safe_terrain_study import (  # noqa: E402
  _PROTOCOL_SHA256 as _V36_PROTOCOL_SHA256,
)
from launch_smp_v36_93d_safe_terrain_study import (
  _validate_protocol as _validate_v36_protocol,
)


class V35ResetTerrainWrenchTest(unittest.TestCase):
  def test_four_arms_share_the_same_one_frame_actor_contract(self) -> None:
    builders = (
      g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg,
      g1_getup_escape_plate_v35_93d_reset_stability_wrench_smp_env_cfg,
      g1_getup_escape_plate_v35_93d_reset_stability_terrain_smp_env_cfg,
      g1_getup_escape_plate_v35_93d_reset_stability_terrain_wrench_smp_env_cfg,
    )
    configs = [builder(play=False) for builder in builders]
    actor_terms = tuple(configs[0].observations["actor"].terms)
    critic_terms = tuple(configs[0].observations["critic"].terms)
    for cfg in configs[1:]:
      self.assertEqual(tuple(cfg.observations["actor"].terms), actor_terms)
      self.assertEqual(tuple(cfg.observations["critic"].terms), critic_terms)
    self.assertNotIn("base_lin_vel", actor_terms)
    # ``None`` is ObservationManager's one-frame/no-stacking representation.
    self.assertIsNone(configs[0].observations["actor"].history_length)

  def test_reset_and_standing_contract_is_shared(self) -> None:
    builders = (
      g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg,
      g1_getup_escape_plate_v35_93d_reset_stability_wrench_smp_env_cfg,
      g1_getup_escape_plate_v35_93d_reset_stability_terrain_smp_env_cfg,
      g1_getup_escape_plate_v35_93d_reset_stability_terrain_wrench_smp_env_cfg,
    )
    for builder in builders:
      cfg = builder(play=False)
      reset = cfg.events["curriculum_validated_fall_reset"]
      self.assertEqual(reset.params["balanced_probability"], 0.90)
      self.assertEqual(reset.params["target_probability"], 0.90)
      self.assertEqual(reset.params["mode_weights"], (4.0, 4.0, 1.0, 1.0))
      self.assertEqual(reset.params["joint_noise_levels"], (0.10, 0.20, 0.30))
      self.assertEqual(reset.params["orientation_noise"], 0.30)
      stage = cfg.events["update_recovery_stage"].params
      self.assertEqual(stage["standing_hold_steps"], 100)

  def test_wrench_is_the_only_rd_increment(self) -> None:
    reset = g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg(play=False)
    wrench = g1_getup_escape_plate_v35_93d_reset_stability_wrench_smp_env_cfg(
      play=False
    )
    self.assertEqual(
      set(wrench.events) - set(reset.events), {"stratified_post_stand_wrench"}
    )
    self.assertEqual(tuple(reset.rewards), tuple(wrench.rewards))
    term = wrench.events["stratified_post_stand_wrench"]
    self.assertEqual(term.params["duration_steps"], (10, 18))
    self.assertEqual(term.params["recovery_steps"], 150)

  def test_terrain_increment_is_grounded_and_deployable(self) -> None:
    terrain = g1_getup_escape_plate_v35_93d_reset_stability_terrain_smp_env_cfg(
      play=False
    )
    generator = terrain.scene.terrain.terrain_generator
    self.assertIsNotNone(generator)
    self.assertEqual(tuple(generator.sub_terrains), ("flat", "slope", "stairs", "rough"))
    self.assertEqual(terrain.scene.terrain.max_init_terrain_level, 2)
    ground = terrain.events["ground_fall_on_training_terrain"]
    self.assertEqual(ground.params["eligible_reset_types"], (0, 1, 2, 3, 4, 5))
    self.assertTrue(ground.params["align_to_surface_normal"])
    self.assertTrue(ground.params["use_stair_height_profile"])
    self.assertIn("validate_training_terrain_contact", terrain.events)
    reset_order = tuple(terrain.events)
    self.assertLess(
      reset_order.index("failure_state_replay_reset"),
      reset_order.index("ground_fall_on_training_terrain"),
    )
    self.assertLess(
      reset_order.index("validate_training_terrain_contact"),
      reset_order.index("reset_escape_obstacle"),
    )
    self.assertLess(
      reset_order.index("reset_escape_obstacle"),
      reset_order.index("reset_recovery_stage"),
    )
    self.assertTrue(
      terrain.events["update_recovery_stage"].params["relative_to_env_origin"]
    )
    self.assertIn("terrain_foot_slip", terrain.rewards)
    self.assertIn("terrain_patch_exit", terrain.terminations)

  def test_play_terrain_selector_is_not_ignored(self) -> None:
    with patch.dict(
      os.environ,
      {
        "SMP_PLAY_TERRAIN_TYPE": "stairs",
        "SMP_PLAY_TERRAIN_LEVEL": "2",
        "SMP_PLAY_TERRAIN_RESET_POSE": "prone",
      },
    ):
      cfg = g1_getup_escape_plate_v35_93d_reset_stability_terrain_smp_env_cfg(
        play=True
      )
    generator = cfg.scene.terrain.terrain_generator
    self.assertEqual(tuple(generator.sub_terrains), ("stairs",))
    stairs = generator.sub_terrains["stairs"]
    self.assertEqual(stairs.step_height_range, (0.15, 0.15))
    reset = cfg.events["curriculum_validated_fall_reset"]
    self.assertEqual(reset.params["mode_weights"], (1.0, 0.0, 0.0, 0.0))

  def test_safe_mixed_and_stairs_are_no_plate_matched_pairs(self) -> None:
    builders = (
      g1_getup_v36_93d_safe_mixed_smp_env_cfg,
      g1_getup_v36_93d_safe_mixed_wrench_smp_env_cfg,
      g1_getup_v36_93d_safe_stairs_smp_env_cfg,
      g1_getup_v36_93d_safe_stairs_wrench_smp_env_cfg,
    )
    configs = [builder(play=False) for builder in builders]
    for cfg in configs:
      self.assertEqual(
        cfg.events["reset_escape_obstacle"].params["obstacle_probability"], 0.0
      )
      self.assertNotIn("failure_state_replay_reset", cfg.events)
      self.assertNotIn("record_failure_states", cfg.events)
      self.assertIn("validate_training_terrain_contact", cfg.events)
      self.assertNotIn("base_lin_vel", cfg.observations["actor"].terms)
    self.assertEqual(
      tuple(configs[0].scene.terrain.terrain_generator.sub_terrains),
      ("flat", "slope", "stairs", "rough"),
    )
    self.assertEqual(
      tuple(configs[2].scene.terrain.terrain_generator.sub_terrains),
      ("stairs",),
    )
    self.assertEqual(
      configs[0].scene.terrain.terrain_generator.sub_terrains[
        "stairs"
      ].platform_width,
      2.40,
    )
    self.assertEqual(
      configs[2].scene.terrain.terrain_generator.sub_terrains[
        "stairs"
      ].platform_width,
      2.40,
    )
    self.assertNotIn("stratified_post_stand_wrench", configs[0].events)
    self.assertIn("stratified_post_stand_wrench", configs[1].events)
    self.assertNotIn("stratified_post_stand_wrench", configs[2].events)
    self.assertIn("stratified_post_stand_wrench", configs[3].events)

  def test_play_wrench_requires_explicit_opt_in(self) -> None:
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("SMP_PLAY_AUTO_DISTURBANCES", None)
      cfg = g1_getup_escape_plate_v35_93d_reset_stability_wrench_smp_env_cfg(
        play=True
      )
      self.assertNotIn("stratified_post_stand_wrench", cfg.events)
    with patch.dict(os.environ, {"SMP_PLAY_AUTO_DISTURBANCES": "1"}):
      cfg = g1_getup_escape_plate_v35_93d_reset_stability_wrench_smp_env_cfg(
        play=True
      )
      self.assertIn("stratified_post_stand_wrench", cfg.events)

  def test_all_tasks_use_fresh_optimizer_warm_start_runner(self) -> None:
    tasks = (
      "Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-G1",
      "Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-Wrench-G1",
      "Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-Terrain-G1",
      "Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-Terrain-Wrench-G1",
      "Smp-Getup-V36-93D-Safe-Mixed-G1",
      "Smp-Getup-V36-93D-Safe-Mixed-Wrench-G1",
      "Smp-Getup-V36-93D-Safe-Stairs-G1",
      "Smp-Getup-V36-93D-Safe-Stairs-Wrench-G1",
    )
    for task in tasks:
      self.assertIs(load_runner_cls(task), SmpCurriculumWarmStartRunner)

  def test_protocol_is_hash_locked_and_non_evidence(self) -> None:
    path = (
      Path(__file__).parents[1]
      / "docs/ral_v34_93d_reset_stability_finetune_v1.json"
    )
    protocol, digest = _validate_protocol(path)
    self.assertEqual(digest, _PROTOCOL_SHA256)
    self.assertEqual(protocol["training_protocol"]["max_iterations"], 6000)
    self.assertEqual(len(protocol["training_protocol"]["gpu_assignment"]), 8)
    self.assertTrue(protocol["claim_boundary"]["not_ral_evidence"])
    self.assertTrue(protocol["claim_boundary"]["no_hardware_authorization"])

  def test_v36_safe_terrain_protocol_is_hash_locked(self) -> None:
    path = (
      Path(__file__).parents[1]
      / "docs/v36_93d_safe_terrain_stairs_study_v1.json"
    )
    protocol, digest = _validate_v36_protocol(path)
    self.assertEqual(digest, _V36_PROTOCOL_SHA256)
    self.assertEqual(protocol["training_protocol"]["max_iterations"], 6000)
    self.assertEqual(len(protocol["training_protocol"]["gpu_assignment"]), 8)
    self.assertEqual(
      protocol["shared_environment_contract"]["guided_plate_probability"], 0.0
    )
    self.assertFalse(protocol["shared_environment_contract"]["failure_state_replay"])
    self.assertTrue(protocol["claim_boundary"]["not_ral_evidence"])


if __name__ == "__main__":
  unittest.main()
