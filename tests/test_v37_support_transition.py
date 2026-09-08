from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from mjlab.tasks.registry import load_runner_cls

from smp.rl.actions import RateLimitedJointPositionActionCfg
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v37_support_transition_env_cfg import (
  g1_getup_v37_93d_actuator_env_cfg,
  g1_getup_v37_93d_support_env_cfg,
)
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from launch_smp_v37_support_transition import (  # noqa: E402
  _PROTOCOL_SHA256,
  _validate_protocol,
)


class V37SupportTransitionTest(unittest.TestCase):
  def test_both_arms_keep_the_same_93d_one_frame_actor(self) -> None:
    a = g1_getup_v37_93d_actuator_env_cfg(play=False)
    b = g1_getup_v37_93d_support_env_cfg(play=False)
    expected = (
      "base_ang_vel",
      "projected_gravity",
      "joint_pos",
      "joint_vel",
      "actions",
    )
    self.assertEqual(tuple(a.observations["actor"].terms), expected)
    self.assertEqual(tuple(b.observations["actor"].terms), expected)
    self.assertNotIn("base_lin_vel", a.observations["actor"].terms)
    self.assertIsNone(a.observations["actor"].history_length)
    self.assertEqual(
      tuple(a.observations["actor"].terms),
      tuple(b.observations["actor"].terms),
    )

  def test_hard_action_envelope_and_effort_derating_are_shared(self) -> None:
    a = g1_getup_v37_93d_actuator_env_cfg(play=False)
    b = g1_getup_v37_93d_support_env_cfg(play=False)
    for cfg in (a, b):
      action = cfg.actions["joint_pos"]
      self.assertIsInstance(action, RateLimitedJointPositionActionCfg)
      self.assertEqual(action.max_target_velocity, 2.5)
      self.assertEqual(action.max_target_acceleration, 15.0)
      self.assertEqual(
        cfg.rewards["joint_speed_excess"].params["speed_limits"],
        (4.5, 4.0, 3.5, 3.0),
      )
      self.assertEqual(
        cfg.rewards["joint_power_excess"].params["power_limits"],
        (100.0, 90.0, 80.0, 70.0),
      )
    a_limits = [
      actuator.effort_limit
      for actuator in a.scene.entities["robot"].articulation.actuators
    ]
    b_limits = [
      actuator.effort_limit
      for actuator in b.scene.entities["robot"].articulation.actuators
    ]
    self.assertEqual(a_limits, b_limits)
    self.assertTrue(all(limit is None or limit > 0.0 for limit in a_limits))

  def test_shared_environment_has_no_plate_wrench_or_replay(self) -> None:
    for cfg in (
      g1_getup_v37_93d_actuator_env_cfg(play=False),
      g1_getup_v37_93d_support_env_cfg(play=False),
    ):
      self.assertEqual(
        cfg.events["reset_escape_obstacle"].params["obstacle_probability"], 0.0
      )
      for name in (
        "failure_state_replay_reset",
        "record_failure_states",
        "stratified_post_stand_wrench",
        "post_stand_body_wrench",
      ):
        self.assertNotIn(name, cfg.events)
      self.assertEqual(cfg.scene.terrain.terrain_type, "plane")

  def test_support_treatment_is_additive_and_ordered(self) -> None:
    a = g1_getup_v37_93d_actuator_env_cfg(play=False)
    b = g1_getup_v37_93d_support_env_cfg(play=False)
    self.assertNotIn("photo_informed_seated_trap_reset", a.events)
    trap = b.events["photo_informed_seated_trap_reset"]
    self.assertEqual(trap.params["probability"], 0.15)
    order = tuple(b.events)
    self.assertEqual(
      order.index("photo_informed_seated_trap_reset"),
      order.index("curriculum_validated_fall_reset") + 1,
    )
    stage = b.events["update_recovery_stage"]
    self.assertIs(stage.func, mdp.update_recovery_stage_with_bilateral_support)
    self.assertEqual(stage.params["standing_hold_steps"], 100)
    self.assertEqual(stage.params["crouched_min_load_share"], 0.18)
    self.assertEqual(stage.params["standing_max_stance_width"], 0.55)
    self.assertIn("bilateral_support_route", b.rewards)
    self.assertIn("transition_leg_asymmetry", b.rewards)
    self.assertIn("transition_stance_width", b.rewards)
    self.assertNotIn("bilateral_support_route", a.rewards)

  def test_play_trap_requires_explicit_opt_in(self) -> None:
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("SMP_PLAY_V37_TRAP_RESET", None)
      cfg = g1_getup_v37_93d_support_env_cfg(play=True)
      self.assertEqual(
        cfg.events["photo_informed_seated_trap_reset"].params["probability"],
        0.0,
      )
    with patch.dict(os.environ, {"SMP_PLAY_V37_TRAP_RESET": "1"}):
      cfg = g1_getup_v37_93d_support_env_cfg(play=True)
      self.assertEqual(
        cfg.events["photo_informed_seated_trap_reset"].params["probability"],
        1.0,
      )

  def test_registered_tasks_use_fresh_optimizer_warm_start(self) -> None:
    for task in (
      "Smp-Getup-V37-93D-Actuator-G1",
      "Smp-Getup-V37-93D-Support-G1",
    ):
      self.assertIs(load_runner_cls(task), SmpCurriculumWarmStartRunner)

  def test_protocol_is_hash_locked_and_non_evidence(self) -> None:
    path = Path(__file__).parents[1] / "docs/v37_93d_support_transition_study_v1.json"
    protocol, digest = _validate_protocol(path)
    self.assertEqual(digest, _PROTOCOL_SHA256)
    self.assertEqual(protocol["training_protocol"]["max_iterations"], 6000)
    self.assertEqual(len(protocol["training_protocol"]["gpu_assignment"]), 6)
    self.assertEqual(protocol["shared_environment_contract"]["terrain"], "flat")
    self.assertTrue(protocol["diagnostic_origin"]["photo_is_not_state_replay"])
    self.assertTrue(protocol["claim_boundary"]["not_ral_evidence"])
    self.assertTrue(protocol["claim_boundary"]["no_hardware_authorization"])


if __name__ == "__main__":
  unittest.main()
