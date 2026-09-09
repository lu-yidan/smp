from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from mjlab.tasks.registry import load_runner_cls

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v38_recovery_route_env_cfg import (
  g1_getup_v38_93d_control_env_cfg,
  g1_getup_v38_93d_route_env_cfg,
)
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from launch_smp_v38_recovery_route import (  # noqa: E402
  _PROTOCOL_SHA256,
  _validate_protocol,
)


class V38RecoveryRouteTest(unittest.TestCase):
  def test_matched_93d_actor_and_actuator_envelope(self) -> None:
    a = g1_getup_v38_93d_control_env_cfg(play=False)
    b = g1_getup_v38_93d_route_env_cfg(play=False)
    expected = (
      "base_ang_vel",
      "projected_gravity",
      "joint_pos",
      "joint_vel",
      "actions",
    )
    for cfg in (a, b):
      self.assertEqual(tuple(cfg.observations["actor"].terms), expected)
      self.assertNotIn("base_lin_vel", cfg.observations["actor"].terms)
      self.assertIsNone(cfg.observations["actor"].history_length)
      self.assertEqual(cfg.episode_length_s, 20.0)
      self.assertEqual(cfg.actions["joint_pos"].max_target_velocity, 2.5)
      self.assertEqual(cfg.actions["joint_pos"].max_target_acceleration, 15.0)

  def test_control_is_v37_b_and_treatment_is_additive(self) -> None:
    a = g1_getup_v38_93d_control_env_cfg(play=False)
    b = g1_getup_v38_93d_route_env_cfg(play=False)
    self.assertNotIn("post_roll_supine_failure_reset", a.events)
    self.assertNotIn("v38_route_progress", a.rewards)
    self.assertIn("post_roll_supine_failure_reset", b.events)
    self.assertEqual(
      b.events["post_roll_supine_failure_reset"].params["probability"], 0.20
    )
    self.assertIs(
      b.events["update_recovery_stage"].func,
      mdp.update_recovery_stage_with_support_graph,
    )
    self.assertEqual(
      b.rewards["bilateral_support_route"].params["minimum_stage"], 1
    )
    self.assertIn("reset_v38_route_progress", b.events)
    self.assertIn("update_v38_route_progress", b.events)
    self.assertIn("prone_support_route", b.rewards)

  def test_post_roll_reset_is_after_existing_trap_and_before_stage_reset(self) -> None:
    cfg = g1_getup_v38_93d_route_env_cfg(play=False)
    order = tuple(cfg.events)
    self.assertEqual(
      order.index("post_roll_supine_failure_reset"),
      order.index("photo_informed_seated_trap_reset") + 1,
    )
    self.assertLess(
      order.index("post_roll_supine_failure_reset"),
      order.index("reset_recovery_stage"),
    )
    self.assertEqual(
      order.index("reset_v38_route_progress"),
      order.index("reset_recovery_stage") + 1,
    )
    self.assertEqual(
      order.index("update_v38_route_progress"),
      order.index("update_recovery_stage") + 1,
    )

  def test_play_post_roll_reset_requires_explicit_opt_in(self) -> None:
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("SMP_PLAY_V38_POST_ROLL_RESET", None)
      cfg = g1_getup_v38_93d_route_env_cfg(play=True)
      self.assertEqual(
        cfg.events["post_roll_supine_failure_reset"].params["probability"], 0.0
      )
    with patch.dict(os.environ, {"SMP_PLAY_V38_POST_ROLL_RESET": "1"}):
      cfg = g1_getup_v38_93d_route_env_cfg(play=True)
      self.assertEqual(
        cfg.events["post_roll_supine_failure_reset"].params["probability"], 1.0
      )

  def test_registered_tasks_use_fresh_optimizer_warm_start(self) -> None:
    for task in (
      "Smp-Getup-V38-93D-Control-G1",
      "Smp-Getup-V38-93D-Route-G1",
    ):
      self.assertIs(load_runner_cls(task), SmpCurriculumWarmStartRunner)

  def test_protocol_is_hash_locked_and_telemetry_complete(self) -> None:
    path = Path(__file__).parents[1] / "docs/v38_93d_recovery_route_study_v1.json"
    protocol, digest = _validate_protocol(path)
    self.assertEqual(digest, _PROTOCOL_SHA256)
    self.assertEqual(protocol["training_protocol"]["max_iterations"], 8000)
    self.assertEqual(len(protocol["training_protocol"]["gpu_assignment"]), 6)
    telemetry = protocol["evaluation_protocol"][
      "required_per_environment_telemetry"
    ]
    self.assertIn("max_head_z", telemetry)
    self.assertIn("max_upright", telemetry)
    self.assertTrue(protocol["claim_boundary"]["no_hardware_authorization"])


if __name__ == "__main__":
  unittest.main()
