from __future__ import annotations

import sys
import unittest
from pathlib import Path

from mjlab.tasks.registry import load_runner_cls

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v39_support_exit_env_cfg import (
  g1_getup_v39_93d_control_env_cfg,
  g1_getup_v39_93d_support_exit_env_cfg,
)
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from launch_smp_v39_support_exit import (  # noqa: E402
  _PROTOCOL_SHA256,
  _validate_protocol,
)


class V39SupportExitTest(unittest.TestCase):
  def test_matched_deployable_actor_and_envelope(self) -> None:
    a = g1_getup_v39_93d_control_env_cfg(play=False)
    b = g1_getup_v39_93d_support_exit_env_cfg(play=False)
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

  def test_treatment_is_additive_and_control_is_exact_v38_b(self) -> None:
    a = g1_getup_v39_93d_control_env_cfg(play=False)
    b = g1_getup_v39_93d_support_exit_env_cfg(play=False)
    self.assertNotIn("reset_v39_support_exit", a.events)
    self.assertNotIn("v39_support_dwell", a.rewards)
    self.assertIn("reset_v39_support_exit", b.events)
    self.assertIn("update_v39_support_exit", b.events)
    self.assertIs(b.events["reset_v39_support_exit"].func, mdp.reset_v39_support_exit)
    self.assertIs(b.events["update_v39_support_exit"].func, mdp.update_v39_support_exit)
    self.assertEqual(b.rewards["v39_support_dwell"].weight, -0.24)
    self.assertEqual(b.rewards["v39_head_best_progress"].weight, 0.55)
    self.assertEqual(b.rewards["v39_foot_transfer"].weight, 0.35)
    self.assertEqual(b.rewards["v38_route_progress"].weight, 0.20)
    self.assertEqual(b.rewards["prone_support_route"].weight, 0.04)

  def test_support_exit_event_order_and_parameters(self) -> None:
    cfg = g1_getup_v39_93d_support_exit_env_cfg(play=False)
    order = tuple(cfg.events)
    self.assertEqual(
      order.index("reset_v39_support_exit"),
      order.index("reset_v38_route_progress") + 1,
    )
    self.assertEqual(
      order.index("update_v39_support_exit"),
      order.index("update_v38_route_progress") + 1,
    )
    params = cfg.events["update_v39_support_exit"].params
    self.assertEqual(params["grace_steps"], 60)
    self.assertEqual(params["full_penalty_steps"], 180)
    self.assertEqual(params["exit_height"], 0.55)
    self.assertEqual(params["exit_upright"], 0.55)

  def test_registered_tasks_use_fresh_optimizer_warm_start(self) -> None:
    for task in (
      "Smp-Getup-V39-93D-Control-G1",
      "Smp-Getup-V39-93D-Support-Exit-G1",
    ):
      self.assertIs(load_runner_cls(task), SmpCurriculumWarmStartRunner)

  def test_protocol_is_hash_locked(self) -> None:
    path = Path(__file__).parents[1] / "docs/v39_93d_support_exit_study_v1.json"
    protocol, digest = _validate_protocol(path)
    self.assertEqual(digest, _PROTOCOL_SHA256)
    self.assertEqual(protocol["training_protocol"]["max_iterations"], 12000)
    self.assertEqual(len(protocol["training_protocol"]["gpu_assignment"]), 6)
    self.assertTrue(protocol["claim_boundary"]["no_hardware_authorization"])


if __name__ == "__main__":
  unittest.main()
