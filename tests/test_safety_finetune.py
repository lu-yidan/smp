from __future__ import annotations

import unittest

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_runner_cls

from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  g1_scratch_a10_f2s2_physical_reset_env_cfg,
  g1_scratch_a11_f2s2_grounded_safety_env_cfg,
)
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner


class SafetyFinetuneTest(unittest.TestCase):
  def test_a11_keeps_a10_observations_terminations_and_primary_objective(self) -> None:
    a10 = g1_scratch_a10_f2s2_physical_reset_env_cfg(play=False)
    a11 = g1_scratch_a11_f2s2_grounded_safety_env_cfg(play=False)
    self.assertEqual(tuple(a10.observations), tuple(a11.observations))
    self.assertEqual(
      tuple(a10.observations["actor"].terms),
      tuple(a11.observations["actor"].terms),
    )
    self.assertEqual(tuple(a10.terminations), tuple(a11.terminations))
    for name in a10.terminations:
      self.assertIs(a10.terminations[name].func, a11.terminations[name].func)
      self.assertEqual(a10.terminations[name].params, a11.terminations[name].params)
    self.assertIs(
      a10.rewards["task_smp_product"].func,
      a11.rewards["task_smp_product"].func,
    )
    self.assertEqual(
      a10.rewards["task_smp_product"].params,
      a11.rewards["task_smp_product"].params,
    )

  def test_a11_freezes_balanced_grounded_procedural_reset(self) -> None:
    cfg = g1_scratch_a11_f2s2_grounded_safety_env_cfg(play=False)
    reset = cfg.events["curriculum_validated_fall_reset"]
    self.assertEqual(reset.params["balanced_probability"], 1.0)
    self.assertEqual(reset.params["target_probability"], 1.0)
    self.assertEqual(reset.params["mode_weights"], (1.0, 1.0, 1.0, 1.0))
    self.assertGreater(reset.params["all_procedural_until_step"], 10**12)
    self.assertEqual(
      cfg.events["gsi_reset"].func.__name__, "physically_validated_gsi_reset"
    )

  def test_a11_safety_terms_and_shuffling_gate_are_frozen(self) -> None:
    cfg = g1_scratch_a11_f2s2_grounded_safety_env_cfg(play=False)
    expected = {
      "joint_speed_excess": (-2.0e-4, {"speed_limit": 10.0}),
      "joint_power_excess": (-5.0e-7, {"power_limit": 250.0}),
      "action_rate_l2": (-5.0e-4, {}),
      "quiet_action_acc_l2": (-3.0e-4, {}),
      "head_vertical_overspeed": (-0.10, {"speed_limit": 0.25}),
      "quiet_foot_speed_l2": (-0.05, {}),
      "quiet_base_angular_speed_l2": (-0.01, {}),
    }
    for name, (weight, params) in expected.items():
      self.assertIn(name, cfg.rewards)
      self.assertEqual(cfg.rewards[name].weight, weight)
      self.assertEqual(cfg.rewards[name].params, params)
    self.assertIn("quiet_foot_speed", cfg.metrics)
    self.assertIn("quiet_action_acc", cfg.metrics)

  def test_a11_training_uses_warm_start_but_play_uses_plain_runner(self) -> None:
    task = "Smp-Getup-Scratch-A11-F2S2-Grounded-Safety-G1"
    self.assertIs(load_runner_cls(task), SmpCurriculumWarmStartRunner)
    self.assertTrue(issubclass(load_runner_cls(task), MjlabOnPolicyRunner))


if __name__ == "__main__":
  unittest.main()
