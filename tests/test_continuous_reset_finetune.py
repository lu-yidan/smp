from __future__ import annotations

import unittest

from mjlab.tasks.registry import load_runner_cls

from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  g1_scratch_a12_f2s2_prone_coverage_env_cfg,
  g1_scratch_a13_f2s2_continuous_reset_env_cfg,
)
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner


class ContinuousResetFinetuneTest(unittest.TestCase):
  def test_a13_changes_only_reset_coverage(self) -> None:
    a12 = g1_scratch_a12_f2s2_prone_coverage_env_cfg(play=False)
    a13 = g1_scratch_a13_f2s2_continuous_reset_env_cfg(play=False)
    self.assertEqual(tuple(a12.observations), tuple(a13.observations))
    self.assertEqual(tuple(a12.rewards), tuple(a13.rewards))
    self.assertEqual(tuple(a12.terminations), tuple(a13.terminations))
    for name in a12.rewards:
      self.assertIs(a12.rewards[name].func, a13.rewards[name].func)
      self.assertEqual(a12.rewards[name].weight, a13.rewards[name].weight)
      self.assertEqual(a12.rewards[name].params, a13.rewards[name].params)
    for name in a12.terminations:
      self.assertIs(a12.terminations[name].func, a13.terminations[name].func)
      self.assertEqual(a12.terminations[name].params, a13.terminations[name].params)

    a12_reset = a12.events["curriculum_validated_fall_reset"]
    a13_reset = a13.events["curriculum_validated_fall_reset"]
    self.assertEqual(a13_reset.params["mode_weights"], (3.0, 1.0, 1.0, 1.0))
    self.assertEqual(a13_reset.params["orientation_noise"], 0.35)
    self.assertEqual(a13_reset.params["joint_noise_levels"], (0.12, 0.20, 0.30))
    self.assertEqual(a13_reset.params["joint_noise_weights"], (0.70, 0.20, 0.10))
    self.assertEqual(a13_reset.params["joint_limit_margin"], 0.02)
    changed = {
      "orientation_noise",
      "joint_noise_levels",
      "joint_noise_weights",
      "joint_limit_margin",
    }
    for key, value in a12_reset.params.items():
      if key not in changed:
        self.assertEqual(value, a13_reset.params[key])

  def test_a13_is_registered_with_fresh_optimizer_warm_start_runner(self) -> None:
    task = "Smp-Getup-Scratch-A13-F2S2-Continuous-Reset-G1"
    self.assertIs(load_runner_cls(task), SmpCurriculumWarmStartRunner)


if __name__ == "__main__":
  unittest.main()
