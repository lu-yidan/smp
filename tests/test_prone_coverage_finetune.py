from __future__ import annotations

import unittest

from mjlab.tasks.registry import load_runner_cls

from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  g1_scratch_a11_f2s2_grounded_safety_env_cfg,
  g1_scratch_a12_f2s2_prone_coverage_env_cfg,
)
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner


class ProneCoverageFinetuneTest(unittest.TestCase):
  def test_a12_changes_only_reset_pose_weights(self) -> None:
    a11 = g1_scratch_a11_f2s2_grounded_safety_env_cfg(play=False)
    a12 = g1_scratch_a12_f2s2_prone_coverage_env_cfg(play=False)
    self.assertEqual(tuple(a11.observations), tuple(a12.observations))
    self.assertEqual(tuple(a11.rewards), tuple(a12.rewards))
    self.assertEqual(tuple(a11.terminations), tuple(a12.terminations))
    for name in a11.rewards:
      self.assertIs(a11.rewards[name].func, a12.rewards[name].func)
      self.assertEqual(a11.rewards[name].weight, a12.rewards[name].weight)
      self.assertEqual(a11.rewards[name].params, a12.rewards[name].params)
    for name in a11.terminations:
      self.assertIs(a11.terminations[name].func, a12.terminations[name].func)
      self.assertEqual(a11.terminations[name].params, a12.terminations[name].params)
    a11_reset = a11.events["curriculum_validated_fall_reset"]
    a12_reset = a12.events["curriculum_validated_fall_reset"]
    self.assertEqual(a11_reset.params["mode_weights"], (1.0, 1.0, 1.0, 1.0))
    self.assertEqual(a12_reset.params["mode_weights"], (3.0, 1.0, 1.0, 1.0))
    for key, value in a11_reset.params.items():
      if key != "mode_weights":
        self.assertEqual(value, a12_reset.params[key])

  def test_a12_is_registered_with_fresh_optimizer_warm_start_runner(self) -> None:
    task = "Smp-Getup-Scratch-A12-F2S2-Prone-Coverage-G1"
    self.assertIs(load_runner_cls(task), SmpCurriculumWarmStartRunner)


if __name__ == "__main__":
  unittest.main()
