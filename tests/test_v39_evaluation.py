from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_smp_v39_support_exit_eval import (  # noqa: E402
  _AUDIT_PER_ENV,
  _GATES,
  _LAUNCH_PLAN_ID,
  _POSES,
  _PROTOCOL_SHA256,
  _REASONS,
  _REQUIRED_PER_ENV,
  _SEEDS,
  _TRAINING_COMMIT,
  _validate_protocol,
)


class V39EvaluationTest(unittest.TestCase):
  def test_frozen_protocol_matches_launcher(self) -> None:
    repo = Path(__file__).parents[1]
    protocol = _validate_protocol(repo / "docs/v39_93d_support_exit_study_v1.json")
    evaluation = protocol["evaluation_protocol"]
    self.assertEqual(
      _PROTOCOL_SHA256,
      "08b2535c39dcaf0448a01353132c0f295ee3452b7cac6b17ebfe8fa76a036e6d",
    )
    self.assertEqual(
      _LAUNCH_PLAN_ID,
      "964f685cbf7d62e85f317ed5ce40d4aece9adddc122239dfa9dfd5d28c1493e2",
    )
    self.assertEqual(_TRAINING_COMMIT, "600943d28ba517a79bfe256d2411c12348bb3858")
    self.assertEqual(tuple(evaluation["pose_strata"]), _POSES)
    self.assertEqual(_GATES, (0, 2000, 4000, 8000, 11999))
    self.assertEqual(_SEEDS, (20262201, 20262202, 20262203))
    self.assertEqual(len(_REASONS), 8)
    self.assertIn("strict_success", _REQUIRED_PER_ENV)

  def test_audit_requires_support_exit_diagnostics(self) -> None:
    for key in (
      "max_head_z",
      "max_upright",
      "hand_support_fraction",
      "knee_support_fraction",
      "peak_joint_torque",
      "peak_joint_power",
    ):
      self.assertIn(key, _AUDIT_PER_ENV)

  def test_shared_evaluator_supports_post_roll_pose(self) -> None:
    source = (
      Path(__file__).parents[1] / "scripts/evaluate_terrain_recovery.py"
    ).read_text()
    self.assertIn('reset_mode == "post_roll_supine_crossed"', source)
    self.assertIn('"_v38_post_roll_supine_reset"', source)
    self.assertIn('"first_head_vertical_speed_step"', source)
    self.assertIn('"longest_strict_hold_steps"', source)


if __name__ == "__main__":
  unittest.main()
