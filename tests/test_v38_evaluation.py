from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_smp_v38_recovery_route_eval import (  # noqa: E402
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


class V38EvaluationTest(unittest.TestCase):
  def test_frozen_protocol_matches_launcher(self) -> None:
    repo = Path(__file__).parents[1]
    protocol = _validate_protocol(repo / "docs/v38_93d_recovery_route_study_v1.json")
    evaluation = protocol["evaluation_protocol"]
    self.assertEqual(
      _PROTOCOL_SHA256,
      "922c0b16c92ffb2bc8c8faa80b24efd39e51728bfb758a9025f3cbc7d96a9ae0",
    )
    self.assertEqual(
      _LAUNCH_PLAN_ID,
      "ee2bafde9d4a0ca0cc868496b255157b4fcd4f42b48211c0ece33477c66faf24",
    )
    self.assertEqual(_TRAINING_COMMIT, "9a2f8b9d2d6ddc1580779c033795440d5d6d82a3")
    self.assertEqual(tuple(evaluation["pose_strata"]), _POSES)
    self.assertEqual(tuple(evaluation["failure_reason_codebook"]), _REASONS)
    self.assertEqual(
      tuple(evaluation["required_per_environment_telemetry"]),
      _REQUIRED_PER_ENV,
    )
    self.assertEqual(_GATES, (0, 1000, 2000, 4000, 6000, 7999))
    self.assertEqual(_SEEDS, (20262101, 20262102, 20262103))

  def test_audit_requires_missing_v37_diagnostics(self) -> None:
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
