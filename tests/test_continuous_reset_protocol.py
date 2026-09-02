from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from launch_smp_continuous_reset_finetune import (  # noqa: E402
  _PROTOCOL_SHA256,
  _validate_protocol,
)


class ContinuousResetProtocolTest(unittest.TestCase):
  def test_protocol_is_hash_locked_and_reset_only(self) -> None:
    path = Path(__file__).parents[1] / "docs/ral_continuous_reset_finetune_v1.json"
    protocol, digest = _validate_protocol(path)
    self.assertEqual(digest, _PROTOCOL_SHA256)
    self.assertEqual(protocol["source_policy"]["checkpoint_name"], "model_3500.pt")
    reset = protocol["treatment"]["reset_distribution"]
    self.assertEqual(reset["orientation_roll_pitch_noise_rad"], 0.35)
    self.assertEqual(reset["joint_noise_levels_rad"], [0.12, 0.20, 0.30])
    self.assertEqual(reset["joint_noise_weights"], [0.70, 0.20, 0.10])
    self.assertEqual(protocol["training_protocol"]["max_iterations"], 5000)
    self.assertEqual(protocol["training_protocol"]["policy_seed"], 20261501)
    self.assertEqual(protocol["training_protocol"]["wandb_mode"], "offline")
    incident = protocol["launch_incident_resolution"]
    self.assertEqual(incident["status"], "RESOLVED_BY_PREREGISTERED_OFFLINE_RETRY")
    self.assertEqual(incident["checkpoint_count"], 0)
    self.assertEqual(incident["gpu_processes_after_exit"], 0)
    self.assertEqual(
      incident["preserved_artifacts"]["launch_manifest"]["sha256"],
      "7f2b6605f097e4493eb4520d339ac015a3b27195d2f9ba8b46ebdd2880742633",
    )
    self.assertTrue(protocol["claim_boundary"]["not_ral_evidence"])

  def test_evaluator_exports_a13_settling_and_reset_coverage_metrics(self) -> None:
    source = (
      Path(__file__).parents[1] / "scripts/evaluate_smp_baseline.py"
    ).read_text()
    required = (
      '"settled_stand_definition"',
      '"settled_stand_success_rate"',
      '"base_height_at_strict_success_median_m"',
      '"post_success_shuffling_fraction_p95"',
      '"post_success_contact_foot_slip_p95_m_s"',
      '"max_torque_p95_nm"',
      '"max_power_p95_w"',
      '"procedural_joint_noise_level_rad"',
      '"procedural_orientation_offset_rad"',
    )
    for token in required:
      self.assertIn(token, source)


if __name__ == "__main__":
  unittest.main()
