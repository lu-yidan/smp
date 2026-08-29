from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from analyze_smp_hardware_trials import _POSE_COUNTS, HardwareAnalysisCfg, analyze


class HardwareTrialAnalysisTest(unittest.TestCase):
  def _ledger(self, root: Path, *, dirty: bool = False) -> Path:
    ledger = root / "trials.csv"
    rows = []
    order = 0
    for pose, count in _POSE_COUNTS.items():
      for index in range(count):
        trial_id = f"{pose}-{index:02d}"
        binary = root / f"{trial_id}.bin"
        metadata = root / f"{trial_id}.json"
        binary.write_bytes(b"evidence")
        metadata.write_text(
          json.dumps(
            {
              "logger_schema_version": 2,
              "deploy_git_commit": "abcdef1",
              "deploy_repository_dirty": dirty,
              "total_steps": 100,
            }
          )
        )
        success = index % 2 == 0
        rows.append(
          {
            "block_id": "block-01",
            "trial_id": trial_id,
            "order_index": order,
            "randomization_seed": 123,
            "policy_seed": 456,
            "checkpoint_sha256": "a" * 64,
            "onnx_sha256": "b" * 64,
            "deploy_git_commit": "abcdef1",
            "deploy_repository_dirty": str(dirty).lower(),
            "logger_schema_version": 2,
            "robot_id": "g1-test",
            "operator_id": "operator",
            "initial_pose": pose,
            "condition": "flat_core",
            "surface": "mat-a",
            "policy_start_time_utc": "2026-08-30T00:00:00Z",
            "valid_initialization": "true",
            "success": str(success).lower(),
            "first_stand": str(success).lower(),
            "recovery_time_s": "4.0" if success else "",
            "secondary_fall": "false",
            "safety_abort": "false",
            "abort_reason": "",
            "human_contact": "false",
            "tether_assist": "false",
            "max_abs_joint_velocity_rad_s": 3.0,
            "max_abs_tau_est_nm": 40.0,
            "max_abs_tau_cmd_est_nm": 42.0,
            "max_abs_imu_angular_velocity_rad_s": 2.0,
            "action_delta_rms": 0.1,
            "action_second_difference_rms": 0.05,
            "final_stance_width_m": 0.3,
            "failure_class": "no_progress" if not success else "",
            "log_bin_path": binary.name,
            "log_json_path": metadata.name,
            "video_uri": f"video://{trial_id}",
            "notes": "",
          }
        )
        order += 1
    with ledger.open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
    return ledger

  def test_complete_matrix_is_summarized_by_pose(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger = self._ledger(root)
      result = analyze(
        HardwareAnalysisCfg(trials=ledger, output_json=root / "result.json")
      )
      self.assertEqual(result["status"], "COMPLETE")
      self.assertEqual(result["valid_trial_count"], 80)
      self.assertEqual(result["pose_counts"], _POSE_COUNTS)
      self.assertEqual(result["safety"]["max_abs_tau_est_nm"]["max"], 40.0)

  def test_dirty_deployment_is_rejected_for_final_evidence(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger = self._ledger(root, dirty=True)
      with self.assertRaisesRegex(ValueError, "dirty deployment"):
        analyze(HardwareAnalysisCfg(trials=ledger, output_json=root / "result.json"))

  def test_partial_matrix_requires_explicit_opt_out(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger = self._ledger(root)
      with ledger.open(newline="") as stream:
        rows = list(csv.DictReader(stream))[:-1]
      with ledger.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
      with self.assertRaisesRegex(ValueError, "incomplete preregistered matrix"):
        analyze(HardwareAnalysisCfg(trials=ledger, output_json=root / "result.json"))
      result = analyze(
        HardwareAnalysisCfg(
          trials=ledger,
          output_json=root / "result.json",
          require_complete=False,
        )
      )
      self.assertEqual(result["status"], "VALID_PARTIAL")


if __name__ == "__main__":
  unittest.main()
