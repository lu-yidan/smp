from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from analyze_smp_hardware_trials import _POSE_COUNTS, HardwareAnalysisCfg, analyze


class HardwareTrialAnalysisTest(unittest.TestCase):
  def _limits(self, root: Path, *, joint_velocity: float = 4.0) -> Path:
    path = root / "safety_limits.json"
    path.write_text(
      json.dumps(
        {
          "schema_version": 1,
          "robot_model": "Unitree G1",
          "robot_id": "g1-test",
          "source": {
            "kind": "deployed_controller_config",
            "reference": "controller/limits.yaml@abcdef1",
            "sha256": "c" * 64,
            "frozen_before_trial_utc": "2026-08-29T23:00:00Z",
          },
          "joint_names": [f"joint_{index}" for index in range(29)],
          "limits": {
            "max_abs_joint_velocity_rad_s": [joint_velocity] * 29,
            "max_abs_tau_est_nm": [50.0] * 29,
            "max_abs_tau_cmd_est_nm": [60.0] * 29,
            "max_abs_imu_angular_velocity_rad_s": 3.0,
            "action_delta_rms": 0.2,
            "action_second_difference_rms": 0.2,
          },
        }
      )
    )
    return path

  def _ledger(self, root: Path, *, dirty: bool = False) -> Path:
    ledger = root / "trials.csv"
    rows = []
    order = 0
    for pose, count in _POSE_COUNTS.items():
      for index in range(count):
        trial_id = f"{pose}-{index:02d}"
        binary = root / f"{trial_id}.bin"
        metadata = root / f"{trial_id}.json"
        fields = {
          "dq": 29,
          "tau_est": 29,
          "tau_cmd_est": 29,
          "ang_vel": 3,
          "actions": 29,
        }
        actions = np.asarray([0.0, 0.1, 0.1, 0.2, 0.2], dtype=np.float32)
        frames = []
        for action in actions:
          frames.append(
            np.concatenate(
              (
                np.full(29, 3.0, dtype=np.float32),
                np.full(29, 40.0, dtype=np.float32),
                np.full(29, 42.0, dtype=np.float32),
                np.full(3, 2.0, dtype=np.float32),
                np.full(29, action, dtype=np.float32),
              )
            )
          )
        np.stack(frames).tofile(binary)
        metadata.write_text(
          json.dumps(
            {
              "logger_schema_version": 2,
              "deploy_git_commit": "abcdef1",
              "deploy_repository_dirty": dirty,
              "record_dim": sum(fields.values()),
              "fields": fields,
              "total_steps": len(frames),
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
            "action_delta_rms": np.sqrt(0.005),
            "action_second_difference_rms": 0.1,
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
        HardwareAnalysisCfg(
          trials=ledger,
          output_json=root / "result.json",
          safety_limits=self._limits(root),
        )
      )
      self.assertEqual(result["status"], "COMPLETE")
      self.assertTrue(result["safety_gate"]["pass"])
      self.assertEqual(result["valid_trial_count"], 80)
      self.assertEqual(result["pose_counts"], _POSE_COUNTS)
      self.assertEqual(result["safety"]["max_abs_tau_est_nm"]["max"], 40.0)

  def test_dirty_deployment_is_rejected_for_final_evidence(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger = self._ledger(root, dirty=True)
      with self.assertRaisesRegex(ValueError, "dirty deployment"):
        analyze(
          HardwareAnalysisCfg(
            trials=ledger,
            output_json=root / "result.json",
            safety_limits=self._limits(root),
          )
        )

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
        analyze(
          HardwareAnalysisCfg(
            trials=ledger,
            output_json=root / "result.json",
            safety_limits=self._limits(root),
          )
        )
      result = analyze(
        HardwareAnalysisCfg(
          trials=ledger,
          output_json=root / "result.json",
          safety_limits=self._limits(root),
          require_complete=False,
        )
      )
      self.assertEqual(result["status"], "VALID_PARTIAL")

  def test_per_joint_safety_exceedance_is_retained_and_fails_gate(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger = self._ledger(root)
      result = analyze(
        HardwareAnalysisCfg(
          trials=ledger,
          output_json=root / "result.json",
          safety_limits=self._limits(root, joint_velocity=2.5),
        )
      )
      self.assertEqual(result["status"], "COMPLETE_WITH_SAFETY_LIMIT_EXCEEDANCE")
      self.assertFalse(result["safety_gate"]["pass"])
      self.assertEqual(result["safety_gate"]["violation_trial_count"], 80)

  def test_hand_entered_safety_summary_must_match_raw_log(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger = self._ledger(root)
      with ledger.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
      rows[0]["max_abs_tau_est_nm"] = "39.0"
      with ledger.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
      with self.assertRaisesRegex(ValueError, "does not match raw log"):
        analyze(
          HardwareAnalysisCfg(
            trials=ledger,
            output_json=root / "result.json",
            safety_limits=self._limits(root),
          )
        )

  def test_safety_limits_must_be_frozen_before_trials(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger = self._ledger(root)
      limits = self._limits(root)
      content = json.loads(limits.read_text())
      content["source"]["frozen_before_trial_utc"] = "2026-08-30T00:01:00Z"
      limits.write_text(json.dumps(content))
      with self.assertRaisesRegex(ValueError, "not frozen before the trial"):
        analyze(
          HardwareAnalysisCfg(
            trials=ledger,
            output_json=root / "result.json",
            safety_limits=limits,
          )
        )


if __name__ == "__main__":
  unittest.main()
