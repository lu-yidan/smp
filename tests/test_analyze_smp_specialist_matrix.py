from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_smp_specialist_matrix as analyzer
import run_smp_specialist_eval_matrix as matrix


def _row(stratum: dict, success: float = 1.0) -> dict:
  return {
    "evaluation_schema_version": 2,
    "actor_observation_dim": 93,
    "policy_seed": 11,
    "stratum": stratum,
    "reset_mode": stratum["fall_pose"],
    "strict_success_rate": success,
    "terrain_exit_rate": 0.0,
    "invalid_dynamics_rate": 0.0,
    "invalid_escape_setup_rate": 0.0,
    "secondary_fall_rate_after_success": 0.0,
    "hand_support_rate": 1.0 if stratum["plate_present"] else None,
    "finite_action_rate": 1.0,
    "max_joint_speed_p95_rad_s": 1.0,
    "max_power_mean_w": 1.0,
    "contact_foot_slip_p95_m_s": 1.0,
    "action_delta_rms_p95": 1.0,
    "action_second_difference_rms_p95": 1.0,
  }


class SpecialistAnalysisTest(unittest.TestCase):
  def _fixture(
    self, root: Path, phase: str, rows: list[dict]
  ) -> analyzer.SpecialistAnalysisCfg:
    flat = root / "flat.json"
    flat.write_text(
      json.dumps(
        {
          "evaluations": [
            {
              "arm": "a6_f2s2_mix_bridge",
              "max_joint_speed_p95_rad_s": 1.0,
              "max_power_mean_w": 1.0,
              "contact_foot_slip_p95_m_s": 1.0,
              "action_delta_rms_p95": 1.0,
              "action_second_difference_rms_p95": 1.0,
            }
          ]
        }
      )
    )
    manifest = root / "manifest.json"
    manifest.write_text(
      json.dumps(
        {
          "phase": phase,
          "policy_seed": 11,
          "checkpoint_step": 19999,
          "flat_summary": str(flat),
          "flat_summary_sha256": analyzer._sha256(flat),
          "runs": [{"arm": "a6_f2s2_mix_bridge"}],
        }
      )
    )
    summary = root / "summary.json"
    summary.write_text(
      json.dumps(
        {
          "phase": phase,
          "policy_seed": 11,
          "checkpoint_step": 19999,
          "manifest": str(manifest),
          "manifest_sha256": analyzer._sha256(manifest),
          "evaluations": rows,
        }
      )
    )
    return analyzer.SpecialistAnalysisCfg(
      summary=summary,
      protocol=Path(__file__).parents[1] / "docs/ral_terrain_plate_protocol.json",
    )

  def test_complete_terrain_matrix_passes_per_seed_gate(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      rows = [_row(stratum) for stratum in matrix.terrain_strata()]
      result = analyzer.analyze(self._fixture(Path(temporary), "T", rows))
      self.assertEqual(result["status"], "PASS")
      self.assertEqual(result["metrics"]["level1_each_terrain_macro"]["stairs"], 1.0)

  def test_complete_plate_matrix_passes_and_heavy_failure_is_detected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      rows = [_row(stratum) for stratum in matrix.plate_strata()]
      cfg = self._fixture(Path(temporary), "P", rows)
      self.assertEqual(analyzer.analyze(cfg)["status"], "PASS")
      payload = json.loads(cfg.summary.read_text())
      for row in payload["evaluations"]:
        if row["stratum"]["plate_mass_kg"] == 12.0:
          row["strict_success_rate"] = 0.0
      cfg.summary.write_text(json.dumps(payload))
      result = analyzer.analyze(cfg)
      self.assertEqual(result["status"], "NO_PROMOTION")
      self.assertIn("plate_12kg", result["failures"])

  def test_incomplete_matrix_is_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      rows = [_row(stratum) for stratum in matrix.terrain_strata()[:-1]]
      cfg = self._fixture(Path(temporary), "T", rows)
      with self.assertRaisesRegex(ValueError, "exactly 76"):
        analyzer.analyze(cfg)


if __name__ == "__main__":
  unittest.main()
