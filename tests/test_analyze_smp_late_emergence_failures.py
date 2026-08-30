from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from analyze_smp_late_emergence_failures import DiagnosticCfg, analyze, write_diagnostic


_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(seed: int, mode: str, successes: int, *, schema: int = 2) -> dict:
  count = 4
  steps = [index if index < successes else -1 for index in range(count)]
  values = [float(index + 1) for index in range(count)]
  rate = successes / count
  return {
    "evaluation_schema_version": schema,
    "seed": 7,
    "num_envs": count,
    "steps": 5,
    "policy_seed": seed,
    "reset_mode": mode,
    "actor_observation_dim": 93,
    "checkpoint": "model_29999.pt",
    "checkpoint_sha256": "a" * 64,
    "task": "Smp-Getup-Test-G1",
    "finite_action_rate": 1.0,
    "strict_successes": successes,
    "strict_success_rate": rate,
    "contact_foot_slip_p95_m_s": 0.2 + rate,
    "root_planar_excursion_p95_m": 0.3 + rate,
    "max_root_linear_speed_mean_m_s": 0.4 + rate,
    "max_root_angular_speed_mean_rad_s": 0.5 + rate,
    "max_joint_speed_p95_rad_s": 10.0 + rate,
    "max_power_mean_w": 100.0 + rate,
    "action_delta_rms_p95": 0.6 + rate,
    "action_second_difference_rms_p95": 0.7 + rate,
    "per_env": {
      "strict_first_step": steps,
      "initial_head_z_m": [0.4, 0.5, 0.6, 0.7],
      "root_planar_excursion_m": values,
      "contact_foot_slip_m_s": values,
      "max_joint_speed_rad_s": values,
      "max_power_w": values,
      "action_delta_rms": values,
      "action_second_difference_rms": values,
    },
  }


class LateEmergenceFailureDiagnosisTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    root = Path(self.temporary.name)
    self.evidence = root / "evidence"
    self.aggregate = root / "aggregate.json"
    self.promotion = root / "promotion.json"
    self.output = root / "diagnosis.json"
    self.seeds = (1, 2, 3)
    self.arms = ("a1", "a6")
    for seed in self.seeds:
      seed_dir = self.evidence / f"seed_{seed}"
      seed_dir.mkdir(parents=True)
      (seed_dir / "_COMPLETE.json").write_text(
        json.dumps({
          "evaluation_schema_version": 2,
          "eval_seeds": [7],
          "num_envs": 4,
          "steps": 5,
          "modes": list(_MODES),
          "result_count": len(self.arms) * len(_MODES),
          "devices": ["cuda:0"],
          "manifest": f"manifests/seed_{seed}.json",
        })
      )
      for arm in self.arms:
        for mode in _MODES:
          successes = 0
          if mode == "native_gsi":
            successes = 1 if arm == "a1" else 3
          elif arm == "a6" and seed != 3:
            successes = 2
          path = seed_dir / f"{arm}__model_29999__{mode}__eval7.json"
          path.write_text(json.dumps(_cell(seed, mode, successes)))
    self.aggregate.write_text(
      json.dumps(
        {
          "status": "MINIMUM_POLICY_SEEDS_MET",
          "policy_seeds": list(self.seeds),
        }
      )
    )
    self.promotion.write_text(
      json.dumps(
        {
          "status": "NO_PROMOTION",
          "selected_arm": None,
          "aggregate_sha256": _sha256(self.aggregate),
          "promotion_id": "frozen",
        }
      )
    )
    self.cfg = DiagnosticCfg(
      evidence_dir=self.evidence,
      promotion=self.promotion,
      aggregate=self.aggregate,
      output_json=self.output,
      expected_policy_seeds=self.seeds,
      expected_arms=self.arms,
      evaluation_seed=7,
      num_envs=4,
      steps=5,
    )

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def test_complete_null_result_is_diagnosed_without_retraining(self) -> None:
    result = analyze(self.cfg)
    self.assertEqual(result["validated_cell_count"], 30)
    self.assertEqual(result["automatic_action"], "STOP_NO_SAFE_AUTOMATIC_RETRAINING")
    self.assertEqual(
      result["arms"]["a1"]["mode_patterns"]["prone"]["pattern"],
      "ALL_SEEDS_ZERO_SUCCESS",
    )
    self.assertEqual(
      result["arms"]["a6"]["mode_patterns"]["prone"]["pattern"],
      "SEED_COLLAPSE_TO_ZERO",
    )

  def test_output_is_immutable_and_deterministic(self) -> None:
    first = write_diagnostic(self.cfg)
    second = write_diagnostic(self.cfg)
    self.assertEqual(first, second)
    self.assertTrue(self.output.is_file())
    self.assertNotIn("NaN", self.output.read_text())

  def test_missing_cell_fails_closed(self) -> None:
    target = self.evidence / "seed_3/a6__model_29999__right_side__eval7.json"
    target.unlink()
    with self.assertRaisesRegex(ValueError, "required artifact is missing"):
      analyze(self.cfg)

  def test_schema_drift_fails_closed(self) -> None:
    target = self.evidence / "seed_1/a1__model_29999__prone__eval7.json"
    target.write_text(json.dumps(_cell(1, "prone", 0, schema=1)))
    with self.assertRaisesRegex(ValueError, "evaluation_schema_version"):
      analyze(self.cfg)

  def test_non_terminal_promotion_is_rejected(self) -> None:
    payload = json.loads(self.promotion.read_text())
    payload["status"] = "PROMOTE_TP_SPECIALISTS"
    self.promotion.write_text(json.dumps(payload))
    with self.assertRaisesRegex(ValueError, "only valid for terminal NO_PROMOTION"):
      analyze(self.cfg)


if __name__ == "__main__":
  unittest.main()
