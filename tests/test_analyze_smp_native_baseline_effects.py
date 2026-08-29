from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_smp_native_baseline_effects as effects


class NativeBaselineEffectsTest(unittest.TestCase):
  def _fixture(self, root: Path, *, proposed_advantage: bool = True):
    registry = json.loads(
      (Path(__file__).parents[1] / "docs/ral_baseline_registry.json").read_text()
    )
    registry_path = root / "registry.json"
    registry_path.write_text(json.dumps(registry))
    base_counts = {
      "task_only_ppo": [496, 455, 450, 445, 440],
      "original_product_smp": [490, 430, 425, 420, 415],
      "proposed_smp_recovery": (
        [500, 480, 475, 470, 465] if proposed_advantage else [490, 420, 415, 410, 405]
      ),
    }
    for gate in effects._GATES:
      for seed_index, seed in enumerate(effects._SEEDS):
        rows = []
        for method in effects._METHODS:
          for mode_index, mode in enumerate(effects._MODES):
            successes = min(512, base_counts[method][mode_index] + seed_index)
            rows.append(
              {
                "arm": method,
                "reset_mode": mode,
                "evaluation_schema_version": 2,
                "policy_seed": seed,
                "seed": 20260829,
                "num_envs": 512,
                "steps": 500,
                "checkpoint": str(root / f"model_{gate}.pt"),
                "matched_eval_manifest_sha256": "a" * 64,
                "strict_successes": successes,
                "strict_success_rate": successes / 512,
                "finite_action_rate": 1.0,
                "max_joint_speed_p95_rad_s": 4.0,
                "max_power_mean_w": 100.0,
                "contact_foot_slip_p95_m_s": 0.1,
                "post_success_root_drift_p95_m": 0.05,
                "secondary_fall_rate_after_success": 0.01,
                "foot_separation_at_success_p95_m": 0.3,
                "action_delta_rms_p95": 0.2,
                "action_second_difference_rms_p95": 0.1,
              }
            )
        path = root / f"gate_{gate}" / f"seed_{seed}" / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"evaluations": rows}))
    return effects.NativeEffectCfg(
      evidence_dir=root,
      registry=registry_path,
      output_json=root / "paired.json",
    )

  def test_supports_paired_advantage_only_from_complete_matched_matrices(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      result = effects.write_analysis(cfg)
      self.assertEqual(result["status"], "PROPOSED_PAIRED_ADVANTAGE_SUPPORTED")
      self.assertTrue(all(result["support_rule_checks"].values()))
      primary = result["contrasts"]["proposed_smp_recovery_minus_original_product_smp"][
        "gates"
      ]["29999"]["fixed_worst"]
      self.assertGreater(primary["ci95_low"], 0.0)
      self.assertEqual(primary["positive_seed_count"], 3)
      self.assertTrue(cfg.output_json.is_file())

  def test_complete_null_result_does_not_claim_advantage(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary), proposed_advantage=False)
      result = effects.analyze(cfg)
      self.assertEqual(result["status"], "NATIVE_COMPARISON_COMPLETE_NO_ADVANTAGE")
      self.assertFalse(result["support_rule_checks"]["primary_fixed_worst_superiority"])

  def test_missing_factorial_or_mixed_held_out_manifest_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      path = root / "gate_8000" / "seed_20260901" / "summary.json"
      payload = json.loads(path.read_text())
      payload["evaluations"].pop()
      path.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "exact three-method five-mode"):
        effects.analyze(cfg)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      path = root / "gate_15000" / "seed_20260902" / "summary.json"
      payload = json.loads(path.read_text())
      for row in payload["evaluations"]:
        row["matched_eval_manifest_sha256"] = "b" * 64
      path.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "do not share one held-out"):
        effects.analyze(cfg)


if __name__ == "__main__":
  unittest.main()
