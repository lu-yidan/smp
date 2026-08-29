from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import select_smp_stable_arm as selector


def _mode(rate: float, total: int = 512) -> dict:
  successes = round(rate * total)
  low, high = selector._wilson(successes, total)
  return {
    "successes": successes,
    "total": total,
    "rate": successes / total,
    "ci95_low": low,
    "ci95_high": high,
  }


def _arm(macro: float, *, gsi: float = 0.98, screen_pass: bool = True) -> dict:
  return {
    "modes": {
      "native_gsi": _mode(gsi),
      "prone": _mode(macro),
      "supine": _mode(macro),
      "left_side": _mode(macro),
      "right_side": _mode(macro),
    },
    "gsi": gsi,
    "fixed_macro": macro,
    "fixed_worst": macro,
    "finite_action_rate_min": 1.0,
    "screen_pass": screen_pass,
    "secondary_fall_rate_after_success_max": 0.02,
    "post_success_root_drift_p95_m": 0.03,
    "contact_foot_slip_p95_m_s": 0.04,
    "max_power_mean_w": 100.0,
    "max_joint_speed_p95_rad_s": 8.0,
  }


class StableSelectionTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def _write(self, gate: int, arms: dict, policy_seed: int = 7) -> None:
    directory = self.root / f"gate_{gate}"
    directory.mkdir()
    (directory / "analysis.json").write_text(
      json.dumps(
        {
          "status": "SCREEN_PASS_NOT_FINAL",
          "checkpoint": f"model_{gate}.pt",
          "policy_seed": policy_seed,
          "arms": arms,
        }
      )
    )

  def test_promotes_at_most_two_checkpoint_stable_arms(self) -> None:
    for gate, rates in (
      (8000, (0.50, 0.45, 0.30)),
      (15000, (0.82, 0.80, 0.50)),
      (25000, (0.88, 0.84, 0.55)),
      (29999, (0.90, 0.86, 0.58)),
    ):
      self._write(
        gate,
        {
          "arm_a": _arm(rates[0]),
          "arm_b": _arm(rates[1]),
          "arm_c": _arm(rates[2], screen_pass=rates[2] >= 0.4),
        },
      )
    result = selector.select(selector.SelectionCfg(evidence_dir=self.root))
    self.assertEqual(result["status"], "PROMOTE_FOR_POLICY_SEEDS")
    self.assertEqual(result["promoted_candidates"], ["arm_a", "arm_b"])
    self.assertFalse(result["arms"]["arm_c"]["eligible"])

  def test_rejects_late_checkpoint_regression(self) -> None:
    for gate, rate in ((8000, 0.5), (15000, 0.82), (25000, 0.95), (29999, 0.81)):
      self._write(gate, {"arm_a": _arm(rate)})
    result = selector.select(selector.SelectionCfg(evidence_dir=self.root))
    self.assertEqual(result["status"], "NO_PROMOTION")
    self.assertFalse(result["arms"]["arm_a"]["eligibility"]["macro_late_regression"])

  def test_requires_every_frozen_gate(self) -> None:
    for gate in (8000, 15000, 25000):
      self._write(gate, {"arm_a": _arm(0.9)})
    with self.assertRaises(FileNotFoundError):
      selector.select(selector.SelectionCfg(evidence_dir=self.root))

  def test_rejects_inconsistent_policy_seed(self) -> None:
    for gate in (8000, 15000, 25000, 29999):
      self._write(gate, {"arm_a": _arm(0.9)}, policy_seed=gate)
    with self.assertRaises(ValueError):
      selector.select(selector.SelectionCfg(evidence_dir=self.root))


if __name__ == "__main__":
  unittest.main()
