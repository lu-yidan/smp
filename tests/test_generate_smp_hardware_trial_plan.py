from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from generate_smp_hardware_trial_plan import HardwareTrialPlanCfg, generate_plan


class HardwareTrialPlanTest(unittest.TestCase):
  def _cfg(self, root: Path, seed: int) -> HardwareTrialPlanCfg:
    return HardwareTrialPlanCfg(
      output_json=root / f"plan-{seed}.json",
      block_id="block-01",
      frozen_before_trial_utc="2026-08-29T22:00:00Z",
      randomization_seed=seed,
      policy_seed=20260901,
      checkpoint_sha256="a" * 64,
      onnx_sha256="b" * 64,
      deploy_git_commit="abcdef1",
      robot_id="g1-test",
      surface="mat-a",
    )

  def test_seeded_plan_is_deterministic_and_stratified(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      first = generate_plan(self._cfg(root, 123))
      second = generate_plan(self._cfg(root, 123))
      self.assertEqual(first, second)
      assignments = first["assignments"]
      self.assertEqual([item["planned_slot"] for item in assignments], list(range(80)))
      self.assertEqual(
        Counter(item["initial_pose"] for item in assignments),
        {
          "prone": 15,
          "supine": 15,
          "left_side": 15,
          "right_side": 15,
          "random_fall_state": 20,
        },
      )

  def test_different_seed_changes_assignment_order(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      first = generate_plan(self._cfg(root, 123))
      second = generate_plan(self._cfg(root, 124))
      self.assertNotEqual(first["assignments"], second["assignments"])


if __name__ == "__main__":
  unittest.main()
