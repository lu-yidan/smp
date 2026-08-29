from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from audit_smp_baseline_registry import EXPECTED_METHODS, audit


class BaselineRegistryTest(unittest.TestCase):
  def setUp(self) -> None:
    self.repo_root = Path(__file__).parents[1]
    self.registry_path = self.repo_root / "docs/ral_baseline_registry.json"
    self.registry = json.loads(self.registry_path.read_text())

  def test_frozen_registry_is_valid_and_blocked(self) -> None:
    report = audit(copy.deepcopy(self.registry), self.registry_path)
    self.assertEqual(report["status"], "BASELINES_BLOCKED")
    self.assertFalse(report["reset_bank_ready"])
    self.assertEqual({item["id"] for item in report["methods"]}, EXPECTED_METHODS)

  def test_actor_privilege_and_budget_drift_are_rejected(self) -> None:
    privileged = copy.deepcopy(self.registry)
    privileged["methods"][0]["actor_extra_inputs"] = ["reset_family"]
    with self.assertRaisesRegex(ValueError, "extra actor inputs"):
      audit(privileged, self.registry_path)
    budget = copy.deepcopy(self.registry)
    budget["training_budget"]["max_updates"] = 40000
    with self.assertRaisesRegex(ValueError, "update budget"):
      audit(budget, self.registry_path)
    objective = copy.deepcopy(self.registry)
    objective["methods"][0]["uses_motion_prior_objective"] = True
    with self.assertRaisesRegex(ValueError, "objective contract"):
      audit(objective, self.registry_path)
    termination = copy.deepcopy(self.registry)
    termination["methods"][1]["uses_motion_prior_termination"] = False
    with self.assertRaisesRegex(ValueError, "termination contract"):
      audit(termination, self.registry_path)

  def test_ready_method_requires_hash_locked_bank_and_implementation(self) -> None:
    ready = copy.deepcopy(self.registry)
    ready["methods"][0]["status"] = "ready_for_training"
    ready["methods"][0]["blocked_on"] = []
    ready["methods"][0]["implementation"] = {
      "task": "docs/ral_baseline_comparison_protocol.md"
    }
    with self.assertRaisesRegex(ValueError, "before reset bank"):
      audit(ready, self.registry_path)

  def test_hash_locked_bank_can_unlock_fully_proven_implementations(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      docs = root / "docs"
      docs.mkdir()
      bank = root / "run_control/reset_bank.npz"
      bank.parent.mkdir()
      bank.write_bytes(b"immutable reset states")
      bank_hash = hashlib.sha256(bank.read_bytes()).hexdigest()
      manifest = root / "run_control/reset_bank.json"
      manifest.write_text(
        json.dumps(
          {
            "status": "READY",
            "bank_sha256": bank_hash,
            "num_states": 262144,
            "tensor_shapes": {"smp_window": [262144, 10, 59]},
          }
        )
      )
      registry = copy.deepcopy(self.registry)
      registry["shared_reset_bank"].update(
        {
          "status": "ready",
          "result_path": "run_control/reset_bank.npz",
          "sha256": bank_hash,
          "manifest_path": "run_control/reset_bank.json",
          "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
      )
      implementation = docs / "implementation.txt"
      implementation.write_text("frozen implementation\n")
      for method in registry["methods"]:
        method["status"] = "ready_for_training"
        method["blocked_on"] = []
        method["implementation"] = {"task_or_adapter": "docs/implementation.txt"}
      registry_path = docs / "registry.json"
      registry_path.write_text(json.dumps(registry))
      report = audit(registry, registry_path)
      self.assertEqual(report["status"], "BASELINES_READY_FOR_TRAINING")
      self.assertTrue(report["reset_bank_ready"])

  def test_complete_method_requires_nonempty_result(self) -> None:
    complete = copy.deepcopy(self.registry)
    complete["methods"][0]["status"] = "complete"
    complete["methods"][0]["blocked_on"] = []
    complete["methods"][0]["implementation"] = {
      "task": "docs/ral_baseline_comparison_protocol.md"
    }
    complete["methods"][0]["result_path"] = "results/missing.json"
    with self.assertRaisesRegex(ValueError, "before reset bank"):
      audit(complete, self.registry_path)


if __name__ == "__main__":
  unittest.main()
