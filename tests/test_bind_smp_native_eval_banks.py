from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import bind_smp_native_eval_banks as binder
import run_smp_frozen_eval_matrix as matrix


class NativeEvalBindingTest(unittest.TestCase):
  def _fixture(self, root: Path) -> binder.EvalBindingCfg:
    checkpoint_rows = []
    for seed in binder._SEEDS:
      for gate in binder._GATES:
        runs = []
        for method in binder._METHODS:
          checkpoint = root / f"{method}_{seed}_{gate}.pt"
          checkpoint.write_bytes(f"{method}-{seed}-{gate}".encode())
          runs.append(
            {
              "name": method,
              "task": f"Task-{method}",
              "checkpoint": str(checkpoint),
              "checkpoint_sha256": binder._sha256(checkpoint),
              "policy_seed": seed,
              "environment_seed": seed,
            }
          )
        payload = {
          "schema_version": 1,
          "manifest_id": f"manifest-{seed}-{gate}",
          "promotion_id": "promotion",
          "training_reset_bank_sha256": "a" * 64,
          "checkpoint_step": gate,
          "policy_seed": seed,
          "evaluation_status": "BLOCKED_ON_MATCHED_HELD_OUT_RESET_BANK",
          "evaluation_protocol": {
            "reset_modes": list(binder._MODES),
            "num_envs": 512,
            "steps": 500,
            "evaluation_seed": 20260829,
          },
          "runs": runs,
        }
        path = root / f"source_{seed}_{gate}.json"
        path.write_text(json.dumps(payload))
        checkpoint_rows.append(
          {
            "policy_seed": seed,
            "checkpoint_step": gate,
            "path": str(path),
            "sha256": binder._sha256(path),
          }
        )
    checkpoint_index = root / "checkpoint_index.json"
    checkpoint_index.write_text(
      json.dumps(
        {
          "status": "CHECKPOINTS_READY_EVALUATION_BANK_BLOCKED",
          "index_id": "checkpoint-index",
          "policy_seeds": list(binder._SEEDS),
          "checkpoint_steps": list(binder._GATES),
          "methods": sorted(binder._METHODS),
          "manifests": checkpoint_rows,
        }
      )
    )
    banks = {}
    for index, mode in enumerate(binder._MODES):
      bank = root / f"{mode}.pt"
      bank.write_bytes(mode.encode())
      counts = [0] * 5
      counts[index] = 512
      banks[mode] = {
        "path": str(bank),
        "sha256": binder._sha256(bank),
        "num_states": 512,
        "reset_type_counts": counts,
      }
    eval_manifest = root / "eval_banks.json"
    eval_manifest.write_text(
      json.dumps(
        {
          "status": "READY",
          "plan_id": "eval-plan",
          "promotion_id": "promotion",
          "generation_seed": 20260829,
          "num_states_per_mode": 512,
          "modes": list(binder._MODES),
          "training_bank_sha256": "a" * 64,
          "exact_training_overlap_count": 0,
          "banks": banks,
        }
      )
    )
    self.eval_manifest = eval_manifest
    self.checkpoint_index = checkpoint_index
    return binder.EvalBindingCfg(
      checkpoint_index=checkpoint_index,
      eval_bank_manifest=eval_manifest,
      output_dir=root / "formal",
    )

  def test_binds_twelve_manifests_with_exact_lineage(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      index = binder.write_bindings(cfg)
      self.assertEqual(index["status"], "READY")
      self.assertEqual(len(index["manifests"]), 12)
      payload = json.loads(Path(index["manifests"][0]["path"]).read_text())
      self.assertEqual(
        payload["evaluation_status"], "READY_WITH_MATCHED_HELD_OUT_RESET_BANK"
      )
      self.assertEqual(
        payload["matched_eval_manifest_sha256"], binder._sha256(self.eval_manifest)
      )
      self.assertEqual(len(payload["runs"]), 3)
      metadata, runs = matrix._load_manifest(Path(index["manifests"][0]["path"]))
      self.assertEqual(
        metadata["evaluation_status"],
        "READY_WITH_MATCHED_HELD_OUT_RESET_BANK",
      )
      self.assertEqual(len(runs), 3)

  def test_bank_or_checkpoint_lineage_change_is_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      payload = json.loads(self.eval_manifest.read_text())
      payload["training_bank_sha256"] = "b" * 64
      self.eval_manifest.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "lineage differs"):
        binder.build(cfg)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      source = Path(
        json.loads(self.checkpoint_index.read_text())["manifests"][0]["path"]
      )
      source.write_text("{}")
      with self.assertRaisesRegex(ValueError, "checkpoint manifest changed"):
        binder.build(cfg)

  def test_index_detects_binding_tampering(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      index = binder.write_bindings(cfg)
      path = Path(index["manifests"][0]["path"])
      payload = json.loads(path.read_text())
      payload["claim_boundary"] = "tampered"
      path.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "changed after indexing"):
        binder.write_bindings(cfg)


if __name__ == "__main__":
  unittest.main()
