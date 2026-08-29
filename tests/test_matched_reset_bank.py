from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from generate_smp_matched_reset_bank import (
  ResetBankCfg,
  _materialize_runtime_registry,
  _sha256,
  _validate_reset_distribution,
  build_plan,
)
from smp.rl.tasks.getup.mdp.events import _validate_matched_reset_bank_payload


def _payload(num_states: int = 8) -> dict[str, torch.Tensor]:
  root = torch.zeros(num_states, 13)
  root[:, 3] = 1.0
  joint = torch.randn(num_states, 29)
  window = torch.zeros(num_states, 10, 59)
  window[:, -1, 9:38] = joint
  return {
    "root_state": root,
    "joint_pos": joint,
    "joint_vel": torch.zeros_like(joint),
    "smp_window": window,
    "reset_type": torch.arange(num_states, dtype=torch.int8) % 5,
  }


class MatchedResetBankTest(unittest.TestCase):
  def setUp(self) -> None:
    self.repo_root = Path(__file__).parents[1]

  def test_valid_payload_locks_current_state_and_history(self) -> None:
    self.assertEqual(_validate_matched_reset_bank_payload(_payload(), 8), 8)

  def test_invalid_shape_nonfinite_quaternion_and_history_are_rejected(self) -> None:
    shape = _payload()
    shape["joint_vel"] = shape["joint_vel"][:, :-1]
    with self.assertRaisesRegex(ValueError, "joint_vel has invalid shape"):
      _validate_matched_reset_bank_payload(shape)
    nonfinite = _payload()
    nonfinite["root_state"][0, 0] = torch.nan
    with self.assertRaisesRegex(ValueError, "finite"):
      _validate_matched_reset_bank_payload(nonfinite)
    quaternion = _payload()
    quaternion["root_state"][0, 3:7] = 0.0
    with self.assertRaisesRegex(ValueError, "quaternion"):
      _validate_matched_reset_bank_payload(quaternion)
    history = _payload()
    history["smp_window"][0, -1, 9] += 1.0
    with self.assertRaisesRegex(ValueError, "joint position disagree"):
      _validate_matched_reset_bank_payload(history)

  def test_reset_distribution_must_match_frozen_mixture(self) -> None:
    reset_type = torch.tensor([0] * 800 + [1] * 50 + [2] * 50 + [3] * 50 + [4] * 50)
    self.assertEqual(
      _validate_reset_distribution(reset_type, 0.2, (1.0, 1.0, 1.0, 1.0)),
      [800, 50, 50, 50, 50],
    )
    with self.assertRaisesRegex(ValueError, "procedural share"):
      _validate_reset_distribution(reset_type, 0.0, None)
    imbalanced = torch.tensor([0] * 800 + [1] * 200)
    with self.assertRaisesRegex(ValueError, "pose balance"):
      _validate_reset_distribution(imbalanced, 0.2, None)

  def test_runtime_registry_is_immutable_and_hash_locked(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      template = root / "template.json"
      output = root / "runtime.json"
      bank = root / "bank.pt"
      manifest = root / "manifest.json"
      template.write_text(
        json.dumps(
          {
            "shared_reset_bank": {
              "status": "missing",
              "result_path": None,
              "sha256": None,
            }
          }
        )
      )
      bank.write_bytes(b"bank")
      manifest.write_text("{}")
      bank_hash = hashlib.sha256(bank.read_bytes()).hexdigest()
      manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
      _materialize_runtime_registry(
        template, output, bank, bank_hash, manifest, manifest_hash
      )
      payload = json.loads(output.read_text())
      self.assertEqual(payload["shared_reset_bank"]["sha256"], bank_hash)
      _materialize_runtime_registry(
        template, output, bank, bank_hash, manifest, manifest_hash
      )
      changed = copy.deepcopy(payload)
      changed["shared_reset_bank"]["sha256"] = "0" * 64
      output.write_text(json.dumps(changed))
      with self.assertRaisesRegex(ValueError, "conflicts"):
        _materialize_runtime_registry(
          template, output, bank, bank_hash, manifest, manifest_hash
        )

  def test_plan_binds_promoted_arm_prior_and_source_hashes(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      aggregate = root / "aggregate.json"
      confirmation = root / "confirmation.json"
      protocol = root / "protocol.json"
      for path in (aggregate, confirmation, protocol):
        path.write_text("{}")
      promotion = root / "promotion.json"
      promotion.write_text(
        json.dumps(
          {
            "status": "PROMOTE_TP_SPECIALISTS",
            "selected_arm": "a6_f2s2_mix_bridge",
            "promotion_id": "promoted-flat-arm",
            "aggregate": str(aggregate),
            "aggregate_sha256": _sha256(aggregate),
            "confirmation_manifest_index": str(confirmation),
            "confirmation_manifest_index_sha256": _sha256(confirmation),
            "protocol": str(protocol),
            "protocol_sha256": _sha256(protocol),
          }
        )
      )
      plan = build_plan(
        ResetBankCfg(
          promotion=promotion,
          registry_template=self.repo_root / "docs/ral_baseline_registry.json",
        )
      )
      self.assertEqual(plan["task"], "Smp-Getup-Scratch-A6-F2S2-Mix-Bridge-G1")
      self.assertEqual(plan["procedural_probability"], 0.2)
      self.assertEqual((plan["window_size"], plan["feature_dim"]), (10, 59))
      self.assertEqual(len(plan["prior_sha256"]), 64)
      changed = json.loads(promotion.read_text())
      changed["aggregate_sha256"] = "0" * 64
      promotion.write_text(json.dumps(changed))
      with self.assertRaisesRegex(ValueError, "aggregate source changed"):
        build_plan(
          ResetBankCfg(
            promotion=promotion,
            registry_template=self.repo_root / "docs/ral_baseline_registry.json",
          )
        )


if __name__ == "__main__":
  unittest.main()
