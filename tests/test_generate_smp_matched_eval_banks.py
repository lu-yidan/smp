from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import generate_smp_matched_eval_banks as generator


def _payload(count: int, offset: float = 0.0, reset_type: int = 0):
  root = torch.zeros(count, 13)
  root[:, 0] = torch.arange(count) + offset
  root[:, 3] = 1.0
  joint = torch.zeros(count, 29)
  velocity = torch.zeros_like(joint)
  window = torch.zeros(count, 10, 59)
  window[:, -1, 9:38] = joint
  return {
    "root_state": root,
    "joint_pos": joint,
    "joint_vel": velocity,
    "smp_window": window,
    "reset_type": torch.full((count,), reset_type, dtype=torch.int8),
  }


class MatchedEvalBankTest(unittest.TestCase):
  def test_plan_binds_promotion_training_bank_prior_and_registry(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      sources = {}
      for name in ("aggregate", "confirmation", "protocol"):
        path = root / f"{name}.json"
        path.write_text("{}")
        sources[name] = (path, generator._sha256(path))
      promotion = root / "promotion.json"
      promotion.write_text(
        json.dumps(
          {
            "status": "PROMOTE_TP_SPECIALISTS",
            "promotion_id": "promotion",
            "selected_arm": "selected",
            "aggregate": str(sources["aggregate"][0]),
            "aggregate_sha256": sources["aggregate"][1],
            "confirmation_manifest_index": str(sources["confirmation"][0]),
            "confirmation_manifest_index_sha256": sources["confirmation"][1],
            "protocol": str(sources["protocol"][0]),
            "protocol_sha256": sources["protocol"][1],
          }
        )
      )
      training_bank = root / "training.pt"
      training_bank.write_bytes(b"training")
      training_manifest = root / "training.json"
      training_manifest.write_text(
        json.dumps(
          {
            "status": "READY",
            "promotion_id": "promotion",
            "bank": str(training_bank.resolve()),
            "bank_sha256": generator._sha256(training_bank),
          }
        )
      )
      registry = root / "registry.json"
      registry.write_text(
        json.dumps(
          {
            "held_out_evaluation_banks": {
              "status": "preregistered",
              "generation_seed": 20260829,
              "num_states_per_mode": 512,
              "modes": list(generator._MODES),
              "training_bank_disjoint_required": True,
            }
          }
        )
      )
      prior = root / "prior.pt"
      prior.write_bytes(b"prior")
      env_cfg = SimpleNamespace(
        events={"init_smp_state": SimpleNamespace(params={"ckpt_path": str(prior)})}
      )
      cfg = generator.EvalBankCfg(
        promotion=promotion,
        training_bank=training_bank,
        training_bank_manifest=training_manifest,
        registry=registry,
        output_dir=root / "output",
        manifest=root / "manifest.json",
      )
      with (
        mock.patch.object(generator, "_ARMS", ({"name": "selected", "task": "Task"},)),
        mock.patch.object(generator, "load_env_cfg", return_value=env_cfg),
        mock.patch.object(
          generator.torch,
          "load",
          return_value={"cfg": {"window_size": 10, "feature_dim": 59}},
        ),
        mock.patch.object(generator, "_git_commit", return_value="commit"),
      ):
        plan = generator.build_plan(cfg)
      self.assertEqual(plan["promotion_id"], "promotion")
      self.assertEqual(plan["training_bank_sha256"], generator._sha256(training_bank))
      self.assertEqual(plan["prior_sha256"], generator._sha256(prior))
      self.assertEqual(plan["modes"], list(generator._MODES))

  def test_each_mode_requires_exact_reset_family(self) -> None:
    for index, mode in enumerate(generator._MODES):
      payload = _payload(4, offset=10.0 * index, reset_type=index)
      counts = generator._validate_mode(payload, mode, 4)
      self.assertEqual(counts[index], 4)
    wrong = _payload(4, reset_type=2)
    with self.assertRaisesRegex(ValueError, "reset types drifted"):
      generator._validate_mode(wrong, "prone", 4)

  def test_exact_overlap_and_internal_duplicates_are_rejected(self) -> None:
    training = _payload(10)
    evaluations = {
      mode: _payload(2, offset=100.0 + 10.0 * index, reset_type=index)
      for index, mode in enumerate(generator._MODES)
    }
    self.assertEqual(generator._exact_overlap_count(training, evaluations), 0)
    evaluations["native_gsi"]["root_state"][0] = training["root_state"][3]
    evaluations["native_gsi"]["joint_pos"][0] = training["joint_pos"][3]
    evaluations["native_gsi"]["joint_vel"][0] = training["joint_vel"][3]
    self.assertEqual(generator._exact_overlap_count(training, evaluations), 1)
    duplicate = {
      mode: _payload(2, offset=100.0, reset_type=0) for mode in generator._MODES
    }
    with self.assertRaisesRegex(ValueError, "duplicate exact states"):
      generator._exact_overlap_count(training, duplicate)

  def test_partial_existing_artifacts_are_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = generator.EvalBankCfg(
        promotion=root / "promotion.json",
        output_dir=root / "banks",
        manifest=root / "manifest.json",
      )
      cfg.output_dir.mkdir()
      (cfg.output_dir / "native_gsi.pt").write_bytes(b"partial")
      with self.assertRaisesRegex(ValueError, "partial"):
        generator._validate_existing(cfg, {"plan_id": "plan"})


if __name__ == "__main__":
  unittest.main()
