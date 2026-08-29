from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import evaluate_smp_baseline as evaluator
import run_smp_frozen_eval_matrix as matrix


def _baseline_env():
  return SimpleNamespace(events={"init_matched_reset_bank": SimpleNamespace(params={})})


class MatchedEvalBindingTest(unittest.TestCase):
  def _fixture(self, root: Path, mode: str = "prone"):
    bank = root / f"{mode}.pt"
    bank.write_bytes(b"bank")
    index = ("native_gsi", "prone", "supine", "left_side", "right_side").index(mode)
    counts = [0] * 5
    counts[index] = 512
    manifest = root / "held_out.json"
    manifest.write_text(
      json.dumps(
        {
          "status": "READY",
          "generation_seed": 20260829,
          "num_states_per_mode": 512,
          "modes": [
            "native_gsi",
            "prone",
            "supine",
            "left_side",
            "right_side",
          ],
          "training_bank_sha256": "a" * 64,
          "exact_training_overlap_count": 0,
          "banks": {
            mode: {
              "path": str(bank),
              "sha256": evaluator._sha256(bank),
              "num_states": 512,
              "reset_type_counts": counts,
            }
          },
        }
      )
    )
    cfg = evaluator.EvalCfg(
      checkpoint=root / "model.pt",
      task="Baseline",
      reset_mode=mode,
      matched_eval_manifest=manifest,
      matched_eval_manifest_sha256=evaluator._sha256(manifest),
    )
    return cfg, manifest, bank

  def test_native_baseline_requires_and_binds_exact_mode_bank(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg, manifest, bank = self._fixture(root)
      env = _baseline_env()
      provenance = evaluator._configure_matched_eval_bank(env, cfg)
      self.assertEqual(provenance["manifest"], str(manifest.resolve()))
      self.assertEqual(provenance["bank"], str(bank.resolve()))
      self.assertEqual(
        env.events["init_matched_reset_bank"].params["expected_num_states"], 512
      )
      self.assertEqual(
        env.events["init_matched_reset_bank"].params["sampling_seed"], 20260829
      )
    with self.assertRaisesRegex(ValueError, "requires a matched held-out"):
      evaluator._configure_matched_eval_bank(
        _baseline_env(), evaluator.EvalCfg(checkpoint=Path("model.pt"))
      )

  def test_changed_manifest_or_wrong_reset_counts_are_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg, manifest, _bank = self._fixture(root)
      cfg = evaluator.EvalCfg(
        **{
          **cfg.__dict__,
          "matched_eval_manifest_sha256": "0" * 64,
        }
      )
      with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
        evaluator._configure_matched_eval_bank(_baseline_env(), cfg)
      cfg, manifest, _bank = self._fixture(root)
      payload = json.loads(manifest.read_text())
      payload["banks"]["prone"]["reset_type_counts"] = [512, 0, 0, 0, 0]
      manifest.write_text(json.dumps(payload))
      cfg = evaluator.EvalCfg(
        **{
          **cfg.__dict__,
          "matched_eval_manifest_sha256": evaluator._sha256(manifest),
        }
      )
      with self.assertRaisesRegex(ValueError, "bank or provenance changed"):
        evaluator._configure_matched_eval_bank(_baseline_env(), cfg)

  def test_frozen_matrix_rejects_blocked_manifest_and_changed_checkpoint(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      checkpoint = root / "model.pt"
      checkpoint.write_bytes(b"checkpoint")
      manifest = root / "manifest.json"
      manifest.write_text(
        json.dumps(
          {
            "evaluation_status": "BLOCKED_ON_MATCHED_HELD_OUT_RESET_BANK",
            "runs": [
              {
                "name": "method",
                "task": "Task",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": matrix._sha256(checkpoint),
              }
            ],
          }
        )
      )
      with self.assertRaisesRegex(ValueError, "not evaluation-ready"):
        matrix._load_manifest(manifest)
      payload = json.loads(manifest.read_text())
      payload.pop("evaluation_status")
      payload["runs"][0]["checkpoint_sha256"] = "0" * 64
      manifest.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "checkpoint changed"):
        matrix._load_manifest(manifest)


if __name__ == "__main__":
  unittest.main()
