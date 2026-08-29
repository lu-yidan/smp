from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import aggregate_smp_specialist_seeds as aggregator


class SpecialistAggregateTest(unittest.TestCase):
  def _fixture(self, root: Path, phase: str = "T") -> aggregator.SpecialistAggregateCfg:
    protocol = Path(__file__).parents[1] / "docs/ral_terrain_plate_protocol.json"
    analyses = []
    for seed in (20260901, 20260902, 20260903):
      summary = root / f"summary_{seed}.json"
      summary.write_text(json.dumps({"seed": seed}))
      manifest = root / f"manifest_{seed}.json"
      manifest.write_text(json.dumps({"seed": seed}))
      metrics = {
        "invalid_dynamics_rate_max": 0.0,
        "secondary_fall_rate_max": 0.01,
      }
      if phase == "T":
        metrics.update(
          {
            "flat_macro": 0.9,
            "level1_each_terrain_macro": {
              "slope": 0.7,
              "stairs": 0.6,
              "rough": 0.8,
            },
          }
        )
      else:
        metrics.update(
          {
            "unpinned_flat_macro": 0.9,
            "plate_pose_macro": {"prone": 0.8, "supine": 0.75},
          }
        )
      analysis = root / f"analysis_{seed}.json"
      analysis.write_text(
        json.dumps(
          {
            "schema_version": 1,
            "status": "PASS",
            "phase": phase,
            "policy_seed": seed,
            "checkpoint_step": 19999,
            "arm": "a6_f2s2_mix_bridge",
            "summary": str(summary.resolve()),
            "summary_sha256": aggregator._sha256(summary),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": aggregator._sha256(manifest),
            "protocol": str(protocol.resolve()),
            "protocol_sha256": aggregator._sha256(protocol),
            "metrics": metrics,
            "safety_ratio_max": 1.1,
          }
        )
      )
      analyses.append(analysis)
    return aggregator.SpecialistAggregateCfg(
      analyses=tuple(analyses),
      output_json=root / "aggregate.json",
      protocol=protocol,
      bootstrap_replicates=100,
    )

  def test_three_passing_final_seeds_promote_phase(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      result = aggregator.aggregate(self._fixture(Path(temporary)))
      self.assertEqual(result["status"], "PHASE_PASS")
      self.assertEqual(result["policy_seed_count"], 3)
      self.assertIn("level1_each_terrain_worst", result["metrics"])

  def test_one_failed_seed_cannot_be_hidden_by_mean(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary), phase="P")
      failed = json.loads(cfg.analyses[1].read_text())
      failed["status"] = "NO_PROMOTION"
      cfg.analyses[1].write_text(json.dumps(failed))
      result = aggregator.aggregate(cfg)
      self.assertEqual(result["status"], "NO_PROMOTION")
      self.assertEqual(result["each_seed_status"]["20260902"], "NO_PROMOTION")

  def test_nonfinal_checkpoint_cannot_promote(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      payload = json.loads(cfg.analyses[0].read_text())
      payload["checkpoint_step"] = 10000
      cfg.analyses[0].write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "final checkpoint"):
        aggregator.aggregate(cfg)


if __name__ == "__main__":
  unittest.main()
