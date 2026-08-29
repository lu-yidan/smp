from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import select_smp_unified_prerequisites as selector


class UnifiedPrerequisiteTest(unittest.TestCase):
  def _fixture(
    self, root: Path, plate_status: str = "PHASE_PASS"
  ) -> selector.UnifiedPrerequisiteCfg:
    protocol = Path(__file__).parents[1] / "docs/ral_terrain_plate_protocol.json"
    flat_promotion = root / "flat_promotion.json"
    flat_promotion.write_text(json.dumps({"promotion_id": "promotion"}))
    launch = root / "launch.json"
    launch.write_text(
      json.dumps(
        {
          "promotion": str(flat_promotion),
          "promotion_sha256": selector._sha256(flat_promotion),
        }
      )
    )

    def make_aggregate(phase: str, status: str) -> Path:
      sources = []
      for seed in (20260901, 20260902, 20260903):
        manifest = root / f"manifest_{phase}_{seed}.json"
        manifest.write_text(
          json.dumps(
            {
              "launch_plan_id": "launch-plan",
              "promotion_id": "promotion",
              "launch_manifest": str(launch),
              "launch_manifest_sha256": selector._sha256(launch),
            }
          )
        )
        analysis = root / f"analysis_{phase}_{seed}.json"
        analysis.write_text(
          json.dumps(
            {
              "manifest": str(manifest),
              "manifest_sha256": selector._sha256(manifest),
            }
          )
        )
        sources.append({"path": str(analysis), "sha256": selector._sha256(analysis)})
      aggregate = root / f"aggregate_{phase}.json"
      aggregate.write_text(
        json.dumps(
          {
            "phase": phase,
            "checkpoint_step": 19999,
            "status": status,
            "arm": "a6_f2s2_mix_bridge",
            "policy_seeds": [20260901, 20260902, 20260903],
            "protocol": str(protocol.resolve()),
            "protocol_sha256": selector._sha256(protocol),
            "source_analyses": sources,
          }
        )
      )
      return aggregate

    return selector.UnifiedPrerequisiteCfg(
      terrain_aggregate=make_aggregate("T", "PHASE_PASS"),
      plate_aggregate=make_aggregate("P", plate_status),
      output=root / "unified.json",
      protocol=protocol,
    )

  def test_matched_t_and_p_phase_passes_promote_u_budget(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      result = selector.select(self._fixture(Path(temporary)))
      self.assertEqual(result["status"], "PROMOTE_U")
      self.assertEqual(result["specialist_launch_plan_id"], "launch-plan")

  def test_failed_plate_phase_blocks_u(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      result = selector.select(
        self._fixture(Path(temporary), plate_status="NO_PROMOTION")
      )
      self.assertEqual(result["status"], "NO_PROMOTION")

  def test_different_arm_lineage_is_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      payload = json.loads(cfg.plate_aggregate.read_text())
      payload["arm"] = "a7_v7_mix_bridge"
      cfg.plate_aggregate.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "flat arm lineage"):
        selector.select(cfg)


if __name__ == "__main__":
  unittest.main()
