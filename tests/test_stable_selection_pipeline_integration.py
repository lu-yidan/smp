from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import advance_smp_ral_pipeline as pipeline


class StableSelectionPipelineTest(unittest.TestCase):
  def test_complete_gates_emit_stable_selection(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = pipeline.PipelineCfg(
        control_dir=root / "control",
        evidence_dir=root / "evidence",
        state=root / "state.json",
        launch_when_ready=True,
      )
      health = {
        "healthy": True,
        "jobs": [
          {
            "log": f"gpu{index}.log",
            "iteration": 29999,
            "completed": True,
          }
          for index in range(8)
        ],
      }
      manifests = [
        {
          "gate": gate,
          "path": str(root / f"evidence/manifests/gate_{gate}.json"),
          "sha256": f"hash-{gate}",
        }
        for gate in cfg.gates
      ]
      selection = {
        "status": "PROMOTE_FOR_POLICY_SEEDS",
        "promoted_candidates": ["arm_a"],
      }
      with (
        mock.patch.object(pipeline, "inspect", return_value=health),
        mock.patch.object(pipeline, "_ensure_manifests", return_value=(manifests, [])),
        mock.patch.object(pipeline, "_active_eval", return_value=None),
        mock.patch.object(pipeline, "_gpu_processes", return_value=[]),
        mock.patch.object(pipeline, "_analysis_complete", return_value=True),
        mock.patch.object(pipeline, "write_selection", return_value=selection),
      ):
        state = pipeline.advance(cfg)
      self.assertEqual(state["status"], "ANALYSIS_COMPLETE")
      self.assertEqual(state["stable_selection"]["promoted_candidates"], ["arm_a"])


if __name__ == "__main__":
  unittest.main()
