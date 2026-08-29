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
        mock.patch.object(
          pipeline,
          "_LOCKED_MANIFEST_HASHES",
          {gate: f"hash-{gate}" for gate in cfg.gates},
        ),
        mock.patch.object(pipeline, "_active_eval", return_value=None),
        mock.patch.object(pipeline, "_gpu_processes", return_value=[]),
        mock.patch.object(pipeline, "_analysis_complete", return_value=True),
        mock.patch.object(pipeline, "write_selection", return_value=selection),
      ):
        state = pipeline.advance(cfg)
      self.assertEqual(state["status"], "ANALYSIS_COMPLETE")
      self.assertEqual(state["stable_selection"]["promoted_candidates"], ["arm_a"])

  def test_existing_confirmation_is_monitored_while_its_gpus_are_busy(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = pipeline.PipelineCfg(
        control_dir=root / "control",
        evidence_dir=root / "evidence",
        state=root / "state.json",
        launch_when_ready=True,
        launch_confirmation_when_ready=True,
        confirmation_control_dir=root / "confirmation",
      )
      cfg.confirmation_control_dir.mkdir()
      (cfg.confirmation_control_dir / "launch_manifest.json").write_text("{}")
      health = {
        "healthy": True,
        "jobs": [
          {"log": f"gpu{index}.log", "iteration": 29999, "completed": True}
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
        "promoted_candidates": ["a0_f2s2_gsi"],
      }
      confirmation = {"status": "LAUNCHED", "plan_id": "p", "jobs": []}
      progress = {
        "status": "CONFIRMATION_TRAINING",
        "action": "still running",
      }
      with (
        mock.patch.object(pipeline, "inspect", return_value=health),
        mock.patch.object(pipeline, "_ensure_manifests", return_value=(manifests, [])),
        mock.patch.object(
          pipeline,
          "_LOCKED_MANIFEST_HASHES",
          {gate: f"hash-{gate}" for gate in cfg.gates},
        ),
        mock.patch.object(pipeline, "_active_eval", return_value=None),
        mock.patch.object(pipeline, "_gpu_processes", return_value=["101"]),
        mock.patch.object(pipeline, "_analysis_complete", return_value=True),
        mock.patch.object(pipeline, "write_selection", return_value=selection),
        mock.patch.object(
          pipeline, "launch_confirmation", return_value=confirmation
        ) as launch,
        mock.patch.object(
          pipeline, "_advance_confirmation", return_value=progress
        ) as advance,
      ):
        state = pipeline.advance(cfg)
      self.assertEqual(state["status"], "CONFIRMATION_TRAINING")
      launch.assert_called_once()
      advance.assert_called_once()


if __name__ == "__main__":
  unittest.main()
