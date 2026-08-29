from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import advance_smp_ral_pipeline as pipeline


def _health(*, completed: bool) -> dict:
  return {
    "healthy": True,
    "jobs": [
      {
        "log": f"gpu{index}.log",
        "iteration": 29999 if completed else 10000,
        "completed": completed,
      }
      for index in range(8)
    ],
  }


class PipelineStateTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    root = Path(self.temporary.name)
    self.cfg = pipeline.PipelineCfg(
      control_dir=root / "control",
      evidence_dir=root / "evidence",
      state=root / "state.json",
      gates=(8000, 15000),
      launch_when_ready=True,
    )
    self.manifests = [
      {
        "gate": 8000,
        "path": str(root / "evidence/manifests/gate_8000.json"),
        "sha256": "hash-8000",
      },
      {
        "gate": 15000,
        "path": str(root / "evidence/manifests/gate_15000.json"),
        "sha256": "hash-15000",
      },
    ]

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def _advance(
    self,
    *,
    completed: bool,
    gpu_processes: list[str],
    completed_gates: set[int] | None = None,
    active_evaluation: dict | None = None,
    pending_gates: list[int] | None = None,
  ) -> tuple[dict, mock.Mock]:
    completed_gates = completed_gates or set()
    pending_gates = pending_gates or []
    launcher = mock.Mock(return_value={"pid": 12345})
    with (
      mock.patch.object(pipeline, "inspect", return_value=_health(completed=completed)),
      mock.patch.object(
        pipeline, "_ensure_manifests", return_value=(self.manifests, pending_gates)
      ),
      mock.patch.object(pipeline, "_active_eval", return_value=active_evaluation),
      mock.patch.object(pipeline, "_gpu_processes", return_value=gpu_processes),
      mock.patch.object(
        pipeline,
        "_analysis_complete",
        side_effect=lambda _cfg, output, _manifest: (
          int(output.name.split("_")[-1]) in completed_gates
        ),
      ),
      mock.patch.object(pipeline, "_launch_evaluation", launcher),
    ):
      return pipeline.advance(self.cfg), launcher

  def test_training_never_launches_evaluation(self) -> None:
    state, launcher = self._advance(completed=False, gpu_processes=["101"])
    self.assertEqual(state["status"], "TRAINING_ACTIVE")
    launcher.assert_not_called()

  def test_completed_training_waits_for_gpu_processes(self) -> None:
    state, launcher = self._advance(completed=True, gpu_processes=["101"])
    self.assertEqual(state["status"], "WAITING_FREE_GPU")
    launcher.assert_not_called()

  def test_free_gpu_launches_only_earliest_incomplete_gate(self) -> None:
    state, launcher = self._advance(
      completed=True,
      gpu_processes=[],
      completed_gates={8000},
    )
    self.assertEqual(state["status"], "EVAL_RUNNING")
    self.assertEqual(state["active_evaluation"], {"pid": 12345})
    launcher.assert_called_once()
    self.assertEqual(launcher.call_args.args[2], 15000)

  def test_dead_incomplete_evaluator_is_alerted_not_relaunched(self) -> None:
    state, launcher = self._advance(
      completed=True,
      gpu_processes=[],
      active_evaluation={"gate": 8000, "pid": 999999, "process_alive": False},
    )
    self.assertEqual(state["status"], "EVAL_ALERT")
    self.assertEqual(state["active_evaluation"]["gate"], 8000)
    launcher.assert_not_called()

  def test_dead_successful_evaluator_clears_marker_and_advances(self) -> None:
    self.cfg.evidence_dir.mkdir(parents=True)
    (self.cfg.evidence_dir / "active_evaluation.json").write_text("{}")
    state, launcher = self._advance(
      completed=True,
      gpu_processes=[],
      completed_gates={8000},
      active_evaluation={"gate": 8000, "pid": 999999, "process_alive": False},
    )
    self.assertEqual(state["status"], "EVAL_RUNNING")
    self.assertFalse((self.cfg.evidence_dir / "active_evaluation.json").exists())
    self.assertEqual(launcher.call_args.args[2], 15000)

  def test_active_eval_preserves_dead_marker_for_reconciliation(self) -> None:
    self.cfg.evidence_dir.mkdir(parents=True)
    marker = self.cfg.evidence_dir / "active_evaluation.json"
    marker.write_text(json.dumps({"gate": 8000, "pid": 999999}))
    with mock.patch.object(pipeline, "_pid_alive", return_value=False):
      active = pipeline._active_eval(self.cfg.evidence_dir)
    self.assertFalse(active["process_alive"])
    self.assertTrue(marker.is_file())

  def test_completed_training_rejects_missing_gate_manifest(self) -> None:
    self.cfg = pipeline.PipelineCfg(
      control_dir=self.cfg.control_dir,
      evidence_dir=self.cfg.evidence_dir,
      state=self.cfg.state,
      gates=(8000, 15000, 25000),
      launch_when_ready=True,
    )
    state, launcher = self._advance(
      completed=True,
      gpu_processes=[],
      pending_gates=[25000],
    )
    self.assertEqual(state["status"], "MANIFESTS_INCOMPLETE_ALERT")
    launcher.assert_not_called()

  def test_completed_training_requires_versioned_manifest_lock(self) -> None:
    self.cfg = pipeline.PipelineCfg(
      control_dir=self.cfg.control_dir,
      evidence_dir=self.cfg.evidence_dir,
      state=self.cfg.state,
      gates=(8000, 15000, 29999),
      launch_when_ready=True,
    )
    self.manifests.append(
      {
        "gate": 29999,
        "path": str(self.cfg.evidence_dir / "manifests/gate_29999.json"),
        "sha256": "hash-29999",
      }
    )
    locks_without_final = {
      gate: sha256
      for gate, sha256 in pipeline._LOCKED_MANIFEST_HASHES.items()
      if gate != 29999
    }
    with mock.patch.object(
      pipeline,
      "_LOCKED_MANIFEST_HASHES",
      locks_without_final,
    ):
      state, launcher = self._advance(completed=True, gpu_processes=[])
    self.assertEqual(state["status"], "MANIFEST_LOCK_REQUIRED")
    self.assertEqual(state["unlocked_manifest_gates"], [29999])
    launcher.assert_not_called()


class CompletionValidationTest(unittest.TestCase):
  def test_expected_causal_manifest_hashes_are_versioned(self) -> None:
    self.assertEqual(
      pipeline._LOCKED_MANIFEST_HASHES,
      {
        8000: "64506f71e85b69b58bb5579621b10a8aa6969a172428c2d213115fb54a08c333",
        15000: "2e99d3af1bc6a3f9d5c01bf17f15d3588b73b673853af78a6dd711820382f7d9",
        25000: "1709b96d2a71f0821315cc98a43412a7f71af4181d367110233b59410ea93029",
        29999: "154d59b92489c73398918dd08a06b5057d4338be7bf5dd208b397db218a5a404",
      },
    )

  def test_rejects_modified_locked_15k_manifest(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      runs = []
      for index in range(8):
        checkpoint = root / f"model_{index}.pt"
        checkpoint.write_bytes(b"checkpoint")
        runs.append(
          {
            "name": f"arm_{index}",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": "checkpoint-hash",
            "policy_seed": 42,
            "environment_seed": 42,
          }
        )
      manifest = root / "gate_15000.json"
      manifest.write_text(
        json.dumps(
          {
            "checkpoint_step": 15000,
            "policy_seed": 42,
            "environment_seed": 42,
            "runs": runs,
          }
        )
      )

      def fake_sha256(path: Path) -> str:
        return "tampered-manifest" if path == manifest else "checkpoint-hash"

      with (
        mock.patch.object(pipeline, "_sha256", side_effect=fake_sha256),
        self.assertRaisesRegex(ValueError, "locked gate 15000 manifest hash changed"),
      ):
        pipeline._validate_manifest(manifest, 15000, 42, 42)

  def test_rejects_incomplete_or_wrong_protocol_marker(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      output = root / "gate_8000"
      output.mkdir()
      manifest = root / "manifest.json"
      manifest.write_text("{}")
      cfg = pipeline.PipelineCfg(
        control_dir=root / "control",
        evidence_dir=root / "evidence",
        state=root / "state.json",
      )
      for name in ("summary.json", "analysis.md"):
        (output / name).write_text("{}")
      (output / "analysis.json").write_text(
        json.dumps({"status": "SCREEN_PASS_NOT_FINAL"})
      )
      (output / "_COMPLETE.json").write_text(
        json.dumps(
          {
            "evaluation_schema_version": 1,
            "manifest": str(manifest.resolve()),
            "result_count": 40,
            "modes": list(cfg.modes),
            "eval_seeds": [cfg.eval_seed],
            "num_envs": cfg.num_envs,
            "steps": cfg.steps,
          }
        )
      )
      self.assertFalse(pipeline._analysis_complete(cfg, output, manifest))


if __name__ == "__main__":
  unittest.main()
