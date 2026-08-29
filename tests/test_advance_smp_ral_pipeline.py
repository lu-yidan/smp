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
  ) -> tuple[dict, mock.Mock]:
    completed_gates = completed_gates or set()
    launcher = mock.Mock(return_value={"pid": 12345})
    with (
      mock.patch.object(pipeline, "inspect", return_value=_health(completed=completed)),
      mock.patch.object(
        pipeline, "_ensure_manifests", return_value=(self.manifests, [])
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


class CompletionValidationTest(unittest.TestCase):
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
