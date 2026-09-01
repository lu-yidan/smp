from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import advance_smp_flat_objective_alignment as advance


class FlatObjectiveAdvanceTest(unittest.TestCase):
  def _cfg(self, root: Path, launch: bool = False) -> advance.FlatObjectiveAdvanceCfg:
    training = root / "training"
    training.mkdir()
    (training / "launch_manifest.json").write_text("{}")
    return advance.FlatObjectiveAdvanceCfg(
      protocol=root / "protocol.json",
      training_control_dir=training,
      manifest_dir=root / "manifests",
      evaluation_root=root / "formal",
      analysis_json=root / "analysis.json",
      analysis_markdown=root / "analysis.md",
      state=root / "state.json",
      logs_root=root / "logs",
      launch_evaluations_when_ready=launch,
    )

  def _launch(self) -> dict:
    return {
      "jobs": [
        {
          "arm": "a6_replication_control",
          "policy_seed": 20261101,
          "gpu": 0,
          "pid": 100,
          "log": "/tmp/job.log",
          "run_name": "test",
        }
      ]
    }

  def _health(self, *, alive: bool, final: bool, error: str | None = None) -> list[dict]:
    return [
      {
        "pid": 100,
        "process_alive": alive,
        "final_checkpoint_present": final,
        "error_match": error,
        "log_exists": True,
        "log_age_minutes": 0.0,
      }
    ]

  def _common(self, health: list[dict]):
    index = {
      "status": "READY_FOR_FROZEN_EVALUATION",
      "index_id": "index",
      "manifests": [{}] * 12,
      "checkpoint_entry_count": 24,
    }
    rows = {
      (seed, gate): {"path": f"/tmp/manifest-{seed}-{gate}.json", "sha256": "x"}
      for seed in advance._SEEDS
      for gate in advance._GATES
    }
    return mock.patch.multiple(
      advance,
      _git=mock.DEFAULT,
      _validate_protocol=mock.DEFAULT,
      _validate_launch=mock.DEFAULT,
      _training_health=mock.DEFAULT,
      write_manifests=mock.DEFAULT,
      _validate_index=mock.DEFAULT,
      _gpu_processes=mock.DEFAULT,
    ), index, rows

  def _configure_common(self, mocks, health, index, rows) -> None:
    mocks["_git"].side_effect = lambda _repo, *args: (
      "" if args and args[0] == "status" else "deadbeef"
    )
    mocks["_validate_protocol"].return_value = {}
    mocks["_validate_launch"].return_value = (self._launch(), {})
    mocks["_training_health"].return_value = health
    mocks["write_manifests"].return_value = index
    mocks["_validate_index"].return_value = (index, rows)
    mocks["_gpu_processes"].return_value = []

  def test_training_active_and_dead_incomplete_are_distinct(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._cfg(Path(temporary))
      patcher, index, rows = self._common(self._health(alive=True, final=False))
      with patcher as mocks:
        self._configure_common(mocks, self._health(alive=True, final=False), index, rows)
        result = advance.advance(cfg)
      self.assertEqual(result["status"], "FLAT_OBJECTIVE_TRAINING_ACTIVE")
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._cfg(Path(temporary))
      patcher, index, rows = self._common(self._health(alive=False, final=False))
      with patcher as mocks:
        self._configure_common(mocks, self._health(alive=False, final=False), index, rows)
        result = advance.advance(cfg)
      self.assertEqual(result["status"], "FLAT_OBJECTIVE_TRAINING_ALERT")
      self.assertTrue(result["alert"]["automatic_restart_forbidden"])

  def test_complete_training_reaches_eval_ready_and_launches_once(self) -> None:
    for launch, expected in (
      (False, "FLAT_OBJECTIVE_READY_FOR_FROZEN_EVALUATION"),
      (True, "FLAT_OBJECTIVE_EVALUATION_LAUNCHED"),
    ):
      with self.subTest(launch=launch), tempfile.TemporaryDirectory() as temporary:
        cfg = self._cfg(Path(temporary), launch=launch)
        patcher, index, rows = self._common(self._health(alive=False, final=True))
        with patcher as mocks, mock.patch.object(
          advance, "_disk_preflight", return_value={"free_gib": 1000, "inode_free_fraction": 0.9}
        ), mock.patch.object(
          advance,
          "_launch_matrix",
          return_value={"pid": 123, "attempt": 1},
        ) as launcher:
          self._configure_common(mocks, self._health(alive=False, final=True), index, rows)
          result = advance.advance(cfg)
        self.assertEqual(result["status"], expected)
        self.assertEqual(launcher.call_count, 1 if launch else 0)

  def test_dead_incomplete_evaluator_is_not_restarted(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._cfg(root, launch=True)
      cfg.evaluation_root.mkdir()
      marker = {
        "plan_id": advance._PLAN_ID,
        "protocol_sha256": advance._PROTOCOL_SHA256,
        "attempt": 1,
        "pid": 999,
        "policy_seed": 20261101,
        "checkpoint_step": 8000,
      }
      marker_path = cfg.evaluation_root / "active_evaluation.json"
      marker_path.write_text(json.dumps(marker))
      patcher, index, rows = self._common(self._health(alive=False, final=True))
      with patcher as mocks, mock.patch.object(
        advance, "_pid_alive", return_value=False
      ), mock.patch.object(
        advance, "_audit_matrix", side_effect=ValueError("incomplete")
      ), mock.patch.object(advance, "_launch_matrix") as launcher:
        self._configure_common(mocks, self._health(alive=False, final=True), index, rows)
        result = advance.advance(cfg)
      self.assertEqual(result["status"], "FLAT_OBJECTIVE_EVAL_ALERT")
      self.assertTrue(result["automatic_restart_forbidden"])
      self.assertTrue(marker_path.is_file())
      launcher.assert_not_called()

  def test_training_health_parses_iteration_throughput_and_wandb(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._cfg(root)
      log = root / "job.log"
      log.write_text(
        "wandb.ai/tabletennis/smp/runs/abc123\n"
        "Learning iteration 123/30000\n"
        "Steps per second: 45,678\n"
      )
      run = root / "run"
      run.mkdir()
      (run / "model_29999.pt").write_bytes(b"checkpoint")
      launch = {
        "jobs": [
          {
            "arm": "a6_replication_control",
            "policy_seed": 20261101,
            "gpu": 0,
            "pid": 100,
            "log": str(log),
            "run_name": "test",
          }
        ]
      }
      with mock.patch.object(advance, "_pid_alive", return_value=True), mock.patch.object(
        advance, "_discover_run", return_value=run
      ):
        rows = advance._training_health(cfg, launch, datetime.now(timezone.utc))
      self.assertEqual(rows[0]["latest_iteration"], 123)
      self.assertEqual(rows[0]["latest_checkpoint_iteration"], 29999)
      self.assertEqual(rows[0]["progress_iteration_lower_bound"], 29999)
      self.assertEqual(rows[0]["latest_throughput_steps_s"], 45678)
      self.assertEqual(rows[0]["wandb_run_id"], "abc123")
      self.assertTrue(rows[0]["final_checkpoint_present"])

  def test_training_health_reports_checkpoint_progress_when_log_lags(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._cfg(root)
      log = root / "job.log"
      log.write_text("Learning iteration 2453/30000\n")
      run = root / "run"
      run.mkdir()
      (run / "model_2000.pt").write_bytes(b"older")
      (run / "model_3000.pt").write_bytes(b"newer")
      launch = {
        "jobs": [
          {
            "arm": "a6_replication_control",
            "policy_seed": 20261101,
            "gpu": 0,
            "pid": 100,
            "log": str(log),
            "run_name": "test",
          }
        ]
      }
      with mock.patch.object(advance, "_pid_alive", return_value=True), mock.patch.object(
        advance, "_discover_run", return_value=run
      ):
        rows = advance._training_health(cfg, launch, datetime.now(timezone.utc))
      self.assertEqual(rows[0]["latest_iteration"], 2453)
      self.assertEqual(rows[0]["latest_checkpoint_iteration"], 3000)
      self.assertEqual(rows[0]["progress_iteration_lower_bound"], 3000)

  def test_matrix_launch_records_one_immutable_attempt(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._cfg(root, launch=True)
      manifest = root / "manifest.json"
      manifest.write_text("{}")
      process = mock.Mock(pid=321)
      with mock.patch.object(advance.subprocess, "Popen", return_value=process) as popen:
        marker = advance._launch_matrix(
          cfg,
          manifest,
          20261101,
          8000,
          root / "formal" / "gate_8000" / "seed_20261101",
          {"free_gib": 1000, "inode_free_fraction": 0.9},
        )
      self.assertEqual(marker["attempt"], 1)
      self.assertIn("--include-per-env", marker["command"])
      self.assertEqual(marker["devices"], list(cfg.devices))
      self.assertTrue((cfg.evaluation_root / "active_evaluation.json").is_file())
      popen.assert_called_once()


if __name__ == "__main__":
  unittest.main()
