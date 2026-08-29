from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import advance_smp_ral_pipeline as pipeline


class ResetBankPipelineControlTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.cfg = pipeline.PipelineCfg(
      control_dir=self.root / "control",
      evidence_dir=self.root / "evidence",
      state=self.root / "state.json",
      baseline_control_dir=self.root / "baselines",
      baseline_registry_template=self.root / "registry_template.json",
      baseline_runtime_registry=self.root / "baselines/registry.json",
      baseline_bank_output=self.root / "baselines/bank.pt",
      baseline_bank_manifest=self.root / "baselines/bank.json",
      launch_baseline_bank_when_ready=True,
    )
    self.promotion = self.root / "promotion.json"
    self.promotion.write_text("{}")

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def test_launch_is_backgrounded_and_records_exact_plan(self) -> None:
    process = mock.Mock(pid=12345)
    with (
      mock.patch.object(
        pipeline, "build_reset_bank_plan", return_value={"plan_id": "plan"}
      ),
      mock.patch.object(pipeline.subprocess, "Popen", return_value=process) as popen,
    ):
      result = pipeline._launch_reset_bank(self.cfg, self.promotion)
    self.assertEqual(result["pid"], 12345)
    self.assertEqual(result["plan_id"], "plan")
    command = popen.call_args.args[0]
    self.assertIn("--run", command)
    self.assertIn("cuda:6", command)
    state = json.loads(
      (self.cfg.baseline_control_dir / "active_generation.json").read_text()
    )
    self.assertEqual(state["pid"], 12345)

  def test_dead_incomplete_generator_remains_fail_closed(self) -> None:
    self.cfg.baseline_control_dir.mkdir(parents=True)
    state = self.cfg.baseline_control_dir / "active_generation.json"
    state.write_text(json.dumps({"pid": 99, "plan_id": "plan"}))
    with (
      mock.patch.object(pipeline, "_pid_alive", return_value=False),
      mock.patch.object(
        pipeline,
        "_validate_reset_bank_artifacts",
        side_effect=ValueError("bad bank"),
      ),
    ):
      result = pipeline._active_reset_bank(self.cfg, self.promotion)
    self.assertTrue(result["failed"])
    self.assertIn("bad bank", result["error"])
    self.assertTrue(state.is_file())

  def test_promoted_pipeline_waits_or_launches_bank_before_tp_smoke(self) -> None:
    promotion = {"status": "PROMOTE_TP_SPECIALISTS", "selected_arm": "a6"}
    with (
      mock.patch.object(pipeline, "write_flat_selection", return_value=promotion),
      mock.patch.object(pipeline, "_active_reset_bank", return_value=None),
      mock.patch.object(pipeline, "_validate_reset_bank_artifacts", return_value=None),
      mock.patch.object(
        pipeline, "_launch_reset_bank", return_value={"pid": 7}
      ) as launch,
    ):
      waiting = pipeline._advance_specialists(self.cfg, ["training-pid"])
      launched = pipeline._advance_specialists(self.cfg, [])
    self.assertEqual(waiting["status"], "WAITING_FREE_GPU")
    self.assertEqual(launched["status"], "RESET_BANK_RUNNING")
    launch.assert_called_once()


if __name__ == "__main__":
  unittest.main()
