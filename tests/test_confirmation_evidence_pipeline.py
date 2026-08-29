from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import advance_smp_ral_pipeline as pipeline


def _health() -> dict:
  return {
    "healthy": True,
    "jobs": [{"log": f"gpu{index}.log", "completed": True} for index in range(3)],
  }


class ConfirmationEvidencePipelineTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    root = Path(self.temporary.name)
    self.cfg = pipeline.PipelineCfg(
      control_dir=root / "screen",
      evidence_dir=root / "screen_eval",
      state=root / "state.json",
      confirmation_control_dir=root / "confirmation",
      confirmation_evidence_dir=root / "confirmation_eval",
      launch_when_ready=True,
    )
    self.confirmation = {"jobs": [{"pid": 100 + index} for index in range(3)]}
    self.index = {
      "policy_seeds": [11, 12, 13],
      "manifests": [
        {
          "policy_seed": seed,
          "path": str(root / f"confirmation_eval/manifests/seed_{seed}.json"),
        }
        for seed in (11, 12, 13)
      ],
    }

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def test_live_confirmation_jobs_do_not_build_manifests(self) -> None:
    with (
      mock.patch.object(pipeline, "_pid_alive", return_value=True),
      mock.patch.object(pipeline, "write_confirmation_manifests") as writer,
    ):
      result = pipeline._advance_confirmation(self.cfg, self.confirmation, ["1"])
    self.assertEqual(result["status"], "CONFIRMATION_TRAINING")
    writer.assert_not_called()

  def test_completed_training_launches_earliest_seed_matrix(self) -> None:
    launched = {"pid": 999}
    with (
      mock.patch.object(pipeline, "_pid_alive", return_value=False),
      mock.patch.object(pipeline, "inspect", return_value=_health()),
      mock.patch.object(
        pipeline, "write_confirmation_manifests", return_value=self.index
      ),
      mock.patch.object(pipeline, "_active_eval", return_value=None),
      mock.patch.object(pipeline, "_analysis_complete", return_value=False),
      mock.patch.object(
        pipeline, "_launch_evaluation", return_value=launched
      ) as launcher,
    ):
      result = pipeline._advance_confirmation(self.cfg, self.confirmation, [])
    self.assertEqual(result["status"], "CONFIRMATION_EVAL_RUNNING")
    self.assertEqual(result["active_evaluation"], launched)
    self.assertEqual(launcher.call_args.args[2], 11)

  def test_complete_seed_matrices_write_policy_level_aggregate(self) -> None:
    aggregate = {"status": "MINIMUM_POLICY_SEEDS_MET", "policy_seeds": [11, 12, 13]}
    with (
      mock.patch.object(pipeline, "_pid_alive", return_value=False),
      mock.patch.object(pipeline, "inspect", return_value=_health()),
      mock.patch.object(
        pipeline, "write_confirmation_manifests", return_value=self.index
      ),
      mock.patch.object(pipeline, "_active_eval", return_value=None),
      mock.patch.object(pipeline, "_analysis_complete", return_value=True),
      mock.patch.object(pipeline, "write_aggregate", return_value=aggregate),
    ):
      result = pipeline._advance_confirmation(self.cfg, self.confirmation, [])
    self.assertEqual(result["status"], "CONFIRMATION_ANALYSIS_COMPLETE")
    self.assertEqual(result["aggregate"]["status"], "MINIMUM_POLICY_SEEDS_MET")


if __name__ == "__main__":
  unittest.main()
