from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import launch_smp_policy_seed_confirmation as launcher


class ConfirmationLaunchTest(unittest.TestCase):
  def _selection(self, root: Path, candidates: list[str]) -> Path:
    source = root / "analysis.json"
    source.write_text("{}")
    path = root / "stable_selection.json"
    path.write_text(
      json.dumps(
        {
          "status": "PROMOTE_FOR_POLICY_SEEDS",
          "policy_seed": 42,
          "promoted_candidates": candidates,
          "sources": [
            {
              "path": str(source),
              "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
          ],
        }
      )
    )
    return path

  def test_plan_forces_agent_and_environment_seed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      selection = self._selection(root, ["a0_f2s2_gsi"])
      with mock.patch.object(launcher, "_git_commit", return_value="commit"):
        plan = launcher.build_plan(
          launcher.ConfirmationCfg(
            selection=selection,
            control_dir=root / "control",
          )
        )
      self.assertEqual(len(plan["jobs"]), 3)
      for job in plan["jobs"]:
        command = job["command"]
        self.assertIn("--agent.seed", command)
        self.assertIn("--env.seed", command)
        self.assertEqual(job["policy_seed"], job["environment_seed"])

  def test_two_candidates_use_six_unique_gpus(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      selection = self._selection(root, ["a0_f2s2_gsi", "a1_v7_gsi"])
      with mock.patch.object(launcher, "_git_commit", return_value="commit"):
        plan = launcher.build_plan(
          launcher.ConfirmationCfg(
            selection=selection,
            control_dir=root / "control",
          )
        )
      self.assertEqual(len(plan["jobs"]), 6)
      self.assertEqual(len({job["gpu"] for job in plan["jobs"]}), 6)

  def test_changed_analysis_invalidates_selection(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      selection = self._selection(root, ["a0_f2s2_gsi"])
      (root / "analysis.json").write_text("changed")
      with self.assertRaisesRegex(ValueError, "source changed"):
        launcher.build_plan(
          launcher.ConfirmationCfg(
            selection=selection,
            control_dir=root / "control",
          )
        )

  def test_launch_is_idempotent(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      selection = self._selection(root, ["a0_f2s2_gsi"])
      cfg = launcher.ConfirmationCfg(
        selection=selection,
        control_dir=root / "control",
        launch=True,
      )
      processes = [mock.Mock(pid=100 + index) for index in range(3)]
      with (
        mock.patch.object(launcher, "_git_commit", return_value="commit"),
        mock.patch.object(launcher, "_gpu_processes", return_value=[]),
        mock.patch.object(launcher.subprocess, "Popen", side_effect=processes) as popen,
      ):
        first = launcher.launch_confirmation(cfg)
        second = launcher.launch_confirmation(cfg)
      self.assertEqual(first["status"], "LAUNCHED")
      self.assertEqual(first["plan_id"], second["plan_id"])
      self.assertEqual(popen.call_count, 3)


if __name__ == "__main__":
  unittest.main()
