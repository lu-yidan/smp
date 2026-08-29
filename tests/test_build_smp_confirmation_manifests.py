from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import build_smp_confirmation_manifests as builder


class ConfirmationManifestTest(unittest.TestCase):
  def _fixture(self, root: Path, seeds=(11, 12, 13)) -> builder.ConfirmationManifestCfg:
    arm = {
      "name": "arm",
      "task": "Task",
      "log_dir": "task",
      "run_suffix": "unused",
      "wandb_run_id": "unused",
    }
    jobs = []
    for _index, seed in enumerate(seeds):
      run_name = f"confirm_arm_seed{seed}"
      run = root / "logs" / "task" / f"timestamp_{run_name}"
      (run / "params").mkdir(parents=True)
      (run / "params" / "agent.yaml").write_text(f"seed: {seed}\n")
      (run / "params" / "env.yaml").write_text(f"seed: {seed}\n")
      (run / "model_29999.pt").write_bytes(f"checkpoint-{seed}".encode())
      log = root / f"job_{seed}.log"
      log.write_text(f"https://wandb.ai/tabletennis/smp/runs/run{seed}\n")
      jobs.append(
        {
          "arm": "arm",
          "policy_seed": seed,
          "environment_seed": seed,
          "run_name": run_name,
          "log": str(log),
        }
      )
    selection = root / "stable_selection.json"
    selection.write_text("{}")
    launch = root / "launch.json"
    launch.write_text(
      json.dumps(
        {
          "status": "LAUNCHED",
          "plan_id": "plan",
          "code_commit": "commit",
          "selection": str(selection),
          "selection_sha256": "selection-hash",
          "jobs": jobs,
        }
      )
    )
    self.arm = arm
    return builder.ConfirmationManifestCfg(
      launch_manifest=launch,
      output_dir=root / "manifests",
      logs_root=root / "logs",
      expected_seeds=tuple(seeds),
    )

  def test_writes_one_manifest_per_policy_seed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      with mock.patch.object(builder, "_ARMS", (self.arm,)):
        index = builder.write_manifests(cfg)
      self.assertEqual(index["status"], "READY")
      self.assertEqual(len(index["manifests"]), 3)
      for row in index["manifests"]:
        payload = json.loads(Path(row["path"]).read_text())
        self.assertEqual(payload["policy_seed"], row["policy_seed"])
        self.assertEqual(payload["runs"][0]["policy_seed"], row["policy_seed"])

  def test_rejects_saved_seed_mismatch(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      agent = next((root / "logs").glob("**/params/agent.yaml"))
      agent.write_text("seed: 99\n")
      with mock.patch.object(builder, "_ARMS", (self.arm,)):
        with self.assertRaisesRegex(ValueError, "saved seed mismatch"):
          builder.build(cfg)


if __name__ == "__main__":
  unittest.main()
