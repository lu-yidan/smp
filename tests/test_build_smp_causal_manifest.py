from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import build_smp_causal_manifest as manifest


class SeedProvenanceTest(unittest.TestCase):
  def _fixture(self, root: Path, agent_seed: int, environment_seed: int) -> None:
    run = root / "task" / "run_suffix"
    (run / "params").mkdir(parents=True)
    (run / "params" / "agent.yaml").write_text(f"seed: {agent_seed}\n")
    (run / "params" / "env.yaml").write_text(f"other: true\nseed: {environment_seed}\n")
    (run / "model_8000.pt").write_bytes(b"checkpoint")

  def test_manifest_uses_saved_effective_seeds(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      self._fixture(root, 42, 42)
      arm = {
        "name": "arm",
        "task": "Task",
        "log_dir": "task",
        "run_suffix": "suffix",
        "wandb_run_id": "run",
      }
      cfg = manifest.ManifestCfg(
        checkpoint_step=8000,
        output=root / "manifest.json",
        logs_root=root,
      )
      with (
        mock.patch.object(manifest, "_ARMS", (arm,)),
        mock.patch.object(manifest, "_git_commit", return_value="commit"),
      ):
        payload = manifest.build_manifest(cfg)
      self.assertEqual(payload["policy_seed"], 42)
      self.assertEqual(payload["environment_seed"], 42)
      self.assertEqual(payload["runs"][0]["policy_seed"], 42)
      self.assertEqual(payload["runs"][0]["environment_seed"], 42)

  def test_manifest_rejects_seed_label_instead_of_saved_seed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      self._fixture(root, 42, 42)
      arm = {
        "name": "arm",
        "task": "Task",
        "log_dir": "task",
        "run_suffix": "suffix",
        "wandb_run_id": "run",
      }
      cfg = manifest.ManifestCfg(
        checkpoint_step=8000,
        output=root / "manifest.json",
        logs_root=root,
        policy_seed=20260830,
        environment_seed=20260830,
      )
      with mock.patch.object(manifest, "_ARMS", (arm,)):
        with self.assertRaisesRegex(FileNotFoundError, "recorded policy/environment"):
          manifest.build_manifest(cfg)


if __name__ == "__main__":
  unittest.main()
