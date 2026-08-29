from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import build_smp_causal_manifest as manifest


class SeedProvenanceTest(unittest.TestCase):
  def _fixture(self, root: Path, agent_seed: int, environment_seed: int) -> None:
    run = root / "task" / "run_suffix"
    (run / "params").mkdir(parents=True)
    (run / "params" / "agent.yaml").write_text(f"seed: {agent_seed}\n")
    (run / "params" / "env.yaml").write_text(f"other: true\nseed: {environment_seed}\n")
    torch.save(
      {
        "iter": 8000,
        "actor_state_dict": {"weight": torch.ones(1)},
        "critic_state_dict": {"weight": torch.ones(1)},
      },
      run / "model_8000.pt",
    )

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

  def test_rejects_modified_locked_a6_continuation_provenance(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      run = root / "run"
      run.mkdir()
      source = root / "model_25000.pt"
      source.write_bytes(b"source")
      (run / "resume_provenance.json").write_text(
        json.dumps(
          {
            "status": "AUDITED_CONTINUATION",
            "arm": "a6_f2s2_mix_bridge",
            "policy_seed": 42,
            "environment_seed": 42,
            "continuation_wandb_run_id": "resume",
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": manifest._sha256(source),
          }
        )
      )
      arm = {"name": "a6_f2s2_mix_bridge"}
      with self.assertRaisesRegex(
        ValueError, "locked continuation provenance hash changed"
      ):
        manifest._continuation_provenance(run, arm, 42, 42)

  def test_gate_selects_run_segment_that_contains_checkpoint(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      self._fixture(root, 42, 42)
      resumed = root / "task" / "newer_run_suffix"
      (resumed / "params").mkdir(parents=True)
      (resumed / "params" / "agent.yaml").write_text("seed: 42\n")
      (resumed / "params" / "env.yaml").write_text("seed: 42\n")
      torch.save(
        {
          "iter": 29999,
          "actor_state_dict": {"weight": torch.ones(1)},
          "critic_state_dict": {"weight": torch.ones(1)},
        },
        resumed / "model_29999.pt",
      )
      source = root / "task" / "run_suffix" / "model_8000.pt"
      (resumed / "resume_provenance.json").write_text(
        json.dumps(
          {
            "status": "AUDITED_CONTINUATION",
            "arm": "arm",
            "policy_seed": 42,
            "environment_seed": 42,
            "continuation_wandb_run_id": "resume-run",
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": manifest._sha256(source),
          }
        )
      )
      arm = {
        "name": "arm",
        "task": "Task",
        "log_dir": "task",
        "run_suffix": "run_suffix",
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
      self.assertEqual(Path(payload["runs"][0]["run_dir"]).name, "run_suffix")
      self.assertEqual(payload["runs"][0]["checkpoint_integrity"]["iteration"], 8000)
      resumed_cfg = manifest.ManifestCfg(
        checkpoint_step=29999,
        output=root / "final_manifest.json",
        logs_root=root,
      )
      with (
        mock.patch.object(manifest, "_ARMS", (arm,)),
        mock.patch.object(manifest, "_git_commit", return_value="commit"),
      ):
        resumed_payload = manifest.build_manifest(resumed_cfg)
      resumed_row = resumed_payload["runs"][0]
      self.assertEqual(resumed_row["wandb_run_id"], "resume-run")
      self.assertEqual(
        resumed_row["continuation_provenance"]["record"]["status"],
        "AUDITED_CONTINUATION",
      )
      self.assertTrue(
        resumed_row["checkpoint_integrity"]["actor_state_dict_all_finite"]
      )

  def test_manifest_rejects_nonfinite_actor_checkpoint(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      self._fixture(root, 42, 42)
      checkpoint = root / "task" / "run_suffix" / "model_8000.pt"
      torch.save(
        {
          "iter": 8000,
          "actor_state_dict": {"weight": torch.tensor([float("nan")])},
          "critic_state_dict": {"weight": torch.ones(1)},
        },
        checkpoint,
      )
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
      with mock.patch.object(manifest, "_ARMS", (arm,)):
        with self.assertRaisesRegex(FileNotFoundError, "nonfinite tensors"):
          manifest.build_manifest(cfg)


if __name__ == "__main__":
  unittest.main()
