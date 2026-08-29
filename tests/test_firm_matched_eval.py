from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import build_firm_matched_eval_manifest as builder
import evaluate_smp_baseline as evaluator
import run_smp_frozen_eval_matrix as matrix
from smp.firm.deployable_policy import (
  FirmActionDiffusion,
  FirmDeployablePolicy,
  FirmGoalAdapter,
  sha256_file,
)
from smp.pretrain.model import DiffusionDenoiser


def _action_checkpoint(root: Path, seed: int, observation_dim: int = 93) -> Path:
  model = FirmActionDiffusion(
    horizon=2,
    observation_dim=observation_dim,
    goal_latent_dim=8,
    d_model=32,
    nhead=4,
    num_layers=1,
    dropout=0.0,
  )
  normalization = {
    "observation_mean": torch.zeros(observation_dim),
    "observation_std": torch.ones(observation_dim),
    "joint_mean": torch.zeros(29),
    "joint_std": torch.ones(29),
    "action_mean": torch.zeros(29),
    "action_std": torch.ones(29),
  }
  path = root / f"action_{seed}.pt"
  torch.save(
    {
      "config": {
        "seed": seed,
        "observation_dim": observation_dim,
        "horizon": 2,
        "goal_latent_dim": 8,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "num_timesteps": 2,
      },
      "model_ema": model.state_dict(),
      "normalization": normalization,
    },
    path,
  )
  return path


def _adapter_checkpoint(
  root: Path, seed: int, action: Path, history_steps: int
) -> Path:
  model = FirmGoalAdapter(
    observation_dim=93,
    history_steps=history_steps,
    latent_dim=8,
    channels=(8, 8, 8),
  )
  goals = torch.randn(4, 29)
  features = F.normalize(torch.randn(4, 8), dim=-1)
  path = root / f"adapter_{history_steps}_{seed}.pt"
  torch.save(
    {
      "config": {
        "seed": seed,
        "observation_dim": 93,
        "history_steps": history_steps,
        "latent_dim": 8,
        "channels": (8, 8, 8),
        "num_epochs": 20,
      },
      "model": model.state_dict(),
      "observation_mean": torch.zeros(93),
      "observation_std": torch.ones(93),
      "codebook_goals": goals,
      "codebook_features": features,
      "artifacts": {"action_checkpoint_sha256": sha256_file(action)},
    },
    path,
  )
  return path


class FirmDeployablePolicyTest(unittest.TestCase):
  def test_evaluator_requires_explicit_firm_adapter(self) -> None:
    with self.assertRaisesRegex(ValueError, "requires a deployable adapter"):
      evaluator._validate_policy_configuration(
        evaluator.EvalCfg(checkpoint=Path("action.pt"), policy_kind="firm_r")
      )

  def test_loads_and_executes_one_frame_93d_policy(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      action = _action_checkpoint(root, 7)
      adapter = _adapter_checkpoint(root, 7, action, 1)
      policy = FirmDeployablePolicy(
        action,
        adapter,
        device="cpu",
        expected_seed=7,
        goal_refresh_steps=1,
      )
      torch.manual_seed(3)
      result = policy({"actor": torch.zeros(3, 93)})
      self.assertEqual(tuple(result.shape), (3, 29))
      self.assertTrue(torch.isfinite(result).all())
      self.assertEqual(policy.metadata()["history_steps"], 1)

  def test_rejects_legacy_90d_action_checkpoint(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      action = _action_checkpoint(root, 7, observation_dim=90)
      adapter = root / "unused.pt"
      with self.assertRaisesRegex(ValueError, "requires 93-D action"):
        FirmDeployablePolicy(action, adapter, device="cpu", expected_seed=7)

  def test_conditional_extension_preserves_unconditional_state_dict(self) -> None:
    unconditional = DiffusionDenoiser(feature_dim=4, window_size=2, d_model=8)
    self.assertFalse(any("condition_proj" in key for key in unconditional.state_dict()))
    value = unconditional(torch.zeros(2, 2, 4), torch.zeros(2, dtype=torch.long))
    self.assertEqual(tuple(value.shape), (2, 2, 4))


class FirmManifestTest(unittest.TestCase):
  def _held_out(self, root: Path) -> Path:
    modes = ("native_gsi", "prone", "supine", "left_side", "right_side")
    banks = {}
    for mode in modes:
      bank = root / f"{mode}.pt"
      bank.write_bytes(mode.encode())
      banks[mode] = {
        "path": str(bank),
        "sha256": builder._sha256(bank),
        "num_states": 512,
      }
    manifest = root / "held_out.json"
    manifest.write_text(
      json.dumps(
        {
          "status": "READY",
          "generation_seed": 20260829,
          "num_states_per_mode": 512,
          "modes": modes,
          "exact_training_overlap_count": 0,
          "training_bank_sha256": "a" * 64,
          "banks": banks,
        }
      )
    )
    return manifest

  def test_builds_six_run_external_reference_manifest(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      action_root = root / "actions"
      adapter_root = root / "adapters"
      seeds = (11, 12, 13)
      variants = (
        {"name": "firm_r_1f", "history_steps": 1, "reporting_class": "one"},
        {"name": "firm_r_50f", "history_steps": 50, "reporting_class": "causal"},
      )
      for seed in seeds:
        action_dir = action_root / f"seed_{seed}"
        action_dir.mkdir(parents=True)
        source_action = _action_checkpoint(root, seed)
        action = action_dir / "firm_action_diffusion.pt"
        action.write_bytes(source_action.read_bytes())
        for variant in variants:
          adapter_dir = adapter_root / variant["name"] / f"seed_{seed}"
          adapter_dir.mkdir(parents=True)
          source_adapter = _adapter_checkpoint(
            root, seed, action, variant["history_steps"]
          )
          (adapter_dir / "firm_goal_adapter.pt").write_bytes(
            source_adapter.read_bytes()
          )
      protocol = root / "protocol.json"
      protocol.write_text(
        json.dumps(
          {
            "status": "FROZEN_BEFORE_DATA_COLLECTION",
            "tier_a_eligible": False,
            "eligibility_note": "external only",
            "replicate_seeds": seeds,
            "adapter_variants": variants,
            "outputs": {
              "action_root": str(action_root),
              "adapter_root": str(adapter_root),
            },
          }
        )
      )
      state = root / "state.json"
      state.write_text(
        json.dumps(
          {
            "status": "READY_FOR_MATCHED_EVAL_ADAPTER",
            "protocol_sha256": builder._sha256(protocol),
          }
        )
      )
      held_out = self._held_out(root)
      output = root / "firm_manifest.json"
      result = builder.build(
        builder.BuildCfg(
          firm_protocol=protocol,
          firm_state=state,
          matched_eval_manifest=held_out,
          matched_eval_manifest_sha256=builder._sha256(held_out),
          output=output,
        )
      )
      self.assertEqual(len(result["runs"]), 6)
      self.assertFalse(result["tier_a_eligible"])
      metadata, runs = matrix._load_manifest(output)
      self.assertEqual(metadata["comparison_class"], "external_reference")
      self.assertEqual(len(runs), 6)

  def test_matrix_rejects_changed_firm_adapter(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      action = root / "action.pt"
      adapter = root / "adapter.pt"
      action.write_bytes(b"action")
      adapter.write_bytes(b"adapter")
      manifest = root / "manifest.json"
      manifest.write_text(
        json.dumps(
          {
            "runs": [
              {
                "name": "firm",
                "task": "Task",
                "policy_kind": "firm_r",
                "checkpoint": str(action),
                "checkpoint_sha256": matrix._sha256(action),
                "firm_adapter_checkpoint": str(adapter),
                "firm_adapter_checkpoint_sha256": matrix._sha256(adapter),
              }
            ]
          }
        )
      )
      adapter.write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "adapter changed"):
        matrix._load_manifest(manifest)

  def test_matrix_routes_firm_artifacts_to_common_evaluator(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      action = root / "action.pt"
      adapter = root / "adapter.pt"
      action.write_bytes(b"action")
      adapter.write_bytes(b"adapter")
      manifest = root / "manifest.json"
      manifest.write_text(
        json.dumps(
          {
            "runs": [
              {
                "name": "firm_r_1f_seed_7",
                "task": "Smp-Getup-Matched-TaskOnly-G1",
                "policy_kind": "firm_r",
                "policy_seed": 7,
                "checkpoint": str(action),
                "checkpoint_sha256": matrix._sha256(action),
                "firm_adapter_checkpoint": str(adapter),
                "firm_adapter_checkpoint_sha256": matrix._sha256(adapter),
                "firm_goal_refresh_steps": 5,
                "firm_num_action_samples": 1,
              }
            ]
          }
        )
      )
      output = io.StringIO()
      with redirect_stdout(output):
        matrix.main(
          matrix.MatrixCfg(
            manifest=manifest,
            output_dir=root / "evaluation",
            modes=("prone",),
            num_envs=8,
            steps=10,
            device="cpu",
            dry_run=True,
          )
        )
      command = output.getvalue()
      self.assertIn("--policy-kind firm_r", command)
      self.assertIn(f"--firm-adapter-checkpoint {adapter}", command)


if __name__ == "__main__":
  unittest.main()
