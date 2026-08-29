from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
  Path(__file__).parents[1] / "scripts" / "firm" / "advance_deployable_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("advance_deployable_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value))


class AdvanceDeployableBaselineTest(unittest.TestCase):
  def _protocol(self, root: Path) -> dict:
    expert = root / "expert.pt"
    motion = root / "motion.npz"
    expert.write_bytes(b"expert")
    motion.write_bytes(b"motion")
    upstream = root / "upstream.json"
    return {
      "protocol_id": "test",
      "expert": {
        "task_id": "Firm-Keyframe-Deployable-G1",
        "checkpoint_file": str(expert),
        "checkpoint_sha256": MODULE.sha256_file(expert),
        "motion_file": str(motion),
        "motion_sha256": MODULE.sha256_file(motion),
        "deployable_state_dim": 93,
      },
      "replicate_seeds": [1, 2, 3],
      "collection": {
        "num_start_frames": 25,
        "episodes_per_frame": 32,
        "max_steps": 500,
        "standing_hold_steps": 25,
        "observation_corruption": True,
        "physical_disturbances": True,
        "minimum_success_fraction": 0.25,
      },
      "launch": {
        "upstream_state_file": str(upstream),
        "upstream_terminal_statuses": ["NATIVE_BASELINE_EVIDENCE_COMPLETE"],
      },
      "outputs": {
        "rollout_root": str(root / "rollouts"),
        "action_root": str(root / "action"),
        "adapter_root": str(root / "adapter"),
      },
    }

  def test_rejects_legacy_90d_rollout(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol = self._protocol(root)
      manifest_path = root / "manifest.json"
      manifest = {
        "task_id": "Firm-Keyframe-Deployable-G1",
        "config": {
          "seed": 1,
          "num_start_frames": 25,
          "episodes_per_frame": 32,
          "max_steps": 500,
          "standing_hold_steps": 25,
          "observation_corruption": True,
          "physical_disturbances": True,
        },
        "layout": {"observation": {"shape": [90]}},
      }
      _write_json(manifest_path, manifest)
      with self.assertRaisesRegex(RuntimeError, "not frozen 93D"):
        MODULE.validate_rollout_manifest(
          manifest_path, protocol, 1, verify_shards=False
        )

  def test_protocol_drift_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol_path = root / "protocol.json"
      state_path = root / "state.json"
      _write_json(protocol_path, self._protocol(root))
      _write_json(state_path, {"protocol_sha256": "wrong", "jobs": {}})
      result = MODULE.advance(
        repo_root=root,
        protocol_path=protocol_path,
        state_path=state_path,
        launch_when_ready=False,
      )
      self.assertEqual(result["status"], "PROTOCOL_DRIFT_ALERT")

  def test_waits_for_upstream_without_querying_gpus(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol = self._protocol(root)
      upstream_path = Path(protocol["launch"]["upstream_state_file"])
      _write_json(upstream_path, {"status": "TRAINING_ACTIVE"})
      protocol_path = root / "protocol.json"
      state_path = root / "state.json"
      _write_json(protocol_path, protocol)
      result = MODULE.advance(
        repo_root=root,
        protocol_path=protocol_path,
        state_path=state_path,
        launch_when_ready=True,
      )
      self.assertEqual(result["status"], "WAITING_UPSTREAM")
      self.assertEqual(result["upstream_status"], "TRAINING_ACTIVE")

  def test_generated_commands_use_tyro_boolean_flags(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      protocol = self._protocol(root)
      protocol["collection"]["shard_size"] = 50000
      protocol["action_training"] = {
        "horizon": 12,
        "successful_only": True,
        "train_fraction": 0.9,
        "batch_size": 512,
        "num_epochs": 100,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "num_timesteps": 50,
        "save_interval": 50,
      }
      protocol["adapter_training"] = {
        "train_fraction": 0.9,
        "balance_goal_sampling": True,
        "batch_size": 1024,
        "num_epochs": 20,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
      }
      collection = MODULE._collection_command(root, protocol, 1, root / "out")
      action = MODULE._action_command(
        protocol, 1, root / "manifest.json", root / "action"
      )
      adapter = MODULE._adapter_command(
        protocol,
        1,
        50,
        root / "action.pt",
        root / "manifest.json",
        root / "adapter",
        "firm_r_50f",
      )
      self.assertIn("--physical-disturbances", collection)
      self.assertIn("--successful-only", action)
      self.assertIn("--balance-goal-sampling", adapter)
      self.assertNotIn("True", collection + action + adapter)


if __name__ == "__main__":
  unittest.main()
