from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import launch_smp_native_baselines as launcher


class NativeBaselineLaunchTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.promotion = self.root / "flat_promotion.json"
    self.promotion.write_text("{}")
    self.bank = self.root / "bank.pt"
    self.bank.write_bytes(b"bank")
    self.bank_sha = hashlib.sha256(self.bank.read_bytes()).hexdigest()
    self.bank_manifest = self.root / "bank.json"
    self.bank_manifest.write_text(
      json.dumps({"status": "READY", "promotion_id": "promotion"})
    )
    self.registry_path = self.root / "registry.json"
    self.registry = {
      "training_budget": {
        "policy_seeds": [20260901, 20260902, 20260903],
        "num_envs": 4096,
        "transitions_per_env_per_update": 24,
        "max_updates": 30000,
        "save_interval": 1000,
      },
      "shared_reset_bank": {
        "result_path": str(self.bank),
        "sha256": self.bank_sha,
        "manifest_path": str(self.bank_manifest),
        "num_states": 262144,
      },
    }
    self.registry_path.write_text(json.dumps(self.registry))
    self.promotion_payload = {
      "promotion_id": "promotion",
      "selected_arm": "a6_f2s2_mix_bridge",
      "selected_arm_index": 6,
    }
    self.readiness = {
      "status": "BASELINES_BLOCKED",
      "reset_bank_ready": True,
      "methods": [
        {"id": method, "status": "ready_for_training", "blocked_on": []}
        for method in launcher._NATIVE_METHOD_TASK_NAMES
      ]
      + [
        {"id": method, "status": "blocked", "blocked_on": ["adapter"]}
        for method in launcher._ADAPTER_METHODS
      ],
    }

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def _cfg(self, *, launch: bool = False) -> launcher.NativeBaselineLaunchCfg:
    return launcher.NativeBaselineLaunchCfg(
      promotion=self.promotion,
      runtime_registry=self.registry_path,
      control_dir=self.root / "control",
      launch=launch,
    )

  def _build(self) -> dict:
    with (
      mock.patch.object(
        launcher, "_validate_promotion", return_value=self.promotion_payload
      ),
      mock.patch.object(
        launcher,
        "_validate_registry",
        return_value=(self.registry, self.readiness),
      ),
      mock.patch.object(launcher, "_git_commit", return_value="commit"),
    ):
      return launcher.build_plan(self._cfg())

  def test_plan_covers_three_methods_and_three_matched_seeds(self) -> None:
    plan = self._build()
    self.assertEqual(len(plan["jobs"]), 9)
    self.assertEqual(len(plan["workers"]), 8)
    self.assertEqual(
      {job["method"] for job in plan["jobs"]}, set(launcher._NATIVE_METHOD_TASK_NAMES)
    )
    self.assertEqual(
      {job["policy_seed"] for job in plan["jobs"]}, {20260901, 20260902, 20260903}
    )
    self.assertTrue(
      all(job["policy_seed"] == job["environment_seed"] for job in plan["jobs"])
    )
    self.assertEqual(sum(len(worker["job_ids"]) for worker in plan["workers"]), 9)
    self.assertEqual(max(len(worker["job_ids"]) for worker in plan["workers"]), 2)

  def test_commands_force_random_init_and_hash_locked_bank(self) -> None:
    plan = self._build()
    for job in plan["jobs"]:
      command = job["command"]
      self.assertIn("--agent.resume", command)
      self.assertEqual(command[command.index("--agent.resume") + 1], "False")
      self.assertEqual(
        command[
          command.index("--env.events.init-matched-reset-bank.params.bank-sha256") + 1
        ],
        self.bank_sha,
      )
      self.assertEqual(
        job["policy_seed"], int(command[command.index("--agent.seed") + 1])
      )
      self.assertEqual(
        job["environment_seed"], int(command[command.index("--env.seed") + 1])
      )

  def test_registry_rejects_adapter_substitution(self) -> None:
    changed = json.loads(json.dumps(self.readiness))
    for method in changed["methods"]:
      if method["id"] == "firm_r_deployable":
        method["status"] = "ready_for_training"
        method["blocked_on"] = []
    with (
      mock.patch.object(launcher, "audit_registry", return_value=changed),
    ):
      with self.assertRaisesRegex(ValueError, "separately blocked"):
        launcher._validate_registry(self.registry_path, "promotion")

  def test_launch_uses_eight_workers_and_is_idempotent(self) -> None:
    processes = [mock.Mock(pid=100 + index) for index in range(8)]
    with (
      mock.patch.object(
        launcher, "_validate_promotion", return_value=self.promotion_payload
      ),
      mock.patch.object(
        launcher,
        "_validate_registry",
        return_value=(self.registry, self.readiness),
      ),
      mock.patch.object(launcher, "_git_commit", return_value="commit"),
      mock.patch.object(launcher, "_gpu_processes", return_value=[]),
      mock.patch.object(launcher, "_pid_alive", return_value=True),
      mock.patch.object(launcher.subprocess, "Popen", side_effect=processes) as popen,
    ):
      first = launcher.launch_baselines(self._cfg(launch=True))
      second = launcher.launch_baselines(self._cfg(launch=True))
    self.assertEqual(first["status"], "LAUNCHED")
    self.assertEqual(first["plan_id"], second["plan_id"])
    self.assertEqual(popen.call_count, 8)

  def test_dead_worker_with_incomplete_queue_is_fail_closed(self) -> None:
    processes = [mock.Mock(pid=100 + index) for index in range(8)]
    with (
      mock.patch.object(
        launcher, "_validate_promotion", return_value=self.promotion_payload
      ),
      mock.patch.object(
        launcher,
        "_validate_registry",
        return_value=(self.registry, self.readiness),
      ),
      mock.patch.object(launcher, "_git_commit", return_value="commit"),
      mock.patch.object(launcher, "_gpu_processes", return_value=[]),
      mock.patch.object(launcher.subprocess, "Popen", side_effect=processes),
    ):
      launcher.launch_baselines(self._cfg(launch=True))
      with (
        mock.patch.object(launcher, "_pid_alive", return_value=False),
        self.assertRaisesRegex(RuntimeError, "immutable queue"),
      ):
        launcher.launch_baselines(self._cfg(launch=True))

  def test_active_gpu_blocks_launch(self) -> None:
    with (
      mock.patch.object(
        launcher, "_validate_promotion", return_value=self.promotion_payload
      ),
      mock.patch.object(
        launcher,
        "_validate_registry",
        return_value=(self.registry, self.readiness),
      ),
      mock.patch.object(launcher, "_git_commit", return_value="commit"),
      mock.patch.object(launcher, "_gpu_processes", return_value=["42"]),
    ):
      with self.assertRaisesRegex(RuntimeError, "GPU process"):
        launcher.launch_baselines(self._cfg(launch=True))


if __name__ == "__main__":
  unittest.main()
