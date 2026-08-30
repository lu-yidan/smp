from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import build_smp_flat_method_manifests as builder


class FlatMethodManifestTest(unittest.TestCase):
  def _fixture(self, root: Path) -> builder.FlatMethodManifestCfg:
    protocol = Path(__file__).parents[1] / "docs" / "ral_flat_method_study_v1.json"
    jobs = []
    for gpu, (arm, seed) in enumerate(
      (arm, seed) for arm in builder._ARMS for seed in builder._SEEDS
    ):
      spec = builder._ARMS[arm]
      run_name = f"flat_method_v1_{arm}_30k_seed{seed}"
      run = root / "logs" / spec["experiment"] / f"timestamp_{run_name}"
      (run / "params").mkdir(parents=True)
      (run / "params" / "agent.yaml").write_text(
        "\n".join(
          (
            f"seed: {seed}",
            "num_steps_per_env: 24",
            "max_iterations: 30000",
            "save_interval: 1000",
            f"experiment_name: {spec['experiment']}",
            f"run_name: {run_name}",
            "resume: false",
          )
        )
        + "\n"
      )
      actor_terms = "\n".join(
        f"      {term}:\n        func: test" for term in builder._ACTOR_TERMS
      )
      (run / "params" / "env.yaml").write_text(
        "scene:\n  num_envs: 4096\n"
        f"observations:\n  actor:\n    terms:\n{actor_terms}\n"
        "    concatenate_terms: true\n    history_length: null\n"
        "  critic:\n    terms: {}\n"
        "events:\n  mixed_fall_reset:\n    func: test\n    params:\n"
        f"      procedural_probability: {spec['procedural_probability']}\n"
        "rewards:\n  task_smp_product:\n    func: test\n    params:\n"
        "      procedural_smp_floor: 0.1\n"
        "terminations:\n  smp_too_low:\n"
        "    func: terminate_low_smp_for_gsi_resets\n"
        f"seed: {seed}\n"
      )
      for gate in builder._GATES:
        torch.save(
          {
            "iter": gate,
            "actor_state_dict": {
              "obs_normalizer._mean": torch.zeros(1, 93),
              "weight": torch.ones(2),
            },
            "critic_state_dict": {
              "obs_normalizer._mean": torch.zeros(1, 960),
              "weight": torch.ones(3),
            },
            "optimizer_state_dict": {"state": {0: {"exp_avg": torch.zeros(4)}}},
          },
          run / f"model_{gate}.pt",
        )
      log = root / f"gpu{gpu}_{arm}_seed{seed}.log"
      log.write_text(f"https://wandb.ai/tabletennis/smp/runs/run{gpu}\n")
      jobs.append(
        {
          "arm": arm,
          "task": spec["task"],
          "promotion_eligible": spec["promotion_eligible"],
          "policy_seed": seed,
          "environment_seed": seed,
          "gpu": gpu,
          "run_name": run_name,
          "log": str(log),
          "pid_file": str(root / f"gpu{gpu}.pid"),
          "command": builder._expected_command(arm, seed, run_name),
          "pid": 100 + gpu,
        }
      )
    launch = {
      "schema_version": 1,
      "status": "LAUNCHED",
      "study_id": "smp-flat-procedural-coverage-v1",
      "plan_id": builder._PLAN_ID,
      "protocol": str(protocol.resolve()),
      "protocol_sha256": builder._PROTOCOL_SHA256,
      "protocol_status": "PREREGISTERED_READY_FOR_TRAINING",
      "code_commit": builder._LAUNCH_COMMIT,
      "policy_seeds": list(builder._SEEDS),
      "devices": [0, 1, 2, 3, 4, 5],
      "reserved_idle_devices": [6, 7],
      "random_actor_critic_and_normalizers": True,
      "actor_observation_dim": 93,
      "actor_history_steps": 1,
      "num_envs": 4096,
      "rollout_steps_per_update": 24,
      "max_iterations": 30000,
      "save_interval": 1000,
      "implementation_smoke": {"status": "PASSED_REAL_MUJOCO_SMOKE"},
      "resource_preflight": {"free_gib": 2500.0, "inode_free_fraction": 0.95},
      "jobs": jobs,
    }
    launch_path = root / "launch.json"
    launch_path.write_text(json.dumps(launch))
    return builder.FlatMethodManifestCfg(
      launch_manifest=launch_path,
      output_dir=root / "manifests",
      protocol=protocol,
      logs_root=root / "logs",
    )

  def test_writes_twelve_manifests_covering_twenty_four_checkpoints(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      index = builder.write_manifests(cfg)
      self.assertEqual(index["status"], "READY_FOR_FROZEN_EVALUATION")
      self.assertEqual(len(index["manifests"]), 12)
      self.assertEqual(index["checkpoint_entry_count"], 24)
      payload = json.loads(Path(index["manifests"][0]["path"]).read_text())
      self.assertEqual(len(payload["runs"]), 2)
      self.assertFalse(payload["runs"][0]["promotion_eligible"])
      self.assertTrue(payload["runs"][1]["promotion_eligible"])
      self.assertTrue(
        all(run["checkpoint_integrity"]["all_tensors_finite"] for run in payload["runs"])
      )

  def test_seed_actor_and_single_factor_drift_are_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      agent = next((root / "logs").glob("**/params/agent.yaml"))
      seed = builder._top_level_scalar(agent, "seed")
      agent.write_text(agent.read_text().replace(f"seed: {seed}", "seed: 9"))
      with self.assertRaisesRegex(ValueError, "policy seed mismatch"):
        builder.build(cfg)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      env = next((root / "logs").glob("**/params/env.yaml"))
      env.write_text(
        env.read_text().replace(
          "      base_ang_vel:\n",
          "      base_lin_vel:\n        func: test\n      base_ang_vel:\n",
        )
      )
      with self.assertRaisesRegex(ValueError, "actor observation contract"):
        builder.build(cfg)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      env = next((root / "logs").glob("**/params/env.yaml"))
      env.write_text(env.read_text().replace("procedural_probability: 0.2", "procedural_probability: 0.3"))
      with self.assertRaisesRegex(ValueError, "procedural probability"):
        builder.build(cfg)

  def test_missing_gate_and_nonfinite_checkpoint_are_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      next((root / "logs").glob("**/model_25000.pt")).unlink()
      with self.assertRaisesRegex(FileNotFoundError, "refusing partial"):
        builder.build(cfg)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      checkpoint = next((root / "logs").glob("**/model_8000.pt"))
      payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
      payload["actor_state_dict"]["weight"][0] = float("nan")
      torch.save(payload, checkpoint)
      with self.assertRaisesRegex(ValueError, "checkpoint integrity failed"):
        builder.build(cfg)

  def test_named_block_scans_large_yaml_linearly(self) -> None:
    filler = "\n".join(
      f"  filler_{index}:\n    value: {index}" for index in range(2000)
    )
    text = (
      f"events:\n{filler}\n"
      "  mixed_fall_reset:\n    params:\n"
      "      procedural_probability: 0.5\n"
      "  next_event:\n    value: true\n"
      "rewards:\n  task_smp_product:\n    value: true\n"
    )
    block = builder._named_block(text, "events", "mixed_fall_reset")
    self.assertIn("procedural_probability: 0.5", block)
    self.assertNotIn("next_event", block)

  def test_index_detects_tampering_and_partial_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      index = builder.write_manifests(cfg)
      manifest = Path(index["manifests"][0]["path"])
      payload = json.loads(manifest.read_text())
      payload["claim_boundary"] = "tampered"
      manifest.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "changed after indexing"):
        builder.write_manifests(cfg)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      cfg.output_dir.mkdir(parents=True)
      (cfg.output_dir / "gate_8000_seed_20261001.json").write_text("{}")
      with self.assertRaisesRegex(ValueError, "without an index"):
        builder.write_manifests(cfg)


if __name__ == "__main__":
  unittest.main()
