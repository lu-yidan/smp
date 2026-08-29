from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import build_smp_native_baseline_manifests as builder


class NativeBaselineManifestTest(unittest.TestCase):
  def _fixture(self, root: Path) -> builder.NativeBaselineManifestCfg:
    sources = {}
    for name in ("promotion", "runtime_registry", "bank", "bank_manifest"):
      path = root / name
      path.write_bytes(name.encode())
      sources[name] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    jobs = []
    workers = []
    arm_index = 6
    combinations = (
      (seed, method)
      for seed in (20260901, 20260902, 20260903)
      for method in builder._METHOD_TASK_NAMES
    )
    for index, (seed, method) in enumerate(combinations):
      gpu = index % 8
      task_name = builder._METHOD_TASK_NAMES[method]
      task = f"Smp-Getup-RAL-B-{task_name}-A{arm_index}-G1"
      experiment = f"smp_ral_b_{method}_a{arm_index}_g1"
      run_name = f"ral_b_{method}_a{arm_index}_30k_seed{seed}"
      run = root / "logs" / experiment / f"timestamp_{run_name}"
      (run / "params").mkdir(parents=True)
      (run / "params" / "agent.yaml").write_text(
        "\n".join(
          (
            f"seed: {seed}",
            "num_steps_per_env: 24",
            "max_iterations: 30000",
            "save_interval: 1000",
            f"experiment_name: {experiment}",
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
        f"scene:\n  num_envs: 4096\n"
        f"observations:\n  actor:\n    terms:\n{actor_terms}\n"
        "    concatenate_terms: true\n    history_length: null\n"
        "  critic:\n    terms: {}\n"
        f"bank_path: {sources['bank'][0].resolve()}\n"
        f"bank_sha256: {sources['bank'][1]}\n"
        f"seed: {seed}\n"
      )
      for gate in (8000, 15000, 25000, 29999):
        (run / f"model_{gate}.pt").write_bytes(f"{method}-{seed}-{gate}".encode())
      log = root / f"gpu{gpu}_{method}_seed{seed}.log"
      log.write_text(f"https://wandb.ai/tabletennis/smp/runs/run{index}\n")
      job_id = f"{method}_seed{seed}"
      jobs.append(
        {
          "job_id": job_id,
          "method": method,
          "task": task,
          "policy_seed": seed,
          "environment_seed": seed,
          "run_name": run_name,
          "log": str(log),
        }
      )
      workers.append({"gpu": gpu, "job_ids": [job_id], "pid": 100 + index})
    launch = {
      "schema_version": 1,
      "status": "LAUNCHED",
      "plan_id": "plan",
      "code_commit": "commit",
      "promotion": str(sources["promotion"][0]),
      "promotion_sha256": sources["promotion"][1],
      "promotion_id": "promotion-id",
      "runtime_registry": str(sources["runtime_registry"][0]),
      "runtime_registry_sha256": sources["runtime_registry"][1],
      "bank": str(sources["bank"][0]),
      "bank_sha256": sources["bank"][1],
      "bank_manifest": str(sources["bank_manifest"][0]),
      "bank_manifest_sha256": sources["bank_manifest"][1],
      "selected_arm": "a6_f2s2_mix_bridge",
      "arm_index": arm_index,
      "policy_seeds": [20260901, 20260902, 20260903],
      "num_envs": 4096,
      "max_updates": 30000,
      "save_interval": 1000,
      "jobs": jobs,
      "workers": workers,
    }
    launch_path = root / "launch.json"
    launch_path.write_text(json.dumps(launch))
    self.sources = sources
    return builder.NativeBaselineManifestCfg(
      launch_manifest=launch_path,
      output_dir=root / "manifests",
      logs_root=root / "logs",
    )

  def test_writes_seed_gate_factorial_and_remains_eval_blocked(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      index = builder.write_manifests(cfg)
      self.assertEqual(index["status"], "CHECKPOINTS_READY_EVALUATION_BANK_BLOCKED")
      self.assertEqual(len(index["manifests"]), 12)
      payload = json.loads(Path(index["manifests"][0]["path"]).read_text())
      self.assertEqual(len(payload["runs"]), 3)
      self.assertEqual(
        payload["evaluation_status"], "BLOCKED_ON_MATCHED_HELD_OUT_RESET_BANK"
      )
      self.assertTrue(
        payload["evaluation_protocol"]["requires_disjoint_matched_reset_bank"]
      )

  def test_saved_seed_and_actor_privilege_drift_are_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      agent = next((root / "logs").glob("**/params/agent.yaml"))
      old_seed = builder._top_level_scalar(agent, "seed")
      agent.write_text(agent.read_text().replace(f"seed: {old_seed}", "seed: 9"))
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

  def test_changed_source_and_partial_gate_are_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      self.sources["bank"][0].write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "reset bank changed"):
        builder.build(cfg)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      next((root / "logs").glob("**/model_25000.pt")).unlink()
      with self.assertRaisesRegex(FileNotFoundError, "refusing partial"):
        builder.build(cfg)

  def test_index_detects_manifest_tampering_and_partial_artifacts(self) -> None:
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
      (cfg.output_dir / "gate_8000_seed_20260901.json").write_text("{}")
      with self.assertRaisesRegex(ValueError, "without an index"):
        builder.write_manifests(cfg)


if __name__ == "__main__":
  unittest.main()
