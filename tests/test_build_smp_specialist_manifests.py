from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import build_smp_specialist_manifests as builder


class SpecialistManifestTest(unittest.TestCase):
  def _fixture(self, root: Path) -> builder.SpecialistManifestCfg:
    seeds = (11, 12, 13)
    source = root / "flat.pt"
    source.write_bytes(b"flat-source")
    jobs = []
    for phase in ("T", "P"):
      for seed in seeds:
        experiment = f"phase_{phase.lower()}"
        run_name = f"specialist_{phase.lower()}_seed{seed}"
        run = root / "logs" / experiment / f"timestamp_{run_name}"
        (run / "params").mkdir(parents=True)
        (run / "params" / "agent.yaml").write_text(f"seed: {seed}\n")
        (run / "params" / "env.yaml").write_text(f"seed: {seed}\n")
        for step in (2, 5, 10, 19):
          (run / f"model_{step}.pt").write_bytes(f"{phase}-{seed}-{step}".encode())
        log = root / f"{phase}_{seed}.log"
        log.write_text(f"https://wandb.ai/tabletennis/smp/runs/{phase}{seed}\n")
        jobs.append(
          {
            "phase": phase,
            "arm": "a6_f2s2_mix_bridge",
            "task": f"Task-{phase}",
            "policy_seed": seed,
            "environment_seed": seed,
            "experiment": experiment,
            "run_name": run_name,
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": builder._sha256(source),
            "log": str(log),
          }
        )
    protocol = root / "protocol.json"
    protocol.write_text("{}")
    summary_sources = []
    for seed in seeds:
      summary = root / f"flat_summary_{seed}.json"
      summary.write_text(
        json.dumps(
          {"evaluations": [{"policy_seed": seed, "arm": "a6_f2s2_mix_bridge"}]}
        )
      )
      summary_sources.append({"path": str(summary), "sha256": builder._sha256(summary)})
    aggregate = root / "aggregate.json"
    aggregate.write_text(json.dumps({"source_summaries": summary_sources}))
    promotion = root / "promotion.json"
    promotion.write_text(
      json.dumps(
        {
          "promotion_id": "promotion",
          "aggregate": str(aggregate),
          "aggregate_sha256": builder._sha256(aggregate),
        }
      )
    )
    launch = root / "launch.json"
    launch.write_text(
      json.dumps(
        {
          "status": "LAUNCHED",
          "plan_id": "plan",
          "promotion_id": "promotion",
          "code_commit": "commit",
          "protocol": str(protocol),
          "protocol_sha256": builder._sha256(protocol),
          "promotion": str(promotion),
          "promotion_sha256": builder._sha256(promotion),
          "jobs": jobs,
        }
      )
    )
    return builder.SpecialistManifestCfg(
      launch_manifest=launch,
      output_dir=root / "manifests",
      logs_root=root / "logs",
      checkpoint_steps=(2, 5, 10, 19),
      expected_seeds=seeds,
    )

  def test_writes_phase_seed_gate_factorial(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      index = builder.write_manifests(cfg)
      self.assertEqual(index["status"], "READY")
      self.assertEqual(len(index["manifests"]), 24)
      for row in index["manifests"]:
        payload = json.loads(Path(row["path"]).read_text())
        self.assertEqual(payload["phase"], row["phase"])
        self.assertEqual(payload["policy_seed"], row["policy_seed"])
        self.assertEqual(payload["checkpoint_step"], row["checkpoint_step"])
        self.assertEqual(len(payload["runs"]), 1)

  def test_rejects_missing_factorial_job(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      launch = json.loads(cfg.launch_manifest.read_text())
      launch["jobs"].pop()
      cfg.launch_manifest.write_text(json.dumps(launch))
      with self.assertRaisesRegex(ValueError, "exactly one job"):
        builder.build(cfg)

  def test_rejects_saved_seed_mismatch(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      agent = next(cfg.logs_root.glob("**/params/agent.yaml"))
      agent.write_text("seed: 999\n")
      with self.assertRaisesRegex(ValueError, "saved seed mismatch"):
        builder.build(cfg)

  def test_existing_manifest_is_immutable(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      index = builder.write_manifests(cfg)
      path = Path(index["manifests"][0]["path"])
      payload = json.loads(path.read_text())
      payload["runs"][0]["checkpoint_sha256"] = "changed"
      path.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "conflicts"):
        builder.write_manifests(cfg)


if __name__ == "__main__":
  unittest.main()
