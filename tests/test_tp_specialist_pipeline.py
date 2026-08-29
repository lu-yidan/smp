from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import launch_smp_tp_specialists as launcher
import run_smp_tp_physics_smoke as smoke_runner
import select_smp_confirmed_flat_arm as selector


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float, values: list[float] | None = None) -> dict:
  values = values or [value, value, value]
  return {
    "mean": value,
    "ci95_low": value - 0.01,
    "ci95_high": value + 0.01,
    "policy_seed_values": values,
  }


class TpSpecialistPipelineTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.protocol = Path(__file__).parents[1] / "docs/ral_terrain_plate_protocol.json"
    self.seeds = (20260901, 20260902, 20260903)
    self.arm = "a6_f2s2_mix_bridge"

    sources = []
    manifests = []
    for seed in self.seeds:
      summary = self.root / f"summary_{seed}.json"
      summary.write_text(json.dumps({"seed": seed}))
      sources.append({"path": str(summary), "sha256": _sha256(summary)})
      checkpoint = self.root / f"model_{seed}.pt"
      checkpoint.write_bytes(f"checkpoint-{seed}".encode())
      run_dir = self.root / f"run_{seed}"
      run_dir.mkdir()
      manifest = self.root / f"manifest_{seed}.json"
      manifest.write_text(
        json.dumps(
          {
            "policy_seed": seed,
            "environment_seed": seed,
            "runs": [
              {
                "name": self.arm,
                "policy_seed": seed,
                "environment_seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "run_dir": str(run_dir),
              }
            ],
          }
        )
      )
      manifests.append(
        {"policy_seed": seed, "path": str(manifest), "sha256": _sha256(manifest)}
      )
    self.index = self.root / "index.json"
    self.index.write_text(
      json.dumps(
        {
          "status": "READY",
          "policy_seeds": list(self.seeds),
          "manifests": manifests,
        }
      )
    )

    metrics = {
      "gsi": _metric(0.98),
      "fixed_macro": _metric(0.85, [0.82, 0.86, 0.87]),
      "fixed_worst": _metric(0.70, [0.68, 0.72, 0.70]),
      "finite_action_rate": _metric(1.0),
      "secondary_fall_rate_after_success": _metric(0.02),
      "post_success_root_drift_p95_m": _metric(0.10),
      "contact_foot_slip_p95_m_s": _metric(0.12),
      "max_power_mean_w": _metric(100.0),
      "max_joint_speed_p95_rad_s": _metric(3.0),
    }
    self.aggregate = self.root / "aggregate.json"
    self.aggregate.write_text(
      json.dumps(
        {
          "status": "MINIMUM_POLICY_SEEDS_MET",
          "policy_seeds": list(self.seeds),
          "policy_seed_count": 3,
          "source_summaries": sources,
          "arms": {self.arm: {"metrics": metrics}},
        }
      )
    )
    self.promotion = self.root / "promotion.json"

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def _select(self) -> dict:
    return selector.write_selection(
      selector.FlatPromotionCfg(
        aggregate=self.aggregate,
        confirmation_manifest_index=self.index,
        protocol=self.protocol,
        output=self.promotion,
      )
    )

  def test_confirmed_arm_is_promoted_with_matched_seed_checkpoints(self) -> None:
    result = self._select()
    self.assertEqual(result["status"], "PROMOTE_TP_SPECIALISTS")
    self.assertEqual(result["selected_arm"], self.arm)
    self.assertEqual(result["selected_arm_index"], 6)
    self.assertEqual(set(result["matched_flat_checkpoints"]), set(self.seeds))

  def test_nonfinite_seed_blocks_promotion(self) -> None:
    payload = json.loads(self.aggregate.read_text())
    payload["arms"][self.arm]["metrics"]["finite_action_rate"] = _metric(
      0.99, [1.0, 0.97, 1.0]
    )
    self.aggregate.write_text(json.dumps(payload))
    result = selector.select(
      selector.FlatPromotionCfg(
        aggregate=self.aggregate,
        confirmation_manifest_index=self.index,
        protocol=self.protocol,
        output=self.promotion,
      )
    )
    self.assertEqual(result["status"], "NO_PROMOTION")

  def test_specialist_plan_has_matched_t_and_p_warm_starts(self) -> None:
    promotion = self._select()
    smoke = self.root / "smoke.json"
    cfg = launcher.SpecialistLaunchCfg(
      promotion=self.promotion,
      control_dir=self.root / "control",
      smoke_test=smoke,
      protocol=self.protocol,
      logs_root=self.root / "logs",
    )
    plan = launcher.build_plan(cfg)
    self.assertEqual(len(plan["jobs"]), 6)
    self.assertEqual({job["phase"] for job in plan["jobs"]}, {"T", "P"})
    self.assertEqual({job["policy_seed"] for job in plan["jobs"]}, set(self.seeds))
    self.assertEqual({job["arm"] for job in plan["jobs"]}, {self.arm})
    for job in plan["jobs"]:
      command = job["command"]
      self.assertIn("--agent.resume", command)
      self.assertIn("--agent.algorithm.learning-rate", command)
      self.assertIn("20000", command)

    smoke.write_text(
      json.dumps(
        {
          "status": "PASS",
          "promotion_id": promotion["promotion_id"],
          "code_commit": plan["code_commit"],
          "tasks": sorted({job["task"] for job in plan["jobs"]}),
        }
      )
    )
    with mock.patch.object(launcher, "_gpu_processes", return_value=["99"]):
      with self.assertRaisesRegex(RuntimeError, "GPU process"):
        launcher.launch_specialists(
          launcher.SpecialistLaunchCfg(
            promotion=self.promotion,
            control_dir=self.root / "control",
            smoke_test=smoke,
            protocol=self.protocol,
            logs_root=self.root / "logs",
            launch=True,
          )
        )

  def test_physics_smoke_is_planned_but_not_faked_without_run(self) -> None:
    promotion = self._select()
    output = self.root / "smoke_result.json"
    result = smoke_runner.run_smoke(
      smoke_runner.TpSmokeCfg(
        promotion=self.promotion,
        output=output,
        work_dir=self.root / "smoke",
        protocol=self.protocol,
        run=False,
      )
    )
    self.assertEqual(result["status"], "PLANNED")
    self.assertFalse(output.exists())
    self.assertEqual(result["promotion_id"], promotion["promotion_id"])

    with mock.patch.object(smoke_runner, "_gpu_processes", return_value=["77"]):
      with self.assertRaisesRegex(RuntimeError, "GPU process"):
        smoke_runner.run_smoke(
          smoke_runner.TpSmokeCfg(
            promotion=self.promotion,
            output=output,
            work_dir=self.root / "smoke",
            protocol=self.protocol,
            run=True,
          )
        )


if __name__ == "__main__":
  unittest.main()
