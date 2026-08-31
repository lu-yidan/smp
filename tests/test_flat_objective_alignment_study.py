from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from mjlab.tasks.registry import load_env_cfg

import smp.rl.tasks  # noqa: F401
from smp.rl.tasks.getup.ral_progression_env_cfg import ACTOR_TERMS
from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  F2S2_PRIOR_PATH,
  g1_scratch_a6_f2s2_mix_bridge_env_cfg,
  g1_scratch_a9_f2s2_objective_aligned_env_cfg,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import advance_smp_flat_objective_alignment as monitor
import evaluate_smp_baseline as evaluator
import launch_smp_flat_objective_alignment as launcher

_REPO = Path(__file__).parents[1]
_PROTOCOL = _REPO / "docs/ral_flat_objective_alignment_v1.json"
_PROTOCOL_SHA256 = "00c6f21db33bee1ff76d659ee85ec7e99817905e49fb92b7d7ed4c8a4a60dbbb"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class FlatObjectiveAlignmentStudyTest(unittest.TestCase):
  def test_a9_holds_prior_reset_observations_and_ppo_inputs_fixed(self) -> None:
    control = g1_scratch_a6_f2s2_mix_bridge_env_cfg(play=False)
    proposed = g1_scratch_a9_f2s2_objective_aligned_env_cfg(play=False)
    for cfg in (control, proposed):
      actor = cfg.observations["actor"]
      self.assertEqual(tuple(actor.terms), ACTOR_TERMS)
      self.assertIsNone(actor.history_length)
      self.assertNotIn("base_lin_vel", actor.terms)
      self.assertEqual(cfg.events["init_smp_state"].params["ckpt_path"], F2S2_PRIOR_PATH)
      reset = cfg.events["mixed_fall_reset"]
      self.assertEqual(reset.params["procedural_probability"], 0.20)
      self.assertEqual(reset.params["mode_weights"], (1.0, 1.0, 1.0, 1.0))
    self.assertIs(
      control.events["mixed_fall_reset"].func,
      proposed.events["mixed_fall_reset"].func,
    )
    self.assertEqual(
      control.events["mixed_fall_reset"].params,
      proposed.events["mixed_fall_reset"].params,
    )
    self.assertIs(
      control.terminations["smp_too_low"].func,
      proposed.terminations["smp_too_low"].func,
    )

  def test_a9_objective_and_terminal_match_preregistration(self) -> None:
    cfg = g1_scratch_a9_f2s2_objective_aligned_env_cfg(play=False)
    task = cfg.rewards["task_smp_product"]
    self.assertEqual(task.params["smp_floor"], 0.35)
    terms = task.params["task_terms"]
    self.assertEqual([row[1] for row in terms], [0.25, 0.20, 0.20, 0.10, 0.10, 0.15])
    self.assertAlmostEqual(sum(row[1] for row in terms), 1.0)
    self.assertEqual(cfg.rewards["head_vertical_overspeed"].weight, -0.50)
    self.assertEqual(cfg.rewards["action_rate_l2"].weight, -0.001)
    self.assertEqual(
      cfg.terminations["stood_up"].params,
      {
        "head_height": 1.10,
        "max_speed": 0.50,
        "hold_steps": 25,
        "min_upright": 0.85,
        "max_angular_speed": 1.00,
      },
    )

  def test_a9_task_is_registered(self) -> None:
    cfg = load_env_cfg("Smp-Getup-Scratch-A9-F2S2-Objective-Aligned-G1")
    self.assertEqual(cfg.rewards["task_smp_product"].params["smp_floor"], 0.35)

  def test_protocol_is_hash_locked_and_uses_fresh_seeds(self) -> None:
    self.assertEqual(_sha256(_PROTOCOL), _PROTOCOL_SHA256)
    protocol = json.loads(_PROTOCOL.read_text())
    self.assertEqual(protocol["status"], "PREREGISTERED_READY_FOR_TRAINING")
    self.assertEqual(protocol["training_protocol"]["policy_seeds"], [20261101, 20261102, 20261103])
    self.assertEqual(protocol["evaluation_protocol"]["evaluation_seed"], 20261110)
    self.assertEqual(protocol["primary_contrast"]["treatment"], "success_aligned_training_objective")
    self.assertFalse(protocol["arms"]["a6_replication_control"]["promotion_eligible"])
    self.assertTrue(protocol["arms"]["a9_objective_aligned"]["promotion_eligible"])
    self.assertEqual(
      protocol["failure_reason_codebook"],
      {str(index): name for index, name in enumerate(evaluator._FAILURE_REASON_NAMES)},
    )
    self.assertEqual(
      protocol["retained_results"]["procedural_coverage_study"]["status"],
      "FLAT_METHOD_COMPLETE_NO_PROMOTION",
    )

  def test_launch_plan_is_exactly_two_arms_by_three_paired_seeds(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = launcher.FlatObjectiveAlignmentCfg(
        protocol=_PROTOCOL,
        control_dir=Path(temporary) / "training",
      )
      first = launcher.build_plan(cfg)
      second = launcher.build_plan(cfg)
    self.assertEqual(first["plan_id"], second["plan_id"])
    self.assertEqual(first["protocol_sha256"], _PROTOCOL_SHA256)
    self.assertEqual(len(first["jobs"]), 6)
    self.assertEqual(
      [job["arm"] for job in first["jobs"]],
      ["a6_replication_control"] * 3 + ["a9_objective_aligned"] * 3,
    )
    for job in first["jobs"]:
      self.assertEqual(job["policy_seed"], job["environment_seed"])
      self.assertIn(str(job["policy_seed"]), job["command"])
      self.assertNotIn("--checkpoint", job["command"])
      self.assertNotIn("--resume", job["command"])
    self.assertTrue(all(not job["promotion_eligible"] for job in first["jobs"][:3]))
    self.assertTrue(all(job["promotion_eligible"] for job in first["jobs"][3:]))

  def test_monitor_validates_exact_launched_plan(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      plan = launcher.build_plan(
        launcher.FlatObjectiveAlignmentCfg(
          protocol=_PROTOCOL,
          control_dir=Path(temporary) / "training",
        )
      )
    plan["status"] = "LAUNCHED"
    for index, job in enumerate(plan["jobs"], start=1000):
      job["pid"] = index
    monitor._validate_launch(plan)
    plan["jobs"][0]["environment_seed"] += 1
    with self.assertRaisesRegex(ValueError, "job drifted"):
      monitor._validate_launch(plan)

  def test_smoke_validation_fails_closed_when_runtime_is_missing(self) -> None:
    protocol = json.loads(_PROTOCOL.read_text())
    with tempfile.TemporaryDirectory() as temporary:
      with self.assertRaisesRegex(RuntimeError, "missing or drifted"):
        launcher._validate_smoke(protocol, Path(temporary))


if __name__ == "__main__":
  unittest.main()
