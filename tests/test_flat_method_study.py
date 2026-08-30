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
  g1_scratch_a8_f2s2_balanced_bridge_env_cfg,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import evaluate_smp_baseline as evaluator
import launch_smp_flat_method_study as launcher

_REPO = Path(__file__).parents[1]
_PROTOCOL = _REPO / "docs/ral_flat_method_study_v1.json"
_PROTOCOL_SHA256 = "610a48471888199709b71f3ecaf42a813df497d3f55dc00dcfe2dde06738c496"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class FlatMethodStudyTest(unittest.TestCase):
  def test_a8_changes_only_procedural_exposure_from_a6(self) -> None:
    control = g1_scratch_a6_f2s2_mix_bridge_env_cfg(play=False)
    proposed = g1_scratch_a8_f2s2_balanced_bridge_env_cfg(play=False)
    for cfg in (control, proposed):
      actor = cfg.observations["actor"]
      self.assertEqual(tuple(actor.terms), ACTOR_TERMS)
      self.assertIsNone(actor.history_length)
      self.assertNotIn("base_lin_vel", actor.terms)
      self.assertEqual(
        cfg.events["init_smp_state"].params["ckpt_path"],
        F2S2_PRIOR_PATH,
      )
      self.assertEqual(
        cfg.rewards["task_smp_product"].params["procedural_smp_floor"],
        0.10,
      )

    control_reset = control.events["mixed_fall_reset"]
    proposed_reset = proposed.events["mixed_fall_reset"]
    self.assertIs(control_reset.func, proposed_reset.func)
    self.assertEqual(control_reset.params["procedural_probability"], 0.20)
    self.assertEqual(proposed_reset.params["procedural_probability"], 0.50)
    self.assertEqual(control_reset.params["mode_weights"], (1.0, 1.0, 1.0, 1.0))
    control_params = dict(control_reset.params)
    proposed_params = dict(proposed_reset.params)
    control_params.pop("procedural_probability")
    proposed_params.pop("procedural_probability")
    self.assertEqual(control_params, proposed_params)
    self.assertIs(
      control.terminations["smp_too_low"].func,
      proposed.terminations["smp_too_low"].func,
    )
    self.assertIs(
      control.rewards["task_smp_product"].func,
      proposed.rewards["task_smp_product"].func,
    )
    self.assertEqual(
      control.rewards["task_smp_product"].params,
      proposed.rewards["task_smp_product"].params,
    )

  def test_a8_task_is_registered(self) -> None:
    cfg = load_env_cfg("Smp-Getup-Scratch-A8-F2S2-Balanced-Bridge-G1")
    self.assertEqual(
      cfg.events["mixed_fall_reset"].params["procedural_probability"],
      0.50,
    )

  def test_preregistration_is_hash_locked_and_matches_code(self) -> None:
    self.assertEqual(_sha256(_PROTOCOL), _PROTOCOL_SHA256)
    protocol = json.loads(_PROTOCOL.read_text())
    self.assertEqual(protocol["status"], "PREREGISTERED_READY_FOR_IMPLEMENTATION_AUDIT")
    self.assertEqual(protocol["training_protocol"]["policy_seeds"], [20261001, 20261002, 20261003])
    self.assertEqual(protocol["training_protocol"]["devices"], [0, 1, 2, 3, 4, 5])
    self.assertEqual(protocol["training_protocol"]["reserved_idle_devices"], [6, 7])
    self.assertFalse(protocol["arms"]["a6_replication_control"]["promotion_eligible"])
    self.assertTrue(protocol["arms"]["a8_balanced_bridge"]["promotion_eligible"])
    codebook = {
      str(index): name for index, name in enumerate(evaluator._FAILURE_REASON_NAMES)
    }
    self.assertEqual(protocol["failure_reason_codebook"], codebook)
    for source in protocol["sources"]:
      path = _REPO / source["path"]
      self.assertEqual(_sha256(path), source["sha256"])

  def test_launch_plan_is_exactly_two_arms_by_three_fresh_seeds(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = launcher.FlatMethodStudyCfg(
        protocol=_PROTOCOL,
        control_dir=Path(temporary) / "training",
      )
      first = launcher.build_plan(cfg)
      second = launcher.build_plan(cfg)
    self.assertEqual(first["plan_id"], second["plan_id"])
    self.assertEqual(
      first["protocol_status"], "PREREGISTERED_READY_FOR_IMPLEMENTATION_AUDIT"
    )
    self.assertEqual(first["protocol_sha256"], _PROTOCOL_SHA256)
    self.assertEqual(first["policy_seeds"], [20261001, 20261002, 20261003])
    self.assertEqual(first["devices"], [0, 1, 2, 3, 4, 5])
    self.assertEqual(first["reserved_idle_devices"], [6, 7])
    self.assertTrue(first["random_actor_critic_and_normalizers"])
    self.assertEqual(len(first["jobs"]), 6)
    self.assertEqual(
      [job["arm"] for job in first["jobs"]],
      ["a6_replication_control"] * 3 + ["a8_balanced_bridge"] * 3,
    )
    for job in first["jobs"]:
      self.assertEqual(job["policy_seed"], job["environment_seed"])
      self.assertIn(str(job["policy_seed"]), job["command"])
      self.assertNotIn("--checkpoint", job["command"])
      self.assertNotIn("--resume", job["command"])
    self.assertTrue(all(not job["promotion_eligible"] for job in first["jobs"][:3]))
    self.assertTrue(all(job["promotion_eligible"] for job in first["jobs"][3:]))

  def test_launch_is_forbidden_before_protocol_audit_is_promoted(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = launcher.FlatMethodStudyCfg(
        protocol=_PROTOCOL,
        control_dir=Path(temporary) / "training",
        launch=True,
      )
      with self.assertRaisesRegex(RuntimeError, "awaiting implementation audit"):
        launcher.launch_study(cfg)
      self.assertFalse(cfg.control_dir.exists())

  def test_protocol_drift_fails_closed_before_plan_creation(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      changed = root / "protocol.json"
      payload = json.loads(_PROTOCOL.read_text())
      payload["training_protocol"]["num_envs"] = 2048
      changed.write_text(json.dumps(payload))
      cfg = launcher.FlatMethodStudyCfg(
        protocol=changed,
        control_dir=root / "training",
      )
      with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
        launcher.build_plan(cfg)


if __name__ == "__main__":
  unittest.main()
