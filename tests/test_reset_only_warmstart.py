from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_runner_cls

from smp.rl.tasks.getup.mdp.events import (
  _physical_gsi_window_precheck,
  _physical_reset_procedural_probability,
)
from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  g1_scratch_a6_f2s2_mix_bridge_env_cfg,
  g1_scratch_a10_f2s2_physical_reset_env_cfg,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import evaluate_smp_baseline as evaluator  # noqa: E402
import play as play_wrapper  # noqa: E402
from launch_smp_reset_only_warmstart import (  # noqa: E402
  _PROTOCOL_SHA256,
  _sha256,
  _validate_protocol,
)


class ResetOnlyWarmStartTest(unittest.TestCase):
  def test_curriculum_boundaries_are_update_aligned(self) -> None:
    self.assertEqual(_physical_reset_procedural_probability(0), 1.0)
    self.assertEqual(_physical_reset_procedural_probability(23_999), 1.0)
    self.assertEqual(_physical_reset_procedural_probability(24_000), 0.5)
    self.assertEqual(_physical_reset_procedural_probability(71_999), 0.5)
    self.assertEqual(_physical_reset_procedural_probability(72_000), 0.2)

  def test_window_precheck_rejects_nonfinite_airborne_and_fast_states(self) -> None:
    window = torch.zeros(5, 10, 59)
    window[:, :, 2] = 0.55
    window[1, -1, 2] = 1.30
    window[2, -1, 0] = torch.nan
    window[3, -1, 9] = 1.0
    window[4, -1, 53] = 3.0
    valid = _physical_gsi_window_precheck(window, control_dt=0.02)
    self.assertEqual(valid.tolist(), [True, False, False, False, False])

  def test_a10_changes_reset_only(self) -> None:
    a6 = g1_scratch_a6_f2s2_mix_bridge_env_cfg(play=False)
    a10 = g1_scratch_a10_f2s2_physical_reset_env_cfg(play=False)
    self.assertEqual(tuple(a6.observations), tuple(a10.observations))
    self.assertEqual(
      tuple(a6.observations["actor"].terms), tuple(a10.observations["actor"].terms)
    )
    self.assertEqual(
      a6.observations["actor"].history_length, a10.observations["actor"].history_length
    )
    self.assertEqual(tuple(a6.rewards), tuple(a10.rewards))
    self.assertEqual(tuple(a6.terminations), tuple(a10.terminations))
    for name in a6.rewards:
      self.assertIs(a6.rewards[name].func, a10.rewards[name].func)
      self.assertEqual(a6.rewards[name].weight, a10.rewards[name].weight)
      self.assertEqual(a6.rewards[name].params, a10.rewards[name].params)
    for name in a6.terminations:
      self.assertIs(a6.terminations[name].func, a10.terminations[name].func)
      self.assertEqual(a6.terminations[name].params, a10.terminations[name].params)
    self.assertNotIn("mixed_fall_reset", a10.events)
    self.assertIn("curriculum_validated_fall_reset", a10.events)
    self.assertEqual(
      a10.events["gsi_reset"].func.__name__, "physically_validated_gsi_reset"
    )

  def test_a10_uses_fresh_optimizer_warm_start_runner(self) -> None:
    runner = load_runner_cls("Smp-Getup-Scratch-A10-F2S2-Physical-Reset-G1")
    self.assertIsNotNone(runner)
    self.assertEqual(runner.__name__, "SmpCurriculumWarmStartRunner")
    self.assertIs(
      evaluator._load_inference_runner_cls(
        "Smp-Getup-Scratch-A10-F2S2-Physical-Reset-G1"
      ),
      MjlabOnPolicyRunner,
    )
    self.assertIs(
      play_wrapper._load_inference_runner_cls(
        "Smp-Getup-Scratch-A10-F2S2-Physical-Reset-G1"
      ),
      MjlabOnPolicyRunner,
    )

  def test_physical_eval_freezes_grounded_pose_without_legacy_reset(self) -> None:
    env_cfg = g1_scratch_a10_f2s2_physical_reset_env_cfg(play=False)
    cfg = evaluator.EvalCfg(
      checkpoint=Path("model.pt"),
      reset_mode="prone",
      physical_reset_validation=True,
    )
    evaluator._configure_physical_reset_validation(env_cfg, cfg)
    reset = env_cfg.events["curriculum_validated_fall_reset"]
    self.assertEqual(reset.params["mode_weights"], (1.0, 0.0, 0.0, 0.0))
    self.assertEqual(reset.params["target_probability"], 1.0)
    self.assertNotIn("forced_fall_reset", env_cfg.events)
    self.assertNotIn("push_robot", env_cfg.events)

  def test_physical_eval_native_mode_uses_only_valid_gsi_or_grounded_fallback(
    self,
  ) -> None:
    env_cfg = g1_scratch_a10_f2s2_physical_reset_env_cfg(play=False)
    cfg = evaluator.EvalCfg(
      checkpoint=Path("model.pt"),
      reset_mode="native_gsi",
      native_pushes=False,
      physical_reset_validation=True,
    )
    evaluator._configure_physical_reset_validation(env_cfg, cfg)
    reset = env_cfg.events["curriculum_validated_fall_reset"]
    self.assertEqual(reset.params["target_probability"], 0.0)
    self.assertEqual(reset.params["balanced_probability"], 0.0)
    self.assertNotIn("forced_fall_reset", env_cfg.events)
    self.assertNotIn("push_robot", env_cfg.events)

  def test_protocol_is_non_evidence_and_source_hash_locked(self) -> None:
    path = Path(__file__).parents[1] / "docs/ral_reset_only_warmstart_v1.json"
    protocol = json.loads(path.read_text())
    self.assertEqual(_sha256(path), _PROTOCOL_SHA256)
    self.assertEqual(_validate_protocol(path)[0], protocol)
    self.assertEqual(protocol["status"], "PREREGISTERED_READY_FOR_CANARY")
    self.assertEqual(
      protocol["study_role"],
      "ENGINEERING_CANARY_NOT_PROMOTION_OR_PERFORMANCE_EVIDENCE",
    )
    self.assertEqual(
      protocol["source_policy"]["checkpoint_sha256"],
      "533fdb6c2072fc9f3436c3e33593c36b29a9c33df7a38667c593845922f016fc",
    )
    self.assertEqual(
      protocol["treatment"]["only_changed_component"],
      "reset_sampling_and_physical_validation",
    )
    self.assertEqual(protocol["training_protocol"]["max_iterations"], 5000)
    self.assertEqual(
      protocol["warm_start_path_audit"]["status"],
      "PASSED_REAL_MUJOCO_WARM_START_SMOKE",
    )
    self.assertFalse(
      protocol["warm_start_path_audit"]["verified"]["optimizer_restored"]
    )


if __name__ == "__main__":
  unittest.main()
