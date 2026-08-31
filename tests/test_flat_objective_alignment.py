from __future__ import annotations

import unittest

from mjlab.tasks.registry import load_env_cfg

import smp.rl.tasks  # noqa: F401
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.ral_progression_env_cfg import ACTOR_TERMS
from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  F2S2_PRIOR_PATH,
  g1_scratch_a6_f2s2_mix_bridge_env_cfg,
  g1_scratch_a9_f2s2_objective_aligned_env_cfg,
)


class FlatObjectiveAlignmentTest(unittest.TestCase):
  def test_a9_preserves_a6_training_inputs_and_changes_only_objective(self) -> None:
    control = g1_scratch_a6_f2s2_mix_bridge_env_cfg(play=False)
    proposed = g1_scratch_a9_f2s2_objective_aligned_env_cfg(play=False)

    for cfg in (control, proposed):
      actor = cfg.observations["actor"]
      self.assertEqual(tuple(actor.terms), ACTOR_TERMS)
      self.assertIsNone(actor.history_length)
      self.assertNotIn("base_lin_vel", actor.terms)
      self.assertEqual(
        cfg.events["init_smp_state"].params["ckpt_path"], F2S2_PRIOR_PATH
      )
      reset = cfg.events["mixed_fall_reset"]
      self.assertEqual(reset.params["procedural_probability"], 0.20)
      self.assertEqual(reset.params["mode_weights"], (1.0, 1.0, 1.0, 1.0))
      self.assertIs(cfg.terminations["smp_too_low"].func, mdp.smp_too_low_gsi_only)

    self.assertIs(proposed.rewards["task_smp_product"].func, task_smp_product)
    self.assertEqual(proposed.rewards["task_smp_product"].params["smp_floor"], 0.35)
    terms = proposed.rewards["task_smp_product"].params["task_terms"]
    self.assertEqual(
      [term[0] for term in terms],
      [
        mdp.recovery_initiation_progress,
        mdp.track_head_height,
        mdp.upright_posture,
        mdp.feet_stationary_when_upright,
        mdp.base_stationary_when_upright,
        mdp.stable_stand_metric,
      ],
    )
    self.assertAlmostEqual(sum(term[1] for term in terms), 1.0)
    self.assertEqual(proposed.rewards["head_vertical_overspeed"].weight, -0.50)
    self.assertEqual(proposed.rewards["action_rate_l2"].weight, -0.001)

  def test_a9_training_success_matches_frozen_flat_success(self) -> None:
    cfg = g1_scratch_a9_f2s2_objective_aligned_env_cfg(play=False)
    params = cfg.terminations["stood_up"].params
    self.assertEqual(params["head_height"], 1.10)
    self.assertEqual(params["max_speed"], 0.50)
    self.assertEqual(params["hold_steps"], 25)
    self.assertEqual(params["min_upright"], 0.85)
    self.assertEqual(params["max_angular_speed"], 1.00)

  def test_a9_task_is_registered_and_deployable(self) -> None:
    cfg = load_env_cfg("Smp-Getup-Scratch-A9-F2S2-Objective-Aligned-G1")
    actor = cfg.observations["actor"]
    self.assertEqual(tuple(actor.terms), ACTOR_TERMS)
    self.assertNotIn("base_lin_vel", actor.terms)
    self.assertEqual(cfg.events["mixed_fall_reset"].params["procedural_probability"], 0.20)


if __name__ == "__main__":
  unittest.main()
