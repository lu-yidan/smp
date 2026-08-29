from __future__ import annotations

import unittest

from mjlab.tasks.registry import load_env_cfg

import smp.rl.tasks  # noqa: F401
from smp.rl.rewards import task_only_reward, task_smp_product
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.ral_baseline_env_cfg import (
  BASELINE_METHOD_TASK_NAMES,
  RESET_BANK_NUM_STATES,
  g1_ral_baseline_env_cfg,
)
from smp.rl.tasks.getup.ral_progression_env_cfg import ACTOR_TERMS


class RalBaselineCfgTest(unittest.TestCase):
  def test_all_methods_keep_the_93d_one_frame_actor(self) -> None:
    for method in BASELINE_METHOD_TASK_NAMES:
      for arm in ("a0", "a6"):
        with self.subTest(method=method, arm=arm):
          cfg = g1_ral_baseline_env_cfg(method, arm)
          actor = cfg.observations["actor"]
          self.assertEqual(tuple(actor.terms), ACTOR_TERMS)
          self.assertNotIn("base_lin_vel", actor.terms)
          self.assertIsNone(actor.history_length)

  def test_task_only_has_no_motion_prior_dependency(self) -> None:
    cfg = g1_ral_baseline_env_cfg("task_only_ppo", "a6")
    self.assertNotIn("init_smp_state", cfg.events)
    self.assertNotIn("gsi_refresh", cfg.events)
    self.assertNotIn("gsi_reset", cfg.events)
    self.assertNotIn("mixed_fall_reset", cfg.events)
    self.assertNotIn("smp_too_low", cfg.terminations)
    self.assertIs(cfg.rewards["task_only"].func, task_only_reward)
    self.assertNotIn("task_smp_product", cfg.rewards)
    self.assertNotIn("smp_score", cfg.metrics)
    self.assertFalse(cfg.events["init_matched_reset_bank"].params["include_smp_window"])

  def test_original_product_restores_strict_smp_semantics(self) -> None:
    cfg = g1_ral_baseline_env_cfg("original_product_smp", "a6")
    reward = cfg.rewards["task_smp_product"]
    self.assertIs(reward.func, task_smp_product)
    self.assertNotIn("procedural_smp_floor", reward.params)
    self.assertIs(cfg.terminations["smp_too_low"].func, mdp.smp_too_low)
    self.assertTrue(cfg.events["init_matched_reset_bank"].params["include_smp_window"])

  def test_proposed_preserves_selected_arm_bridge_and_termination(self) -> None:
    cfg = g1_ral_baseline_env_cfg("proposed_smp_recovery", "a6")
    reward = cfg.rewards["task_smp_product"]
    self.assertIs(reward.func, mdp.procedural_bridge_task_smp_product)
    self.assertEqual(reward.params["procedural_smp_floor"], 0.10)
    self.assertIs(cfg.terminations["smp_too_low"].func, mdp.smp_too_low_gsi_only)

  def test_reset_bank_contract_is_identical_across_methods(self) -> None:
    for method in BASELINE_METHOD_TASK_NAMES:
      cfg = g1_ral_baseline_env_cfg(method, "a6")
      loader = cfg.events["init_matched_reset_bank"]
      self.assertEqual(loader.params["bank_path"], "")
      self.assertEqual(loader.params["bank_sha256"], "")
      self.assertEqual(loader.params["expected_num_states"], RESET_BANK_NUM_STATES)
      self.assertIsNone(loader.params["sampling_seed"])
      self.assertIs(
        cfg.events["matched_reset_bank_reset"].func, mdp.matched_reset_bank_reset
      )
      self.assertLess(
        list(cfg.events).index("init_matched_reset_bank"),
        list(cfg.events).index("matched_reset_bank_reset"),
      )
      startup = [name for name, term in cfg.events.items() if term.mode == "startup"]
      for common in ("foot_friction", "encoder_bias", "base_com"):
        self.assertLess(startup.index(common), startup.index("init_matched_reset_bank"))
      if method == "task_only_ppo":
        self.assertNotIn("init_smp_state", startup)
      else:
        self.assertLess(
          startup.index("init_smp_state"),
          startup.index("init_matched_reset_bank"),
        )

  def test_all_preregistered_task_ids_load(self) -> None:
    for _method, task_name in BASELINE_METHOD_TASK_NAMES.items():
      for arm in (f"a{index}" for index in range(8)):
        task = f"Smp-Getup-RAL-B-{task_name}-{arm.upper()}-G1"
        cfg = load_env_cfg(task)
        self.assertEqual(tuple(cfg.observations["actor"].terms), ACTOR_TERMS)


if __name__ == "__main__":
  unittest.main()
