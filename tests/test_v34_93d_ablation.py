from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from mjlab.tasks.registry import load_env_cfg, load_runner_cls

import smp.rl.tasks  # noqa: F401
from smp.rl.tasks.getup.escape_v34_93d_env_cfg import (
  g1_getup_escape_plate_v34_93d_smp_env_cfg,
)
from smp.rl.tasks.getup.escape_v34_env_cfg import (
  g1_getup_escape_plate_v34_smp_env_cfg,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import launch_smp_v34_93d_ablation as launcher
import project_smp_v34_checkpoint_to_93d as projection

_REPO = Path(__file__).parents[1]
_PROTOCOL = _REPO / "docs/ral_v34_93d_observation_ablation_v1.json"
_PROTOCOL_SHA256 = "11a0828e38ce83c27693b7b50a59778a5a67c690b4329bdf0fc2357a036d18ca"


class V34NinetyThreeDimAblationTest(unittest.TestCase):
  def test_only_actor_base_linear_velocity_term_is_removed(self) -> None:
    original = g1_getup_escape_plate_v34_smp_env_cfg(play=False)
    ablation = g1_getup_escape_plate_v34_93d_smp_env_cfg(play=False)
    original_terms = tuple(original.observations["actor"].terms)
    ablation_terms = tuple(ablation.observations["actor"].terms)
    self.assertEqual(original_terms[0], "base_lin_vel")
    self.assertEqual(ablation_terms, original_terms[1:])
    self.assertEqual(original.observations["critic"], ablation.observations["critic"])
    self.assertEqual(original.events, ablation.events)
    self.assertEqual(original.rewards, ablation.rewards)
    self.assertEqual(original.terminations, ablation.terminations)
    self.assertEqual(original.episode_length_s, ablation.episode_length_s)
    self.assertEqual(original.sim.mujoco.timestep, ablation.sim.mujoco.timestep)
    self.assertEqual(original.decimation, ablation.decimation)

  def test_registered_task_uses_fresh_optimizer_warm_start(self) -> None:
    cfg = load_env_cfg("Smp-Getup-Escape-Plate-V34-93D-G1")
    self.assertNotIn("base_lin_vel", cfg.observations["actor"].terms)
    self.assertEqual(
      load_runner_cls("Smp-Getup-Escape-Plate-V34-93D-G1").__name__,
      "SmpCurriculumWarmStartRunner",
    )

  def test_protocol_and_plan_freeze_the_single_factor(self) -> None:
    self.assertEqual(hashlib.sha256(_PROTOCOL.read_bytes()).hexdigest(), _PROTOCOL_SHA256)
    protocol = json.loads(_PROTOCOL.read_text())
    self.assertEqual(protocol["status"], "PREREGISTERED_READY_FOR_TRAINING")
    self.assertEqual(
      protocol["treatment"]["only_environment_change"],
      "remove the first three actor entries named base_lin_vel",
    )
    self.assertIn("A14 action target limiter", protocol["treatment"]["excluded"])
    plan = launcher.build_plan(
      launcher.V34NinetyThreeDimAblationCfg(protocol=_PROTOCOL)
    )
    self.assertEqual(plan["physical_device"], 1)
    self.assertEqual(plan["num_envs"], 4096)
    self.assertEqual(plan["max_iterations"], 12000)
    self.assertEqual(plan["policy_seed"], 20261701)
    self.assertEqual(plan["policy_seed"], plan["environment_seed"])
    self.assertEqual(plan["runner"], "SmpCurriculumWarmStartRunner")
    self.assertIn("--agent.resume", plan["command"])

  def test_projection_is_zero_velocity_function_preserving(self) -> None:
    generator = torch.Generator().manual_seed(7)
    mean = torch.randn(1, 96, generator=generator)
    std = torch.rand(1, 96, generator=generator) + 0.2
    actor = {
      "obs_normalizer._mean": mean,
      "obs_normalizer._var": std.square(),
      "obs_normalizer._std": std,
      "obs_normalizer.count": torch.tensor(100),
      "distribution.std_param": torch.ones(29),
      "mlp.0.weight": torch.randn(512, 96, generator=generator),
      "mlp.0.bias": torch.randn(512, generator=generator),
    }
    critic = {"mlp.0.weight": torch.randn(512, 960, generator=generator)}
    checkpoint = {
      "iter": 98000,
      "actor_state_dict": actor,
      "critic_state_dict": critic,
      "optimizer_state_dict": {"state": {1: {"exp_avg": torch.ones(2)}}},
      "infos": {},
    }
    source = Path(self.id())
    with mock.patch.object(torch, "load", return_value=checkpoint):
      projected, audit = projection.project_checkpoint(source)
    self.assertEqual(tuple(projected["actor_state_dict"]["mlp.0.weight"].shape), (512, 93))
    self.assertEqual(
      tuple(projected["actor_state_dict"]["obs_normalizer._mean"].shape),
      (1, 93),
    )
    self.assertEqual(tuple(projected["critic_state_dict"]["mlp.0.weight"].shape), (512, 960))
    self.assertEqual(projected["optimizer_state_dict"], {})
    self.assertLessEqual(audit["zero_velocity_first_layer_max_abs_error"], 5.0e-5)
    self.assertTrue(audit["all_tensors_finite"])


if __name__ == "__main__":
  unittest.main()
