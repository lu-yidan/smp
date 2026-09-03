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
import evaluate_escape_checkpoint as escape_evaluator
import launch_smp_v34_93d_ablation as launcher
import project_smp_v34_checkpoint_to_93d as projection
import run_smp_v34_93d_eval_matrix as eval_launcher

_REPO = Path(__file__).parents[1]
_PROTOCOL = _REPO / "docs/ral_v34_93d_observation_ablation_v1.json"
_PROTOCOL_SHA256 = "11a0828e38ce83c27693b7b50a59778a5a67c690b4329bdf0fc2357a036d18ca"
_EVAL_PROTOCOL = _REPO / "docs/ral_v34_93d_evaluation_v1.json"
_EVAL_PROTOCOL_SHA256 = (
  "1f212abf7e627ac031fcca6637e69b5f43c2c94e6f703bfbc8de6e676734f52e"
)


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
    self.assertEqual(
      escape_evaluator._inference_runner_cls(
        "Smp-Getup-Escape-Plate-V34-93D-G1"
      ).__name__,
      "MjlabOnPolicyRunner",
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

  def test_evaluation_protocol_and_matrix_are_frozen(self) -> None:
    self.assertEqual(
      hashlib.sha256(_EVAL_PROTOCOL.read_bytes()).hexdigest(),
      _EVAL_PROTOCOL_SHA256,
    )
    integrity = {
      "embedded_iteration": 0,
      "actor_input_dim": 93,
      "critic_input_dim": 960,
      "all_tensors_finite": True,
    }
    with mock.patch.object(
      eval_launcher, "_checkpoint_integrity", return_value=integrity
    ):
      plan = eval_launcher.build_plan(
        eval_launcher.V34NinetyThreeDimEvalCfg(protocol=_EVAL_PROTOCOL)
      )
    self.assertEqual(plan["total_cells"], 14)
    self.assertEqual(plan["matrix"]["num_envs_per_cell"], 512)
    self.assertEqual(plan["matrix"]["steps_per_environment"], 1000)
    self.assertEqual(plan["matrix"]["evaluation_seed"], 20261710)
    self.assertEqual(
      [job["cell_id"] for job in plan["jobs"][:4]],
      [
        "v34_96d_gate98000_prone",
        "v34_96d_gate98000_supine",
        "v34_93d_gate0_prone",
        "v34_93d_gate0_supine",
      ],
    )
    self.assertEqual(
      [job["physical_device"] for job in plan["jobs"]],
      [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5],
    )


if __name__ == "__main__":
  unittest.main()
