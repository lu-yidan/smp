from __future__ import annotations

import unittest

import torch
from mjlab.tasks.registry import load_runner_cls

from smp.rl.actions import (
  RateLimitedJointPositionActionCfg,
  bounded_target_step,
)
from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  g1_scratch_a13_f2s2_continuous_reset_env_cfg,
  g1_scratch_a14_f2s2_velocity_envelope_env_cfg,
)
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner


class VelocityEnvelopeFinetuneTest(unittest.TestCase):
  def test_discrete_target_envelope_is_hard_bounded(self) -> None:
    current = torch.zeros(2, 3)
    velocity = torch.zeros_like(current)
    desired = torch.tensor([[10.0, -10.0, 0.1], [-2.0, 3.0, -4.0]])
    next_target, next_velocity = bounded_target_step(
      current,
      velocity,
      desired,
      dt=0.02,
      max_velocity=4.0,
      max_acceleration=30.0,
    )
    self.assertLessEqual(float(next_velocity.abs().max()), 4.0 + 1.0e-6)
    self.assertLessEqual(
      float((next_velocity - velocity).abs().max() / 0.02), 30.0 + 1.0e-5
    )
    self.assertTrue(torch.allclose(next_target, next_velocity * 0.02))

  def test_a14_changes_action_transform_and_adds_tail_costs_only(self) -> None:
    a13 = g1_scratch_a13_f2s2_continuous_reset_env_cfg(play=False)
    a14 = g1_scratch_a14_f2s2_velocity_envelope_env_cfg(play=False)
    self.assertEqual(tuple(a13.observations), tuple(a14.observations))
    self.assertEqual(tuple(a13.events), tuple(a14.events))
    self.assertEqual(tuple(a13.terminations), tuple(a14.terminations))
    for name in a13.events:
      self.assertIs(a13.events[name].func, a14.events[name].func)
      self.assertEqual(a13.events[name].params, a14.events[name].params)
    self.assertIsInstance(a14.actions["joint_pos"], RateLimitedJointPositionActionCfg)
    self.assertEqual(a14.actions["joint_pos"].max_target_velocity, 4.0)
    self.assertEqual(a14.actions["joint_pos"].max_target_acceleration, 30.0)
    self.assertEqual(a14.sim.mujoco.timestep, 0.005)
    self.assertEqual(a14.decimation, 4)
    self.assertEqual(set(a14.rewards) - set(a13.rewards), {
      "joint_speed_tail_barrier",
      "target_velocity_soft_barrier",
    })

  def test_a14_uses_fresh_optimizer_warm_start_runner(self) -> None:
    task = "Smp-Getup-Scratch-A14-F2S2-Velocity-Envelope-G1"
    self.assertIs(load_runner_cls(task), SmpCurriculumWarmStartRunner)


if __name__ == "__main__":
  unittest.main()
