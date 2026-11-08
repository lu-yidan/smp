"""Verify the final-target score cannot saturate at a seated waypoint."""

import unittest
from types import SimpleNamespace

import torch

from smp.rl.tasks.getup.stair_recovery_finish_env_cfg import final_posture_progress


class Scene(dict):
  env_origins = torch.tensor([[0.0, 0.0, 1.4]]).repeat(5, 1)


class FinishRewardTests(unittest.TestCase):
  def test_fixed_final_target_and_local_height(self):
    # Floor sit, crouch, near-standing, standing, tall but tilted.
    height = torch.tensor([0.62, 0.86, 1.0, 1.15, 1.15])
    upright = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.5])
    data = SimpleNamespace(
      site_pos_w=torch.zeros(5, 1, 3), projected_gravity_b=torch.zeros(5, 3)
    )
    data.site_pos_w[:, 0, 2] = height
    data.projected_gravity_b[:, 2] = -upright
    robot = SimpleNamespace(data=data, find_sites=lambda *a, **kw: ([0], ["head"]))
    env = SimpleNamespace(
      scene=Scene(robot=robot),
      _recovery_support_height=torch.zeros(5),
      _v4_recovery_stage=torch.zeros(5, dtype=torch.long),
    )
    scores = final_posture_progress(env)
    torch.testing.assert_close(scores, torch.tensor([0.0, 0.1764, 0.49, 1.0, 0.0]))
    env._v4_recovery_stage.fill_(3)
    torch.testing.assert_close(final_posture_progress(env), scores)
    # Translating both robot and its supporting surface leaves the reward unchanged.
    data.site_pos_w[:, 0, 2] += 0.4
    env._recovery_support_height += 0.4
    torch.testing.assert_close(final_posture_progress(env), scores)


if __name__ == "__main__":
  unittest.main()
