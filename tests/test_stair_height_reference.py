"""Height rewards must be invariant to an unrelated upper stair platform."""

import unittest
from types import SimpleNamespace

import torch

from smp.rl.tasks.getup.mdp import rewards


class Scene(dict):
  env_origins = torch.tensor([[0.0, 0.0, 1.2]])


class HeightReferenceTests(unittest.TestCase):
  def make_env(self):
    robot = SimpleNamespace(
      find_sites=lambda *args, **kwargs: ([0], ["head"]),
      data=SimpleNamespace(
        site_pos_w=torch.tensor([[[0.0, 0.0, 1.15]]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
      ),
    )
    return SimpleNamespace(
      scene=Scene(robot=robot),
      _v4_recovery_stage=torch.zeros(1, dtype=torch.long),
    )

  def test_bottom_standing_rewards_use_local_floor(self):
    env = self.make_env()
    env._recovery_support_height = torch.zeros(1)
    height = rewards.track_head_height(
      env, target_height=1.15, relative_to_env_origin=True
    )
    initiation = rewards.recovery_initiation_progress(env, relative_to_env_origin=True)
    torch.testing.assert_close(height, torch.ones(1))
    torch.testing.assert_close(initiation, torch.ones(1))

  def test_original_task_keeps_original_reference(self):
    env = self.make_env()
    torch.testing.assert_close(
      rewards._head_height(env, relative_to_env_origin=True), torch.tensor([-0.05])
    )
    torch.testing.assert_close(rewards._head_height(env), torch.tensor([1.15]))


if __name__ == "__main__":
  unittest.main()
