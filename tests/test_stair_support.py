"""Check invariances and avoid penalizing necessary staggered recovery support."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from smp.rl.tasks.getup.stair_recovery_93d_env_cfg import surface_height
from smp.rl.tasks.getup.stair_support_env_cfg import (
  descending_support,
  heading_components,
  stance_cost,
  support_gate,
)


class StairSupportTests(unittest.TestCase):
  def test_yaw_invariance_and_staggered_stance(self):
    delta = torch.tensor([[0.4, 0.24], [0.4, 0.80], [0.4, -0.1]])
    quat = torch.tensor([[1.0, 0, 0, 0]]).repeat(3, 1)
    forward, lateral = heading_components(delta, quat)
    cost = stance_cost(forward, lateral, torch.ones(3), 0.24)
    self.assertEqual(float(cost[0]), 0.0)  # adjacent-tread fore-aft offset allowed
    self.assertGreater(float(cost[1]), 0)
    self.assertGreater(float(cost[2]), 0)  # crossed feet are not rewarded
    rotated = torch.stack((-delta[:, 1], delta[:, 0]), -1)
    quat[:] = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    f2, l2 = heading_components(rotated, quat)
    torch.testing.assert_close(f2, forward)
    torch.testing.assert_close(l2, lateral)

  def test_low_pose_free_and_crouch_activation(self):
    gate = support_gate(torch.tensor([0.4, 0.85, 1.15]), torch.tensor([0.9] * 3))
    torch.testing.assert_close(gate, torch.tensor([0.0, 0.5, 1.0]))
    cost = stance_cost(torch.zeros(3), torch.full((3,), 0.8), gate, 0.24)
    self.assertEqual(float(cost[0]), 0.0)
    self.assertGreater(float(cost[2]), float(cost[1]))

  def test_actual_nested_tread_heights_and_descent_reference(self):
    # Base slab with a raised 30cm tread centered at zero.
    boxes = torch.tensor(
      [[[0.0, 0.0, -0.1, 2.0, 2.0, 0.1], [0.0, 0.0, 0.075, 0.15, 2.0, 0.075]]]
    )
    xy = torch.tensor([[[0.0, 0.0], [0.14, 0.0], [0.20, 0.0]]])
    with patch(
      "smp.rl.tasks.getup.stair_recovery_93d_env_cfg._boxes", return_value=boxes
    ):
      h = surface_height(SimpleNamespace(), torch.tensor([0]), xy)
    torch.testing.assert_close(h, torch.tensor([[0.15, 0.15, 0.0]]))
    start = torch.tensor([0.15, 0.15, 0.0])
    current = torch.tensor([0.15, 0.0, 0.0])
    loss, dropped = descending_support(start, current, torch.full((3,), 0.15))
    self.assertEqual(dropped.tolist(), [False, True, False])
    # Moving world origin up does not change drop detection; base crouch height
    # is deliberately absent from this computation.
    l2, d2 = descending_support(start + 2, current + 2, torch.full((3,), 0.15))
    torch.testing.assert_close(loss, l2)
    self.assertTrue(torch.equal(dropped, d2))


if __name__ == "__main__":
  unittest.main()
