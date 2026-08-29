from __future__ import annotations

import unittest

import torch

from smp.firm.goal_adapter import FirmGoalAdapter


class FirmGoalAdapterShapeTest(unittest.TestCase):
  def test_one_frame_93d_adapter(self) -> None:
    adapter = FirmGoalAdapter(observation_dim=93, history_steps=1)
    output = adapter(torch.zeros(4, 1, 93))
    self.assertEqual(tuple(output.shape), (4, 64))
    self.assertTrue(torch.isfinite(output).all())

  def test_fifty_frame_93d_adapter(self) -> None:
    adapter = FirmGoalAdapter(observation_dim=93, history_steps=50)
    output = adapter(torch.zeros(4, 50, 93))
    self.assertEqual(tuple(output.shape), (4, 64))

  def test_legacy_fifty_frame_layout_remains_loadable(self) -> None:
    original = FirmGoalAdapter(observation_dim=90, history_steps=50)
    restored = FirmGoalAdapter(observation_dim=90, history_steps=50)
    restored.load_state_dict(original.state_dict(), strict=True)


if __name__ == "__main__":
  unittest.main()
