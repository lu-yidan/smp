from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import evaluate_smp_baseline as evaluator


class FailureReasonTelemetryTest(unittest.TestCase):
  def test_first_step_is_one_based_and_immutable_after_first_hit(self) -> None:
    first = torch.tensor([-1, -1, 4], dtype=torch.long)
    evaluator._record_first_step(
      first,
      torch.tensor([True, False, True]),
      step=7,
    )
    self.assertEqual(first.tolist(), [8, -1, 4])

  def test_every_frozen_reason_is_mutually_exclusive(self) -> None:
    count = len(evaluator._FAILURE_REASON_NAMES)
    strict_success = torch.zeros(count, dtype=torch.bool)
    strict_success[0] = True
    finite = torch.ones(count, dtype=torch.bool)
    invalid_dynamics = torch.zeros(count, dtype=torch.bool)
    terrain_exit = torch.zeros(count, dtype=torch.bool)
    invalid_escape_setup = torch.zeros(count, dtype=torch.bool)
    invalid_escape_contact = torch.zeros(count, dtype=torch.bool)
    plate_present = torch.zeros(count, dtype=torch.bool)
    escape_reached = torch.ones(count, dtype=torch.bool)
    head_reached = torch.ones(count, dtype=torch.bool)
    upright_reached = torch.ones(count, dtype=torch.bool)
    linear_speed_reached = torch.ones(count, dtype=torch.bool)
    angular_speed_reached = torch.ones(count, dtype=torch.bool)

    finite[1] = False
    invalid_dynamics[1] = True
    invalid_dynamics[2] = True
    terrain_exit[3] = True
    invalid_escape_setup[4] = True
    invalid_escape_contact[5] = True
    plate_present[6] = True
    escape_reached[6] = False
    head_reached[7] = False
    upright_reached[8] = False
    linear_speed_reached[9] = False
    angular_speed_reached[10] = False

    reason, counts = evaluator._classify_strict_failure_reasons(
      strict_success=strict_success,
      finite_action=finite,
      invalid_dynamics=invalid_dynamics,
      terrain_exit=terrain_exit,
      invalid_escape_setup=invalid_escape_setup,
      invalid_escape_contact=invalid_escape_contact,
      plate_present=plate_present,
      escape_reached=escape_reached,
      head_reached=head_reached,
      upright_reached=upright_reached,
      linear_speed_reached=linear_speed_reached,
      angular_speed_reached=angular_speed_reached,
    )

    self.assertEqual(reason.tolist(), list(range(count)))
    self.assertEqual(counts, {name: 1 for name in evaluator._FAILURE_REASON_NAMES})
    self.assertEqual(sum(counts.values()), count)

  def test_shape_drift_fails_closed(self) -> None:
    values = torch.ones(2, dtype=torch.bool)
    kwargs = {
      "strict_success": values,
      "finite_action": values[:1],
      "invalid_dynamics": ~values,
      "terrain_exit": ~values,
      "invalid_escape_setup": ~values,
      "invalid_escape_contact": ~values,
      "plate_present": ~values,
      "escape_reached": values,
      "head_reached": values,
      "upright_reached": values,
      "linear_speed_reached": values,
      "angular_speed_reached": values,
    }
    with self.assertRaisesRegex(ValueError, "identical shapes"):
      evaluator._classify_strict_failure_reasons(**kwargs)


if __name__ == "__main__":
  unittest.main()
