from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import validate_smp_progression_protocol as validator


class ProgressionProtocolTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    path = Path(__file__).parents[1] / "docs" / "ral_terrain_plate_protocol.json"
    cls.payload = json.loads(path.read_text())

  def test_frozen_protocol_is_valid(self) -> None:
    result = validator.validate(copy.deepcopy(self.payload))
    self.assertEqual(result["status"], "VALID")
    self.assertEqual(result["actor_dimension"], 93)
    self.assertEqual(result["phases"], ["T", "P", "U"])

  def test_rejects_plate_in_terrain_specialist(self) -> None:
    payload = copy.deepcopy(self.payload)
    payload["phases"]["T"]["training_distribution"]["plate_probability"] = 0.1
    with self.assertRaisesRegex(ValueError, "T must not contain a plate"):
      validator.validate(payload)

  def test_rejects_unprivileged_actor_regression(self) -> None:
    payload = copy.deepcopy(self.payload)
    payload["actor_contract"]["dimension"] = 96
    with self.assertRaisesRegex(ValueError, "93D"):
      validator.validate(payload)

  def test_rejects_unified_phase_without_both_specialists(self) -> None:
    payload = copy.deepcopy(self.payload)
    payload["phases"]["U"]["promotion_gates"]["must_pass_all_P_gates"] = False
    with self.assertRaisesRegex(ValueError, "both specialist"):
      validator.validate(payload)


if __name__ == "__main__":
  unittest.main()
