from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from launch_smp_prone_coverage_finetune import (  # noqa: E402
  _PROTOCOL_SHA256,
  _validate_protocol,
)


class ProneCoverageProtocolTest(unittest.TestCase):
  def test_protocol_is_hash_locked_and_single_factor(self) -> None:
    path = Path(__file__).parents[1] / "docs/ral_prone_coverage_finetune_v1.json"
    protocol, digest = _validate_protocol(path)
    self.assertEqual(digest, _PROTOCOL_SHA256)
    self.assertEqual(
      protocol["treatment"]["reset_distribution"]["mode_weights"],
      [3.0, 1.0, 1.0, 1.0],
    )
    self.assertEqual(protocol["training_protocol"]["max_iterations"], 5000)
    self.assertEqual(protocol["training_protocol"]["policy_seed"], 20261401)
    self.assertTrue(protocol["claim_boundary"]["not_ral_evidence"])


if __name__ == "__main__":
  unittest.main()
