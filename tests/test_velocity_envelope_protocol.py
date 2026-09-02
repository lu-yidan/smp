from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from launch_smp_velocity_envelope_finetune import (  # noqa: E402
  _PROTOCOL_SHA256,
  _validate_protocol,
)


class VelocityEnvelopeProtocolTest(unittest.TestCase):
  def test_protocol_is_hash_locked_and_launch_eligible(self) -> None:
    path = Path(__file__).parents[1] / "docs/ral_velocity_envelope_finetune_v1.json"
    protocol, digest = _validate_protocol(path)
    self.assertEqual(digest, _PROTOCOL_SHA256)
    self.assertEqual(protocol["source_policy"]["checkpoint_name"], "model_4999.pt")
    envelope = protocol["treatment"]["action_envelope"]
    self.assertEqual(envelope["max_joint_position_target_velocity_rad_s"], 4.0)
    self.assertEqual(
      envelope["max_joint_position_target_acceleration_rad_s2"], 30.0
    )
    training = protocol["training_protocol"]
    self.assertEqual(training["max_iterations"], 8000)
    self.assertEqual(training["policy_seed"], 20261601)
    self.assertEqual(training["wandb_mode"], "offline")
    audit = protocol["implementation_audit"]
    self.assertEqual(audit["status"], "PASSED_REAL_MUJOCO_WARM_START_SMOKE")
    self.assertEqual(audit["verified"]["actor_input_dim"], 93)
    self.assertTrue(protocol["claim_boundary"]["not_ral_evidence"])
    self.assertTrue(
      protocol["claim_boundary"][
        "not_authorized_for_unprotected_real_robot_deployment"
      ]
    )


if __name__ == "__main__":
  unittest.main()
