"""Prompt-load, body-aligned guided-board escape curriculum."""

from __future__ import annotations

import mujoco
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.escape_v3_env_cfg import (
  g1_getup_escape_plate_v3_smp_env_cfg,
)


def get_body_aligned_escape_board_spec() -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Create a larger 8 kg board with one passive vertical slide joint."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="escape_plate")
  body.add_joint(
    name="escape_plate_slide",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=(0.0, 0.0, 1.0),
    limited=True,
    range=(-1.20, 0.0),
    damping=60.0,
  )
  body.add_geom(
    name="escape_plate_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    # Full dimensions: 0.90 x 0.64 x 0.07 m.
    size=(0.45, 0.32, 0.035),
    mass=8.0,
    friction=(1.2, 0.01, 0.001),
    rgba=(0.10, 0.68, 0.32, 0.82),
  )
  return spec


def g1_getup_escape_plate_v31_smp_env_cfg(play: bool = False):
  """Build V3.1 with prompt prone loading and required supported progress."""
  cfg = g1_getup_escape_plate_v3_smp_env_cfg(play=play)
  cfg.scene.entities["escape_obstacle"].spec_fn = get_body_aligned_escape_board_spec

  cfg.events["reset_escape_obstacle"].params.update(
    {
      "body_origin_clearance": 0.22,
      "align_to_body": True,
      # Shift away from the head so the long board covers chest and pelvis.
      "longitudinal_offset": -0.10,
      "xy_offset_range": 0.010,
    }
  )
  cfg.events["update_escape_phase"].params.update(
    {
      "separation_threshold": 0.50,
      # Contact must occur within 0.5 s and before the robot gets tall.
      "max_wait_steps": 35,
      "max_initial_contact_head_height": 0.75,
      # Escape is not credited without actual hand-supported translation.
      "hand_sensor_name": "hand_ground_contact",
      "min_hand_support_steps": 5,
      "min_hand_supported_progress": 0.04,
    }
  )

  # Do not reward initiating an upright recovery while the board is descending.
  cfg.rewards.pop("recovery_initiation", None)
  cfg.rewards["task_smp_product"].params["constrained_scale"] = 0.08
  cfg.rewards["hand_supported_escape_progress"].weight = 0.40
  cfg.rewards["escape_separation_progress"].weight = 0.05
  cfg.rewards["escape_completion"].weight = 0.30

  cfg.terminations.pop("invalid_escape_contact", None)
  cfg.terminations["invalid_escape_episode"] = TerminationTermCfg(
    func=mdp.invalid_escape_episode
  )
  cfg.metrics.update(
    {
      "escape_invalid_setup": MetricsTermCfg(
        func=mdp.escape_invalid_setup_metric, reduce="last"
      ),
      "escape_first_contact_head_height": MetricsTermCfg(
        func=mdp.escape_first_contact_head_height_metric, reduce="last"
      ),
      "escape_hand_support_steps": MetricsTermCfg(
        func=mdp.escape_hand_support_steps_metric, reduce="last"
      ),
      "escape_hand_supported_progress": MetricsTermCfg(
        func=mdp.escape_hand_supported_progress_metric, reduce="last"
      ),
    }
  )
  return cfg
