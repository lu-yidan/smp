"""Settled, crawl-ready initial contact for guided-board escape."""

from __future__ import annotations

import os

import mujoco
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.escape_v31_env_cfg import (
  g1_getup_escape_plate_v31_smp_env_cfg,
)


def get_settled_escape_board_spec() -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Create the V3.2 board with locally stiff, load-bearing contacts."""
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
  geom = body.add_geom(
    name="escape_plate_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.45, 0.32, 0.035),
    mass=8.0,
    friction=(1.2, 0.01, 0.001),
    rgba=(0.12, 0.72, 0.24, 0.82),
  )
  # Higher priority makes these parameters win only for plate contacts.  The
  # floor and all non-plate robot contacts retain their original compliance.
  geom.priority = 1
  geom.solref = (0.01, 1.0)
  geom.solimp = (0.98, 0.995, 0.001, 0.5, 2.0)
  return spec


def g1_getup_escape_plate_v32_smp_env_cfg(play: bool = False):
  """Build V3.2: begin prone, grounded, and within millimetres of the load."""
  cfg = g1_getup_escape_plate_v31_smp_env_cfg(play=play)
  cfg.scene.entities["escape_obstacle"].spec_fn = get_settled_escape_board_spec
  cfg.events["mixed_fall_reset"].params.update(
    {
      # Keep continuous pose coverage without tilting the prepared prone body
      # far enough that the head or one shoulder becomes the sole support.
      "joint_noise": 0.10,
      "orientation_noise": 0.04,
    }
  )
  cfg.events["reset_escape_obstacle"].params.update(
    {
      "crawl_ready_prone": True,
      "crawl_arm_noise": 0.035,
      "ground_clearance": 0.004,
      "surface_gap": 0.001,
      "plate_half_extents": (0.45, 0.32, 0.035),
      "collision_geom_pattern": r".*_collision$",
    }
  )
  cfg.events["update_escape_phase"].params.update(
    {
      # With a 1 mm reset gap the board must contact before any recovery motion.
      "max_wait_steps": 12,
      "max_initial_contact_head_height": 0.40,
    }
  )
  cfg.terminations["unstable_sim_state"] = TerminationTermCfg(
    func=mdp.unstable_sim_state
  )
  if play:
    # The plate is the task condition, not an optional automatic disturbance.
    # It defaults on and has its own explicit play override.
    obstacle_enabled = os.environ.get("SMP_PLAY_ESCAPE_OBSTACLE", "1") == "1"
    cfg.events["mixed_fall_reset"].params["procedural_probability"] = 1.0
    cfg.events["reset_escape_obstacle"].params["obstacle_probability"] = float(
      obstacle_enabled
    )
  return cfg
