"""Contact-safe guided-plate escape curriculum on top of the V7 policy."""

from __future__ import annotations

import os

import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v7_env_cfg import g1_getup_v7_route_smp_env_cfg


def get_guided_escape_plate_spec() -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Create a passive plate whose only physical degree of freedom is vertical.

  MjLab automatically wraps this fixed-base articulated entity in a per-environment
  mocap root.  The root is positioned over the reset torso once; the slide joint
  then lets gravity, the robot, and contact forces move the plate only along z.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="escape_plate")
  body.add_joint(
    name="escape_plate_slide",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=(0.0, 0.0, 1.0),
    limited=True,
    range=(-1.20, 0.0),
    damping=120.0,
  )
  body.add_geom(
    name="escape_plate_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.26, 0.26, 0.035),
    mass=5.0,
    friction=(1.2, 0.01, 0.001),
    rgba=(0.10, 0.42, 0.90, 0.82),
  )
  return spec


def g1_getup_escape_plate_v3_smp_env_cfg(play: bool = False):
  """Build V3: hand-supported lateral escape under a guided physical plate."""
  cfg = g1_getup_v7_route_smp_env_cfg(play=play)
  cfg.episode_length_s = 20.0
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 2500

  cfg.scene.entities["escape_obstacle"] = EntityCfg(
    spec_fn=get_guided_escape_plate_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(1.20, 1.20, 0.80),
      joint_pos={"escape_plate_slide": 0.0},
      joint_vel={"escape_plate_slide": 0.0},
    ),
  )

  hand_ground = ContactSensorCfg(
    name="hand_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"(left|right)_hand_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  robot_plate = ContactSensorCfg(
    name="robot_obstacle_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="body",
      pattern="escape_plate",
      entity="escape_obstacle",
    ),
    fields=("found", "force", "dist"),
    reduce="mindist",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (hand_ground, robot_plate)

  # Training isolates the first escape skill.  Playback may explicitly retain
  # V6's post-stand wrench, independently of whether the plate is enabled.
  if not (play and os.environ.get("SMP_PLAY_AUTO_DISTURBANCES") == "1"):
    cfg.events.pop("stratified_post_stand_wrench", None)
  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 0.95,
      "mode_weights": (0.0, 1.0, 0.0, 0.0),
      "root_height_range": (0.38, 0.50),
      "root_xy_range": 0.04,
      "joint_noise": 0.18,
      "orientation_noise": 0.10,
      "root_linear_velocity": 0.02,
      "root_angular_velocity": 0.05,
    }
  )
  obstacle_probability = 0.90
  if play:
    obstacle_probability = float(
      os.environ.get("SMP_PLAY_ESCAPE_OBSTACLE", "1") == "1"
    )
  cfg.events["reset_escape_obstacle"] = EventTermCfg(
    func=mdp.reset_guided_escape_plate,
    mode="reset",
    params={
      "obstacle_probability": obstacle_probability,
      "target_body_name": "torso_link",
      "eligible_reset_types": (2,),
      "xy_offset_range": 0.015,
      # V2's torso-centre offset intersected raised hands/feet.  V3 places the
      # plate above the highest robot body origin, then lets gravity establish
      # contact.  The margin includes the plate half-thickness.
      "body_origin_clearance": 0.26,
      "inactive_xy": (1.20, 1.20),
    },
  )
  cfg.events["update_escape_phase"] = EventTermCfg(
    func=mdp.update_escape_phase,
    mode="step",
    params={
      "sensor_name": "robot_obstacle_contact",
      "clear_hold_steps": 15,
      "separation_threshold": 0.38,
      "max_penetration": 0.020,
      "max_contact_force": 1500.0,
    },
  )

  cfg.rewards["task_smp_product"].func = mdp.escape_gated_task_smp_product
  cfg.rewards["task_smp_product"].params["constrained_scale"] = 0.10
  cfg.rewards.update(
    {
      "hand_supported_escape_progress": RewardTermCfg(
        func=mdp.hand_supported_escape_progress,
        weight=0.30,
        params={
          "sensor_name": "hand_ground_contact",
          "progress_scale": 0.025,
          "max_head_height": 0.90,
        },
      ),
      "escape_separation_progress": RewardTermCfg(
        func=mdp.escape_separation_progress,
        weight=0.10,
        params={"progress_scale": 0.025},
      ),
      "escape_contact_force_excess": RewardTermCfg(
        func=mdp.escape_contact_force_excess_l2,
        weight=-0.03,
        params={
          "sensor_name": "robot_obstacle_contact",
          "force_limit": 300.0,
          "force_scale": 300.0,
        },
      ),
      "escape_completion": RewardTermCfg(
        func=mdp.escape_completion,
        weight=0.35,
      ),
    }
  )
  cfg.terminations["invalid_escape_contact"] = TerminationTermCfg(
    func=mdp.invalid_escape_contact
  )
  cfg.metrics.update(
    {
      "escape_phase": MetricsTermCfg(func=mdp.escape_phase_metric, reduce="last"),
      "escape_completion": MetricsTermCfg(func=mdp.escape_completion, reduce="last"),
      "escape_obstacle_episode": MetricsTermCfg(
        func=mdp.escape_obstacle_episode_metric, reduce="last"
      ),
      "escape_invalid_contact": MetricsTermCfg(
        func=mdp.escape_invalid_contact_metric, reduce="last"
      ),
      "escape_peak_penetration": MetricsTermCfg(
        func=mdp.escape_peak_penetration_metric, reduce="last"
      ),
      "escape_peak_contact_force": MetricsTermCfg(
        func=mdp.escape_peak_contact_force_metric, reduce="last"
      ),
      "hand_support_contact": MetricsTermCfg(
        func=mdp.hand_support_contact_metric,
        params={"sensor_name": "hand_ground_contact"},
      ),
      "hand_supported_escape_progress": MetricsTermCfg(
        func=mdp.hand_supported_escape_progress,
        params={"sensor_name": "hand_ground_contact"},
      ),
    }
  )
  return cfg
