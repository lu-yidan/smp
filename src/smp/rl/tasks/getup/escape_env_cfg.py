"""Physical-object escape curriculum on top of the V7 get-up policy."""

from __future__ import annotations

import os

import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v7_env_cfg import g1_getup_v7_route_smp_env_cfg


def get_escape_obstacle_spec() -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Create a visible padded plate that can be pushed or crawled out from."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="escape_obstacle")
  body.add_freejoint(name="escape_obstacle_joint")
  body.add_geom(
    name="escape_obstacle_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.20, 0.42, 0.045),
    mass=8.0,
    friction=(1.2, 0.01, 0.001),
    rgba=(0.85, 0.25, 0.08, 0.85),
  )
  return spec


def g1_getup_escape_smp_env_cfg(play: bool = False):
  """Build V2: hand-supported escape from a movable physical obstacle."""
  cfg = g1_getup_v7_route_smp_env_cfg(play=play)
  cfg.episode_length_s = 20.0
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 2500

  cfg.scene.entities["escape_obstacle"] = EntityCfg(
    spec_fn=get_escape_obstacle_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.72, 0.72, 0.055)),
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
  robot_obstacle = ContactSensorCfg(
    name="robot_obstacle_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="body",
      pattern="escape_obstacle",
      entity="escape_obstacle",
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (hand_ground, robot_obstacle)

  # Training isolates object escape from V6's post-standing knockdown
  # curriculum. Playback can request it independently from the obstacle.
  if not (play and os.environ.get("SMP_PLAY_AUTO_DISTURBANCES") == "1"):
    cfg.events.pop("stratified_post_stand_wrench", None)
  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 0.90,
      # Historical V2 used reset type 2, which is physically supine.
      "mode_weights": (0.0, 1.0, 0.0, 0.0),
      "root_height_range": (0.38, 0.54),
      "joint_noise": 0.22,
    }
  )
  obstacle_probability = 0.80
  if play:
    obstacle_probability = float(
      os.environ.get("SMP_PLAY_ESCAPE_OBSTACLE", "1") == "1"
    )
  cfg.events["reset_escape_obstacle"] = EventTermCfg(
    func=mdp.reset_escape_obstacle,
    mode="reset",
    params={
      "obstacle_probability": obstacle_probability,
      "target_body_names": ("torso_link",),
      "target_weights": (1.0,),
      "eligible_reset_types": (2,),
      "xy_offset_range": 0.015,
      "clearance": 0.080,
    },
  )
  cfg.events["update_escape_phase"] = EventTermCfg(
    func=mdp.update_escape_phase,
    mode="step",
    params={
      "sensor_name": "robot_obstacle_contact",
      "clear_hold_steps": 15,
      "separation_threshold": 0.24,
    },
  )

  # While pinned, reduce the height/upright objective and reward controlled
  # hand-supported translation plus genuine new separation. Once escaped, the
  # original V7 get-up objective automatically resumes at full strength.
  cfg.rewards["task_smp_product"].func = mdp.escape_gated_task_smp_product
  cfg.rewards["task_smp_product"].params["constrained_scale"] = 0.15
  cfg.rewards.update(
    {
      "crawl_with_hand_support": RewardTermCfg(
        func=mdp.crawl_with_hand_support,
        weight=0.10,
        params={
          "sensor_name": "hand_ground_contact",
          "target_speed": 0.10,
          "speed_scale": 55.0,
          "max_head_height": 0.90,
        },
      ),
      "escape_separation_progress": RewardTermCfg(
        func=mdp.escape_separation_progress,
        weight=0.20,
        params={"progress_scale": 0.025},
      ),
      "escape_completion": RewardTermCfg(
        func=mdp.escape_completion,
        weight=0.25,
      ),
    }
  )
  cfg.metrics.update(
    {
      "escape_phase": MetricsTermCfg(func=mdp.escape_phase_metric, reduce="last"),
      "escape_completion": MetricsTermCfg(
        func=mdp.escape_completion, reduce="last"
      ),
      "escape_object_displacement": MetricsTermCfg(
        func=mdp.escape_object_displacement_metric, reduce="last"
      ),
      "escape_obstacle_episode": MetricsTermCfg(
        func=mdp.escape_obstacle_episode_metric, reduce="last"
      ),
      "hand_support_contact": MetricsTermCfg(
        func=mdp.hand_support_contact_metric,
        params={"sensor_name": "hand_ground_contact"},
      ),
      "crawl_with_hand_support": MetricsTermCfg(
        func=mdp.crawl_with_hand_support,
        params={"sensor_name": "hand_ground_contact"},
      ),
    }
  )
  return cfg
