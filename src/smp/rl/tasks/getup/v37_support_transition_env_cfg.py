"""V37 flat actuator-feasibility and bilateral-support continuation study."""

from __future__ import annotations

import copy
import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.actions import RateLimitedJointPositionActionCfg
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.escape_v35_93d_reset_stability_env_cfg import (
  g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg,
)


def _insert_after_event(cfg, after: str, name: str, term) -> None:
  reordered = {}
  inserted = False
  for event_name, event_term in cfg.events.items():
    reordered[event_name] = event_term
    if event_name == after:
      reordered[name] = term
      inserted = True
  if not inserted:
    raise RuntimeError(f"required reset event {after!r} is missing")
  cfg.events = reordered


def _additional_effort_derating(cfg, factor: float) -> None:
  """Clone shared actuator configs and apply one additional fixed derating."""
  robot_cfg = copy.deepcopy(cfg.scene.entities["robot"])
  cfg.scene.entities["robot"] = robot_cfg
  for actuator in robot_cfg.articulation.actuators:
    if actuator.effort_limit is not None:
      actuator.effort_limit *= factor


def _v37_actuator_feasibility_cfg(play: bool):
  """Matched V37-A base: flat reset coverage plus deployable hard envelopes."""
  cfg = g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg(play=play)

  # This study is flat get-up only: neither a plate nor automatic wrench is an
  # implicit treatment.  Failure replay is removed because its evolving GPU
  # ring would make the paired arm comparison state-history dependent.
  cfg.events["reset_escape_obstacle"].params.update(
    {"obstacle_probability": 0.0, "inactive_xy": (20.0, 20.0)}
  )
  cfg.events.pop("record_failure_states", None)
  cfg.events.pop("failure_state_replay_reset", None)
  cfg.events.pop("stratified_post_stand_wrench", None)
  cfg.events.pop("post_stand_body_wrench", None)

  previous_action = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = RateLimitedJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=previous_action.scale,
    use_default_offset=True,
    max_target_velocity=2.5,
    max_target_acceleration=15.0,
  )
  # V35/V36 already inherit the V5 0.90 derating.  This isolated 0.80 factor
  # therefore gives a deterministic 0.72 nominal effort ceiling.
  _additional_effort_derating(cfg, factor=0.80)

  foot_ground = ContactSensorCfg(
    name="v37_foot_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"(left|right)_foot[1-7]_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  existing = {sensor.name for sensor in cfg.scene.sensors or ()}
  if foot_ground.name not in existing:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (foot_ground,)

  cfg.rewards["joint_speed_excess"].params["speed_limits"] = (
    4.5,
    4.0,
    3.5,
    3.0,
  )
  cfg.rewards["joint_power_excess"].params["power_limits"] = (
    100.0,
    90.0,
    80.0,
    70.0,
  )
  cfg.rewards.update(
    {
      "joint_speed_tail_barrier": RewardTermCfg(
        func=mdp.joint_speed_tail_barrier,
        weight=-0.12,
        params={"soft_limit": 6.0, "reference_limit": 12.0},
      ),
      "target_velocity_soft_barrier": RewardTermCfg(
        func=mdp.target_velocity_soft_barrier,
        weight=-0.025,
        params={"soft_limit": 2.0},
      ),
    }
  )
  cfg.metrics.update(
    {
      "joint_speed_tail_barrier": MetricsTermCfg(
        func=mdp.joint_speed_tail_barrier,
        params={"soft_limit": 6.0, "reference_limit": 12.0},
      ),
      "target_velocity_max": MetricsTermCfg(func=mdp.target_velocity_max_metric),
      "target_acceleration_max": MetricsTermCfg(
        func=mdp.target_acceleration_max_metric
      ),
      "target_limited_fraction": MetricsTermCfg(
        func=mdp.target_limited_fraction_metric
      ),
      "bilateral_foot_support": MetricsTermCfg(
        func=mdp.bilateral_foot_support_score
      ),
    }
  )
  return cfg


def g1_getup_v37_93d_actuator_env_cfg(play: bool = False):
  """V37-A control: actuator feasibility without route shaping/trap resets."""
  return _v37_actuator_feasibility_cfg(play)


def g1_getup_v37_93d_support_env_cfg(play: bool = False):
  """V37-B treatment: A plus seated-trap coverage and bilateral transition."""
  cfg = _v37_actuator_feasibility_cfg(play)
  trap_probability = (
    float(os.environ.get("SMP_PLAY_V37_TRAP_RESET", "0") == "1")
    if play
    else 0.15
  )
  _insert_after_event(
    cfg,
    "curriculum_validated_fall_reset",
    "photo_informed_seated_trap_reset",
    EventTermCfg(
      func=mdp.photo_informed_seated_trap_reset,
      mode="reset",
      params={
        "probability": trap_probability,
        "joint_noise": 0.08,
        "joint_limit_margin": 0.03,
        "max_penetration": 0.012,
        "max_support_gap": 0.025,
      },
    ),
  )
  stage = cfg.events["update_recovery_stage"]
  stage.func = mdp.update_recovery_stage_with_bilateral_support
  stage.params.update(
    {
      "foot_sensor_name": "v37_foot_ground_contact",
      "seated_hold_steps": 15,
      "crouched_hold_steps": 25,
      "standing_hold_steps": 100,
      "seated_min_load_share": 0.10,
      "crouched_min_load_share": 0.18,
      "standing_min_load_share": 0.15,
      "seated_max_stance_width": 0.70,
      "crouched_max_stance_width": 0.58,
      "standing_max_stance_width": 0.55,
    }
  )
  cfg.rewards.update(
    {
      "bilateral_support_route": RewardTermCfg(
        func=mdp.bilateral_foot_support_score,
        weight=0.18,
        params={"full_score_load_share": 0.35},
      ),
      "bilateral_load_imbalance": RewardTermCfg(
        func=mdp.bilateral_load_imbalance_l2,
        weight=-0.10,
        params={"free_imbalance": 0.25},
      ),
      "transition_leg_asymmetry": RewardTermCfg(
        func=mdp.transition_leg_asymmetry_l2,
        weight=-0.08,
      ),
      "transition_stance_width": RewardTermCfg(
        func=mdp.transition_stance_width_excess_l2,
        weight=-0.20,
        params={"max_width": 0.55, "min_head_height": 0.50},
      ),
    }
  )
  cfg.metrics.update(
    {
      "seated_trap_reset": MetricsTermCfg(func=mdp.seated_trap_reset_metric),
      "bilateral_load_imbalance": MetricsTermCfg(
        func=mdp.bilateral_load_imbalance_l2,
        params={"free_imbalance": 0.25},
      ),
      "transition_leg_asymmetry": MetricsTermCfg(
        func=mdp.transition_leg_asymmetry_l2
      ),
      "transition_stance_width": MetricsTermCfg(
        func=mdp.transition_stance_width_excess_l2,
        params={"max_width": 0.55, "min_head_height": 0.50},
      ),
      "v37_bilateral_contact": MetricsTermCfg(
        func=mdp.v37_bilateral_support_metric
      ),
      "v37_min_foot_load_share": MetricsTermCfg(
        func=mdp.v37_min_foot_load_share_metric
      ),
      "v37_stance_width": MetricsTermCfg(func=mdp.v37_stance_width_metric),
    }
  )
  return cfg


__all__ = [
  "g1_getup_v37_93d_actuator_env_cfg",
  "g1_getup_v37_93d_support_env_cfg",
]
