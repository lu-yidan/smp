"""V38 post-roll recovery-route continuation on the deployable 93D actor."""

from __future__ import annotations

import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v37_support_transition_env_cfg import (
  g1_getup_v37_93d_support_env_cfg,
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


def _v38_matched_base(play: bool):
  """Preserve V37-B exactly as the matched continuation control."""
  return g1_getup_v37_93d_support_env_cfg(play=play)


def g1_getup_v38_93d_control_env_cfg(play: bool = False):
  """V38-A: matched V37-B continuation without a new treatment."""
  return _v38_matched_base(play)


def g1_getup_v38_93d_route_env_cfg(play: bool = False):
  """V38-B: post-roll reset coverage and stage-dependent support graph."""
  cfg = _v38_matched_base(play)
  cfg.sim.nconmax = max(cfg.sim.nconmax, 128)
  cfg.sim.njmax = max(cfg.sim.njmax, 2500)

  hand_ground = ContactSensorCfg(
    name="v38_hand_ground_contact",
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
  knee_ground = ContactSensorCfg(
    name="v38_knee_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"(left|right)_(shin|linkage_brace)_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  existing = {sensor.name for sensor in cfg.scene.sensors or ()}
  cfg.scene.sensors = (cfg.scene.sensors or ()) + tuple(
    sensor for sensor in (hand_ground, knee_ground) if sensor.name not in existing
  )

  post_roll_probability = (
    float(os.environ.get("SMP_PLAY_V38_POST_ROLL_RESET", "0") == "1")
    if play
    else 0.20
  )
  _insert_after_event(
    cfg,
    "photo_informed_seated_trap_reset",
    "post_roll_supine_failure_reset",
    EventTermCfg(
      func=mdp.post_roll_supine_failure_reset,
      mode="reset",
      params={
        "probability": post_roll_probability,
        "joint_noise": 0.08,
        "joint_limit_margin": 0.03,
        "max_penetration": 0.012,
        "max_support_gap": 0.025,
      },
    ),
  )

  stage = cfg.events["update_recovery_stage"]
  stage.func = mdp.update_recovery_stage_with_support_graph
  stage.params.update(
    {
      "hand_sensor_name": "v38_hand_ground_contact",
      "knee_sensor_name": "v38_knee_ground_contact",
    }
  )
  _insert_after_event(
    cfg,
    "reset_recovery_stage",
    "reset_v38_route_progress",
    EventTermCfg(
      func=mdp.reset_v38_route_progress,
      mode="reset",
      params={
        "hand_sensor_name": "v38_hand_ground_contact",
        "knee_sensor_name": "v38_knee_ground_contact",
      },
    ),
  )
  _insert_after_event(
    cfg,
    "update_recovery_stage",
    "update_v38_route_progress",
    EventTermCfg(
      func=mdp.update_v38_route_progress,
      mode="step",
      params={
        "hand_sensor_name": "v38_hand_ground_contact",
        "knee_sensor_name": "v38_knee_ground_contact",
        "delta_scale": 0.025,
      },
    ),
  )

  cfg.rewards["bilateral_support_route"] = RewardTermCfg(
    func=mdp.stage_delayed_bilateral_foot_support_score,
    weight=0.18,
    params={
      "sensor_name": "v37_foot_ground_contact",
      "full_score_load_share": 0.35,
      "minimum_stage": 1,
    },
  )
  cfg.rewards.update(
    {
      "v38_route_progress": RewardTermCfg(
        func=mdp.v38_route_progress_reward,
        weight=0.35,
      ),
      "prone_support_route": RewardTermCfg(
        func=mdp.prone_support_route,
        weight=0.12,
        params={
          "hand_sensor_name": "v38_hand_ground_contact",
          "knee_sensor_name": "v38_knee_ground_contact",
        },
      ),
    }
  )
  cfg.metrics.update(
    {
      "post_roll_supine_reset": MetricsTermCfg(
        func=mdp.post_roll_supine_reset_metric
      ),
      "v38_route_progress": MetricsTermCfg(func=mdp.v38_route_progress_reward),
      "v38_hand_support": MetricsTermCfg(func=mdp.v38_hand_support_metric),
      "v38_knee_support": MetricsTermCfg(func=mdp.v38_knee_support_metric),
    }
  )
  return cfg


__all__ = [
  "g1_getup_v38_93d_control_env_cfg",
  "g1_getup_v38_93d_route_env_cfg",
]
