"""V39 support-exit continuation on the deployable 93D actor."""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v38_recovery_route_env_cfg import (
  g1_getup_v38_93d_route_env_cfg,
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


def _v39_matched_base(play: bool):
  """Preserve V38-B exactly as the continuation control."""
  return g1_getup_v38_93d_route_env_cfg(play=play)


def g1_getup_v39_93d_control_env_cfg(play: bool = False):
  """V39-A: matched V38-B continuation without a new treatment."""
  return _v39_matched_base(play)


def g1_getup_v39_93d_support_exit_env_cfg(play: bool = False):
  """V39-B: make hand/knee support transitional, then transfer to both feet."""
  cfg = _v39_matched_base(play)

  _insert_after_event(
    cfg,
    "reset_v38_route_progress",
    "reset_v39_support_exit",
    EventTermCfg(func=mdp.reset_v39_support_exit, mode="reset"),
  )
  _insert_after_event(
    cfg,
    "update_v38_route_progress",
    "update_v39_support_exit",
    EventTermCfg(
      func=mdp.update_v39_support_exit,
      mode="step",
      params={
        "hand_sensor_name": "v38_hand_ground_contact",
        "knee_sensor_name": "v38_knee_ground_contact",
        "foot_sensor_name": "v37_foot_ground_contact",
        "grace_steps": 60,
        "full_penalty_steps": 180,
        "exit_height": 0.55,
        "exit_upright": 0.55,
        "progress_scale": 0.0125,
      },
    ),
  )

  # The V38 low-route term has done its job once the robot reaches supported
  # kneeling.  Reduce it rather than removing it so prone initiation is kept.
  cfg.rewards["prone_support_route"].weight = 0.04
  cfg.rewards["v38_route_progress"].weight = 0.20
  cfg.rewards.update(
    {
      "v39_head_best_progress": RewardTermCfg(
        func=mdp.v39_head_best_progress_reward,
        weight=0.55,
      ),
      "v39_foot_transfer": RewardTermCfg(
        func=mdp.v39_foot_transfer_reward,
        weight=0.35,
      ),
      "v39_support_dwell": RewardTermCfg(
        func=mdp.v39_support_dwell_penalty,
        weight=-0.24,
      ),
    }
  )
  cfg.metrics.update(
    {
      "v39_support_dwell": MetricsTermCfg(func=mdp.v39_support_dwell_penalty),
      "v39_foot_transfer": MetricsTermCfg(func=mdp.v39_foot_transfer_reward),
      "v39_head_best_progress": MetricsTermCfg(func=mdp.v39_head_best_progress_reward),
    }
  )
  return cfg


__all__ = [
  "g1_getup_v39_93d_control_env_cfg",
  "g1_getup_v39_93d_support_exit_env_cfg",
]
