"""Recoverable, non-ballistic G1 get-up refinement."""

from __future__ import annotations

import copy
import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.staged_env_cfg import (
  g1_getup_robust_staged_smp_env_cfg,
)


def _derate_actuator_effort_limits(cfg, factor: float = 0.90) -> None:
  """Apply an isolated physical effort cap without changing policy shape.

  G1 config factories share nested actuator objects, so clone the entity first
  to prevent V5 construction from mutating V2-V4 or compounding across calls.
  """
  robot_cfg = copy.deepcopy(cfg.scene.entities["robot"])
  cfg.scene.entities["robot"] = robot_cfg
  for actuator in robot_cfg.articulation.actuators:
    if actuator.effort_limit is not None:
      actuator.effort_limit *= factor


def g1_getup_robust_safe_smp_env_cfg(play: bool = False):
  """Build v5: reliable re-knockdown recovery with bounded burst effort."""
  cfg = g1_getup_robust_staged_smp_env_cfg(play=play)
  cfg.episode_length_s = 12.0

  # Absolute reset probabilities:
  # 30% GSI, 15% supine, 35% prone, 10% left side, 10% right side.
  # Slightly lower placement and wider joint noise improve contact-rich prone
  # coverage while retaining a substantial learned-prior component.
  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 0.70,
      "mode_weights": (1.5, 3.5, 1.0, 1.0),
      "root_height_range": (0.42, 0.56),
      "joint_noise": 0.16,
    }
  )

  # Replace generic periodic pushes with one deliberate fall after a stable
  # stand. This trains the same recovery cycle tested by dragging a standing
  # robot down, including nonzero impact velocity and contact history.
  cfg.events.pop("random_body_wrench", None)
  if not play or os.environ.get("SMP_PLAY_AUTO_DISTURBANCES") == "1":
    cfg.events["post_stand_body_wrench"] = EventTermCfg(
      func=mdp.post_stand_body_wrench,
      mode="step",
      params={
        "delay_steps": (20, 60),
        "duration_steps": (10, 18),
        "recovery_steps": 100,
        "force_range": (90.0, 190.0),
        "torque_range": (4.0, 14.0),
        "curriculum_steps": 250_000,
      },
    )

  # V4's full low-pose smoothing can make lying still locally optimal. V5 keeps
  # 30-50% of it while lying, then restores full smoothing through crouch and
  # stand. Joint speed retains a relatively generous low-pose limit; mechanical
  # power and vertical speed directly target explosive motion.
  cfg.rewards.update(
    {
      "action_rate_l2": RewardTermCfg(func=mdp.staged_action_rate_l2, weight=-0.0015),
      "action_acc_l2": RewardTermCfg(func=mdp.staged_action_acc_l2, weight=-0.0012),
      "joint_acc_l2": RewardTermCfg(func=mdp.staged_joint_acc_l2, weight=-5.0e-8),
      "joint_torques_l2": RewardTermCfg(
        func=mdp.staged_joint_torques_l2, weight=-1.0e-6
      ),
      "joint_speed_excess": RewardTermCfg(
        func=mdp.staged_joint_speed_excess_l2,
        weight=-0.02,
        params={"speed_limits": (6.0, 5.0, 4.0, 3.5)},
      ),
      "joint_power_excess": RewardTermCfg(
        func=mdp.staged_joint_power_excess_l2,
        weight=-2.0e-6,
        params={"power_limits": (140.0, 110.0, 90.0, 75.0)},
      ),
      "recovery_initiation": RewardTermCfg(
        func=mdp.recovery_initiation_progress,
        weight=0.12,
      ),
    }
  )
  cfg.rewards["head_vertical_overspeed"].params["speed_limit"] = 0.20

  # A small physical derating catches residual peaks even when reward tradeoffs
  # favor them. Larger reductions are intentionally avoided because knee and hip
  # support torque are necessary for contact-rich prone recovery.
  _derate_actuator_effort_limits(cfg, factor=0.90)

  cfg.metrics.update(
    {
      "post_stand_knockdown": MetricsTermCfg(func=mdp.post_stand_knockdown_metric),
      "max_joint_torque": MetricsTermCfg(func=mdp.max_joint_torque_metric),
      "max_joint_power": MetricsTermCfg(func=mdp.max_joint_power_metric),
      "recovery_initiation": MetricsTermCfg(func=mdp.recovery_initiation_progress),
    }
  )

  return cfg
