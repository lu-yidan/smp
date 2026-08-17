"""Slow ordered get-up refinement with prone-focused reset coverage."""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.smooth_env_cfg import (
  g1_getup_robust_smooth_smp_env_cfg,
)


def g1_getup_robust_staged_smp_env_cfg(play: bool = False):
  """Build v4: prone-focused, seated-crouched-standing, and non-ballistic."""
  cfg = g1_getup_robust_smooth_smp_env_cfg(play=play)

  # Absolute reset probabilities:
  # 40% GSI, 10% supine, 30% prone, 10% left side, 10% right side.
  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 0.60,
      "mode_weights": (3.0, 1.0, 1.0, 1.0),
    }
  )
  cfg.events["reset_recovery_stage"] = EventTermCfg(
    func=mdp.reset_recovery_stage,
    mode="reset",
  )
  cfg.events["update_recovery_stage"] = EventTermCfg(
    func=mdp.update_recovery_stage,
    mode="step",
    params={
      "seated_hold_steps": 10,
      "crouched_hold_steps": 10,
      "standing_hold_steps": 25,
    },
  )

  # The first waypoint rotates and supports the torso into a seated/kneeling
  # configuration before the second waypoint permits a crouch. Only then does
  # the final standing target become active. Vertical targets stay at 0.06,
  # 0.08, and 0.10 m/s, with a separate penalty beyond 0.20 m/s.
  cfg.rewards["task_smp_product"].params["task_terms"] = (
    (mdp.staged_recovery_pose, 0.22, {}),
    (mdp.staged_head_velocity_profile, 0.18, {}),
    (mdp.track_head_height, 0.10, {"target_height": 1.15, "scale": 2.0}),
    (mdp.upright_posture, 0.15, {"power": 2.0}),
    (mdp.feet_stationary_when_upright, 0.08, {"scale": 20.0}),
    (mdp.base_stationary_when_upright, 0.07, {"scale": 8.0}),
    (mdp.low_base_angular_velocity, 0.07, {"scale": 0.8}),
    (mdp.low_joint_velocity, 0.06, {"scale": 0.04}),
    (mdp.smooth_action, 0.07, {"scale": 4.0}),
  )

  # Never remove the smoothing penalties at low height. The ordered waypoint
  # reward supplies a feasible route while these terms suppress abrupt control,
  # vertical launch, high acceleration, and high actuator effort throughout.
  cfg.rewards.update(
    {
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.0015),
      "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.0012),
      "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-5.0e-8),
      "joint_torques_l2": RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
      ),
      "head_vertical_overspeed": RewardTermCfg(
        func=mdp.head_vertical_overspeed_l2,
        weight=-1.0,
        params={"speed_limit": 0.20},
      ),
    }
  )

  # During focused refinement, keep pushes meaningful but below v3's maximum so
  # prone waypoint learning is not drowned out by the disturbance curriculum.
  if "random_body_wrench" in cfg.events:
    cfg.events["random_body_wrench"].params.update(
      {
        "force_range": (25.0, 120.0),
        "torque_range": (2.0, 14.0),
        "curriculum_steps": 250_000,
        "recovery_steps": 50,
      }
    )

  cfg.metrics.update(
    {
      "prone_reset": MetricsTermCfg(func=mdp.prone_reset_metric),
      "recovery_stage": MetricsTermCfg(func=mdp.recovery_stage_metric),
      "recovery_stage_complete": MetricsTermCfg(
        func=mdp.recovery_stage_complete_metric
      ),
      "head_vertical_overspeed": MetricsTermCfg(
        func=mdp.head_vertical_overspeed_l2,
        params={"speed_limit": 0.20},
      ),
      "mean_knee_flexion": MetricsTermCfg(func=mdp.mean_knee_flexion_metric),
    }
  )

  return cfg
