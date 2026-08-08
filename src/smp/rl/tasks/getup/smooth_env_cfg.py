"""Smooth robust G1 get-up refinement with quiet-standing objectives."""

from __future__ import annotations

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.robust_env_cfg import g1_getup_robust_smp_env_cfg


def g1_getup_robust_smooth_smp_env_cfg(play: bool = False):
  """Build the v3 task for slower get-up motion and quieter final stance."""
  cfg = g1_getup_robust_smp_env_cfg(play=play)
  feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))

  # A symmetric velocity profile removes the incentive to rise faster than the
  # requested speed. Quiet-stance terms only activate once tall and upright.
  cfg.rewards["task_smp_product"].params.update(
    {
      "ws": 4.0,
      "task_terms": (
        (
          mdp.track_head_velocity_profile,
          0.15,
          {
            "start_height": 0.5,
            "stop_height": 1.15,
            "max_velocity": 0.15,
            "speed_limit": 0.25,
            "scale": 35.0,
            "overspeed_scale": 80.0,
          },
        ),
        (mdp.track_head_height, 0.15, {"target_height": 1.15, "scale": 2.0}),
        (mdp.upright_posture, 0.20, {"power": 2.0}),
        (
          mdp.feet_stationary_when_upright,
          0.10,
          {"scale": 20.0},
        ),
        (mdp.base_stationary_when_upright, 0.10, {"scale": 8.0}),
        (mdp.low_base_angular_velocity, 0.10, {"scale": 0.8}),
        (mdp.low_joint_velocity, 0.10, {"scale": 0.04}),
        (mdp.smooth_action, 0.10, {"scale": 4.0}),
      ),
    }
  )

  # These penalties sit outside the SMP product so a low prior score cannot
  # hide high-frequency control, joint acceleration, or joint-limit violations.
  cfg.rewards.update(
    {
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.0015),
      "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.001),
      "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-8),
      "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.1),
    }
  )

  # Start refinement with moderate pushes. During and shortly after a wrench,
  # quiet-standing rewards are gated off so recovery steps remain available.
  if "random_body_wrench" in cfg.events:
    cfg.events["random_body_wrench"].params.update(
      {
        "force_range": (30.0, 160.0),
        "torque_range": (3.0, 18.0),
        "curriculum_steps": 200_000,
        "recovery_steps": 50,
      }
    )

  cfg.metrics.update(
    {
      "head_vertical_speed": MetricsTermCfg(func=mdp.head_vertical_speed_metric),
      "mean_foot_speed": MetricsTermCfg(
        func=mdp.mean_foot_speed_metric, params={"asset_cfg": feet}
      ),
      "max_joint_speed": MetricsTermCfg(func=mdp.max_joint_speed_metric),
      "joint_acc_rms": MetricsTermCfg(func=mdp.joint_acc_rms_metric),
      "action_rate_rms": MetricsTermCfg(func=mdp.action_rate_rms_metric),
      "quiet_stance": MetricsTermCfg(func=mdp.quiet_stance_gate),
    }
  )

  return cfg
