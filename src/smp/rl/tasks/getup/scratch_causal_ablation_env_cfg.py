"""Causal from-scratch ablations built on the successful original SMP recipe."""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.smp_observation_factorial_env_cfg import (
  g1_getup_obs_f1_nolinvel_smp_env_cfg,
)

F2S2_PRIOR_PATH = "datasets/pretrain_ckpt/pretrained_getup_f2s2.pt"
V7_ROUTE_PRIOR_PATH = (
  "datasets/pretrain_ckpt/pretrained_getup_lafan_route_v7.pt"
)


def _scratch_causal_cfg(
  *,
  play: bool,
  prior_path: str,
  procedural_probability: float,
  reset_aware_termination: bool,
  procedural_smp_floor: float,
):
  """Vary only prior, reset coverage, termination gate, and reward bridge."""
  cfg = g1_getup_obs_f1_nolinvel_smp_env_cfg(play=play)
  cfg.sim.nconmax = 64
  cfg.events["init_smp_state"].params["ckpt_path"] = prior_path

  if procedural_probability > 0.0:
    cfg.events["mixed_fall_reset"] = EventTermCfg(
      func=mdp.mixed_fall_reset,
      mode="reset",
      params={
        "procedural_probability": procedural_probability,
        "mode_weights": (1.0, 1.0, 1.0, 1.0),
        "root_height_range": (0.48, 0.62),
        "joint_noise": 0.12,
        "orientation_noise": 0.0,
        "root_xy_range": 0.1,
        "root_linear_velocity": 0.1,
        "root_angular_velocity": 0.2,
      },
    )

  if reset_aware_termination:
    cfg.terminations["smp_too_low"].func = mdp.smp_too_low_gsi_only

  if procedural_smp_floor > 0.0:
    cfg.rewards["task_smp_product"].func = (
      mdp.procedural_bridge_task_smp_product
    )
    cfg.rewards["task_smp_product"].params["procedural_smp_floor"] = (
      procedural_smp_floor
    )

  cfg.metrics.update(
    {
      "gsi_reset": MetricsTermCfg(func=mdp.gsi_reset_metric),
      "procedural_reset": MetricsTermCfg(func=mdp.procedural_reset_metric),
      "prone_reset": MetricsTermCfg(func=mdp.prone_reset_metric),
      "supine_reset": MetricsTermCfg(func=mdp.supine_reset_metric),
      "task_score": MetricsTermCfg(func=mdp.cached_task_score),
      "smp_score": MetricsTermCfg(func=mdp.cached_smp_score),
      "raw_smp_score": MetricsTermCfg(
        func=mdp.cached_raw_smp_score, params={"ws": 6.0}
      ),
      "product_score": MetricsTermCfg(func=mdp.cached_product_score),
      "max_joint_speed": MetricsTermCfg(func=mdp.max_joint_speed_metric),
      "max_joint_torque": MetricsTermCfg(func=mdp.max_joint_torque_metric),
      "max_joint_power": MetricsTermCfg(func=mdp.max_joint_power_metric),
    }
  )
  return cfg


def g1_scratch_a0_f2s2_gsi_env_cfg(play: bool = False):
  return _scratch_causal_cfg(
    play=play,
    prior_path=F2S2_PRIOR_PATH,
    procedural_probability=0.0,
    reset_aware_termination=False,
    procedural_smp_floor=0.0,
  )


def g1_scratch_a1_v7_gsi_env_cfg(play: bool = False):
  return _scratch_causal_cfg(
    play=play,
    prior_path=V7_ROUTE_PRIOR_PATH,
    procedural_probability=0.0,
    reset_aware_termination=False,
    procedural_smp_floor=0.0,
  )


def g1_scratch_a2_f2s2_mix_strict_env_cfg(play: bool = False):
  return _scratch_causal_cfg(
    play=play,
    prior_path=F2S2_PRIOR_PATH,
    procedural_probability=0.20,
    reset_aware_termination=False,
    procedural_smp_floor=0.0,
  )


def g1_scratch_a3_v7_mix_strict_env_cfg(play: bool = False):
  return _scratch_causal_cfg(
    play=play,
    prior_path=V7_ROUTE_PRIOR_PATH,
    procedural_probability=0.20,
    reset_aware_termination=False,
    procedural_smp_floor=0.0,
  )


def g1_scratch_a4_f2s2_mix_reset_aware_env_cfg(play: bool = False):
  return _scratch_causal_cfg(
    play=play,
    prior_path=F2S2_PRIOR_PATH,
    procedural_probability=0.20,
    reset_aware_termination=True,
    procedural_smp_floor=0.0,
  )


def g1_scratch_a5_v7_mix_reset_aware_env_cfg(play: bool = False):
  return _scratch_causal_cfg(
    play=play,
    prior_path=V7_ROUTE_PRIOR_PATH,
    procedural_probability=0.20,
    reset_aware_termination=True,
    procedural_smp_floor=0.0,
  )


def g1_scratch_a6_f2s2_mix_bridge_env_cfg(play: bool = False):
  return _scratch_causal_cfg(
    play=play,
    prior_path=F2S2_PRIOR_PATH,
    procedural_probability=0.20,
    reset_aware_termination=True,
    procedural_smp_floor=0.10,
  )


def g1_scratch_a7_v7_mix_bridge_env_cfg(play: bool = False):
  return _scratch_causal_cfg(
    play=play,
    prior_path=V7_ROUTE_PRIOR_PATH,
    procedural_probability=0.20,
    reset_aware_termination=True,
    procedural_smp_floor=0.10,
  )


def g1_scratch_a8_f2s2_balanced_bridge_env_cfg(play: bool = False):
  """One-shot method arm: increase only procedural reset exposure to 50%."""
  return _scratch_causal_cfg(
    play=play,
    prior_path=F2S2_PRIOR_PATH,
    procedural_probability=0.50,
    reset_aware_termination=True,
    procedural_smp_floor=0.10,
  )


def g1_scratch_a9_f2s2_objective_aligned_env_cfg(play: bool = False):
  """Align flat recovery training with the frozen strict success contract.

  This arm deliberately keeps A6's F2S2 prior, reset mixture, observations,
  disturbances, and PPO setup.  Only the recovery objective changes: the task
  score covers initiation, elevation, uprightness, quiet stance, and the exact
  frozen stable-stand state, while a non-zero SMP floor prevents off-prior
  procedural poses from losing the task gradient.
  """
  cfg = _scratch_causal_cfg(
    play=play,
    prior_path=F2S2_PRIOR_PATH,
    procedural_probability=0.20,
    reset_aware_termination=True,
    procedural_smp_floor=0.0,
  )
  cfg.rewards["task_smp_product"].func = task_smp_product
  cfg.rewards["task_smp_product"].params = {
    "task_terms": (
      (mdp.recovery_initiation_progress, 0.25, {}),
      (mdp.track_head_height, 0.20, {"target_height": 1.10, "scale": 2.0}),
      (mdp.upright_posture, 0.20, {"power": 2.0}),
      (mdp.feet_stationary_when_upright, 0.10, {"scale": 20.0}),
      (mdp.base_stationary_when_upright, 0.10, {"scale": 8.0}),
      (
        mdp.stable_stand_metric,
        0.15,
        {
          "head_height": 1.10,
          "min_upright": 0.85,
          "max_linear_speed": 0.50,
          "max_angular_speed": 1.00,
        },
      ),
    ),
    "smp_floor": 0.35,
  }
  cfg.rewards.update(
    {
      "head_vertical_overspeed": RewardTermCfg(
        func=mdp.head_vertical_overspeed_l2,
        weight=-0.50,
        params={"speed_limit": 0.25},
      ),
      "action_rate_l2": RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.001,
      ),
    }
  )
  cfg.terminations["stood_up"].params.update(
    {
      "head_height": 1.10,
      "max_speed": 0.50,
      "hold_steps": 25,
      "min_upright": 0.85,
      "max_angular_speed": 1.00,
    }
  )
  cfg.metrics.update(
    {
      "strict_stable_stand": MetricsTermCfg(
        func=mdp.stable_stand_metric,
        params={
          "head_height": 1.10,
          "min_upright": 0.85,
          "max_linear_speed": 0.50,
          "max_angular_speed": 1.00,
        },
      ),
      "head_vertical_overspeed": MetricsTermCfg(
        func=mdp.head_vertical_overspeed_l2,
        params={"speed_limit": 0.25},
      ),
      "action_rate_rms": MetricsTermCfg(func=mdp.action_rate_rms_metric),
    }
  )
  return cfg


__all__ = [
  "g1_scratch_a0_f2s2_gsi_env_cfg",
  "g1_scratch_a1_v7_gsi_env_cfg",
  "g1_scratch_a2_f2s2_mix_strict_env_cfg",
  "g1_scratch_a3_v7_mix_strict_env_cfg",
  "g1_scratch_a4_f2s2_mix_reset_aware_env_cfg",
  "g1_scratch_a5_v7_mix_reset_aware_env_cfg",
  "g1_scratch_a6_f2s2_mix_bridge_env_cfg",
  "g1_scratch_a7_v7_mix_bridge_env_cfg",
  "g1_scratch_a8_f2s2_balanced_bridge_env_cfg",
  "g1_scratch_a9_f2s2_objective_aligned_env_cfg",
]
