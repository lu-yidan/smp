"""Robust G1 get-up task with mixed falls and physical perturbations."""

from __future__ import annotations

import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.getup_env_cfg import g1_getup_smp_env_cfg


def g1_getup_robust_smp_env_cfg(play: bool = False):
  """Build a robust get-up task while preserving the baseline task unchanged."""
  cfg = g1_getup_smp_env_cfg(play=play)

  # More simultaneous contacts are expected for arbitrary lying poses.
  cfg.sim.nconmax = 64
  cfg.episode_length_s = 10.0

  # Replace instantaneous base-velocity pushes with finite physical wrenches.
  cfg.events.pop("push_robot", None)
  cfg.events["mixed_fall_reset"] = EventTermCfg(
    func=mdp.mixed_fall_reset,
    mode="reset",
    params={
      "procedural_probability": 0.5,
      "root_height_range": (0.48, 0.62),
      "joint_noise": 0.12,
    },
  )
  cfg.events["random_body_wrench"] = EventTermCfg(
    func=mdp.random_body_wrench,
    mode="step",
    params={
      "body_names": ("pelvis", "torso_link"),
      "interval_steps": (50, 150),
      "duration_steps": (5, 15),
      "force_range": (40.0, 250.0),
      "torque_range": (4.0, 30.0),
      "curriculum_steps": 150_000,
    },
  )

  # Playback defaults to clean evaluation; the wrapper flag can opt back into
  # the same automatic wrench event used for training.
  if play and os.environ.get("SMP_PLAY_AUTO_DISTURBANCES") != "1":
    cfg.events.pop("random_body_wrench", None)

  # Task score remains bounded in [0, 1], then receives one SMP gate.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "ws": 4.0,
      "task_terms": (
        (
          mdp.upward_velocity,
          0.30,
          {
            "target_velocity": 0.25,
            "head_height_threshold": 0.9,
            "scale": 100.0,
          },
        ),
        (mdp.track_head_height, 0.20, {"target_height": 1.1, "scale": 1.0}),
        (mdp.upright_posture, 0.20, {"power": 2.0}),
        (mdp.low_base_angular_velocity, 0.10, {"scale": 0.5}),
        (mdp.low_joint_velocity, 0.10, {"scale": 0.02}),
        (mdp.smooth_action, 0.10, {"scale": 2.0}),
      ),
    },
  )

  # Arbitrary falls and later disturbances are intentionally off-prior. Keep
  # SMP as a soft reward gate, but never terminate recovery solely for low SMP.
  cfg.terminations.pop("smp_too_low", None)
  # Keep successful agents alive for standing-time disturbances and recovery.
  cfg.terminations.pop("stood_up", None)

  cfg.metrics.update(
    {
      "active_wrench": MetricsTermCfg(func=mdp.active_wrench_metric),
      "task_score": MetricsTermCfg(func=mdp.cached_task_score),
      "smp_score": MetricsTermCfg(func=mdp.cached_smp_score),
      "raw_smp_score": MetricsTermCfg(
        func=mdp.cached_raw_smp_score, params={"ws": 4.0}
      ),
      "product_score": MetricsTermCfg(func=mdp.cached_product_score),
      "upright": MetricsTermCfg(func=mdp.upright_posture, params={"power": 1.0}),
      "low_base_angular_velocity": MetricsTermCfg(
        func=mdp.low_base_angular_velocity, params={"scale": 0.5}
      ),
      "low_joint_velocity": MetricsTermCfg(
        func=mdp.low_joint_velocity, params={"scale": 0.02}
      ),
      "smooth_action": MetricsTermCfg(func=mdp.smooth_action, params={"scale": 2.0}),
      "stable_stand": MetricsTermCfg(func=mdp.stable_stand_metric),
      "procedural_reset": MetricsTermCfg(func=mdp.procedural_reset_metric),
    }
  )

  return cfg
