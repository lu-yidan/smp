"""V6 multi-prior and hard-state recovery refinements."""

from __future__ import annotations

import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.safe_env_cfg import (
  g1_getup_robust_safe_smp_env_cfg,
)

V6_PRIOR_PATH = "datasets/pretrain_ckpt/pretrained_getup_lafan6_v6.pt"


def g1_getup_v6_prior_smp_env_cfg(play: bool = False):
  """V5 behavior with only the six-sequence recovery prior changed."""
  cfg = g1_getup_robust_safe_smp_env_cfg(play=play)
  cfg.events["init_smp_state"].params["ckpt_path"] = V6_PRIOR_PATH
  return cfg


def g1_getup_v6_smp_env_cfg(play: bool = False):
  """V6 full task: broader resets, hard-state replay, and push cohorts."""
  cfg = g1_getup_v6_prior_smp_env_cfg(play=play)
  cfg.episode_length_s = 20.0

  # Before replay is populated, expected reset shares are 35% new-prior GSI
  # and 65% continuous procedural falls. Once populated, replay replaces 20%
  # of the mixed result. Orientation noise on both roll and pitch covers
  # oblique front/side/back contacts rather than four exact 90-degree poses.
  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 0.65,
      "mode_weights": (2.0, 3.5, 1.5, 1.5),
      "root_height_range": (0.38, 0.64),
      "joint_noise": 0.24,
      "orientation_noise": 0.40,
      "root_xy_range": 0.18,
      "root_linear_velocity": 0.25,
      "root_angular_velocity": 0.60,
    }
  )

  # The prior-only task retains V5's single knockdown. Full V6 replaces it
  # with episode cohorts: 25% clean, 50% one standard knockdown, and 25% up to
  # three stronger knockdowns after successful recoveries.
  cfg.events.pop("post_stand_body_wrench", None)
  disturbances_enabled = not play or os.environ.get("SMP_PLAY_AUTO_DISTURBANCES") == "1"
  if disturbances_enabled:
    cfg.events["stratified_post_stand_wrench"] = EventTermCfg(
      func=mdp.stratified_post_stand_wrench,
      mode="step",
      params={
        "cohort_weights": (0.25, 0.50, 0.25),
        "delay_steps": (20, 60),
        "duration_steps": (10, 18),
        "recovery_steps": 100,
        "standard_force_range": (80.0, 170.0),
        "intensive_force_range": (120.0, 230.0),
        "torque_range": (4.0, 16.0),
        "standard_max_pushes": 1,
        "intensive_max_pushes": 3,
        "curriculum_steps": 300_000,
      },
    )

  # Training records states that fail to improve for 1.5 s and eventually
  # replaces 20% of resets from the GPU replay ring. Playback never mutates or
  # samples this ring, keeping manual evaluation deterministic with respect to
  # the chosen reset seed.
  if not play:
    cfg.events["record_failure_states"] = EventTermCfg(
      func=mdp.record_failure_states,
      mode="step",
      params={
        "capacity": 8192,
        "stagnation_steps": 75,
        "progress_epsilon": 0.02,
        "record_probability": 0.25,
        "max_records_per_step": 64,
      },
    )
    cfg.events["failure_state_replay_reset"] = EventTermCfg(
      func=mdp.failure_state_replay_reset,
      mode="reset",
      params={"replay_probability": 0.20, "minimum_buffer_size": 128},
    )

  cfg.metrics.update(
    {
      "failure_replay_reset": MetricsTermCfg(func=mdp.failure_replay_reset_metric),
      "failure_buffer_fill": MetricsTermCfg(func=mdp.failure_buffer_fill_metric),
      "v6_active_wrench": MetricsTermCfg(func=mdp.v6_active_wrench_metric),
      "v6_push_cohort": MetricsTermCfg(func=mdp.v6_push_cohort_metric),
      "v6_push_count": MetricsTermCfg(func=mdp.v6_push_count_metric),
    }
  )
  return cfg
