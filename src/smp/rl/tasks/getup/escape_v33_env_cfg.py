"""Geometry-aware all-body clearance curriculum for guided-board escape."""

from __future__ import annotations

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.escape_v32_env_cfg import (
  g1_getup_escape_plate_v32_smp_env_cfg,
)


def g1_getup_escape_plate_v33_smp_env_cfg(play: bool = False):
  """Build V3.3: full collision-model clearance under a mass curriculum."""
  cfg = g1_getup_escape_plate_v32_smp_env_cfg(play=play)
  reset = cfg.events["reset_escape_obstacle"]
  reset.func = mdp.reset_guided_escape_plate_curriculum
  reset.params.update(
    {
      # Early episodes expose an edge and a light plate. The curriculum closes
      # the offset and raises the upper mass bound as the policy learns to use
      # hand support and translate its entire body out of the footprint.
      "longitudinal_offset_curriculum": (0.18, 0.04),
      "lateral_offset_curriculum": (0.22, 0.05),
      "overlap_curriculum_steps": 100_000,
      "plate_mass_range": (4.0, 12.0),
      "initial_max_mass": 6.0,
      "mass_curriculum_steps": 100_000,
      "xy_offset_range": 0.005,
    }
  )
  if play:
    # Play should show the intended pinned scenario on every reset instead of
    # sampling the easier, partially covered start of the training curriculum.
    reset.params.update(
      {
        "longitudinal_offset_curriculum": (0.0, 0.0),
        "lateral_offset_curriculum": (0.0, 0.0),
        "overlap_curriculum_steps": 1,
        "plate_mass_range": (8.0, 8.0),
        "initial_max_mass": 8.0,
        "mass_curriculum_steps": 1,
      }
    )

  cfg.events["update_escape_phase"].params.update(
    {
      "geometry_clearance": True,
      "collision_geom_pattern": r".*_collision$",
      "plate_geom_name": "escape_plate_geom",
      "plate_half_extents": (0.45, 0.32, 0.035),
      "min_planar_clearance": 0.025,
      "clear_hold_steps": 15,
    }
  )

  # V3.2's centre-distance reward allowed the pelvis to move 0.5 m while a
  # head, hand, or foot remained covered. V3.3 retains only a tiny directional
  # hint and makes all-body geometry progress the dominant escape objective.
  cfg.rewards["task_smp_product"].params["constrained_scale"] = 0.05
  cfg.rewards["hand_supported_escape_progress"].weight = 0.0
  cfg.rewards["escape_separation_progress"].weight = 0.01
  cfg.rewards["escape_completion"].weight = 0.60
  cfg.rewards.update(
    {
      "escape_geometry_progress": RewardTermCfg(
        func=mdp.escape_geometry_progress,
        weight=0.45,
        params={
          "sensor_name": "hand_ground_contact",
          "coverage_scale": 0.025,
          "clearance_scale": 0.02,
          "max_head_height": 0.90,
        },
      ),
      "escape_geometry_clearance_score": RewardTermCfg(
        func=mdp.escape_geometry_clearance_score,
        weight=0.08,
        params={"target_clearance": 0.04},
      ),
    }
  )
  cfg.metrics.update(
    {
      "escape_covered_geom_count": MetricsTermCfg(
        func=mdp.escape_covered_geom_count_metric, reduce="last"
      ),
      "escape_best_covered_geom_count": MetricsTermCfg(
        func=mdp.escape_best_covered_geom_count_metric, reduce="last"
      ),
      "escape_planar_clearance": MetricsTermCfg(
        func=mdp.escape_planar_clearance_metric, reduce="last"
      ),
      "escape_plate_mass": MetricsTermCfg(
        func=mdp.escape_plate_mass_metric, reduce="last"
      ),
    }
  )
  return cfg
