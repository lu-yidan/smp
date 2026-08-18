"""V3.6 proprioceptive terrain-recovery curriculum."""

from __future__ import annotations

import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxRandomGridTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.terrain_v35_env_cfg import (
  ROUGH_HEIGHTS_M,
  SLOPE_DEGREES,
  STAIR_APRON_WIDTH_M,
  STAIR_HEIGHTS_M,
  TERRAIN_OUTER_BORDER_M,
  TERRAIN_PATCH_SIZE,
  BoxSlopeTerrainCfg,
  g1_getup_terrain_v35_smp_env_cfg,
)


def terrain_generator_v36(seed: int = 20260818) -> TerrainGeneratorCfg:
  """Four terrain families by four adaptive difficulty rows."""
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=True,
    size=TERRAIN_PATCH_SIZE,
    border_width=TERRAIN_OUTER_BORDER_M,
    border_height=1.0,
    num_rows=4,
    num_cols=4,
    color_scheme="height",
    sub_terrains={
      # Proportions allocate the episode cohorts.  Flat remains at level zero
      # and protects the frozen recovery skill from catastrophic forgetting.
      "flat": BoxFlatTerrainCfg(proportion=0.30),
      "slope": BoxSlopeTerrainCfg(
        proportion=0.30,
        level_angles_degrees=SLOPE_DEGREES,
      ),
      "stairs": BoxPyramidStairsTerrainCfg(
        proportion=0.25,
        step_height_range=(STAIR_HEIGHTS_M[0], STAIR_HEIGHTS_M[-1]),
        step_width=0.30,
        platform_width=0.55,
        border_width=STAIR_APRON_WIDTH_M,
      ),
      "rough": BoxRandomGridTerrainCfg(
        proportion=0.15,
        grid_width=0.40,
        grid_height_range=(ROUGH_HEIGHTS_M[0], ROUGH_HEIGHTS_M[-1]),
        platform_width=0.55,
        merge_similar_heights=True,
        height_merge_threshold=0.04,
        max_merge_distance=4,
        border_width=1.0,
      ),
    },
    difficulty_range=(0.0, 1.0),
  )


def _surface_normal_levels() -> tuple[tuple[tuple[float, float, float], ...], ...]:
  vertical = tuple((0.0, 0.0, 1.0) for _ in range(4))
  slopes = tuple(
    (
      math.sin(math.radians(angle)),
      0.0,
      math.cos(math.radians(angle)),
    )
    for angle in SLOPE_DEGREES
  )
  # Column order must match terrain_generator_v36().sub_terrains.
  return (vertical, slopes, vertical, vertical)


def g1_getup_terrain_v36_smp_env_cfg(play: bool = False):
  """Build V3.6: model-98000 fine-tuning without plate or privileged actor input."""
  cfg = g1_getup_terrain_v35_smp_env_cfg(play=play)

  if not play:
    cfg.scene.terrain = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=terrain_generator_v36(),
      env_spacing=None,
      max_init_terrain_level=0,
      debug_vis=False,
    )
    ground = cfg.events["ground_procedural_fall_on_terrain"]
    ground.params.pop("surface_normals", None)
    ground.params["surface_normal_levels"] = _surface_normal_levels()
    cfg.curriculum = {
      "terrain_levels": CurriculumTermCfg(
        func=mdp.terrain_levels_getup,
        params={
          "stand_hold_steps": 25,
          "success_radius": 1.50,
          "minimum_episode_steps": 20,
        },
      )
    }
  else:
    # Playback keeps V3.5's explicit --terrain-type/--terrain-level controls.
    cfg.curriculum = {}

  # V3.6 isolates terrain adaptation.  Plate loads, automatic pushes, and the
  # flat-world failure replay ring are deliberately excluded from this phase.
  for event_name in (
    "stratified_post_stand_wrench",
    "record_failure_states",
    "failure_state_replay_reset",
  ):
    cfg.events.pop(event_name, None)

  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 1.0,
      "mode_weights": (3.5, 2.5, 2.0, 2.0),
      "root_xy_range": 0.04,
      "orientation_noise": 0.12,
      "joint_noise": 0.20,
    }
  )
  cfg.events["reset_recovery_stage"].params["relative_to_env_origin"] = True
  cfg.events["update_recovery_stage"].params["relative_to_env_origin"] = True

  cfg.rewards["task_smp_product"].params["task_terms"] = (
    (
      mdp.staged_recovery_pose,
      0.22,
      {"relative_to_env_origin": True},
    ),
    (
      mdp.staged_head_velocity_profile,
      0.18,
      {"relative_to_env_origin": True},
    ),
    (
      mdp.track_head_height,
      0.10,
      {
        "target_height": 1.15,
        "scale": 2.0,
        "relative_to_env_origin": True,
      },
    ),
    (mdp.upright_posture, 0.15, {"power": 2.0}),
    (
      mdp.feet_stationary_when_upright,
      0.08,
      {"scale": 20.0, "relative_to_env_origin": True},
    ),
    (
      mdp.base_stationary_when_upright,
      0.07,
      {"scale": 8.0, "relative_to_env_origin": True},
    ),
    (mdp.low_base_angular_velocity, 0.07, {"scale": 0.8}),
    (mdp.low_joint_velocity, 0.06, {"scale": 0.04}),
    (mdp.smooth_action, 0.07, {"scale": 4.0}),
  )
  cfg.rewards["recovery_initiation"].params["relative_to_env_origin"] = True
  cfg.rewards["prone_support_route"].params["relative_to_env_origin"] = True
  cfg.rewards["prone_leg_splay"].params["relative_to_env_origin"] = True
  cfg.rewards.update(
    {
      "terrain_planar_displacement": RewardTermCfg(
        func=mdp.terrain_planar_displacement_l2,
        weight=-0.15,
        params={"free_radius": 0.40},
      ),
      "terrain_foot_slip": RewardTermCfg(
        func=mdp.terrain_foot_slip_l2,
        weight=-0.02,
      ),
      "terrain_stance_width": RewardTermCfg(
        func=mdp.terrain_stance_width_excess_l2,
        weight=-0.05,
        params={"max_width": 0.65},
      ),
    }
  )

  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=mdp.stood_up,
    params={
      "head_height": 1.10,
      "min_upright": 0.85,
      "max_speed": 0.50,
      "max_angular_speed": 1.0,
      "hold_steps": 25,
      "relative_to_env_origin": True,
      "max_origin_distance": 1.50,
    },
  )
  cfg.terminations["terrain_patch_exit"] = TerminationTermCfg(
    func=mdp.terrain_patch_exit,
    params={"margin": 0.50},
  )
  cfg.terminations["unstable_sim_state"] = TerminationTermCfg(
    func=mdp.unstable_sim_state
  )
  cfg.episode_length_s = 20.0

  cfg.metrics["stable_stand"].params.update(
    {"head_height": 1.10, "relative_to_env_origin": True}
  )
  cfg.metrics.update(
    {
      "terrain_planar_displacement": MetricsTermCfg(
        func=mdp.terrain_planar_displacement_l2,
        params={"free_radius": 0.40},
      ),
      "terrain_foot_slip": MetricsTermCfg(func=mdp.terrain_foot_slip_l2),
      "terrain_stance_width": MetricsTermCfg(
        func=mdp.terrain_stance_width_excess_l2,
        params={"max_width": 0.65},
      ),
    }
  )
  return cfg


__all__ = ["g1_getup_terrain_v36_smp_env_cfg", "terrain_generator_v36"]
