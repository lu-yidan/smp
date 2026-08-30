"""Preregistered 93D terrain (T) and flat-plate (P) specialist tasks."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Callable

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.escape_v34_env_cfg import (
  g1_getup_escape_plate_v34_smp_env_cfg,
)
from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  g1_scratch_a0_f2s2_gsi_env_cfg,
  g1_scratch_a1_v7_gsi_env_cfg,
  g1_scratch_a2_f2s2_mix_strict_env_cfg,
  g1_scratch_a3_v7_mix_strict_env_cfg,
  g1_scratch_a4_f2s2_mix_reset_aware_env_cfg,
  g1_scratch_a5_v7_mix_reset_aware_env_cfg,
  g1_scratch_a6_f2s2_mix_bridge_env_cfg,
  g1_scratch_a7_v7_mix_bridge_env_cfg,
  g1_scratch_a8_f2s2_balanced_bridge_env_cfg,
)
from smp.rl.tasks.getup.terrain_v35_env_cfg import (
  SLOPE_DEGREES,
  STAIR_APRON_WIDTH_M,
  STAIR_HEIGHTS_M,
  TERRAIN_PATCH_SIZE,
)
from smp.rl.tasks.getup.terrain_v36_env_cfg import terrain_generator_v36

ScratchBuilder = Callable[[bool], object]

SCRATCH_ARM_BUILDERS: dict[str, ScratchBuilder] = {
  "a0": g1_scratch_a0_f2s2_gsi_env_cfg,
  "a1": g1_scratch_a1_v7_gsi_env_cfg,
  "a2": g1_scratch_a2_f2s2_mix_strict_env_cfg,
  "a3": g1_scratch_a3_v7_mix_strict_env_cfg,
  "a4": g1_scratch_a4_f2s2_mix_reset_aware_env_cfg,
  "a5": g1_scratch_a5_v7_mix_reset_aware_env_cfg,
  "a6": g1_scratch_a6_f2s2_mix_bridge_env_cfg,
  "a7": g1_scratch_a7_v7_mix_bridge_env_cfg,
  "a8": g1_scratch_a8_f2s2_balanced_bridge_env_cfg,
}

ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)
TERRAIN_PROPORTIONS = (0.30, 0.20, 0.35, 0.15)
TERRAIN_LEVEL_WEIGHTS = (0.55, 0.30, 0.15, 0.0)
STAIR_EDGE_WEIGHTS = (0.40, 0.25, 0.20, 0.15)
PLATE_POSE_WEIGHTS = (3.0, 3.0, 1.0, 1.0)
PLATE_PROBABILITY_BY_RESET = (2.0 / 3.0, 2.0 / 3.0, 0.0, 0.0)

_PLATE_SENSOR_NAMES = ("hand_ground_contact", "robot_obstacle_contact")
_PLATE_REWARD_NAMES = (
  "hand_supported_escape_progress",
  "escape_separation_progress",
  "escape_contact_force_excess",
  "escape_completion",
  "escape_geometry_progress",
  "escape_geometry_clearance_score",
)
_PLATE_METRIC_NAMES = (
  "escape_phase",
  "escape_completion",
  "escape_obstacle_episode",
  "escape_invalid_contact",
  "escape_peak_penetration",
  "escape_peak_contact_force",
  "hand_support_contact",
  "hand_supported_escape_progress",
  "escape_invalid_setup",
  "escape_first_contact_head_height",
  "escape_hand_support_steps",
  "escape_hand_supported_progress",
  "escape_covered_geom_count",
  "escape_best_covered_geom_count",
  "escape_planar_clearance",
  "escape_plate_mass",
  "supine_reset",
)


def _scratch_cfg(arm: str, play: bool):
  try:
    cfg = SCRATCH_ARM_BUILDERS[arm](play=play)
  except KeyError as exc:
    raise ValueError(f"unknown scratch arm {arm!r}") from exc
  actor = cfg.observations["actor"]
  if tuple(actor.terms) != ACTOR_TERMS or actor.history_length is not None:
    raise RuntimeError("RAL progression requires the exact 93D one-frame actor")
  cfg.events.pop("push_robot", None)
  return cfg


def _insert_event(
  cfg, name: str, term, *, before: str | None = None, after: str | None = None
):
  if (before is None) == (after is None):
    raise ValueError("specify exactly one insertion anchor")
  anchor = before if before is not None else after
  reordered = {}
  inserted = False
  for event_name, event_term in cfg.events.items():
    if before is not None and event_name == anchor:
      reordered[name] = term
      inserted = True
    reordered[event_name] = event_term
    if after is not None and event_name == anchor:
      reordered[name] = term
      inserted = True
  if not inserted:
    raise RuntimeError(f"missing reset event anchor {anchor!r}")
  cfg.events = reordered


def _balanced_fall_reset() -> EventTermCfg:
  return EventTermCfg(
    func=mdp.mixed_fall_reset,
    mode="reset",
    params={
      "procedural_probability": 1.0,
      "mode_weights": (1.0, 1.0, 1.0, 1.0),
      "root_height_range": (0.46, 0.46),
      "root_xy_range": 0.025,
      "root_linear_velocity": 0.0,
      "root_angular_velocity": 0.0,
      "orientation_noise": 0.10,
      "joint_noise": 0.18,
    },
  )


def _surface_normal_levels():
  vertical = tuple((0.0, 0.0, 1.0) for _ in range(4))
  slopes = tuple(
    (
      math.sin(math.radians(angle)),
      0.0,
      math.cos(math.radians(angle)),
    )
    for angle in SLOPE_DEGREES
  )
  return (vertical, slopes, vertical, vertical)


def g1_scratch_ral_terrain_env_cfg(arm: str, play: bool = False):
  """Phase T: selected 93D flat arm on frozen terrain strata, no plate."""
  cfg = _scratch_cfg(arm, play)
  generator = terrain_generator_v36(seed=20260911)
  for name, proportion in zip(
    ("flat", "slope", "stairs", "rough"), TERRAIN_PROPORTIONS, strict=True
  ):
    generator.sub_terrains[name].proportion = proportion
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=generator,
    env_spacing=None,
    max_init_terrain_level=2,
    debug_vis=play,
  )
  cfg.scene.extent = 7.0
  cfg.sim.nconmax = max(cfg.sim.nconmax, 512)
  cfg.sim.njmax = max(cfg.sim.njmax, 5000)
  cfg.episode_length_s = 20.0

  level_event = EventTermCfg(
    func=mdp.sample_weighted_terrain_levels,
    mode="reset",
    params={"level_weights": TERRAIN_LEVEL_WEIGHTS},
  )
  _insert_event(cfg, "sample_weighted_terrain_levels", level_event, before="gsi_reset")
  cfg.events["mixed_fall_reset"] = _balanced_fall_reset()
  edge_event = EventTermCfg(
    func=mdp.sample_terrain_edge_reset,
    mode="reset",
    params={
      "cohort_weights": STAIR_EDGE_WEIGHTS,
      "stair_step_heights": STAIR_HEIGHTS_M,
      "terrain_size": TERRAIN_PATCH_SIZE[0],
      "stair_border_width": STAIR_APRON_WIDTH_M,
      "stair_platform_width": 0.55,
      "stair_step_width": 0.30,
    },
  )
  _insert_event(cfg, "sample_terrain_edge_reset", edge_event, after="mixed_fall_reset")
  ground_event = EventTermCfg(
    func=mdp.ground_procedural_fall_on_terrain,
    mode="reset",
    params={
      "ground_clearance": 0.006,
      "surface_normal_levels": _surface_normal_levels(),
      "use_stair_height_profile": True,
      "stair_step_heights": STAIR_HEIGHTS_M,
      "terrain_size": TERRAIN_PATCH_SIZE[0],
      "stair_border_width": STAIR_APRON_WIDTH_M,
      "stair_platform_width": 0.55,
      "stair_step_width": 0.30,
    },
  )
  _insert_event(
    cfg,
    "ground_procedural_fall_on_terrain",
    ground_event,
    after="sample_terrain_edge_reset",
  )

  task_terms = []
  for func, weight, params in cfg.rewards["task_smp_product"].params["task_terms"]:
    params = dict(params)
    if func in (mdp.upward_velocity, mdp.track_head_height):
      params["relative_to_env_origin"] = True
    task_terms.append((func, weight, params))
  cfg.rewards["task_smp_product"].params["task_terms"] = tuple(task_terms)
  cfg.rewards.update(
    {
      "terrain_planar_displacement": RewardTermCfg(
        func=mdp.terrain_planar_displacement_l2,
        weight=-0.15,
        params={"free_radius": 0.40, "reference_reset_anchor": True},
      ),
      "terrain_foot_slip": RewardTermCfg(func=mdp.terrain_foot_slip_l2, weight=-0.02),
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
    func=mdp.terrain_patch_exit, params={"margin": 0.50}
  )
  cfg.terminations["unstable_sim_state"] = TerminationTermCfg(
    func=mdp.unstable_sim_state
  )
  cfg.metrics.update(
    {
      "stable_stand": MetricsTermCfg(
        func=mdp.stable_stand_metric,
        params={"head_height": 1.10, "relative_to_env_origin": True},
      ),
      "terrain_planar_displacement": MetricsTermCfg(
        func=mdp.terrain_planar_displacement_l2,
        params={"free_radius": 0.40, "reference_reset_anchor": True},
      ),
      "terrain_foot_slip": MetricsTermCfg(func=mdp.terrain_foot_slip_l2),
      "terrain_stance_width": MetricsTermCfg(
        func=mdp.terrain_stance_width_excess_l2, params={"max_width": 0.65}
      ),
      "terrain_edge_reset": MetricsTermCfg(func=mdp.terrain_edge_reset_metric),
      "terrain_reset_offset": MetricsTermCfg(func=mdp.terrain_reset_offset_metric),
    }
  )
  foot_ground = ContactSensorCfg(
    name="terrain_foot_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"(left|right)_foot[1-7]_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=2,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (foot_ground,)
  return cfg


def g1_scratch_ral_plate_env_cfg(arm: str, play: bool = False):
  """Phase P: selected 93D flat arm with a stratified passive plate."""
  cfg = _scratch_cfg(arm, play)
  plate_cfg = g1_getup_escape_plate_v34_smp_env_cfg(play=play)
  cfg.episode_length_s = 20.0
  cfg.sim.nconmax = max(cfg.sim.nconmax, 512)
  cfg.sim.njmax = max(cfg.sim.njmax, 5000)
  cfg.scene.entities["escape_obstacle"] = deepcopy(
    plate_cfg.scene.entities["escape_obstacle"]
  )
  existing = {sensor.name for sensor in cfg.scene.sensors or ()}
  plate_sensors = tuple(
    deepcopy(sensor)
    for sensor in plate_cfg.scene.sensors or ()
    if sensor.name in _PLATE_SENSOR_NAMES and sensor.name not in existing
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + plate_sensors

  cfg.events["mixed_fall_reset"] = _balanced_fall_reset()
  cfg.events["mixed_fall_reset"].params["mode_weights"] = PLATE_POSE_WEIGHTS
  reset_plate = deepcopy(plate_cfg.events["reset_escape_obstacle"])
  reset_plate.func = mdp.reset_stratified_guided_escape_plate
  for key in ("plate_mass_range", "initial_max_mass", "mass_curriculum_steps"):
    reset_plate.params.pop(key, None)
  reset_plate.params.update(
    {
      "plate_masses": (4.0, 8.0, 12.0),
      "mass_weights": (0.25, 0.50, 0.25),
      "friction_range": (0.4, 1.2),
      "obstacle_probability": 0.0,
      "obstacle_probability_by_reset_type": PLATE_PROBABILITY_BY_RESET,
      "eligible_reset_types": (1, 2),
      "longitudinal_offset": 0.0,
      "lateral_offset": 0.0,
      "longitudinal_offset_curriculum": (0.12, 0.12),
      "lateral_offset_curriculum": (0.12, 0.12),
      "overlap_curriculum_steps": 1,
      "xy_offset_range": 0.0,
    }
  )
  _insert_event(cfg, "reset_escape_obstacle", reset_plate, after="mixed_fall_reset")
  cfg.events["update_escape_phase"] = deepcopy(plate_cfg.events["update_escape_phase"])

  floor = cfg.rewards["task_smp_product"].params.pop("procedural_smp_floor", 0.0)
  cfg.rewards[
    "task_smp_product"
  ].func = mdp.escape_gated_procedural_bridge_task_smp_product
  cfg.rewards["task_smp_product"].params.update(
    {"procedural_smp_floor": floor, "constrained_scale": 0.05}
  )
  for name in _PLATE_REWARD_NAMES:
    cfg.rewards[name] = deepcopy(plate_cfg.rewards[name])
  cfg.terminations["invalid_escape_episode"] = deepcopy(
    plate_cfg.terminations["invalid_escape_episode"]
  )
  cfg.terminations["unstable_sim_state"] = TerminationTermCfg(
    func=mdp.unstable_sim_state
  )
  for name in _PLATE_METRIC_NAMES:
    cfg.metrics[name] = deepcopy(plate_cfg.metrics[name])
  cfg.metrics.update(
    {
      "escape_plate_friction": MetricsTermCfg(
        func=mdp.escape_plate_friction_metric, reduce="last"
      ),
      "escape_plate_longitudinal_offset": MetricsTermCfg(
        func=mdp.escape_plate_longitudinal_offset_metric, reduce="last"
      ),
      "escape_plate_lateral_offset": MetricsTermCfg(
        func=mdp.escape_plate_lateral_offset_metric, reduce="last"
      ),
    }
  )
  return cfg


__all__ = [
  "ACTOR_TERMS",
  "PLATE_POSE_WEIGHTS",
  "PLATE_PROBABILITY_BY_RESET",
  "SCRATCH_ARM_BUILDERS",
  "STAIR_EDGE_WEIGHTS",
  "TERRAIN_LEVEL_WEIGHTS",
  "TERRAIN_PROPORTIONS",
  "g1_scratch_ral_plate_env_cfg",
  "g1_scratch_ral_terrain_env_cfg",
]
