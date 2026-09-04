"""Grounded reset-coverage and settled-standing refinement of V34-93D."""

from __future__ import annotations

import math
import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.escape_v34_93d_env_cfg import (
  g1_getup_escape_plate_v34_93d_smp_env_cfg,
)
from smp.rl.tasks.getup.terrain_v35_env_cfg import (
  RESET_POSE_WEIGHTS,
  SLOPE_DEGREES,
  STAIR_APRON_WIDTH_M,
  STAIR_HEIGHTS_M,
  TERRAIN_PATCH_SIZE,
  terrain_generator_v35,
  terrain_surface_normals_v35,
)
from smp.rl.tasks.getup.terrain_v36_env_cfg import terrain_generator_v36

_TERRAIN_NAMES = ("flat", "slope", "stairs", "rough")


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


def g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg(
  play: bool = False,
):
  """Fine-tune V34-93D on grounded pose coverage and sustained standing.

  This phase intentionally does not add terrain, physical wrenches, or the A14
  target-velocity envelope.  Those remain isolated later treatments.
  """
  cfg = g1_getup_escape_plate_v34_93d_smp_env_cfg(play=play)

  original_reset = cfg.events.pop("mixed_fall_reset")
  cfg.events["gsi_reset"].func = mdp.physically_validated_gsi_reset
  cfg.events["gsi_reset"].params = {
    "max_attempts": 4,
    "max_joint_speed": 12.0,
    "max_root_linear_speed": 2.0,
    "max_root_angular_speed": 4.0,
    "max_penetration": 0.012,
    "max_support_gap": 0.025,
  }
  procedural_probability = 1.0 if play else 0.90
  mode_weights = (
    original_reset.params["mode_weights"] if play else (4.0, 4.0, 1.0, 1.0)
  )
  _insert_after_event(
    cfg,
    "gsi_reset",
    "curriculum_validated_fall_reset",
    EventTermCfg(
      func=mdp.curriculum_validated_fall_reset,
      mode="reset",
      params={
        "all_procedural_until_step": 0,
        "balanced_until_step": 0,
        "balanced_probability": procedural_probability,
        "target_probability": procedural_probability,
        "mode_weights": mode_weights,
        "joint_noise_levels": (0.10, 0.20, 0.30),
        "joint_noise_weights": (0.60, 0.30, 0.10),
        "joint_limit_margin": 0.02,
        "orientation_noise": 0.30,
        "max_penetration": 0.012,
        "max_support_gap": 0.025,
      },
    ),
  )

  # Require a real settled state, rather than advancing after only 0.5 s.
  cfg.events["update_recovery_stage"].params.update(
    {
      "seated_hold_steps": 10,
      "crouched_hold_steps": 15,
      "standing_hold_steps": 100,
    }
  )

  task_terms = []
  for func, weight, params in cfg.rewards["task_smp_product"].params["task_terms"]:
    if func is mdp.staged_recovery_pose:
      func = mdp.staged_recovery_pose_band
      params = {}
    task_terms.append((func, weight, params))
  cfg.rewards["task_smp_product"].params["task_terms"] = tuple(task_terms)
  cfg.rewards.update(
    {
      # A one-step bonus cannot be farmed by remaining in a marginal crouch.
      "recovery_stage_transition": RewardTermCfg(
        func=mdp.recovery_stage_transition_reward,
        weight=0.75,
      ),
      # Dense initiation is exposed only after the plate route is clear.
      "recovery_initiation": RewardTermCfg(
        func=mdp.escape_gated_recovery_initiation_progress,
        weight=0.08,
        params={"constrained_scale": 0.0},
      ),
      # These terms are tall/upright gated and do not suppress low-pose escape.
      "quiet_foot_speed_l2": RewardTermCfg(
        func=mdp.quiet_foot_speed_l2,
        weight=-0.02,
      ),
      "quiet_action_acc_l2": RewardTermCfg(
        func=mdp.quiet_action_acc_l2,
        weight=-1.0e-4,
      ),
      "quiet_base_angular_speed_l2": RewardTermCfg(
        func=mdp.quiet_base_angular_speed_l2,
        weight=-0.005,
      ),
    }
  )
  cfg.metrics.update(
    {
      "physical_gsi_rejection": MetricsTermCfg(
        func=mdp.physical_gsi_rejection_metric
      ),
      "physical_procedural_reset": MetricsTermCfg(
        func=mdp.physical_procedural_reset_metric
      ),
      "procedural_joint_noise_level": MetricsTermCfg(
        func=mdp.procedural_joint_noise_level_metric
      ),
      "procedural_orientation_offset": MetricsTermCfg(
        func=mdp.procedural_orientation_offset_metric
      ),
      "recovery_stage_transition": MetricsTermCfg(
        func=mdp.recovery_stage_transition_reward
      ),
      "quiet_foot_speed": MetricsTermCfg(func=mdp.quiet_foot_speed_l2),
      "quiet_action_acc": MetricsTermCfg(func=mdp.quiet_action_acc_l2),
      "quiet_base_angular_speed": MetricsTermCfg(
        func=mdp.quiet_base_angular_speed_l2
      ),
    }
  )
  return cfg


def _add_post_stand_wrench(cfg, play: bool) -> None:
  if play and os.environ.get("SMP_PLAY_AUTO_DISTURBANCES") != "1":
    return
  cfg.events["stratified_post_stand_wrench"] = EventTermCfg(
    func=mdp.stratified_post_stand_wrench,
    mode="step",
    params={
      "cohort_weights": (0.25, 0.50, 0.25),
      "delay_steps": (20, 60),
      "duration_steps": (10, 18),
      "recovery_steps": 150,
      "standard_force_range": (80.0, 170.0),
      "intensive_force_range": (120.0, 230.0),
      "torque_range": (4.0, 16.0),
      "standard_max_pushes": 1,
      "intensive_max_pushes": 3,
      "curriculum_steps": 300_000,
    },
  )


def _terrain_surface_normal_levels(
  terrain_names: tuple[str, ...] = _TERRAIN_NAMES,
):
  vertical = tuple((0.0, 0.0, 1.0) for _ in range(4))
  slopes = tuple(
    (
      math.sin(math.radians(angle)),
      0.0,
      math.cos(math.radians(angle)),
    )
    for angle in SLOPE_DEGREES
  )
  by_name = {
    "flat": vertical,
    "slope": slopes,
    "stairs": vertical,
    "rough": vertical,
  }
  return tuple(by_name[name] for name in terrain_names)


def _training_terrain_generator(family: str, safe_spawn: bool = False):
  generator = terrain_generator_v36(seed=20261820)
  if safe_spawn:
    # A recovery reset must start on a real support patch.  The historical
    # 0.55 m island is narrower than a prone G1 footprint and lets an AABB
    # corner falsely select the upper tread while every real primitive hangs
    # over a lower tread.  Terrain complexity begins outside this 2.4 m
    # landing island; later edge-reset studies can be added as a separate arm.
    generator.sub_terrains["stairs"].platform_width = 2.40
    generator.sub_terrains["rough"].platform_width = 2.40
  if family == "mixed":
    return generator
  if family != "stairs":
    raise ValueError(f"unsupported V36 terrain family {family!r}")
  stairs = generator.sub_terrains["stairs"]
  stairs.proportion = 1.0
  generator.sub_terrains = {"stairs": stairs}
  generator.num_cols = 1
  return generator


def _play_terrain_spec(default_family: str):
  kind = os.environ.get("SMP_PLAY_TERRAIN_TYPE", default_family)
  level_text = os.environ.get("SMP_PLAY_TERRAIN_LEVEL", "1")
  try:
    level = int(level_text)
  except ValueError as exc:
    raise ValueError("terrain level must be an integer from 0 to 3") from exc
  reset_pose = os.environ.get("SMP_PLAY_TERRAIN_RESET_POSE", "mixed")
  if reset_pose not in RESET_POSE_WEIGHTS:
    choices = ", ".join(RESET_POSE_WEIGHTS)
    raise ValueError(f"unknown terrain reset pose {reset_pose!r}; choose {choices}")
  return kind, level, reset_pose


def _add_terrain(
  cfg,
  play: bool,
  family: str = "mixed",
  safe_spawn: bool = True,
) -> None:
  if play:
    kind, level, reset_pose = _play_terrain_spec(family)
    generator = terrain_generator_v35(kind, level, seed=20261820)
    if safe_spawn and "stairs" in generator.sub_terrains:
      generator.sub_terrains["stairs"].platform_width = 2.40
    if safe_spawn and "rough" in generator.sub_terrains:
      generator.sub_terrains["rough"].platform_width = 2.40
    terrain_names = tuple(generator.sub_terrains)
    surface_params = {
      "surface_normals": terrain_surface_normals_v35(kind, level),
    }
    cfg.events["curriculum_validated_fall_reset"].params["mode_weights"] = (
      RESET_POSE_WEIGHTS[reset_pose]
    )
    max_init_level = 0
  else:
    generator = _training_terrain_generator(family, safe_spawn=safe_spawn)
    terrain_names = tuple(generator.sub_terrains)
    surface_params = {
      "surface_normal_levels": _terrain_surface_normal_levels(terrain_names),
    }
    max_init_level = 2
  # Sample the three preregistered train levels without coupling treatment to
  # an online curriculum success signal.  Level 3 remains held out for eval.
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=generator,
    env_spacing=None,
    max_init_terrain_level=max_init_level,
    debug_vis=play,
  )
  cfg.scene.extent = 7.0
  cfg.sim.nconmax = max(cfg.sim.nconmax, 512)
  cfg.sim.njmax = max(cfg.sim.njmax, 5000)

  grounding = EventTermCfg(
    func=mdp.ground_procedural_fall_on_terrain,
    mode="reset",
    params={
      # Type zero is an accepted GSI state; types one through four are the
      # continuous procedural pose families.  Type five is failure replay;
      # it must be re-grounded after replay writes the robot state.
      "eligible_reset_types": (0, 1, 2, 3, 4, 5),
      "ground_clearance": 0.002,
      "align_to_surface_normal": True,
      "use_stair_height_profile": True,
      "stair_step_heights": STAIR_HEIGHTS_M,
      "terrain_size": TERRAIN_PATCH_SIZE[0],
      "stair_border_width": STAIR_APRON_WIDTH_M,
      "stair_platform_width": generator.sub_terrains["stairs"].platform_width
      if "stairs" in generator.sub_terrains
      else 0.55,
      "stair_step_width": 0.30,
      **surface_params,
    },
  )
  contact_validation = EventTermCfg(
    func=mdp.validate_terrain_reset_contact,
    mode="reset",
    params={
      "sensor_name": "terrain_reset_ground_contact",
      "max_refinements": 16,
      "refinement_step": 0.005,
      "max_penetration": 0.012,
    },
  )
  reset_plate = cfg.events.pop("reset_escape_obstacle")
  reset_recovery_stage = cfg.events.pop("reset_recovery_stage")
  reset_plate.params.update(
    {
      "eligible_terrain_names": terrain_names,
      "reground_robot": False,
    }
  )
  reordered = {}
  inserted = False
  for name, term in cfg.events.items():
    reordered[name] = term
    if name == "failure_state_replay_reset":
      reordered["ground_fall_on_training_terrain"] = grounding
      reordered["validate_training_terrain_contact"] = contact_validation
      reordered["reset_escape_obstacle"] = reset_plate
      reordered["reset_recovery_stage"] = reset_recovery_stage
      inserted = True
  if not inserted:
    # The new no-replay study intentionally removes the replay ring.  In that
    # case the validated fall sampler is the last robot-state writer.
    reordered = {}
    for name, term in cfg.events.items():
      reordered[name] = term
      if name == "curriculum_validated_fall_reset":
        reordered["ground_fall_on_training_terrain"] = grounding
        reordered["validate_training_terrain_contact"] = contact_validation
        reordered["reset_escape_obstacle"] = reset_plate
        reordered["reset_recovery_stage"] = reset_recovery_stage
        inserted = True
    if not inserted:
      raise RuntimeError("V35 terrain treatment requires validated reset ordering")
  cfg.events = reordered

  cfg.events["reset_recovery_stage"].params["relative_to_env_origin"] = True
  cfg.events["update_recovery_stage"].params["relative_to_env_origin"] = True
  relative_terms = {
    mdp.staged_recovery_pose_band,
    mdp.staged_head_velocity_profile,
    mdp.track_head_height,
    mdp.feet_stationary_when_upright,
    mdp.base_stationary_when_upright,
  }
  task_terms = []
  for func, weight, params in cfg.rewards["task_smp_product"].params["task_terms"]:
    params = dict(params)
    if func in relative_terms:
      params["relative_to_env_origin"] = True
    task_terms.append((func, weight, params))
  cfg.rewards["task_smp_product"].params["task_terms"] = tuple(task_terms)
  for reward_name in (
    "recovery_initiation",
    "prone_support_route",
    "prone_leg_splay",
    "quiet_foot_speed_l2",
    "quiet_action_acc_l2",
    "quiet_base_angular_speed_l2",
  ):
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["relative_to_env_origin"] = True

  terrain_foot_ground = ContactSensorCfg(
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
  terrain_reset_ground = ContactSensorCfg(
    name="terrain_reset_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "dist"),
    reduce="mindist",
    num_slots=1,
    history_length=1,
  )
  existing_sensors = {sensor.name for sensor in cfg.scene.sensors or ()}
  if terrain_foot_ground.name not in existing_sensors:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_foot_ground,)
  if terrain_reset_ground.name not in existing_sensors:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_reset_ground,)
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
      "terrain_reset_contact_valid": MetricsTermCfg(
        func=mdp.terrain_reset_contact_valid_metric,
      ),
      "terrain_reset_refinement_steps": MetricsTermCfg(
        func=mdp.terrain_reset_refinement_steps_metric,
      ),
      "terrain_reset_min_distance": MetricsTermCfg(
        func=mdp.terrain_reset_min_distance_metric,
      ),
    }
  )
  cfg.terminations["terrain_patch_exit"] = TerminationTermCfg(
    func=mdp.terrain_patch_exit,
    params={"margin": 0.50},
  )


def g1_getup_escape_plate_v35_93d_reset_stability_wrench_smp_env_cfg(
  play: bool = False,
):
  cfg = g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg(play=play)
  _add_post_stand_wrench(cfg, play)
  return cfg


def g1_getup_escape_plate_v35_93d_reset_stability_terrain_smp_env_cfg(
  play: bool = False,
):
  cfg = g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg(play=play)
  _add_terrain(cfg, play)
  return cfg


def g1_getup_escape_plate_v35_93d_reset_stability_terrain_wrench_smp_env_cfg(
  play: bool = False,
):
  cfg = g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg(play=play)
  _add_terrain(cfg, play)
  _add_post_stand_wrench(cfg, play)
  return cfg


def _g1_getup_v36_93d_safe_terrain_cfg(
  play: bool,
  family: str,
  wrench: bool,
):
  """Terrain-only continuation with audited reset contact and no plate.

  Unlike the failed V35 RT/RTD arms, this study cannot mix a guided plate with
  terrain and cannot replay a flat/old-terrain failure state after grounding.
  Mixed and stairs-only arms otherwise share the same reset, reward, actor,
  optimizer warm-start, and optional post-stand wrench contracts.
  """
  cfg = g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg(play=play)
  plate = cfg.events["reset_escape_obstacle"]
  plate.params.update(
    {
      "obstacle_probability": 0.0,
      "inactive_xy": (20.0, 20.0),
    }
  )
  for event_name in ("record_failure_states", "failure_state_replay_reset"):
    cfg.events.pop(event_name, None)
  _add_terrain(cfg, play, family=family, safe_spawn=True)
  if wrench:
    _add_post_stand_wrench(cfg, play)
  return cfg


def g1_getup_v36_93d_safe_mixed_smp_env_cfg(play: bool = False):
  return _g1_getup_v36_93d_safe_terrain_cfg(play, "mixed", False)


def g1_getup_v36_93d_safe_mixed_wrench_smp_env_cfg(play: bool = False):
  return _g1_getup_v36_93d_safe_terrain_cfg(play, "mixed", True)


def g1_getup_v36_93d_safe_stairs_smp_env_cfg(play: bool = False):
  return _g1_getup_v36_93d_safe_terrain_cfg(play, "stairs", False)


def g1_getup_v36_93d_safe_stairs_wrench_smp_env_cfg(play: bool = False):
  return _g1_getup_v36_93d_safe_terrain_cfg(play, "stairs", True)


__all__ = [
  "g1_getup_escape_plate_v35_93d_reset_stability_smp_env_cfg",
  "g1_getup_escape_plate_v35_93d_reset_stability_terrain_smp_env_cfg",
  "g1_getup_escape_plate_v35_93d_reset_stability_terrain_wrench_smp_env_cfg",
  "g1_getup_escape_plate_v35_93d_reset_stability_wrench_smp_env_cfg",
  "g1_getup_v36_93d_safe_mixed_smp_env_cfg",
  "g1_getup_v36_93d_safe_mixed_wrench_smp_env_cfg",
  "g1_getup_v36_93d_safe_stairs_smp_env_cfg",
  "g1_getup_v36_93d_safe_stairs_wrench_smp_env_cfg",
]
