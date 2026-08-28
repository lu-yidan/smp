"""Recovery-Core R1: clean SMP recovery without plate, terrain, or milestones."""

from __future__ import annotations

from copy import deepcopy

from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.plate_terrain_v3874_gsi_milestone_env_cfg import (
  g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg,
)

GSI_PROBABILITY = 0.60
PROCEDURAL_PROBABILITY = 1.0 - GSI_PROBABILITY
SMP_PRODUCT_WEIGHT = 0.50

_PLATE_REWARDS = (
  "hand_supported_escape_progress",
  "escape_separation_progress",
  "escape_contact_force_excess",
  "escape_completion",
  "escape_geometry_progress",
  "escape_geometry_clearance_score",
)
_PLATE_METRICS = (
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
)
_PLATE_SENSORS = {"hand_ground_contact", "robot_obstacle_contact"}


def g1_recovery_core_r1_smp_env_cfg(play: bool = False):
  """Return the isolated deployable recovery task used by the main SMP arm.

  The actor stays identical to the unified task, but the environment contains
  no plate, no non-flat terrain, no LAFAN milestone reset, and no task label.
  GSI remains the primary initialization mechanism while procedural falls cover
  off-prior prone, supine, and side-lying states.
  """
  cfg = g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg(play=play)

  # Physically remove the escape task instead of merely parking a zero-probability
  # obstacle. This keeps both the experiment and the MuJoCo contact graph clean.
  cfg.events.pop("reset_escape_obstacle", None)
  cfg.events.pop("update_escape_phase", None)
  cfg.scene.entities.pop("escape_obstacle", None)
  cfg.scene.sensors = tuple(
    sensor
    for sensor in cfg.scene.sensors or ()
    if sensor.name not in _PLATE_SENSORS
  )
  for name in _PLATE_REWARDS:
    cfg.rewards.pop(name, None)
  for name in _PLATE_METRICS:
    cfg.metrics.pop(name, None)
  cfg.terminations.pop("invalid_escape_episode", None)

  # Milestones were a bootstrap diagnostic, not part of the final method.
  cfg.events.pop("lafan_milestone_reset", None)
  cfg.metrics.pop("lafan_milestone_reset", None)
  cfg.metrics.pop("lafan_milestone_stage", None)
  cfg.events["mixed_fall_reset"].params["procedural_probability"] = (
    PROCEDURAL_PROBABILITY
  )

  # Restore SMP as a material part of the objective after the low-weight
  # exploration gate. The non-zero floor preserves gradients off the prior.
  cfg.rewards["task_smp_product"].func = task_smp_product
  cfg.rewards["task_smp_product"].weight = SMP_PRODUCT_WEIGHT
  cfg.rewards["task_smp_product"].params.pop("constrained_scale", None)
  cfg.rewards["recovery_initiation"].func = mdp.recovery_initiation_progress
  cfg.rewards["recovery_initiation"].params.pop("constrained_scale", None)

  # Keep the generated support surface strictly flat and disable terrain
  # progression. Terrain recovery is evaluated and trained in its own task.
  generator = cfg.scene.terrain.terrain_generator
  if play and generator is not None and "flat" not in generator.sub_terrains:
    training_cfg = (
      g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg(play=False)
    )
    generator = deepcopy(training_cfg.scene.terrain.terrain_generator)
    generator.num_rows = 1
    generator.num_cols = 1
    cfg.scene.terrain.terrain_generator = generator
  if generator is not None:
    for name, terrain_cfg in generator.sub_terrains.items():
      terrain_cfg.proportion = float(name == "flat")
  cfg.curriculum.pop("terrain_levels", None)

  return cfg


__all__ = ["g1_recovery_core_r1_smp_env_cfg"]
