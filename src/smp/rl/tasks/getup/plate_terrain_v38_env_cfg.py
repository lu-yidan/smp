"""V3.8 deployable unified plate-escape and terrain recovery task."""

from __future__ import annotations

import os
from copy import deepcopy

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.escape_v34_env_cfg import (
  g1_getup_escape_plate_v34_smp_env_cfg,
)
from smp.rl.tasks.getup.terrain_v37_env_cfg import (
  g1_getup_terrain_v37_deploy_smp_env_cfg,
)

ACTOR_HISTORY_LENGTH = 4
TRAIN_PLATE_PROBABILITY = 0.90
PLATE_TERRAIN_NAMES = ("flat", "stairs")
PLATE_EDGE_COHORTS = (0,)

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


def _insert_reset_after_terrain_grounding(cfg, name, term) -> None:
  reordered = {}
  inserted = False
  for event_name, event_term in cfg.events.items():
    reordered[event_name] = event_term
    if event_name == "ground_procedural_fall_on_terrain":
      reordered[name] = term
      inserted = True
  if not inserted:
    raise RuntimeError("V3.8 requires terrain grounding before plate placement")
  cfg.events = reordered


def g1_getup_plate_terrain_v38_deploy_smp_env_cfg(play: bool = False):
  """Combine V3.7 terrain recovery and V3.4 plate escape without actor privilege.

  The actor receives four frames of deployable proprioception only: zero base
  linear velocity, IMU angular velocity, projected gravity, joint position,
  joint velocity, and previous actions. Terrain labels, plate state, contact
  truth, and simulator velocities remain absent from the actor.
  """
  cfg = g1_getup_terrain_v37_deploy_smp_env_cfg(play=play)
  plate_cfg = g1_getup_escape_plate_v34_smp_env_cfg(play=play)

  cfg.scene.entities["escape_obstacle"] = deepcopy(
    plate_cfg.scene.entities["escape_obstacle"]
  )
  existing_sensors = {sensor.name for sensor in cfg.scene.sensors or ()}
  plate_sensors = tuple(
    deepcopy(sensor)
    for sensor in plate_cfg.scene.sensors or ()
    if sensor.name in _PLATE_SENSOR_NAMES and sensor.name not in existing_sensors
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + plate_sensors

  reset_plate = deepcopy(plate_cfg.events["reset_escape_obstacle"])
  obstacle_probability = TRAIN_PLATE_PROBABILITY
  if play:
    obstacle_probability = float(os.environ.get("SMP_PLAY_ESCAPE_OBSTACLE", "0") == "1")
  reset_plate.params.update(
    {
      "obstacle_probability": obstacle_probability,
      "eligible_reset_types": (1, 2),
      "eligible_terrain_names": PLATE_TERRAIN_NAMES,
      "eligible_terrain_cohorts": PLATE_EDGE_COHORTS,
      "reground_robot": False,
    }
  )
  _insert_reset_after_terrain_grounding(cfg, "reset_escape_obstacle", reset_plate)
  cfg.events["update_escape_phase"] = deepcopy(plate_cfg.events["update_escape_phase"])
  cfg.events["update_escape_phase"].params["relative_to_env_origin"] = True

  cfg.rewards["task_smp_product"].func = mdp.escape_gated_task_smp_product
  cfg.rewards["task_smp_product"].params["constrained_scale"] = 0.05
  cfg.rewards[
    "recovery_initiation"
  ].func = mdp.escape_gated_recovery_initiation_progress
  cfg.rewards["recovery_initiation"].params["constrained_scale"] = 0.0
  for name in _PLATE_REWARD_NAMES:
    cfg.rewards[name] = deepcopy(plate_cfg.rewards[name])
  cfg.rewards["hand_supported_escape_progress"].params["relative_to_env_origin"] = True
  cfg.rewards["escape_geometry_progress"].params["relative_to_env_origin"] = True

  cfg.terminations["invalid_escape_episode"] = deepcopy(
    plate_cfg.terminations["invalid_escape_episode"]
  )
  for name in _PLATE_METRIC_NAMES:
    cfg.metrics[name] = deepcopy(plate_cfg.metrics[name])

  # ObservationManager stores each term oldest-to-newest before concatenating
  # terms. The matching checkpoint adapter follows this exact layout.
  cfg.observations["actor"].history_length = ACTOR_HISTORY_LENGTH
  cfg.observations["actor"].flatten_history_dim = True
  cfg.sim.nconmax = max(cfg.sim.nconmax, 512)
  cfg.sim.njmax = max(cfg.sim.njmax, 5000)
  return cfg


__all__ = [
  "ACTOR_HISTORY_LENGTH",
  "g1_getup_plate_terrain_v38_deploy_smp_env_cfg",
]
