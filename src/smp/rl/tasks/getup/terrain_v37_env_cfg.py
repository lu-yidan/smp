"""V3.7 edge-aware stair recovery with deployable actor observations."""

from __future__ import annotations

import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.terrain_v35_env_cfg import (
  STAIR_APRON_WIDTH_M,
  STAIR_HEIGHTS_M,
  TERRAIN_PATCH_SIZE,
)
from smp.rl.tasks.getup.terrain_v363_env_cfg import (
  g1_getup_terrain_v363_smp_env_cfg,
)

EDGE_RESET_COHORTS = ("center", "near_edge", "straddle", "lower_tread")
EDGE_RESET_WEIGHTS = (0.40, 0.20, 0.20, 0.20)
TERRAIN_TRAIN_PROPORTIONS = (0.25, 0.20, 0.40, 0.15)
_PLAY_EDGE_COHORT_ENV = "SMP_PLAY_TERRAIN_EDGE_COHORT"


def _edge_weights(play: bool) -> tuple[float, float, float, float]:
  if not play:
    return EDGE_RESET_WEIGHTS
  selected = os.environ.get(_PLAY_EDGE_COHORT_ENV, "mixed")
  if selected == "mixed":
    return EDGE_RESET_WEIGHTS
  if selected not in EDGE_RESET_COHORTS:
    choices = ", ".join(("mixed", *EDGE_RESET_COHORTS))
    raise ValueError(f"unknown terrain edge cohort {selected!r}; choose {choices}")
  index = EDGE_RESET_COHORTS.index(selected)
  return tuple(1.0 if i == index else 0.0 for i in range(4))


def _edge_aware_cfg(play: bool):
  cfg = g1_getup_terrain_v363_smp_env_cfg(play=play)
  if not play:
    generator = cfg.scene.terrain.terrain_generator
    if generator is None:
      raise RuntimeError("V3.7 training requires generated terrain")
    names = ("flat", "slope", "stairs", "rough")
    if tuple(generator.sub_terrains) != names:
      raise RuntimeError("V3.7 terrain columns must keep their audited order")
    for name, proportion in zip(names, TERRAIN_TRAIN_PROPORTIONS, strict=True):
      generator.sub_terrains[name].proportion = proportion
    curriculum = cfg.curriculum["terrain_levels"].params
    curriculum["minimum_level_fractions"] = (0.75, 0.25, 0.0, 0.0)
    curriculum["maximum_terrain_level"] = 1

  edge_event = EventTermCfg(
    func=mdp.sample_terrain_edge_reset,
    mode="reset",
    params={
      "cohort_weights": _edge_weights(play),
      "stair_step_heights": STAIR_HEIGHTS_M,
      "terrain_size": TERRAIN_PATCH_SIZE[0],
      "stair_border_width": STAIR_APRON_WIDTH_M,
      "stair_platform_width": 0.55,
      "stair_step_width": 0.30,
    },
  )
  reordered = {}
  inserted = False
  for name, term in cfg.events.items():
    reordered[name] = term
    if name == "mixed_fall_reset":
      reordered["sample_terrain_edge_reset"] = edge_event
      inserted = True
  if not inserted:
    raise RuntimeError("V3.7 requires mixed_fall_reset before terrain grounding")
  cfg.events = reordered

  ground = cfg.events["ground_procedural_fall_on_terrain"]
  ground.params.update(
    {
      "use_stair_height_profile": True,
      "stair_step_heights": STAIR_HEIGHTS_M,
      "terrain_size": TERRAIN_PATCH_SIZE[0],
      "stair_border_width": STAIR_APRON_WIDTH_M,
      "stair_platform_width": 0.55,
      "stair_step_width": 0.30,
    }
  )
  cfg.rewards["terrain_planar_displacement"].params["reference_reset_anchor"] = True
  cfg.metrics["terrain_planar_displacement"].params["reference_reset_anchor"] = True
  cfg.metrics.update(
    {
      "terrain_edge_reset": MetricsTermCfg(func=mdp.terrain_edge_reset_metric),
      "terrain_reset_offset": MetricsTermCfg(func=mdp.terrain_reset_offset_metric),
    }
  )
  return cfg


def g1_getup_terrain_v37_smp_env_cfg(play: bool = False):
  """Oracle-observation V3.7 task for controlled ablation only."""
  return _edge_aware_cfg(play)


def g1_getup_terrain_v37_deploy_smp_env_cfg(play: bool = False):
  """Deployment task: actor base velocity is identically zero."""
  cfg = _edge_aware_cfg(play)
  cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
    func=mdp.zero_base_linear_velocity
  )
  return cfg


__all__ = [
  "EDGE_RESET_COHORTS",
  "EDGE_RESET_WEIGHTS",
  "g1_getup_terrain_v37_deploy_smp_env_cfg",
  "g1_getup_terrain_v37_smp_env_cfg",
]
