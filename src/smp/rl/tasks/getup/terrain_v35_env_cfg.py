"""V3.5 zero-shot recovery benchmark on discontinuous terrain.

This task intentionally keeps the policy observation vector identical to V3.4.
Terrain identity, difficulty, local heights, reset mode, and contact labels are
evaluation-only state; the actor receives no privileged terrain information.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxRandomGridTerrainCfg,
  SubTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v8_env_cfg import g1_getup_v8_natural_smp_env_cfg

_PLAY_TERRAIN_ENV = "SMP_PLAY_TERRAIN_TYPE"
_PLAY_TERRAIN_LEVEL_ENV = "SMP_PLAY_TERRAIN_LEVEL"
_PLAY_RESET_POSE_ENV = "SMP_PLAY_TERRAIN_RESET_POSE"

TERRAIN_KINDS = ("flat", "slope", "stairs", "rough", "mixed")
RESET_POSE_WEIGHTS = {
  "mixed": (1.0, 1.0, 1.0, 1.0),
  "prone": (1.0, 0.0, 0.0, 0.0),
  "supine": (0.0, 1.0, 0.0, 0.0),
  "left_side": (0.0, 0.0, 1.0, 0.0),
  "right_side": (0.0, 0.0, 0.0, 1.0),
}

# Level zero is deliberately non-trivial for non-flat families.  The four
# levels are fixed benchmark bins, not a training curriculum.
SLOPE_DEGREES = (5.0, 10.0, 15.0, 20.0)
STAIR_HEIGHTS_M = (0.05, 0.10, 0.15, 0.20)
ROUGH_HEIGHTS_M = (0.02, 0.04, 0.06, 0.08)
TERRAIN_PATCH_SIZE = (8.0, 8.0)
TERRAIN_OUTER_BORDER_M = 1.0
STAIR_APRON_WIDTH_M = 1.90


@dataclass(kw_only=True)
class BoxSlopeTerrainCfg(SubTerrainCfg):
  """A single directed planar slope implemented without a heightfield."""

  angle_degrees: float | None = None
  level_angles_degrees: tuple[float, ...] | None = None
  thickness: float = 0.30

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    if self.level_angles_degrees is not None:
      level = min(
        int(difficulty * len(self.level_angles_degrees)),
        len(self.level_angles_degrees) - 1,
      )
      angle_degrees = self.level_angles_degrees[level]
    elif self.angle_degrees is not None:
      angle_degrees = self.angle_degrees
    else:
      raise ValueError("a fixed angle or quantized level angles must be provided")
    angle = math.radians(angle_degrees)
    half_x = self.size[0] / 2
    half_y = self.size[1] / 2
    half_z = self.thickness / 2
    # Positive rotation about y makes +x the downhill direction.  Offset the
    # box centre so its top plane crosses z=0 at the patch centre.
    quat = (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)
    geom = spec.body("terrain").add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(half_x, half_y, half_z),
      pos=(half_x, half_y, -half_z * math.cos(angle)),
      quat=quat,
    )
    origin = np.array((half_x, half_y, 0.0))
    return TerrainOutput(
      origin=origin,
      geometries=[TerrainGeometry(geom=geom, color=(0.24, 0.52, 0.72, 1.0))],
    )


def terrain_generator_v35(kind: str, level: int, seed: int) -> TerrainGeneratorCfg:
  if kind not in TERRAIN_KINDS:
    choices = ", ".join(TERRAIN_KINDS)
    raise ValueError(f"unknown terrain type {kind!r}; choose {choices}")
  if level not in range(4):
    raise ValueError("terrain level must be an integer from 0 to 3")

  # A 0.55 m central support patch is large enough for the reset torso but
  # small enough that arms and legs immediately interact with the surrounding
  # slope, stair edge, or rough cells.
  platform_width = 0.55
  stair_height = STAIR_HEIGHTS_M[level]
  rough_height = ROUGH_HEIGHTS_M[level]
  families = {
    "flat": BoxFlatTerrainCfg(proportion=1.0),
    "slope": BoxSlopeTerrainCfg(
      proportion=1.0,
      angle_degrees=SLOPE_DEGREES[level],
    ),
    "stairs": BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(stair_height, stair_height),
      step_width=0.30,
      platform_width=platform_width,
      # Keep roughly six stair rings, then add a broad flat apron so a failed
      # recovery rolls onto valid ground instead of immediately leaving terrain.
      border_width=STAIR_APRON_WIDTH_M,
    ),
    "rough": BoxRandomGridTerrainCfg(
      proportion=1.0,
      grid_width=0.30,
      grid_height_range=(rough_height, rough_height),
      platform_width=platform_width,
      merge_similar_heights=True,
      height_merge_threshold=0.02,
      max_merge_distance=3,
      border_width=1.0,
    ),
  }
  sub_terrains = families if kind == "mixed" else {kind: families[kind]}
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=kind == "mixed",
    size=TERRAIN_PATCH_SIZE,
    border_width=TERRAIN_OUTER_BORDER_M,
    border_height=1.0,
    num_rows=1,
    num_cols=max(1, len(sub_terrains)),
    color_scheme="height",
    sub_terrains=sub_terrains,
    difficulty_range=(1.0, 1.0),
  )


def terrain_surface_normals_v35(
  kind: str, level: int
) -> tuple[tuple[float, float, float], ...]:
  """Return support normals ordered like the generated terrain columns."""
  if kind not in TERRAIN_KINDS:
    choices = ", ".join(TERRAIN_KINDS)
    raise ValueError(f"unknown terrain type {kind!r}; choose {choices}")
  if level not in range(4):
    raise ValueError("terrain level must be an integer from 0 to 3")
  slope_normal = (
    math.sin(math.radians(SLOPE_DEGREES[level])),
    0.0,
    math.cos(math.radians(SLOPE_DEGREES[level])),
  )
  family_normals = {
    "flat": (0.0, 0.0, 1.0),
    "slope": slope_normal,
    "stairs": (0.0, 0.0, 1.0),
    "rough": (0.0, 0.0, 1.0),
  }
  active_kinds = tuple(family_normals) if kind == "mixed" else (kind,)
  return tuple(family_normals[name] for name in active_kinds)


def g1_getup_terrain_v35_smp_env_cfg(play: bool = False):
  """Build the no-obstacle, zero-shot V3.5 terrain evaluation task."""
  cfg = g1_getup_v8_natural_smp_env_cfg(play=play)
  kind = os.environ.get(_PLAY_TERRAIN_ENV, "stairs") if play else "mixed"
  level_text = os.environ.get(_PLAY_TERRAIN_LEVEL_ENV, "1") if play else "1"
  try:
    level = int(level_text)
  except ValueError as exc:
    raise ValueError("terrain level must be an integer from 0 to 3") from exc
  reset_pose = os.environ.get(_PLAY_RESET_POSE_ENV, "mixed") if play else "mixed"
  if reset_pose not in RESET_POSE_WEIGHTS:
    choices = ", ".join(RESET_POSE_WEIGHTS)
    raise ValueError(f"unknown terrain reset pose {reset_pose!r}; choose {choices}")

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=terrain_generator_v35(kind, level, seed=20260818),
    env_spacing=None,
    max_init_terrain_level=0,
    debug_vis=play,
  )
  cfg.scene.extent = 7.0
  # Keep headroom for highly folded prone poses and simultaneous contacts with
  # multiple step/grid cells during interactive dragging.
  cfg.sim.nconmax = 512
  cfg.sim.njmax = 5000
  cfg.episode_length_s = 20.0

  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 1.0,
      "mode_weights": RESET_POSE_WEIGHTS[reset_pose],
      "root_height_range": (0.46, 0.46),
      "root_xy_range": 0.025,
      "root_linear_velocity": 0.0,
      "root_angular_velocity": 0.0,
      "orientation_noise": 0.10,
      "joint_noise": 0.18,
    }
  )

  surface_normals = terrain_surface_normals_v35(kind, level)

  # Insert grounding before recovery-stage initialization so stage labels and
  # the SMP history both describe the physically valid post-placement state.
  reordered_events = {}
  inserted = False
  for name, term in cfg.events.items():
    if name == "reset_recovery_stage":
      reordered_events["ground_procedural_fall_on_terrain"] = EventTermCfg(
        func=mdp.ground_procedural_fall_on_terrain,
        mode="reset",
        params={
          "ground_clearance": 0.006,
          "surface_normals": surface_normals,
        },
      )
      inserted = True
    reordered_events[name] = term
  if not inserted:
    reordered_events["ground_procedural_fall_on_terrain"] = EventTermCfg(
      func=mdp.ground_procedural_fall_on_terrain,
      mode="reset",
      params={
        "ground_clearance": 0.006,
        "surface_normals": surface_normals,
      },
    )
  cfg.events = reordered_events

  # Training-style pushes are disabled for the benchmark.  Interactive play
  # can still opt in through the existing --auto-disturbances flag.
  if play and os.environ.get("SMP_PLAY_AUTO_DISTURBANCES") != "1":
    cfg.events.pop("stratified_post_stand_wrench", None)

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


__all__ = [
  "RESET_POSE_WEIGHTS",
  "ROUGH_HEIGHTS_M",
  "SLOPE_DEGREES",
  "STAIR_APRON_WIDTH_M",
  "STAIR_HEIGHTS_M",
  "TERRAIN_OUTER_BORDER_M",
  "TERRAIN_PATCH_SIZE",
  "TERRAIN_KINDS",
  "g1_getup_terrain_v35_smp_env_cfg",
  "terrain_generator_v35",
  "terrain_surface_normals_v35",
]
