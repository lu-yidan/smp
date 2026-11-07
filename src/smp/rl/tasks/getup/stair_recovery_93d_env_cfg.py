"""V36 93D recovery across stair positions, with local support and drift costs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)

from smp.rl.tasks.getup.mdp.events import (
  _collision_vertical_geometry,
  _prime_smp_history_from_current_state,
)
from smp.rl.tasks.getup.terrain_v36_93d_env_cfg import (
  g1_getup_terrain_v36_93d_smp_env_cfg,
)

HEIGHTS = (0.05, 0.10, 0.15, 0.20)


@dataclass(kw_only=True)
class BinnedStairsCfg(BoxPyramidStairsTerrainCfg):
  def function(self, difficulty, spec, rng):
    height = HEIGHTS[min(int(difficulty * 4), 3)]
    fixed = replace(self, step_height_range=(height, height))
    return BoxPyramidStairsTerrainCfg.function(fixed, difficulty, spec, rng)


def _ids(env, env_ids):
  return torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids


def _boxes(env, ids):
  """Cache the actual axis-aligned terrain boxes, including lower aprons."""
  terrain = env.scene.terrain
  if not hasattr(env, "_stair_boxes"):
    m, d = env.sim.mj_model, env.sim.mj_data
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    if body < 0:
      raise RuntimeError("terrain body missing")
    ids_cpu = np.flatnonzero(
      (m.geom_bodyid == body) & (m.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    )
    if not len(ids_cpu):
      raise RuntimeError("no terrain boxes")
    rotation = d.geom_xmat[ids_cpu].reshape(-1, 3, 3)
    if not np.allclose(np.abs(rotation), np.eye(3)[None], atol=1e-6):
      raise RuntimeError("stair grounding supports axis-aligned boxes only")
    positions, sizes = d.geom_xpos[ids_cpu], m.geom_size[ids_cpu]
    origins = terrain.terrain_origins.cpu().numpy()
    cells = []
    for row in origins:
      for origin in row:
        keep = np.all(np.abs(positions[:, :2] - origin[:2]) < 4.0 - 1e-5, axis=1)
        boxes = np.concatenate([positions[keep], sizes[keep]], -1)
        if not len(boxes):
          raise RuntimeError("empty terrain cell")
        cells.append(boxes)
    maximum = max(len(x) for x in cells)
    array = np.zeros((len(cells), maximum, 6), dtype=np.float32)
    array[:, :, 2] = -10000  # padded boxes never become support
    for i, x in enumerate(cells):
      array[i, : len(x)] = x
    env._stair_boxes = torch.tensor(
      array.reshape(*origins.shape[:2], maximum, 6), device=env.device
    )
  return env._stair_boxes[terrain.terrain_levels[ids], terrain.terrain_types[ids]]


def surface_height(env, ids, xy):
  boxes = _boxes(env, ids)
  inside = (xy[:, :, None, :] - boxes[:, None, :, :2]).abs() <= boxes[
    :, None, :, 3:5
  ] + 1e-5
  top = boxes[:, :, 2] + boxes[:, :, 5]
  return torch.where(inside.all(-1), top[:, None, :], -10000.0).amax(-1).clamp_min(0.0)


@torch.no_grad()
def sample_stair_level(env, env_ids=None, fixed_level=None):
  ids = _ids(env, env_ids)
  terrain = env.scene.terrain
  # Nonzero exposure to every difficulty from the start, independent of noisy
  # standing counters. Hard cases increase over the first 2,000 PPO updates.
  fraction = min(env.common_step_counter / (2000 * 24), 1.0)
  initial = torch.tensor((0.55, 0.25, 0.15, 0.05), device=env.device)
  final = torch.tensor((0.20, 0.25, 0.30, 0.25), device=env.device)
  sampled = torch.multinomial(
    initial + (final - initial) * fraction, len(ids), replacement=True
  )
  if fixed_level is not None:
    sampled.fill_(fixed_level)
  sampled[terrain.terrain_types[ids] == 0] = 0
  terrain.terrain_levels[ids] = sampled
  terrain.env_origins[ids] = terrain.terrain_origins[
    sampled, terrain.terrain_types[ids]
  ]


def collision_checked_joint_bank(env):
  """Reject invalid articulations offline once; never add unchecked joint noise."""
  if hasattr(env, "_stair_joint_bank"):
    return env._stair_joint_bank
  robot = env.scene["robot"]
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  qadr = robot.indexing.joint_q_adr.cpu().numpy()
  fadr = robot.indexing.free_joint_q_adr.cpu().numpy()
  joint_ids = robot.indexing.joint_ids.cpu().numpy()
  robot_geoms = set(robot.indexing.geom_ids.cpu().tolist())
  default = robot.data.default_joint_pos[0].cpu().numpy()
  lo, hi = m.jnt_range[joint_ids].T
  rng = np.random.default_rng(20261108)
  accepted = []
  attempts = 0
  while len(accepted) < 2048 and attempts < 50000:
    attempts += 1
    angles = np.clip(
      default + rng.uniform(-0.20, 0.20, len(default)), lo + 0.005, hi - 0.005
    )
    d.qpos[:] = m.qpos0
    d.qpos[fadr] = np.array((0.0, 0.0, 20.0, 1.0, 0.0, 0.0, 0.0))
    d.qpos[qadr] = angles
    mujoco.mj_forward(m, d)
    minimum = min(
      (
        c.dist
        for c in d.contact
        if int(c.geom1) in robot_geoms and int(c.geom2) in robot_geoms
      ),
      default=0.0,
    )
    if minimum >= -0.001:
      accepted.append(angles)
  if len(accepted) != 2048:
    raise RuntimeError(
      f"Could not build a valid reset articulation bank: {len(accepted)}/{attempts}"
    )
  env._stair_bank_attempts = attempts
  env._stair_joint_bank = torch.tensor(
    np.stack(accepted), dtype=torch.float32, device=env.device
  )
  return env._stair_joint_bank


@torch.no_grad()
def place_on_stairs(env, env_ids=None):
  ids = _ids(env, env_ids)
  if not len(ids):
    return
  robot, terrain = env.scene["robot"], env.scene.terrain
  n = len(ids)
  bank = collision_checked_joint_bank(env)
  joint_pos = bank[torch.randint(0, len(bank), (n,), device=env.device)]
  robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)
  # 0=upper platform, 1=treads/risers, 2=bottom approach, 3=corners.
  region = torch.multinomial(
    torch.tensor((0.15, 0.40, 0.30, 0.15), device=env.device), n, replacement=True
  )
  radial_lo = torch.tensor((0.0, 0.50, 1.85, 0.80), device=env.device)[region]
  radial_hi = torch.tensor((0.40, 1.75, 2.35, 1.90), device=env.device)[region]
  radial = radial_lo + torch.rand(n, device=env.device) * (radial_hi - radial_lo)
  tangent = (torch.rand(n, device=env.device) * 2 - 1) * radial * 0.75
  tangent = torch.where(
    region == 3, radial * (0.85 + 0.15 * torch.rand(n, device=env.device)), tangent
  )
  side = torch.randint(0, 4, (n,), device=env.device)
  x = torch.where(side % 2 == 0, radial, tangent)
  y = torch.where(side % 2 == 0, tangent, radial)
  x *= torch.where(side < 2, 1.0, -1.0)
  y *= torch.where(side < 2, 1.0, -1.0)
  offset = torch.stack((x, y), -1)
  flat = terrain.terrain_types[ids] == 0
  offset[flat] = (torch.rand(int(flat.sum()), 2, device=env.device) * 2 - 1) * 0.4
  if not hasattr(env, "_stair_region"):
    env._stair_region = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._recovery_start_xy = torch.zeros(env.num_envs, 2, device=env.device)
    env._recovery_support_height = torch.zeros(env.num_envs, device=env.device)
    env._stair_max_displacement = torch.zeros(env.num_envs, device=env.device)
    env._stair_stand_hold = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._stair_success3 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._stair_stand_anchor = torch.zeros(env.num_envs, 2, device=env.device)
    env._stair_anchor_valid = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
  env._stair_region[ids] = region
  root = torch.cat(
    (robot.data.root_link_pose_w[ids].clone(), torch.zeros(n, 6, device=env.device)), -1
  )
  root[:, :2] = terrain.env_origins[ids, :2] + offset
  root[:, 2] = terrain.env_origins[ids, 2] + 2.0
  robot.write_root_state_to_sim(root, env_ids=ids)
  env.sim.forward()
  pos, extent, centers, half, _ = _collision_vertical_geometry(
    env, ids, r".*_collision$"
  )
  boxes = _boxes(env, ids)
  overlap = (centers[:, :, None, :2] - boxes[:, None, :, :2]).abs() <= half[
    :, :, None, :2
  ] + boxes[:, None, :, 3:5]
  bottoms = pos[:, :, 2] - extent
  required = boxes[:, None, :, 2] + boxes[:, None, :, 5] - bottoms[:, :, None] + 0.006
  shift = torch.where(overlap.all(-1), required, -10000.0).amax(dim=(1, 2))
  if not torch.isfinite(shift).all() or (shift < -100).any():
    raise RuntimeError("no support below reset robot")
  root[:, 2] += shift
  robot.write_root_state_to_sim(root, env_ids=ids)
  env.sim.forward()
  env._recovery_start_xy[ids] = root[:, :2]
  env._stair_max_displacement[ids] = 0
  env._stair_stand_hold[ids] = 0
  env._stair_success3[ids] = False
  env._stair_anchor_valid[ids] = False
  update_stair_reference(env, ids, track=False)
  _prime_smp_history_from_current_state(env, ids)


@torch.no_grad()
def update_stair_reference(env, env_ids=None, track=True):
  if not hasattr(env, "_recovery_start_xy"):
    return
  ids = _ids(env, env_ids)
  robot = env.scene["robot"]
  feet = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]
  root_xy = robot.data.root_link_pos_w[ids, :2]
  root_h = surface_height(env, ids, root_xy[:, None, :]).squeeze(-1)
  foot_h = surface_height(env, ids, robot.data.site_pos_w[ids][:, feet, :2]).mean(-1)
  upright = (-robot.data.projected_gravity_b[ids, 2]).clamp(0, 1)
  blend = ((upright - 0.6) / 0.25).clamp(0, 1)
  env._recovery_support_height[ids] = (1 - blend) * root_h + blend * foot_h
  if not track:
    return
  distance = torch.linalg.vector_norm(root_xy - env._recovery_start_xy[ids], dim=-1)
  env._stair_max_displacement[ids] = torch.maximum(
    env._stair_max_displacement[ids], distance
  )
  head = robot.find_sites(["head"], preserve_order=True)[0][0]
  height = robot.data.site_pos_w[ids, head, 2] - env._recovery_support_height[ids]
  stable = (
    (height >= 1.10)
    & (upright >= 0.85)
    & (robot.data.root_link_lin_vel_w[ids].norm(dim=-1) < 0.5)
    & (robot.data.root_link_ang_vel_w[ids].norm(dim=-1) < 1.0)
  )
  env._stair_stand_hold[ids] = torch.where(stable, env._stair_stand_hold[ids] + 1, 0)
  env._stair_success3[ids] |= env._stair_stand_hold[ids] >= 150
  new_anchor = (env._stair_stand_hold[ids] >= 50) & ~env._stair_anchor_valid[ids]
  env._stair_stand_anchor[ids[new_anchor]] = root_xy[new_anchor]
  env._stair_anchor_valid[ids[new_anchor]] = True


def local_displacement(env, free_radius=0.4):
  displacement = (
    env.scene["robot"].data.root_link_pos_w[:, :2] - env._recovery_start_xy
  ).norm(dim=-1)
  return (displacement - free_radius).clamp_min(0).square()


def standing_drift(env):
  distance = (
    env.scene["robot"].data.root_link_pos_w[:, :2] - env._stair_stand_anchor
  ).norm(dim=-1)
  return env._stair_anchor_valid * (distance - 0.15).clamp_min(0).square()


def local_exit(env, radius=1.5):
  return (env.scene["robot"].data.root_link_pos_w[:, :2] - env._recovery_start_xy).norm(
    dim=-1
  ) > radius


def stair_metric(env, field, level=-1):
  values = {
    "sr3": env._stair_success3.float(),
    "local_sr3": (env._stair_success3 & (env._stair_max_displacement <= 0.75)).float(),
    "max_displacement": env._stair_max_displacement,
    "level": env.scene.terrain.terrain_levels.float(),
  }[field]
  if level < 0:
    return values
  mask = (env.scene.terrain.terrain_types > 0) & (
    env.scene.terrain.terrain_levels == level
  )
  # Preserve per-env vectors; separate denominator metric allows exact cohort aggregation.
  return values * mask


def stair_recovery_93d_env_cfg(play=False):
  cfg = g1_getup_terrain_v36_93d_smp_env_cfg(play)
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    env_spacing=None,
    max_init_terrain_level=3,
    terrain_generator=TerrainGeneratorCfg(
      seed=20261108,
      curriculum=True,
      size=(8.0, 8.0),
      num_rows=4,
      num_cols=3,
      border_width=1.0,
      border_height=1.0,
      color_scheme="height",
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.10),
        "stairs_30cm": BinnedStairsCfg(
          proportion=0.45,
          step_height_range=(0.05, 0.20),
          step_width=0.30,
          platform_width=0.55,
          border_width=1.9,
        ),
        "stairs_40cm": BinnedStairsCfg(
          proportion=0.45,
          step_height_range=(0.05, 0.20),
          step_width=0.40,
          platform_width=0.55,
          border_width=1.9,
        ),
      },
    ),
  )
  cfg.curriculum = {}
  cfg.episode_length_s = 20.0
  cfg.events["mixed_fall_reset"].params.update(
    mode_weights=(1.0, 1.0, 1.0, 1.0), orientation_noise=0.18, joint_noise=0.20
  )
  cfg.events["ground_procedural_fall_on_terrain"] = EventTermCfg(
    func=place_on_stairs, mode="reset"
  )
  events = {}
  for name, term in cfg.events.items():
    if name == "mixed_fall_reset":
      events["sample_stair_level"] = EventTermCfg(func=sample_stair_level, mode="reset")
    if name == "update_recovery_stage":
      events["stair_support_reference"] = EventTermCfg(
        func=update_stair_reference, mode="step"
      )
    events[name] = term
  cfg.events = events
  cfg.terminations.pop("stood_up", None)
  cfg.terminations["local_recovery_exit"] = TerminationTermCfg(func=local_exit)
  cfg.rewards["terrain_planar_displacement"] = RewardTermCfg(
    func=local_displacement, weight=-0.30
  )
  cfg.rewards["standing_anchor_drift"] = RewardTermCfg(
    func=standing_drift, weight=-0.50
  )
  cfg.metrics["terrain_planar_displacement"] = MetricsTermCfg(func=local_displacement)
  for field in ("sr3", "local_sr3", "max_displacement", "level"):
    cfg.metrics["stair_" + field] = MetricsTermCfg(
      func=stair_metric, params={"field": field}, reduce="last"
    )
  return cfg
