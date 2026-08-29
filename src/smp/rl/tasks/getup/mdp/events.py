"""Reset events for the getup task."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, yaw_quat

from smp.rl.events import prime_sim_and_buffer

__all__ = [
  "apply_sustained_constraint",
  "failure_state_replay_reset",
  "ground_procedural_fall_on_terrain",
  "lafan_milestone_reset",
  "init_matched_reset_bank",
  "matched_reset_bank_reset",
  "mixed_fall_reset",
  "sample_terrain_edge_reset",
  "sample_weighted_terrain_levels",
  "post_stand_body_wrench",
  "random_body_wrench",
  "record_failure_states",
  "reset_escape_obstacle",
  "reset_guided_escape_plate",
  "reset_guided_escape_plate_curriculum",
  "reset_stratified_guided_escape_plate",
  "reset_recovery_stage",
  "reset_sustained_constraint",
  "reset_stand_counter",
  "stratified_post_stand_wrench",
  "update_escape_phase",
  "update_recovery_stage",
]

_MATCHED_BANK_SHAPES = {
  "root_state": (13,),
  "joint_pos": (29,),
  "joint_vel": (29,),
  "smp_window": (10, 59),
}


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _validate_matched_reset_bank_payload(
  payload: dict[str, torch.Tensor], expected_num_states: int | None = None
) -> int:
  """Validate frozen reset tensors without trusting serialization metadata."""
  required = {*_MATCHED_BANK_SHAPES, "reset_type"}
  if set(payload) != required:
    raise ValueError(f"matched reset bank keys must be {sorted(required)}")
  root = payload["root_state"]
  if root.ndim != 2:
    raise ValueError("root_state must be rank two")
  num_states = int(root.shape[0])
  if expected_num_states is not None and num_states != expected_num_states:
    raise ValueError(
      f"matched reset bank has {num_states} states, expected {expected_num_states}"
    )
  if num_states <= 0:
    raise ValueError("matched reset bank is empty")
  for name, trailing_shape in _MATCHED_BANK_SHAPES.items():
    tensor = payload[name]
    if tuple(tensor.shape) != (num_states, *trailing_shape):
      raise ValueError(f"{name} has invalid shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
      raise ValueError(f"{name} must contain finite floating-point values")
  reset_type = payload["reset_type"]
  if reset_type.shape != (num_states,) or reset_type.dtype not in (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
  ):
    raise ValueError("reset_type must be one integer per state")
  if torch.any((reset_type < 0) | (reset_type > 4)):
    raise ValueError("reset_type must be in [0, 4]")
  quat_norm = torch.linalg.vector_norm(root[:, 3:7], dim=-1)
  if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1.0e-3):
    raise ValueError("root quaternion is not normalized")
  # Feature layout: root position 3, root rotation 6, then 29 joint positions.
  bank_joint = payload["smp_window"][:, -1, 9:38]
  if not torch.allclose(bank_joint, payload["joint_pos"], atol=1.0e-4):
    raise ValueError("SMP window and banked joint position disagree")
  return num_states


@torch.no_grad()
def init_matched_reset_bank(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  bank_path: str = "",
  bank_sha256: str = "",
  expected_num_states: int = 262144,
  include_smp_window: bool = True,
  sampling_seed: int | None = None,
) -> None:
  """Load one SHA-locked Tier-A reset bank onto the environment device."""
  del env_ids
  path = Path(bank_path)
  if not path.is_file():
    raise FileNotFoundError(f"matched reset bank missing: {path}")
  if len(bank_sha256) != 64 or _file_sha256(path) != bank_sha256:
    raise ValueError("matched reset bank SHA-256 mismatch")
  payload = torch.load(path, map_location="cpu", weights_only=True)
  if not isinstance(payload, dict):
    raise ValueError("matched reset bank must contain a tensor dictionary")
  num_states = _validate_matched_reset_bank_payload(payload, expected_num_states)
  selected = dict(payload)
  if not include_smp_window:
    selected.pop("smp_window")
  elif not hasattr(env, "_smp_buffer"):
    raise RuntimeError("SMP-window replay requires init_smp_state to run first")
  env._matched_reset_bank = {  # type: ignore[attr-defined]
    name: tensor.to(env.device) for name, tensor in selected.items()
  }
  env._matched_reset_bank_sha256 = bank_sha256  # type: ignore[attr-defined]
  env._matched_reset_bank_has_window = include_smp_window  # type: ignore[attr-defined]
  seed = int(env.cfg.seed if sampling_seed is None else sampling_seed)
  generator = torch.Generator(device="cpu")
  generator.manual_seed(seed)
  env._matched_reset_bank_permutation = torch.randperm(  # type: ignore[attr-defined]
    num_states, generator=generator, device="cpu"
  ).to(env.device)
  stride = 1 if num_states == 1 else 2 * (seed % (num_states // 2)) + 1
  while math.gcd(stride, num_states) != 1:
    stride = (stride + 1) % num_states
    if stride == 0:
      stride = 1
  env._matched_reset_bank_stride = stride  # type: ignore[attr-defined]
  env._matched_reset_bank_cursor = torch.zeros(  # type: ignore[attr-defined]
    env.num_envs, dtype=torch.long, device=env.device
  )
  env._matched_reset_bank_sampling_seed = seed  # type: ignore[attr-defined]


@torch.no_grad()
def matched_reset_bank_reset(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Replay identical current states, and SMP history when requested, for Tier A."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  bank = getattr(env, "_matched_reset_bank", None)
  if not isinstance(bank, dict):
    raise RuntimeError("matched reset bank was not initialized")
  num_states = int(bank["root_state"].shape[0])
  permutation = getattr(env, "_matched_reset_bank_permutation", None)
  cursors = getattr(env, "_matched_reset_bank_cursor", None)
  stride = getattr(env, "_matched_reset_bank_stride", None)
  if permutation is None or cursors is None or stride is None:
    raise RuntimeError("matched reset-bank sampler was not initialized")
  positions = (env_ids + cursors[env_ids] * int(stride)) % num_states
  indexes = permutation[positions]
  cursors[env_ids] += 1
  root_state = bank["root_state"][indexes].clone()
  joint_pos = bank["joint_pos"][indexes]
  joint_vel = bank["joint_vel"][indexes]
  robot = env.scene["robot"]
  origins = env.scene.env_origins[env_ids]
  if getattr(env, "_matched_reset_bank_has_window", False):
    default_xy = robot.data.default_root_state[env_ids, :2]
    placement_xy = root_state[:, :2] - default_xy
    prime_sim_and_buffer(
      env,
      env_ids,
      bank["smp_window"][indexes],
      placement_xy=placement_xy,
      placement_yaw=yaw_quat(root_state[:, 3:7]),
    )
  root_state[:, :3] += origins
  robot.write_root_state_to_sim(root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  reset_types = getattr(env, "_robust_reset_type", None)
  if reset_types is None:
    reset_types = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._robust_reset_type = reset_types  # type: ignore[attr-defined]
  reset_types[env_ids] = bank["reset_type"][indexes].long()
  env.sim.forward()


def _ensure_escape_state(env: ManagerBasedRlEnv) -> None:
  """Allocate shared state for free-object and guided-plate escape tasks."""
  if not hasattr(env, "_escape_phase"):
    env._escape_phase = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._escape_target_slot = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_phase  # type: ignore[attr-defined]
    )
    env._escape_contact_ever = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.bool, device=env.device
    )
    env._escape_clear_hold = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_phase  # type: ignore[attr-defined]
    )
    env._escape_best_separation = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
    env._escape_separation_delta = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_best_separation  # type: ignore[attr-defined]
    )
    env._escape_start_robot_xy = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, 2, device=env.device
    )
    env._escape_start_obstacle_xy = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_start_robot_xy  # type: ignore[attr-defined]
    )
  if not hasattr(env, "_escape_invalid_contact"):
    env._escape_invalid_contact = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.bool, device=env.device
    )
    env._escape_peak_penetration = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
    env._escape_peak_contact_force = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
    env._escape_sensor_grace = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._escape_invalid_setup = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.bool, device=env.device
    )
    env._escape_wait_steps = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._escape_first_contact_head_height = torch.full(  # type: ignore[attr-defined]
      (env.num_envs,), -1.0, device=env.device
    )
    env._escape_hand_support_steps = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._escape_hand_supported_progress = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
  if not hasattr(env, "_escape_covered_geom_count"):
    env._escape_covered_geom_count = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._escape_initial_covered_geom_count = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_covered_geom_count  # type: ignore[attr-defined]
    )
    env._escape_best_covered_geom_count = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_covered_geom_count  # type: ignore[attr-defined]
    )
    env._escape_coverage_score = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
    env._escape_best_coverage_score = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_coverage_score  # type: ignore[attr-defined]
    )
    env._escape_coverage_delta = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_coverage_score  # type: ignore[attr-defined]
    )
    env._escape_planar_clearance = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_coverage_score  # type: ignore[attr-defined]
    )
    env._escape_best_planar_clearance = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_coverage_score  # type: ignore[attr-defined]
    )
    env._escape_clearance_delta = torch.zeros_like(  # type: ignore[attr-defined]
      env._escape_coverage_score  # type: ignore[attr-defined]
    )
    env._escape_geometry_initialized = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.bool, device=env.device
    )


def _collision_vertical_geometry(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  geom_pattern: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return exact primitive z bounds plus conservative world XY AABBs.

  The returned tensors are geom position, vertical half-extent, world AABB
  centre, world AABB half-size, and global geom ids.  G1 collision geoms are
  spheres/capsules, but the common MuJoCo primitives are handled as well.
  """
  robot = env.scene["robot"]
  local_ids, names = robot.find_geoms(geom_pattern)
  if not local_ids:
    raise ValueError(f"no robot geoms match {geom_pattern!r}")
  local = torch.tensor(local_ids, dtype=torch.long, device=env.device)
  geom_ids = robot.indexing.geom_ids[local].long()
  pos = robot.data.data.geom_xpos[env_ids[:, None], geom_ids[None, :]]
  mat = robot.data.data.geom_xmat[env_ids[:, None], geom_ids[None, :]]
  size = env.sim.model.geom_size[env_ids[:, None], geom_ids[None, :]]
  geom_type = env.sim.model.geom_type[geom_ids]

  row_z = mat[:, :, 2, :]
  abs_row_z = row_z.abs()
  # Box is also the conservative fallback for non-primitive geometry.
  z_extent = (abs_row_z * size).sum(dim=-1)
  sphere = geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
  capsule = geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
  cylinder = geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
  ellipsoid = geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
  z_extent = torch.where(sphere[None, :], size[:, :, 0], z_extent)
  z_extent = torch.where(
    capsule[None, :],
    size[:, :, 0] + size[:, :, 1] * abs_row_z[:, :, 2],
    z_extent,
  )
  radial_projection = torch.sqrt(row_z[:, :, 0].square() + row_z[:, :, 1].square())
  z_extent = torch.where(
    cylinder[None, :],
    size[:, :, 0] * radial_projection + size[:, :, 1] * abs_row_z[:, :, 2],
    z_extent,
  )
  z_extent = torch.where(
    ellipsoid[None, :],
    torch.sqrt(((row_z * size).square()).sum(dim=-1)),
    z_extent,
  )

  aabb = env.sim.model.geom_aabb[env_ids[:, None], geom_ids[None, :]]
  aabb_center = pos + torch.einsum("ngij,ngj->ngi", mat, aabb[:, :, 0])
  aabb_half = torch.einsum("ngij,ngj->ngi", mat.abs(), aabb[:, :, 1])
  return pos, z_extent, aabb_center, aabb_half, geom_ids


def _guided_plate_planar_clearance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  collision_geom_pattern: str,
  plate_geom_name: str,
  plate_half_extents: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Measure full-body footprint coverage relative to the guided plate.

  Returns the number of collision geoms still covered by the plate footprint,
  their summed distance to the nearest footprint edge, and the minimum planar
  clearance after every collision geom is outside. Conservative projected
  AABBs make success harder to obtain, avoiding false clearance at rotated
  hands, feet, and capsules.
  """
  _, _, aabb_center, aabb_half, _ = _collision_vertical_geometry(
    env, env_ids, collision_geom_pattern
  )
  obstacle = env.scene["escape_obstacle"]
  local_plate_ids, _ = obstacle.find_geoms([plate_geom_name], preserve_order=True)
  if len(local_plate_ids) != 1:
    raise ValueError(f"plate geom {plate_geom_name!r} must resolve exactly once")
  local_plate = torch.tensor(local_plate_ids, dtype=torch.long, device=env.device)
  plate_geom_id = obstacle.indexing.geom_ids[local_plate][0].long()
  plate_pos = obstacle.data.data.geom_xpos[env_ids, plate_geom_id]
  plate_mat = obstacle.data.data.geom_xmat[env_ids, plate_geom_id]
  forward_xy = plate_mat[:, 0, :2]
  lateral_xy = plate_mat[:, 1, :2]

  relative_xy = aabb_center[:, :, :2] - plate_pos[:, None, :2]
  along = (relative_xy * forward_xy[:, None, :]).sum(dim=-1)
  across = (relative_xy * lateral_xy[:, None, :]).sum(dim=-1)
  along_extent = (
    aabb_half[:, :, 0] * forward_xy[:, None, 0].abs()
    + aabb_half[:, :, 1] * forward_xy[:, None, 1].abs()
  )
  across_extent = (
    aabb_half[:, :, 0] * lateral_xy[:, None, 0].abs()
    + aabb_half[:, :, 1] * lateral_xy[:, None, 1].abs()
  )
  along_overlap = plate_half_extents[0] + along_extent - along.abs()
  across_overlap = plate_half_extents[1] + across_extent - across.abs()
  covered = (along_overlap > 0.0) & (across_overlap > 0.0)
  # The nearest edge distance supplies dense progress while a geom remains
  # covered; it reaches zero exactly when the geom clears either plate edge.
  coverage_depth = torch.minimum(
    torch.clamp(along_overlap, min=0.0),
    torch.clamp(across_overlap, min=0.0),
  )
  coverage_score = torch.where(covered, coverage_depth, 0.0).sum(dim=-1)

  outside_along = torch.clamp(-along_overlap, min=0.0)
  outside_across = torch.clamp(-across_overlap, min=0.0)
  geom_clearance = torch.sqrt(outside_along.square() + outside_across.square())
  # Any covered geom has zero clearance, so the minimum becomes positive only
  # after the complete collision model, not merely the pelvis, has escaped.
  geom_clearance = torch.where(
    covered, torch.zeros_like(geom_clearance), geom_clearance
  )
  planar_clearance = geom_clearance.amin(dim=-1)
  return covered.sum(dim=-1), coverage_score, planar_clearance


def _prime_smp_history_from_current_state(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor
) -> None:
  """Fill the SMP window with the current, post-reset simulator state."""
  if env_ids.numel() == 0:
    return
  robot = env.scene["robot"]
  origins = env.scene.env_origins[env_ids]
  buffer = env._smp_buffer  # type: ignore[attr-defined]
  window_size = buffer.window_size
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  root_pos = robot.data.root_link_pos_w[env_ids] - origins
  ee_pos = robot.data.body_link_pos_w[env_ids][:, ee_indexes] - origins[:, None, :]
  buffer.reset(
    env_ids,
    root_pos[:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_quat_w[env_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_lin_vel_w[env_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_ang_vel_w[env_ids][:, None, :].expand(-1, window_size, -1),
    ee_pos[:, None, :, :].expand(-1, window_size, -1, -1),
    robot.data.joint_pos[env_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.joint_vel[env_ids][:, None, :].expand(-1, window_size, -1),
  )


@torch.no_grad()
def reset_escape_obstacle(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  obstacle_probability: float = 0.80,
  target_body_names: tuple[str, ...] = ("torso_link", "pelvis"),
  target_weights: tuple[float, ...] = (0.65, 0.35),
  eligible_reset_types: tuple[int, ...] | None = None,
  xy_offset_range: float = 0.035,
  clearance: float = 0.055,
  inactive_xy: tuple[float, float] = (0.72, 0.72),
) -> None:
  """Place a free obstacle on the torso/pelvis after the robot fall reset.

  The obstacle is a physical entity: unlike a body-following wrench, the robot
  can push it away or crawl out from underneath it.  Target/body identity is
  simulator-only state and is not appended to policy observations.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  if len(target_body_names) != len(target_weights):
    raise ValueError("target_body_names and target_weights must have equal length")
  weights = torch.tensor(target_weights, dtype=torch.float, device=env.device)
  if torch.any(weights < 0.0) or weights.sum() <= 0.0:
    raise ValueError("target_weights must be non-negative with positive sum")
  weights /= weights.sum()

  robot = env.scene["robot"]
  obstacle = env.scene["escape_obstacle"]
  body_ids = robot.find_bodies(list(target_body_names), preserve_order=True)[0]
  if len(body_ids) != len(target_body_names):
    raise ValueError("all escape target bodies must resolve exactly once")
  body_ids_tensor = torch.tensor(body_ids, dtype=torch.long, device=env.device)
  n = env_ids.numel()
  active = torch.rand(n, device=env.device) < obstacle_probability
  if eligible_reset_types is not None:
    reset_type = getattr(env, "_robust_reset_type", None)
    if reset_type is None:
      raise RuntimeError("eligible_reset_types requires mixed_fall_reset state")
    eligible = torch.zeros(n, dtype=torch.bool, device=env.device)
    for reset_value in eligible_reset_types:
      eligible |= reset_type[env_ids] == reset_value
    active &= eligible
  target_slot = torch.multinomial(weights, n, replacement=True)
  selected_body_ids = body_ids_tensor[target_slot]
  target_pos = robot.data.body_link_pos_w[env_ids, selected_body_ids].clone()

  xy_noise = torch.empty(n, 2, device=env.device).uniform_(
    -xy_offset_range, xy_offset_range
  )
  active_pos = target_pos
  active_pos[:, :2] += xy_noise
  active_pos[:, 2] += clearance
  origins = env.scene.env_origins[env_ids]
  inactive_pos = origins.clone()
  inactive_pos[:, 0] += inactive_xy[0]
  inactive_pos[:, 1] += inactive_xy[1]
  inactive_pos[:, 2] += clearance
  pos = torch.where(active[:, None], active_pos, inactive_pos)

  roll = torch.zeros(n, device=env.device)
  pitch = torch.zeros(n, device=env.device)
  yaw = torch.empty(n, device=env.device).uniform_(-torch.pi, torch.pi)
  quat = quat_from_euler_xyz(roll, pitch, yaw)
  velocity = torch.zeros(n, 6, device=env.device)
  obstacle.write_root_state_to_sim(
    torch.cat((pos, quat, velocity), dim=-1), env_ids=env_ids
  )
  env.sim.forward()

  _ensure_escape_state(env)

  # phase: 0=clean, 1=waiting for initial contact, 2=constrained, 3=escaped.
  env._escape_phase[env_ids] = active.long()  # type: ignore[attr-defined]
  env._escape_target_slot[env_ids] = target_slot  # type: ignore[attr-defined]
  env._escape_contact_ever[env_ids] = False  # type: ignore[attr-defined]
  env._escape_clear_hold[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_separation_delta[env_ids] = 0.0  # type: ignore[attr-defined]
  robot_xy = robot.data.root_link_pos_w[env_ids, :2]
  obstacle_xy = obstacle.data.root_link_pos_w[env_ids, :2]
  separation = torch.linalg.vector_norm(robot_xy - obstacle_xy, dim=-1)
  env._escape_best_separation[env_ids] = separation  # type: ignore[attr-defined]
  env._escape_start_robot_xy[env_ids] = robot_xy  # type: ignore[attr-defined]
  env._escape_start_obstacle_xy[env_ids] = obstacle_xy  # type: ignore[attr-defined]
  env._escape_invalid_contact[env_ids] = False  # type: ignore[attr-defined]
  env._escape_peak_penetration[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_peak_contact_force[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_sensor_grace[env_ids] = 1  # type: ignore[attr-defined]
  env._escape_invalid_setup[env_ids] = False  # type: ignore[attr-defined]
  env._escape_wait_steps[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_first_contact_head_height[env_ids] = -1.0  # type: ignore[attr-defined]
  env._escape_hand_support_steps[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_hand_supported_progress[env_ids] = 0.0  # type: ignore[attr-defined]


@torch.no_grad()
def reset_guided_escape_plate(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  obstacle_probability: float = 0.90,
  obstacle_probability_by_reset_type: tuple[float, ...] | None = None,
  target_body_name: str = "torso_link",
  eligible_reset_types: tuple[int, ...] | None = None,
  eligible_terrain_names: tuple[str, ...] | None = None,
  eligible_terrain_cohorts: tuple[int, ...] | None = None,
  xy_offset_range: float = 0.015,
  body_origin_clearance: float = 0.26,
  align_to_body: bool = False,
  longitudinal_offset: float = 0.0,
  lateral_offset: float = 0.0,
  longitudinal_offset_curriculum: tuple[float, float] | None = None,
  lateral_offset_curriculum: tuple[float, float] | None = None,
  overlap_curriculum_steps: int = 0,
  support_ready_supine: bool = False,
  support_arm_noise: float = 0.0,
  reground_robot: bool = True,
  ground_clearance: float = 0.004,
  surface_gap: float | None = None,
  plate_half_extents: tuple[float, float, float] = (0.45, 0.32, 0.035),
  collision_geom_pattern: str = r".*_collision$",
  inactive_xy: tuple[float, float] = (1.20, 1.20),
) -> None:
  """Reset a guided plate above the robot with conservative positive clearance.

  The mocap anchor is positioned once at reset.  It never follows the robot.
  A passive slide joint lets the plate descend under gravity and react to contact
  only along the vertical axis, so successful separation must come from robot
  translation rather than an obstacle that is teleported away.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  robot = env.scene["robot"]
  obstacle = env.scene["escape_obstacle"]
  target_ids = robot.find_bodies([target_body_name], preserve_order=True)[0]
  if len(target_ids) != 1:
    raise ValueError(f"target body {target_body_name!r} must resolve exactly once")

  n = env_ids.numel()
  sampled_longitudinal_offset = torch.full(
    (n,), longitudinal_offset, dtype=torch.float, device=env.device
  )
  sampled_lateral_offset = torch.full(
    (n,), lateral_offset, dtype=torch.float, device=env.device
  )
  if not 0.0 <= obstacle_probability <= 1.0:
    raise ValueError("obstacle_probability must be in [0, 1]")
  reset_type = getattr(env, "_robust_reset_type", None)
  probability = torch.full(
    (n,), obstacle_probability, dtype=torch.float, device=env.device
  )
  if obstacle_probability_by_reset_type is not None:
    if reset_type is None:
      raise RuntimeError(
        "obstacle_probability_by_reset_type requires mixed_fall_reset state"
      )
    probability_by_type = torch.tensor(
      obstacle_probability_by_reset_type, dtype=torch.float, device=env.device
    )
    if torch.any((probability_by_type < 0.0) | (probability_by_type > 1.0)):
      raise ValueError("per-reset-type obstacle probabilities must be in [0, 1]")
    reset_value = reset_type[env_ids]
    mapped = (reset_value >= 1) & (reset_value <= probability_by_type.numel())
    probability[mapped] = probability_by_type[reset_value[mapped] - 1]
  active = torch.rand(n, device=env.device) < probability
  if eligible_reset_types is not None:
    if reset_type is None:
      raise RuntimeError("eligible_reset_types requires mixed_fall_reset state")
    eligible = torch.zeros(n, dtype=torch.bool, device=env.device)
    for reset_value in eligible_reset_types:
      eligible |= reset_type[env_ids] == reset_value
    active &= eligible
  if eligible_terrain_names is not None:
    terrain = env.scene["terrain"]
    generator = terrain.cfg.terrain_generator
    if generator is None:
      raise RuntimeError("eligible_terrain_names requires generated terrain")
    terrain_names = list(generator.sub_terrains)
    eligible_columns = torch.tensor(
      [
        terrain_names.index(name)
        for name in eligible_terrain_names
        if name in terrain_names
      ],
      dtype=torch.long,
      device=env.device,
    )
    if eligible_columns.numel() == 0:
      active &= False
    else:
      active &= torch.isin(terrain.terrain_types[env_ids], eligible_columns)
  if eligible_terrain_cohorts is not None:
    cohort = getattr(env, "_terrain_reset_cohort", None)
    if cohort is None:
      raise RuntimeError(
        "eligible_terrain_cohorts requires sample_terrain_edge_reset state"
      )
    eligible_cohorts = torch.tensor(
      eligible_terrain_cohorts, dtype=torch.long, device=env.device
    )
    active &= torch.isin(cohort[env_ids], eligible_cohorts)

  active_ids = env_ids[active]
  if support_ready_supine and active_ids.numel() > 0:
    # V3.2/V3.3 were physically supine (reset type 2), despite their original
    # "prone" labels. Preserve that learned support-ready arm initialization
    # only for supine actors; true prone actors retain independently sampled
    # arms in the mixed-pose task.
    reset_type = getattr(env, "_robust_reset_type", None)
    if reset_type is None:
      raise RuntimeError("support_ready_supine requires mixed_fall_reset state")
    supine_ids = active_ids[reset_type[active_ids] == 2]
    arm_names = (
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    )
    if supine_ids.numel() > 0:
      arm_ids, _ = robot.find_joints(arm_names, preserve_order=True)
      if len(arm_ids) != len(arm_names):
        raise ValueError("all crawl-ready arm joints must resolve exactly once")
      joint_pos = robot.data.joint_pos[supine_ids].clone()
      joint_vel = torch.zeros_like(joint_pos)
      arm_pose = torch.tensor(
        (
          -2.104,
          -1.105,
          0.0,
          0.744,
          0.0,
          0.0,
          0.0,
          -2.104,
          1.105,
          0.0,
          0.744,
          0.0,
          0.0,
          0.0,
        ),
        device=env.device,
      )
      arm_values = arm_pose[None, :].expand(supine_ids.numel(), -1).clone()
      if support_arm_noise > 0.0:
        arm_values += torch.empty_like(arm_values).uniform_(
          -support_arm_noise, support_arm_noise
        )
      arm_local = torch.tensor(arm_ids, dtype=torch.long, device=env.device)
      joint_pos[:, arm_local] = arm_values
      robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=supine_ids)
      env.sim.forward()

    if reground_robot:
      # Flat-world plate tasks historically perform this final grounding pass.
      # Combined terrain tasks disable it because their terrain-aware grounding
      # event has already placed the robot on the exact local support surface.
      geom_pos, z_extent, _, _, _ = _collision_vertical_geometry(
        env, active_ids, collision_geom_pattern
      )
      lowest = (geom_pos[:, :, 2] - z_extent).amin(dim=-1)
      root_state = torch.cat(
        (
          robot.data.root_link_pose_w[active_ids].clone(),
          torch.zeros(active_ids.numel(), 6, device=env.device),
        ),
        dim=-1,
      )
      root_state[:, 2] += (
        env.scene.env_origins[active_ids, 2] + ground_clearance - lowest
      )
      robot.write_root_state_to_sim(root_state, env_ids=active_ids)
      env.sim.forward()
    _prime_smp_history_from_current_state(env, active_ids)

  target_pos = robot.data.body_link_pos_w[env_ids, target_ids[0]].clone()
  forward_xy = torch.zeros(n, 2, device=env.device)
  forward_xy[:, 0] = 1.0
  if align_to_body:
    head_ids = robot.find_sites(["head"], preserve_order=True)[0]
    if len(head_ids) != 1:
      raise ValueError("head site must resolve exactly once for plate alignment")
    head_xy = robot.data.site_pos_w[env_ids, head_ids[0], :2]
    raw_forward = head_xy - target_pos[:, :2]
    forward_norm = torch.linalg.vector_norm(raw_forward, dim=-1, keepdim=True)
    forward_xy = raw_forward / torch.clamp(forward_norm, min=1e-6)
    fallback = forward_norm[:, 0] < 1e-5
    forward_xy[fallback, 0] = 1.0
    forward_xy[fallback, 1] = 0.0
    target_pos[:, :2] += sampled_longitudinal_offset[:, None] * forward_xy
    lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=-1)
    target_pos[:, :2] += sampled_lateral_offset[:, None] * lateral_xy
    if overlap_curriculum_steps > 0:
      progress = min(
        float(env.common_step_counter) / max(overlap_curriculum_steps, 1), 1.0
      )
      if longitudinal_offset_curriculum is not None:
        amplitude = longitudinal_offset_curriculum[0] + progress * (
          longitudinal_offset_curriculum[1] - longitudinal_offset_curriculum[0]
        )
        longitudinal_noise = torch.empty(n, device=env.device).uniform_(
          -amplitude, amplitude
        )
        sampled_longitudinal_offset += longitudinal_noise
        target_pos[:, :2] += longitudinal_noise[:, None] * forward_xy
      if lateral_offset_curriculum is not None:
        amplitude = lateral_offset_curriculum[0] + progress * (
          lateral_offset_curriculum[1] - lateral_offset_curriculum[0]
        )
        lateral_noise = torch.empty(n, device=env.device).uniform_(
          -amplitude, amplitude
        )
        sampled_lateral_offset += lateral_noise
        target_pos[:, :2] += lateral_noise[:, None] * lateral_xy
  target_pos[:, :2] += torch.empty(n, 2, device=env.device).uniform_(
    -xy_offset_range, xy_offset_range
  )
  if surface_gap is None:
    # Targeting only the torso centre recreated V2's bug whenever a hand, foot,
    # or head collision was higher.  This conservative envelope is independent
    # of which link happens to be uppermost in the sampled lying pose.
    target_pos[:, 2] = (
      robot.data.body_link_pos_w[env_ids, :, 2].amax(dim=-1) + body_origin_clearance
    )
  else:
    # Place the board a few millimetres above the exact primitive support
    # surface.  Conservative XY AABBs decide which robot geoms overlap the
    # yaw-aligned plate footprint; exact primitive support avoids the large
    # false clearance caused by rotated hand/capsule AABBs.
    geom_pos, z_extent, aabb_center, aabb_half, _ = _collision_vertical_geometry(
      env, env_ids, collision_geom_pattern
    )
    lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=-1)
    relative_xy = aabb_center[:, :, :2] - target_pos[:, None, :2]
    along = (relative_xy * forward_xy[:, None, :]).sum(dim=-1)
    across = (relative_xy * lateral_xy[:, None, :]).sum(dim=-1)
    along_extent = (
      aabb_half[:, :, 0] * forward_xy[:, None, 0].abs()
      + aabb_half[:, :, 1] * forward_xy[:, None, 1].abs()
    )
    across_extent = (
      aabb_half[:, :, 0] * lateral_xy[:, None, 0].abs()
      + aabb_half[:, :, 1] * lateral_xy[:, None, 1].abs()
    )
    overlaps = (along.abs() <= plate_half_extents[0] + along_extent) & (
      across.abs() <= plate_half_extents[1] + across_extent
    )
    surface_top = torch.where(
      overlaps,
      geom_pos[:, :, 2] + z_extent,
      torch.full_like(along, -torch.inf),
    ).amax(dim=-1)
    if torch.any(~torch.isfinite(surface_top)):
      raise RuntimeError("guided escape plate footprint overlaps no robot geometry")
    target_pos[:, 2] = surface_top + plate_half_extents[2] + surface_gap
  origins = env.scene.env_origins[env_ids]
  inactive_pos = origins.clone()
  inactive_pos[:, 0] += inactive_xy[0]
  inactive_pos[:, 1] += inactive_xy[1]
  inactive_pos[:, 2] += body_origin_clearance
  anchor_pos = torch.where(active[:, None], target_pos, inactive_pos)
  if align_to_body:
    yaw = torch.atan2(forward_xy[:, 1], forward_xy[:, 0])
    anchor_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
  else:
    anchor_quat = torch.zeros(n, 4, device=env.device)
    anchor_quat[:, 0] = 1.0
  obstacle.write_mocap_pose_to_sim(
    torch.cat((anchor_pos, anchor_quat), dim=-1), env_ids=env_ids
  )
  obstacle.write_joint_state_to_sim(
    torch.zeros(n, 1, device=env.device),
    torch.zeros(n, 1, device=env.device),
    env_ids=env_ids,
  )
  _ensure_escape_state(env)
  if not hasattr(env, "_escape_plate_longitudinal_offset"):
    env._escape_plate_longitudinal_offset = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
    env._escape_plate_lateral_offset = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
  env._escape_plate_longitudinal_offset[env_ids] = (  # type: ignore[attr-defined]
    sampled_longitudinal_offset
  )
  env._escape_plate_lateral_offset[env_ids] = (  # type: ignore[attr-defined]
    sampled_lateral_offset
  )
  env._escape_phase[env_ids] = active.long()  # type: ignore[attr-defined]
  env._escape_target_slot[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_contact_ever[env_ids] = False  # type: ignore[attr-defined]
  env._escape_clear_hold[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_separation_delta[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_invalid_contact[env_ids] = False  # type: ignore[attr-defined]
  env._escape_peak_penetration[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_peak_contact_force[env_ids] = 0.0  # type: ignore[attr-defined]
  # Step events run after auto-reset but before the subsequent sim.sense().  Skip
  # one update so a terminated episode's stale sensor sample cannot invalidate
  # the freshly reset episode.
  env._escape_sensor_grace[env_ids] = 1  # type: ignore[attr-defined]
  env._escape_invalid_setup[env_ids] = False  # type: ignore[attr-defined]
  env._escape_wait_steps[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_first_contact_head_height[env_ids] = -1.0  # type: ignore[attr-defined]
  env._escape_hand_support_steps[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_hand_supported_progress[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_covered_geom_count[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_initial_covered_geom_count[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_best_covered_geom_count[env_ids] = 0  # type: ignore[attr-defined]
  env._escape_coverage_score[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_best_coverage_score[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_coverage_delta[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_planar_clearance[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_best_planar_clearance[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_clearance_delta[env_ids] = 0.0  # type: ignore[attr-defined]
  env._escape_geometry_initialized[env_ids] = False  # type: ignore[attr-defined]
  robot_xy = robot.data.root_link_pos_w[env_ids, :2]
  obstacle_xy = anchor_pos[:, :2]
  separation = torch.linalg.vector_norm(robot_xy - obstacle_xy, dim=-1)
  env._escape_best_separation[env_ids] = separation  # type: ignore[attr-defined]
  env._escape_start_robot_xy[env_ids] = robot_xy  # type: ignore[attr-defined]
  env._escape_start_obstacle_xy[env_ids] = obstacle_xy  # type: ignore[attr-defined]


@requires_model_fields(
  "body_mass",
  "body_inertia",
  recompute=RecomputeLevel.set_const,
)
@torch.no_grad()
def reset_guided_escape_plate_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  plate_mass_range: tuple[float, float] = (4.0, 12.0),
  initial_max_mass: float = 6.0,
  mass_curriculum_steps: int = 8_000_000,
  **plate_reset_kwargs,
) -> None:
  """Reset the plate with physically consistent light-to-heavy mass scaling."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  if not (0.0 < plate_mass_range[0] <= initial_max_mass <= plate_mass_range[1]):
    raise ValueError("plate mass curriculum must satisfy 0 < min <= initial <= max")
  obstacle = env.scene["escape_obstacle"]
  local_body_ids, _ = obstacle.find_bodies(["escape_plate"], preserve_order=True)
  if len(local_body_ids) != 1:
    raise ValueError("escape_plate body must resolve exactly once")
  local_body = torch.tensor(local_body_ids, dtype=torch.long, device=env.device)
  body_id = obstacle.indexing.body_ids[local_body][0].long()
  default_mass = env.sim.get_default_field("body_mass")[body_id]
  default_inertia = env.sim.get_default_field("body_inertia")[body_id]
  progress = min(float(env.common_step_counter) / max(mass_curriculum_steps, 1), 1.0)
  current_max = initial_max_mass + progress * (plate_mass_range[1] - initial_max_mass)
  sampled_mass = torch.empty(env_ids.numel(), device=env.device).uniform_(
    plate_mass_range[0], current_max
  )
  mass_scale = sampled_mass / torch.clamp(default_mass, min=1e-6)
  env.sim.model.body_mass[env_ids, body_id] = sampled_mass
  env.sim.model.body_inertia[env_ids, body_id] = default_inertia * mass_scale[:, None]
  reset_guided_escape_plate(
    env,
    env_ids=env_ids,
    **plate_reset_kwargs,
  )


@requires_model_fields(
  "body_mass",
  "body_inertia",
  "geom_friction",
  recompute=RecomputeLevel.set_const,
)
@torch.no_grad()
def reset_stratified_guided_escape_plate(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  plate_masses: tuple[float, ...] = (4.0, 8.0, 12.0),
  mass_weights: tuple[float, ...] = (0.25, 0.50, 0.25),
  friction_range: tuple[float, float] = (0.4, 1.2),
  plate_body_name: str = "escape_plate",
  plate_geom_name: str = "escape_plate_geom",
  **plate_reset_kwargs,
) -> None:
  """Apply a categorical plate load before the contact-safe plate reset.

  Sampling is per world and episode. Only tangential friction is varied;
  torsional and rolling coefficients retain the audited plate defaults.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  masses = torch.tensor(plate_masses, dtype=torch.float, device=env.device)
  weights = torch.tensor(mass_weights, dtype=torch.float, device=env.device)
  if masses.ndim != 1 or masses.numel() == 0 or torch.any(masses <= 0.0):
    raise ValueError("plate_masses must be a non-empty positive sequence")
  if weights.shape != masses.shape or torch.any(weights < 0.0):
    raise ValueError("mass_weights must match plate_masses and be non-negative")
  if not torch.isclose(weights.sum(), weights.new_tensor(1.0)):
    raise ValueError("mass_weights must sum to one")
  if not 0.0 < friction_range[0] <= friction_range[1]:
    raise ValueError("friction_range must satisfy 0 < low <= high")

  obstacle = env.scene["escape_obstacle"]
  local_body_ids, _ = obstacle.find_bodies([plate_body_name], preserve_order=True)
  local_geom_ids, _ = obstacle.find_geoms([plate_geom_name], preserve_order=True)
  if len(local_body_ids) != 1 or len(local_geom_ids) != 1:
    raise ValueError("stratified plate body and geom must each resolve exactly once")
  local_body = torch.tensor(local_body_ids, dtype=torch.long, device=env.device)
  local_geom = torch.tensor(local_geom_ids, dtype=torch.long, device=env.device)
  body_id = obstacle.indexing.body_ids[local_body][0].long()
  geom_id = obstacle.indexing.geom_ids[local_geom][0].long()

  sampled_mass = masses[torch.multinomial(weights, env_ids.numel(), replacement=True)]
  default_mass = env.sim.get_default_field("body_mass")[body_id]
  default_inertia = env.sim.get_default_field("body_inertia")[body_id]
  scale = sampled_mass / torch.clamp(default_mass, min=1e-6)
  env.sim.model.body_mass[env_ids, body_id] = sampled_mass
  env.sim.model.body_inertia[env_ids, body_id] = default_inertia * scale[:, None]

  sampled_friction = torch.empty(env_ids.numel(), device=env.device).uniform_(
    *friction_range
  )
  env.sim.model.geom_friction[env_ids, geom_id, 0] = sampled_friction
  if not hasattr(env, "_escape_plate_friction"):
    env._escape_plate_friction = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
  env._escape_plate_friction[env_ids] = sampled_friction  # type: ignore[attr-defined]
  reset_guided_escape_plate(env, env_ids=env_ids, **plate_reset_kwargs)


@torch.no_grad()
def update_escape_phase(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  sensor_name: str = "robot_obstacle_contact",
  clear_hold_steps: int = 15,
  separation_threshold: float = 0.24,
  max_penetration: float | None = None,
  max_contact_force: float | None = None,
  max_wait_steps: int | None = None,
  max_wait_steps_by_reset_type: tuple[int, ...] | None = None,
  max_initial_contact_head_height: float | None = None,
  relative_to_env_origin: bool = False,
  hand_sensor_name: str | None = None,
  min_hand_support_steps: int = 0,
  min_hand_supported_progress: float = 0.0,
  geometry_clearance: bool = False,
  collision_geom_pattern: str = r".*_collision$",
  plate_geom_name: str = "escape_plate_geom",
  plate_half_extents: tuple[float, float, float] = (0.45, 0.32, 0.035),
  min_planar_clearance: float = 0.02,
) -> None:
  """Track first contact, separation progress, and stable physical escape."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0 or not hasattr(env, "_escape_phase"):
    return
  grace = getattr(env, "_escape_sensor_grace", None)
  if grace is not None:
    waiting = grace[env_ids] > 0
    grace[env_ids] = torch.clamp(grace[env_ids] - 1, min=0)
    env_ids = env_ids[~waiting]
    if env_ids.numel() == 0:
      return
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None:
    raise RuntimeError(f"{sensor_name} must expose the 'found' contact field")
  contact = torch.any(found[env_ids] > 0, dim=-1)
  phase = env._escape_phase  # type: ignore[attr-defined]
  contact_ever = env._escape_contact_ever  # type: ignore[attr-defined]
  clear_hold = env._escape_clear_hold  # type: ignore[attr-defined]
  best = env._escape_best_separation  # type: ignore[attr-defined]
  delta = env._escape_separation_delta  # type: ignore[attr-defined]
  wait_steps = env._escape_wait_steps  # type: ignore[attr-defined]
  first_contact_height = env._escape_first_contact_head_height  # type: ignore[attr-defined]
  invalid_setup = env._escape_invalid_setup  # type: ignore[attr-defined]
  hand_support_steps = env._escape_hand_support_steps  # type: ignore[attr-defined]
  hand_supported_progress = env._escape_hand_supported_progress  # type: ignore[attr-defined]

  # Track contact quality before updating task phase.  V3 rejects episodes that
  # violate conservative solver limits instead of learning from interpenetration.
  penetration = torch.zeros(env_ids.numel(), device=env.device)
  if sensor.data.dist is not None:
    valid = found[env_ids] > 0
    penetration = torch.where(
      valid, torch.clamp(-sensor.data.dist[env_ids], min=0.0), 0.0
    ).amax(dim=-1)
    if sensor.data.dist_history is not None:
      penetration = torch.maximum(
        penetration,
        torch.clamp(-sensor.data.dist_history[env_ids], min=0.0).amax(dim=(-1, -2)),
      )
  contact_force = torch.zeros_like(penetration)
  if sensor.data.force is not None:
    contact_force = torch.linalg.vector_norm(sensor.data.force[env_ids], dim=-1).amax(
      dim=-1
    )
    if sensor.data.force_history is not None:
      contact_force = torch.maximum(
        contact_force,
        torch.linalg.vector_norm(sensor.data.force_history[env_ids], dim=-1).amax(
          dim=(-1, -2)
        ),
      )
  if hasattr(env, "_escape_peak_penetration"):
    peak_penetration = env._escape_peak_penetration  # type: ignore[attr-defined]
    peak_force = env._escape_peak_contact_force  # type: ignore[attr-defined]
    peak_penetration[env_ids] = torch.maximum(peak_penetration[env_ids], penetration)
    peak_force[env_ids] = torch.maximum(peak_force[env_ids], contact_force)
    invalid = torch.zeros_like(contact)
    if max_penetration is not None:
      invalid |= penetration > max_penetration
    if max_contact_force is not None:
      invalid |= contact_force > max_contact_force
    invalid &= phase[env_ids] > 0
    env._escape_invalid_contact[env_ids] |= invalid  # type: ignore[attr-defined]
    phase[env_ids[invalid]] = 4

  active = (phase[env_ids] > 0) & (phase[env_ids] < 4)
  waiting_for_contact = (phase[env_ids] == 1) & active
  wait_steps[env_ids] = torch.where(
    waiting_for_contact, wait_steps[env_ids] + 1, wait_steps[env_ids]
  )
  first_contact = waiting_for_contact & contact
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_height = robot.data.site_pos_w[env_ids, head_idx, 2]
  if relative_to_env_origin:
    support_height = getattr(env, "_terrain_reset_support_height", None)
    if support_height is None:
      support_height = env.scene.env_origins[:, 2]
    head_height = head_height - support_height[env_ids]
  first_contact_height[env_ids[first_contact]] = head_height[first_contact]
  late_contact = torch.zeros_like(contact)
  if max_initial_contact_head_height is not None:
    late_contact = first_contact & (head_height > max_initial_contact_head_height)
  setup_timeout = torch.zeros_like(contact)
  if max_wait_steps_by_reset_type is not None:
    if not max_wait_steps_by_reset_type or any(
      limit <= 0 for limit in max_wait_steps_by_reset_type
    ):
      raise ValueError("per-reset contact wait limits must be positive")
    reset_type = getattr(env, "_robust_reset_type", None)
    if reset_type is None:
      raise RuntimeError("per-reset contact wait limits require reset types")
    local_reset_type = reset_type[env_ids]
    waiting_types = local_reset_type[waiting_for_contact]
    if waiting_types.numel() > 0 and (
      torch.any(waiting_types < 1)
      or torch.any(waiting_types > len(max_wait_steps_by_reset_type))
    ):
      raise ValueError("active plate reset type has no contact wait limit")
    limits_key = tuple(max_wait_steps_by_reset_type)
    limits = getattr(env, "_escape_wait_limits_by_reset_type", None)
    if limits is None or getattr(env, "_escape_wait_limits_key", None) != limits_key:
      limits = torch.tensor(limits_key, device=env.device)
      env._escape_wait_limits_by_reset_type = limits  # type: ignore[attr-defined]
      env._escape_wait_limits_key = limits_key  # type: ignore[attr-defined]
    local_index = torch.clamp(
      local_reset_type - 1,
      min=0,
      max=len(max_wait_steps_by_reset_type) - 1,
    )
    local_limit = limits[local_index]
    setup_timeout = waiting_for_contact & (wait_steps[env_ids] > local_limit)
  elif max_wait_steps is not None:
    setup_timeout = waiting_for_contact & (wait_steps[env_ids] > max_wait_steps)
  setup_invalid = late_contact | setup_timeout
  invalid_setup[env_ids] |= setup_invalid
  phase[env_ids[setup_invalid]] = 4

  active = (phase[env_ids] > 0) & (phase[env_ids] < 4)
  contact_ever[env_ids] |= contact & active
  newly_contacted = env_ids[(phase[env_ids] == 1) & contact]
  phase[newly_contacted] = 2

  robot_xy = env.scene["robot"].data.root_link_pos_w[env_ids, :2]
  obstacle_xy = env.scene["escape_obstacle"].data.root_link_pos_w[env_ids, :2]
  separation = torch.linalg.vector_norm(robot_xy - obstacle_xy, dim=-1)
  previous_best = best[env_ids].clone()
  delta[env_ids] = torch.clamp(separation - previous_best, min=0.0)
  best[env_ids] = torch.maximum(previous_best, separation)

  geometry_ready = torch.ones_like(contact)
  if geometry_clearance:
    covered_count, coverage_score, planar_clearance = _guided_plate_planar_clearance(
      env,
      env_ids,
      collision_geom_pattern,
      plate_geom_name,
      plate_half_extents,
    )
    initialized = env._escape_geometry_initialized  # type: ignore[attr-defined]
    initial_count = env._escape_initial_covered_geom_count  # type: ignore[attr-defined]
    current_count = env._escape_covered_geom_count  # type: ignore[attr-defined]
    best_count = env._escape_best_covered_geom_count  # type: ignore[attr-defined]
    current_score = env._escape_coverage_score  # type: ignore[attr-defined]
    best_score = env._escape_best_coverage_score  # type: ignore[attr-defined]
    coverage_delta = env._escape_coverage_delta  # type: ignore[attr-defined]
    current_clearance = env._escape_planar_clearance  # type: ignore[attr-defined]
    best_clearance = env._escape_best_planar_clearance  # type: ignore[attr-defined]
    clearance_delta = env._escape_clearance_delta  # type: ignore[attr-defined]

    active_geometry = (phase[env_ids] > 0) & (phase[env_ids] < 4)
    first_geometry = (~initialized[env_ids]) & active_geometry
    initial_count[env_ids[first_geometry]] = covered_count[first_geometry]
    best_count[env_ids[first_geometry]] = covered_count[first_geometry]
    best_score[env_ids[first_geometry]] = coverage_score[first_geometry]
    best_clearance[env_ids[first_geometry]] = planar_clearance[first_geometry]
    initialized[env_ids[first_geometry]] = True

    previous_best_score = best_score[env_ids].clone()
    previous_best_clearance = best_clearance[env_ids].clone()
    current_count[env_ids] = covered_count
    current_score[env_ids] = coverage_score
    current_clearance[env_ids] = planar_clearance
    coverage_delta[env_ids] = torch.where(
      initialized[env_ids],
      torch.clamp(previous_best_score - coverage_score, min=0.0),
      torch.zeros_like(coverage_score),
    )
    clearance_delta[env_ids] = torch.where(
      initialized[env_ids],
      torch.clamp(planar_clearance - previous_best_clearance, min=0.0),
      torch.zeros_like(planar_clearance),
    )
    best_count[env_ids] = torch.minimum(best_count[env_ids], covered_count)
    best_score[env_ids] = torch.minimum(previous_best_score, coverage_score)
    best_clearance[env_ids] = torch.maximum(previous_best_clearance, planar_clearance)
    geometry_ready = (covered_count == 0) & (planar_clearance >= min_planar_clearance)

  constrained = phase[env_ids] == 2
  if hand_sensor_name is not None:
    hand_found = env.scene[hand_sensor_name].data.found
    if hand_found is None:
      raise RuntimeError(f"{hand_sensor_name} must expose the 'found' field")
    hand_support = torch.any(hand_found[env_ids] > 0, dim=-1) & constrained
    hand_support_steps[env_ids] += hand_support.long()
    hand_supported_progress[env_ids] += torch.where(
      hand_support, delta[env_ids], torch.zeros_like(delta[env_ids])
    )
  support_valid = (hand_support_steps[env_ids] >= min_hand_support_steps) & (
    hand_supported_progress[env_ids] >= min_hand_supported_progress
  )
  separation_ready = separation >= separation_threshold
  if geometry_clearance:
    separation_ready = geometry_ready
  clear = constrained & (~contact) & separation_ready & support_valid
  already_escaped = phase[env_ids] == 3
  updated_hold = torch.where(clear, clear_hold[env_ids] + 1, 0)
  # Keep the achieved hold count after success so evaluation can distinguish a
  # genuine 15-step clearance from a transient final-frame geometry state.
  clear_hold[env_ids] = torch.where(already_escaped, clear_hold[env_ids], updated_hold)
  escaped_ids = env_ids[clear_hold[env_ids] >= clear_hold_steps]
  phase[escaped_ids] = 3


def _ensure_sustained_constraint_state(
  env: ManagerBasedRlEnv,
  body_names: tuple[str, ...],
  cohort_weights: tuple[float, float, float],
) -> None:
  """Allocate the per-environment state used by the constrained task.

  Cohorts are clean, trunk-constrained, and limb-constrained.  Keeping the
  assignment explicit makes it possible to report a recovery envelope instead
  of averaging unrelated cases into one success number.
  """
  if hasattr(env, "_constraint_body_ids"):
    return
  if len(body_names) < 3:
    raise ValueError("body_names must include two trunk bodies and at least one limb")
  weights = torch.tensor(cohort_weights, dtype=torch.float, device=env.device)
  if weights.shape != (3,) or torch.any(weights < 0.0) or weights.sum() <= 0.0:
    raise ValueError("cohort_weights must be three non-negative values")
  robot = env.scene["robot"]
  body_ids = robot.find_bodies(list(body_names), preserve_order=True)[0]
  if len(body_ids) != len(body_names):
    raise ValueError("all sustained-constraint body names must resolve exactly once")

  env._constraint_body_ids = body_ids  # type: ignore[attr-defined]
  env._constraint_cohort_weights = weights / weights.sum()  # type: ignore[attr-defined]
  env._constraint_cohort = torch.zeros(  # type: ignore[attr-defined]
    env.num_envs, dtype=torch.long, device=env.device
  )
  env._constraint_body_slot = torch.zeros_like(  # type: ignore[attr-defined]
    env._constraint_cohort  # type: ignore[attr-defined]
  )
  env._constraint_wait = torch.zeros_like(  # type: ignore[attr-defined]
    env._constraint_cohort  # type: ignore[attr-defined]
  )
  env._constraint_remaining = torch.zeros_like(  # type: ignore[attr-defined]
    env._constraint_cohort  # type: ignore[attr-defined]
  )
  env._constraint_duration = torch.ones_like(  # type: ignore[attr-defined]
    env._constraint_cohort  # type: ignore[attr-defined]
  )
  env._constraint_force_magnitude = torch.zeros(  # type: ignore[attr-defined]
    env.num_envs, device=env.device
  )
  shape = (env.num_envs, len(body_ids), 3)
  env._constraint_forces = torch.zeros(shape, device=env.device)  # type: ignore[attr-defined]
  env._constraint_torques = torch.zeros(shape, device=env.device)  # type: ignore[attr-defined]


@torch.no_grad()
def reset_sustained_constraint(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  body_names: tuple[str, ...] = (
    "pelvis",
    "torso_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_knee_link",
    "right_knee_link",
  ),
  cohort_weights: tuple[float, float, float] = (0.25, 0.50, 0.25),
  delay_steps: tuple[int, int] = (0, 15),
  duration_steps: tuple[int, int] = (100, 350),
  force_range: tuple[float, float] = (20.0, 120.0),
  torque_range: tuple[float, float] = (0.0, 8.0),
  lateral_force_fraction: float = 0.20,
  curriculum_steps: int = 400_000,
) -> None:
  """Sample a persistent, downward-biased body constraint for each episode.

  This is intentionally a force-based first benchmark, not a claim that a
  wrench is equivalent to rigid-object pinning.  It provides a cheap,
  vectorized curriculum before movable and fixed obstacle contacts are added.
  The policy does not observe the sampled cohort, body, force, or duration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  _ensure_sustained_constraint_state(env, body_names, cohort_weights)

  cohort = env._constraint_cohort  # type: ignore[attr-defined]
  body_slot = env._constraint_body_slot  # type: ignore[attr-defined]
  wait = env._constraint_wait  # type: ignore[attr-defined]
  remaining = env._constraint_remaining  # type: ignore[attr-defined]
  duration = env._constraint_duration  # type: ignore[attr-defined]
  magnitude = env._constraint_force_magnitude  # type: ignore[attr-defined]
  forces = env._constraint_forces  # type: ignore[attr-defined]
  torques = env._constraint_torques  # type: ignore[attr-defined]

  cohort[env_ids] = torch.multinomial(
    env._constraint_cohort_weights,  # type: ignore[attr-defined]
    env_ids.numel(),
    replacement=True,
  )
  trunk = cohort[env_ids] == 1
  limb = cohort[env_ids] == 2
  slots = torch.zeros(env_ids.numel(), dtype=torch.long, device=env.device)
  if trunk.any():
    slots[trunk] = torch.randint(0, 2, (int(trunk.sum()),), device=env.device)
  if limb.any():
    slots[limb] = torch.randint(
      2, len(body_names), (int(limb.sum()),), device=env.device
    )
  body_slot[env_ids] = slots

  wait[env_ids] = torch.randint(
    delay_steps[0], delay_steps[1] + 1, (env_ids.numel(),), device=env.device
  )
  sampled_duration = torch.randint(
    duration_steps[0], duration_steps[1] + 1, (env_ids.numel(),), device=env.device
  )
  sampled_duration[cohort[env_ids] == 0] = 0
  duration[env_ids] = torch.clamp(sampled_duration, min=1)
  remaining[env_ids] = sampled_duration

  progress = min(float(env.common_step_counter) / max(curriculum_steps, 1), 1.0)
  max_force = force_range[0] + progress * (force_range[1] - force_range[0])
  sampled_force = torch.empty(env_ids.numel(), device=env.device).uniform_(
    0.60 * max_force, max_force
  )
  sampled_force[cohort[env_ids] == 0] = 0.0
  magnitude[env_ids] = sampled_force

  angle = torch.empty(env_ids.numel(), device=env.device).uniform_(-torch.pi, torch.pi)
  lateral = lateral_force_fraction * sampled_force
  force_vec = torch.stack(
    (lateral * torch.cos(angle), lateral * torch.sin(angle), -sampled_force),
    dim=-1,
  )
  max_torque = torque_range[0] + progress * (torque_range[1] - torque_range[0])
  torque_vec = torch.empty_like(force_vec).uniform_(-max_torque, max_torque)
  torque_vec[:, 2] *= 0.35
  forces[env_ids] = 0.0
  torques[env_ids] = 0.0
  forces[env_ids, slots] = force_vec
  torques[env_ids, slots] = torque_vec


@torch.no_grad()
def apply_sustained_constraint(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
) -> None:
  """Apply and eventually release the constraint sampled at episode reset."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0 or not hasattr(env, "_constraint_body_ids"):
    return
  wait = env._constraint_wait  # type: ignore[attr-defined]
  remaining = env._constraint_remaining  # type: ignore[attr-defined]
  forces = env._constraint_forces  # type: ignore[attr-defined]
  torques = env._constraint_torques  # type: ignore[attr-defined]

  waiting = wait[env_ids] > 0
  wait[env_ids[waiting]] -= 1
  active = (wait[env_ids] <= 0) & (remaining[env_ids] > 0)
  step_forces = torch.zeros_like(forces[env_ids])
  step_torques = torch.zeros_like(torques[env_ids])
  active_ids = env_ids[active]
  if active_ids.numel() > 0:
    step_forces[active] = forces[active_ids]
    step_torques[active] = torques[active_ids]
    remaining[active_ids] -= 1
  env.scene["robot"].write_external_wrench_to_sim(
    step_forces,
    step_torques,
    env_ids=env_ids,
    body_ids=env._constraint_body_ids,  # type: ignore[attr-defined]
  )


@torch.no_grad()
def reset_stand_counter(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Zero the ``stood_up`` standing-hold counter for the reset envs (no-op until
  ``stood_up`` lazily creates it).  Separate from ``gsi_reset`` so it stays reusable."""
  if not hasattr(env, "_getup_stand_count"):
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env._getup_stand_count[env_ids] = 0  # type: ignore[attr-defined]


@torch.no_grad()
def mixed_fall_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  procedural_probability: float = 0.5,
  root_height_range: tuple[float, float] = (0.48, 0.62),
  joint_noise: float = 0.12,
  orientation_noise: float = 0.0,
  root_xy_range: float = 0.1,
  root_linear_velocity: float = 0.1,
  root_angular_velocity: float = 0.2,
  mode_weights: tuple[float, float, float, float] | None = None,
) -> None:
  """Mix GSI resets with four physically plausible procedural lying poses.

  gsi_reset runs first for every reset. This event replaces a configurable
  subset with prone, supine, left-side, or right-side poses, then re-primes the
  SMP history with the actual simulator state so no stale GSI trajectory leaks
  into the reward.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return

  if mode_weights is not None:
    weights = torch.tensor(mode_weights, dtype=torch.float, device=env.device)
    if weights.shape != (4,) or torch.any(weights < 0.0) or weights.sum() <= 0.0:
      msg = "mode_weights must contain four non-negative values with positive sum"
      raise ValueError(msg)
    weights /= weights.sum()

  reset_types = getattr(env, "_robust_reset_type", None)
  if reset_types is None:
    reset_types = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._robust_reset_type = reset_types  # type: ignore[attr-defined]
  reset_types[env_ids] = 0

  # Clear any active wrench and reschedule the next push for reset environments.
  if hasattr(env, "_robust_push_active"):
    env._robust_push_active[env_ids] = 0  # type: ignore[attr-defined]
    env._robust_push_recovery[env_ids] = 0  # type: ignore[attr-defined]
    env._robust_push_wait[env_ids] = torch.randint(  # type: ignore[attr-defined]
      50, 151, (env_ids.numel(),), device=env.device
    )
    env._robust_forces[env_ids] = 0.0  # type: ignore[attr-defined]
    env._robust_torques[env_ids] = 0.0  # type: ignore[attr-defined]

  # V5 applies at most one targeted knockdown after a stable stand. Reset its
  # scheduler and any residual wrench when a new episode begins.
  if hasattr(env, "_v5_knockdown_active"):
    env._v5_knockdown_active[env_ids] = 0  # type: ignore[attr-defined]
    env._v5_knockdown_recovery[env_ids] = 0  # type: ignore[attr-defined]
    env._v5_knockdown_wait[env_ids] = -1  # type: ignore[attr-defined]
    env._v5_knockdown_done[env_ids] = False  # type: ignore[attr-defined]
    env._v5_knockdown_forces[env_ids] = 0.0  # type: ignore[attr-defined]
    env._v5_knockdown_torques[env_ids] = 0.0  # type: ignore[attr-defined]

  # V6 assigns clean, standard-push, and intensive-push cohorts per episode.
  if hasattr(env, "_v6_push_active"):
    env._v6_push_active[env_ids] = 0  # type: ignore[attr-defined]
    env._v6_push_recovery[env_ids] = 0  # type: ignore[attr-defined]
    env._v6_push_wait[env_ids] = -1  # type: ignore[attr-defined]
    env._v6_push_count[env_ids] = 0  # type: ignore[attr-defined]
    env._v6_push_forces[env_ids] = 0.0  # type: ignore[attr-defined]
    env._v6_push_torques[env_ids] = 0.0  # type: ignore[attr-defined]
    cohort_weights = env._v6_push_cohort_weights  # type: ignore[attr-defined]
    env._v6_push_cohort[env_ids] = torch.multinomial(  # type: ignore[attr-defined]
      cohort_weights, env_ids.numel(), replacement=True
    )

  # Per-environment stagnation trackers reset, while the global replay ring
  # deliberately persists across episodes.
  if hasattr(env, "_v6_failure_stagnant"):
    env._v6_failure_stagnant[env_ids] = 0  # type: ignore[attr-defined]
    env._v6_failure_best[env_ids] = 0.0  # type: ignore[attr-defined]
    env._v6_failure_prev_stage[env_ids] = 0  # type: ignore[attr-defined]

  choose = torch.rand(env_ids.numel(), device=env.device) < procedural_probability
  fall_ids = env_ids[choose]
  if fall_ids.numel() == 0:
    return

  robot = env.scene["robot"]
  n = fall_ids.numel()
  if mode_weights is None:
    modes = torch.randint(0, 4, (n,), device=env.device)
  else:
    modes = torch.multinomial(weights, n, replacement=True)
  roll = torch.zeros(n, device=env.device)
  pitch = torch.zeros(n, device=env.device)
  roll = torch.where(modes == 2, torch.full_like(roll, torch.pi / 2), roll)
  roll = torch.where(modes == 3, torch.full_like(roll, -torch.pi / 2), roll)
  # With G1's root convention, +pi/2 places the torso forward/chest axis down
  # (prone), while -pi/2 points it up (supine).
  pitch = torch.where(modes == 0, torch.full_like(pitch, torch.pi / 2), pitch)
  pitch = torch.where(modes == 1, torch.full_like(pitch, -torch.pi / 2), pitch)
  if orientation_noise > 0.0:
    # Noise on both axes creates continuous oblique front/back/side contacts.
    roll += torch.empty_like(roll).uniform_(-orientation_noise, orientation_noise)
    pitch += torch.empty_like(pitch).uniform_(-orientation_noise, orientation_noise)
  yaw = torch.empty(n, device=env.device).uniform_(-torch.pi, torch.pi)

  default_root = robot.data.default_root_state[fall_ids].clone()
  delta_quat = quat_from_euler_xyz(roll, pitch, yaw)
  quat = quat_mul(default_root[:, 3:7], delta_quat)
  origins = env.scene.env_origins[fall_ids]
  xy = torch.empty(n, 2, device=env.device).uniform_(-root_xy_range, root_xy_range)
  height = torch.empty(n, 1, device=env.device).uniform_(*root_height_range)
  pos = torch.cat([xy, height], dim=-1) + origins
  lin_vel = torch.empty(n, 3, device=env.device).uniform_(
    -root_linear_velocity, root_linear_velocity
  )
  ang_vel = torch.empty(n, 3, device=env.device).uniform_(
    -root_angular_velocity, root_angular_velocity
  )
  robot.write_root_state_to_sim(
    torch.cat([pos, quat, lin_vel, ang_vel], dim=-1), env_ids=fall_ids
  )

  joint_pos = robot.data.default_joint_pos[fall_ids].clone()
  joint_pos += torch.empty_like(joint_pos).uniform_(-joint_noise, joint_noise)
  joint_vel = torch.empty_like(joint_pos).uniform_(-0.1, 0.1)
  robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=fall_ids)
  env.sim.forward()

  # A repeated static window is honest history for a newly placed pose and avoids
  # an artificial GSI-to-procedural discontinuity in the SMP score.
  _prime_smp_history_from_current_state(env, fall_ids)
  reset_types[fall_ids] = modes + 1


@torch.no_grad()
def sample_weighted_terrain_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  level_weights: tuple[float, ...] = (0.55, 0.30, 0.15, 0.0),
  flat_level: int = 0,
  flat_name: str = "flat",
) -> None:
  """Move worlds to frozen terrain-level strata before robot reset.

  Terrain family allocation remains the generator's proportional, fixed
  column assignment. Non-flat levels are sampled from preregistered weights on
  every episode; flat replay remains level zero. The resulting labels are
  simulator-only and never enter actor observations.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  terrain = env.scene["terrain"]
  origins = terrain.terrain_origins
  generator = terrain.cfg.terrain_generator
  if origins is None or generator is None:
    raise RuntimeError("weighted terrain levels require generated terrain")
  weights = torch.tensor(level_weights, dtype=torch.float, device=env.device)
  if weights.shape != (origins.shape[0],) or torch.any(weights < 0.0):
    raise ValueError("level_weights must match terrain rows and be non-negative")
  if not torch.isclose(weights.sum(), weights.new_tensor(1.0)):
    raise ValueError("level_weights must sum to one")
  if not 0 <= flat_level < origins.shape[0]:
    raise ValueError("flat_level must index a generated terrain row")

  sampled = torch.multinomial(weights, env_ids.numel(), replacement=True)
  names = list(generator.sub_terrains)
  if flat_name in names:
    flat_col = names.index(flat_name)
    sampled = torch.where(
      terrain.terrain_types[env_ids] == flat_col,
      torch.full_like(sampled, flat_level),
      sampled,
    )
  terrain.terrain_levels[env_ids] = sampled
  terrain.env_origins[env_ids] = origins[sampled, terrain.terrain_types[env_ids]]


EDGE_RESET_COHORTS = ("center", "near_edge", "straddle", "lower_tread")


def _stair_surface_height(
  radial_offset: torch.Tensor,
  origin_height: torch.Tensor,
  step_height: torch.Tensor,
  *,
  terrain_size: float,
  border_width: float,
  platform_width: float,
  step_width: float,
) -> torch.Tensor:
  """Return the pyramid-stair surface height at a Chebyshev radius."""
  inner_width = terrain_size - 2.0 * border_width
  num_steps = max(
    0,
    int((inner_width - platform_width) / (2.0 * step_width)),
  )
  top_width = inner_width - 2.0 * num_steps * step_width
  top_half = 0.5 * top_width
  ring = torch.ceil(
    torch.clamp((radial_offset - top_half) / step_width, min=0.0)
  ).long()
  ring = torch.clamp(ring, max=num_steps + 1)
  return origin_height - ring * step_height


@torch.no_grad()
def sample_terrain_edge_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  cohort_weights: tuple[float, float, float, float] = (0.50, 0.25, 0.15, 0.10),
  near_edge_range: tuple[float, float] = (0.18, 0.24),
  straddle_range: tuple[float, float] = (0.27, 0.34),
  lower_tread_range: tuple[float, float] = (0.38, 0.52),
  tangent_range: float = 0.12,
  stair_step_heights: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
  stair_name: str = "stairs",
  terrain_size: float = 8.0,
  stair_border_width: float = 1.90,
  stair_platform_width: float = 0.55,
  stair_step_width: float = 0.30,
) -> None:
  """Stratify stair resets around the top edge without privileged observations.

  The selected cohort and local support height are training/evaluation state.
  They never enter the actor observation. Non-stair terrain keeps its original
  centre reset and origin-height reference.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return

  weights = torch.tensor(cohort_weights, dtype=torch.float, device=env.device)
  if weights.numel() != len(EDGE_RESET_COHORTS) or torch.any(weights < 0):
    raise ValueError("cohort_weights must contain four non-negative entries")
  if not torch.isclose(weights.sum(), weights.new_tensor(1.0)):
    raise ValueError("cohort_weights must sum to one")

  terrain = env.scene["terrain"]
  generator = terrain.cfg.terrain_generator
  if generator is None:
    raise RuntimeError("edge reset requires generated terrain")
  terrain_names = list(generator.sub_terrains)
  stair_col = terrain_names.index(stair_name) if stair_name in terrain_names else -1

  if not hasattr(env, "_terrain_reset_cohort"):
    env._terrain_reset_cohort = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._terrain_reset_anchor_xy = env.scene.env_origins[:, :2].clone()  # type: ignore[attr-defined]
    env._terrain_reset_support_height = env.scene.env_origins[:, 2].clone()  # type: ignore[attr-defined]

  robot = env.scene["robot"]
  origins = env.scene.env_origins[env_ids]
  env._terrain_reset_cohort[env_ids] = 0  # type: ignore[attr-defined]
  env._terrain_reset_anchor_xy[env_ids] = robot.data.root_link_pos_w[env_ids, :2]  # type: ignore[attr-defined]
  env._terrain_reset_support_height[env_ids] = origins[:, 2]  # type: ignore[attr-defined]

  stair_mask = terrain.terrain_types[env_ids] == stair_col
  stair_ids = env_ids[stair_mask]
  if stair_ids.numel() == 0:
    return

  n = stair_ids.numel()
  cohorts = torch.multinomial(weights, n, replacement=True)
  offsets = torch.zeros(n, 2, device=env.device)
  axis = torch.randint(0, 2, (n,), device=env.device)
  sign = torch.where(
    torch.rand(n, device=env.device) < 0.5,
    -torch.ones(n, device=env.device),
    torch.ones(n, device=env.device),
  )
  tangent = torch.empty(n, device=env.device).uniform_(-tangent_range, tangent_range)

  ranges = (near_edge_range, straddle_range, lower_tread_range)
  for cohort_index, radial_range in enumerate(ranges, start=1):
    selected = cohorts == cohort_index
    count = int(selected.sum())
    if count == 0:
      continue
    radial = torch.empty(count, device=env.device).uniform_(*radial_range)
    local_axis = axis[selected]
    offsets[selected, 0] = torch.where(
      local_axis == 0, sign[selected] * radial, tangent[selected]
    )
    offsets[selected, 1] = torch.where(
      local_axis == 1, sign[selected] * radial, tangent[selected]
    )

  stair_origins = env.scene.env_origins[stair_ids]
  root_state = torch.cat(
    (
      robot.data.root_link_pose_w[stair_ids].clone(),
      torch.zeros(n, 6, device=env.device),
    ),
    dim=-1,
  )
  centre = cohorts == 0
  offsets[centre] = root_state[centre, :2] - stair_origins[centre, :2]
  root_state[:, :2] = stair_origins[:, :2] + offsets
  robot.write_root_state_to_sim(root_state, env_ids=stair_ids)
  env.sim.forward()

  levels = terrain.terrain_levels[stair_ids]
  heights = torch.tensor(stair_step_heights, device=env.device)[levels]
  radial = offsets.abs().amax(dim=-1)
  support_height = _stair_surface_height(
    radial,
    stair_origins[:, 2],
    heights,
    terrain_size=terrain_size,
    border_width=stair_border_width,
    platform_width=stair_platform_width,
    step_width=stair_step_width,
  )
  env._terrain_reset_cohort[stair_ids] = cohorts  # type: ignore[attr-defined]
  env._terrain_reset_anchor_xy[stair_ids] = root_state[:, :2]  # type: ignore[attr-defined]
  env._terrain_reset_support_height[stair_ids] = support_height  # type: ignore[attr-defined]


@torch.no_grad()
def ground_procedural_fall_on_terrain(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  eligible_reset_types: tuple[int, ...] = (1, 2, 3, 4),
  ground_clearance: float = 0.006,
  collision_geom_pattern: str = r".*_collision$",
  surface_normals: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 1.0),),
  surface_normal_levels: tuple[tuple[tuple[float, float, float], ...], ...]
  | None = None,
  use_stair_height_profile: bool = False,
  stair_name: str = "stairs",
  stair_step_heights: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
  terrain_size: float = 8.0,
  stair_border_width: float = 1.90,
  stair_platform_width: float = 0.55,
  stair_step_width: float = 0.30,
) -> None:
  """Place procedural fall resets on their terrain spawn surface.

  Terrain generators expose a collision-safe spawn origin and support normal.
  ``mixed_fall_reset`` samples pose and articulation first; this event then
  translates the complete robot so the conservative collision AABB support is
  just above that plane.  The flat/stair/rough origins are the highest support
  under the reset footprint, while a directed slope uses its exact plane normal.
  No terrain height, type, or reset label is added to actor observations.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return

  reset_type = getattr(env, "_robust_reset_type", None)
  if reset_type is None:
    raise RuntimeError("terrain grounding requires mixed_fall_reset state")
  eligible = torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device)
  for reset_value in eligible_reset_types:
    eligible |= reset_type[env_ids] == reset_value
  grounded_ids = env_ids[eligible]
  if grounded_ids.numel() == 0:
    return

  robot = env.scene["robot"]
  _, _, aabb_center, aabb_half, _ = _collision_vertical_geometry(
    env, grounded_ids, collision_geom_pattern
  )
  terrain = env.scene["terrain"]
  if surface_normal_levels is not None:
    normal_levels = torch.tensor(
      surface_normal_levels, dtype=aabb_center.dtype, device=env.device
    )
    normal_levels /= torch.clamp(
      torch.linalg.vector_norm(normal_levels, dim=-1, keepdim=True), min=1e-6
    )
    normals = normal_levels[
      terrain.terrain_types[grounded_ids], terrain.terrain_levels[grounded_ids]
    ]
  else:
    normal_options = torch.tensor(
      surface_normals, dtype=aabb_center.dtype, device=env.device
    )
    normal_options /= torch.clamp(
      torch.linalg.vector_norm(normal_options, dim=-1, keepdim=True), min=1e-6
    )
    if len(surface_normals) == 1:
      normals = normal_options.expand(grounded_ids.numel(), -1)
    else:
      terrain_types = env.scene["terrain"].terrain_types[grounded_ids]
      normals = normal_options[terrain_types]
  origins = env.scene.env_origins[grounded_ids]
  signed_center = ((aabb_center - origins[:, None, :]) * normals[:, None, :]).sum(
    dim=-1
  )
  support_extent = (aabb_half * normals[:, None, :].abs()).sum(dim=-1)
  lowest_signed_distance = (signed_center - support_extent).amin(dim=-1)
  translation = (ground_clearance - lowest_signed_distance)[:, None] * normals

  if use_stair_height_profile:
    generator = terrain.cfg.terrain_generator
    if generator is None:
      raise RuntimeError("stair-profile grounding requires generated terrain")
    names = list(generator.sub_terrains)
    stair_col = names.index(stair_name) if stair_name in names else -1
    stair_mask = terrain.terrain_types[grounded_ids] == stair_col
    local_stair_ids = torch.nonzero(stair_mask, as_tuple=False).flatten()
    if local_stair_ids.numel() > 0:
      stair_env_ids = grounded_ids[local_stair_ids]
      stair_origins = env.scene.env_origins[stair_env_ids]
      stair_centers = aabb_center[local_stair_ids]
      stair_halves = aabb_half[local_stair_ids]
      sample_directions = stair_centers.new_tensor(
        (
          (0.0, 0.0),
          (-1.0, 0.0),
          (1.0, 0.0),
          (0.0, -1.0),
          (0.0, 1.0),
          (-1.0, -1.0),
          (-1.0, 1.0),
          (1.0, -1.0),
          (1.0, 1.0),
        )
      )
      sample_xy = stair_centers[:, :, None, :2] + (
        stair_halves[:, :, None, :2] * sample_directions[None, None, :, :]
      )
      local_xy = sample_xy - stair_origins[:, None, None, :2]
      radial = local_xy.abs().amax(dim=-1)
      levels = terrain.terrain_levels[stair_env_ids]
      step_height = stair_centers.new_tensor(stair_step_heights)[levels]
      surface_z = _stair_surface_height(
        radial,
        stair_origins[:, None, None, 2],
        step_height[:, None, None],
        terrain_size=terrain_size,
        border_width=stair_border_width,
        platform_width=stair_platform_width,
        step_width=stair_step_width,
      )
      geom_bottom = stair_centers[:, :, 2] - stair_halves[:, :, 2]
      vertical_shift = (surface_z + ground_clearance - geom_bottom[:, :, None]).amax(
        dim=(1, 2)
      )
      translation[local_stair_ids] = 0.0
      translation[local_stair_ids, 2] = vertical_shift

  root_state = torch.cat(
    (
      robot.data.root_link_pose_w[grounded_ids].clone(),
      torch.zeros(grounded_ids.numel(), 6, device=env.device),
    ),
    dim=-1,
  )
  root_state[:, :3] += translation
  robot.write_root_state_to_sim(root_state, env_ids=grounded_ids)
  env.sim.forward()
  _prime_smp_history_from_current_state(env, grounded_ids)


def _head_height_and_upright(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  relative_to_env_origin: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  knee_ids = robot.find_joints(
    ["left_knee_joint", "right_knee_joint"], preserve_order=True
  )[0]
  head_z = robot.data.site_pos_w[env_ids, head_idx, 2]
  if relative_to_env_origin:
    reference = getattr(env, "_terrain_reset_support_height", None)
    if reference is None:
      reference = env.scene.env_origins[:, 2]
    head_z = head_z - reference[env_ids]
  head_vz = robot.data.site_lin_vel_w[env_ids, head_idx, 2]
  upright = torch.clamp(-robot.data.projected_gravity_b[env_ids, 2], 0.0, 1.0)
  knee_flexion = robot.data.joint_pos[env_ids][:, knee_ids].mean(dim=-1)
  return head_z, head_vz, upright, knee_flexion


@torch.no_grad()
def reset_recovery_stage(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  relative_to_env_origin: bool = False,
) -> None:
  """Initialize the ordered seated-crouched-standing recovery stage."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return

  if not hasattr(env, "_v4_recovery_stage"):
    env._v4_recovery_stage = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._v4_stage_hold = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
  if not hasattr(env, "_v4_stage_transition"):
    env._v4_stage_transition = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )

  head_z, _, upright, knee_flexion = _head_height_and_upright(
    env, env_ids, relative_to_env_origin=relative_to_env_origin
  )
  stage = torch.zeros(env_ids.numel(), dtype=torch.long, device=env.device)
  stage = torch.where(
    (head_z >= 0.55) & (upright >= 0.55) & (knee_flexion >= 1.00),
    torch.ones_like(stage),
    stage,
  )
  stage = torch.where(
    (head_z >= 0.78) & (upright >= 0.72) & (knee_flexion >= 0.80),
    torch.full_like(stage, 2),
    stage,
  )
  stage = torch.where(
    (head_z >= 1.08) & (upright >= 0.85), torch.full_like(stage, 3), stage
  )
  env._v4_recovery_stage[env_ids] = stage  # type: ignore[attr-defined]
  env._v4_stage_hold[env_ids] = 0  # type: ignore[attr-defined]
  env._v4_stage_transition[env_ids] = 0  # type: ignore[attr-defined]


@torch.no_grad()
def update_recovery_stage(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  seated_hold_steps: int = 10,
  crouched_hold_steps: int = 10,
  standing_hold_steps: int = 25,
  relative_to_env_origin: bool = False,
) -> None:
  """Advance recovery only after stable seated, crouched, and standing holds."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  if not hasattr(env, "_v4_recovery_stage"):
    reset_recovery_stage(env, env_ids, relative_to_env_origin=relative_to_env_origin)

  stage = env._v4_recovery_stage  # type: ignore[attr-defined]
  hold = env._v4_stage_hold  # type: ignore[attr-defined]
  transition = env._v4_stage_transition  # type: ignore[attr-defined]
  transition[env_ids] = 0
  head_z, head_vz, upright, knee_flexion = _head_height_and_upright(
    env, env_ids, relative_to_env_origin=relative_to_env_origin
  )
  local_stage = stage[env_ids]

  # A new substantial fall restarts the ordered recovery sequence.
  fallen = (head_z < 0.65) & (upright < 0.45)
  fallen_ids = env_ids[fallen & (local_stage > 0)]
  stage[fallen_ids] = 0
  hold[fallen_ids] = 0
  local_stage = stage[env_ids]

  seated = (
    (local_stage == 0)
    & (head_z >= 0.55)
    & (upright >= 0.55)
    & (knee_flexion >= 1.00)
    & (torch.abs(head_vz) <= 0.16)
  )
  crouched = (
    (local_stage == 1)
    & (head_z >= 0.78)
    & (upright >= 0.72)
    & (knee_flexion >= 0.80)
    & (torch.abs(head_vz) <= 0.18)
  )
  standing = (
    (local_stage == 2)
    & (head_z >= 1.08)
    & (upright >= 0.85)
    & (torch.abs(head_vz) <= 0.12)
  )
  satisfied = seated | crouched | standing
  hold[env_ids] = torch.where(satisfied, hold[env_ids] + 1, 0)

  advance_seated = env_ids[seated & (hold[env_ids] >= seated_hold_steps)]
  advance_crouched = env_ids[crouched & (hold[env_ids] >= crouched_hold_steps)]
  advance_standing = env_ids[standing & (hold[env_ids] >= standing_hold_steps)]
  stage[advance_seated] = 1
  stage[advance_crouched] = 2
  stage[advance_standing] = 3
  transition[advance_seated] = 1
  transition[advance_crouched] = 2
  transition[advance_standing] = 3
  advanced = torch.cat([advance_seated, advance_crouched, advance_standing])
  hold[advanced] = 0


@torch.no_grad()
def random_body_wrench(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  body_names: tuple[str, ...] = ("pelvis", "torso_link"),
  interval_steps: tuple[int, int] = (50, 150),
  duration_steps: tuple[int, int] = (5, 15),
  recovery_steps: int = 40,
  force_range: tuple[float, float] = (40.0, 250.0),
  torque_range: tuple[float, float] = (4.0, 30.0),
  curriculum_steps: int = 150_000,
) -> None:
  """Apply finite-duration world-frame wrenches to a random torso body.

  Force and torque amplitudes ramp linearly with the global control-step count.
  Unlike push_by_setting_velocity, this produces contact-mediated physical
  perturbations and explicitly clears the wrench after 0.1--0.3 seconds.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  robot = env.scene["robot"]

  if not hasattr(env, "_robust_force_body_ids"):
    body_ids = robot.find_bodies(list(body_names), preserve_order=True)[0]
    env._robust_force_body_ids = body_ids  # type: ignore[attr-defined]
    num_bodies = len(body_ids)
    env._robust_push_wait = torch.randint(  # type: ignore[attr-defined]
      interval_steps[0],
      interval_steps[1] + 1,
      (env.num_envs,),
      device=env.device,
    )
    env._robust_push_active = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._robust_push_recovery = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._robust_forces = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, num_bodies, 3, device=env.device
    )
    env._robust_torques = torch.zeros_like(  # type: ignore[attr-defined]
      env._robust_forces  # type: ignore[attr-defined]
    )

  wait = env._robust_push_wait  # type: ignore[attr-defined]
  active = env._robust_push_active  # type: ignore[attr-defined]
  recovery = env._robust_push_recovery  # type: ignore[attr-defined]
  forces = env._robust_forces  # type: ignore[attr-defined]
  torques = env._robust_torques  # type: ignore[attr-defined]
  body_ids = env._robust_force_body_ids  # type: ignore[attr-defined]

  recovering = recovery[env_ids] > 0
  recovery[env_ids[recovering]] -= 1
  inactive = active[env_ids] <= 0
  wait[env_ids[inactive]] -= 1
  start_ids = env_ids[(wait[env_ids] <= 0) & (active[env_ids] <= 0)]
  if start_ids.numel() > 0:
    progress = min(float(env.common_step_counter) / max(curriculum_steps, 1), 1.0)
    force_amp = force_range[0] + progress * (force_range[1] - force_range[0])
    torque_amp = torque_range[0] + progress * (torque_range[1] - torque_range[0])
    chosen = torch.randint(0, len(body_ids), (start_ids.numel(),), device=env.device)
    force_vec = torch.empty(start_ids.numel(), 3, device=env.device).uniform_(
      -force_amp, force_amp
    )
    force_vec[:, 2] *= 0.35
    torque_vec = torch.empty_like(force_vec).uniform_(-torque_amp, torque_amp)
    forces[start_ids] = 0.0
    torques[start_ids] = 0.0
    forces[start_ids, chosen] = force_vec
    torques[start_ids, chosen] = torque_vec
    durations = torch.randint(
      duration_steps[0],
      duration_steps[1] + 1,
      (start_ids.numel(),),
      device=env.device,
    )
    active[start_ids] = durations
    recovery[start_ids] = durations + recovery_steps

  active_ids = env_ids[active[env_ids] > 0]
  inactive_ids = env_ids[active[env_ids] <= 0]
  forces[inactive_ids] = 0.0
  torques[inactive_ids] = 0.0
  robot.write_external_wrench_to_sim(
    forces[env_ids], torques[env_ids], env_ids=env_ids, body_ids=body_ids
  )

  if active_ids.numel() > 0:
    active[active_ids] -= 1
    finished = active_ids[active[active_ids] == 0]
    if finished.numel() > 0:
      wait[finished] = torch.randint(
        interval_steps[0],
        interval_steps[1] + 1,
        (finished.numel(),),
        device=env.device,
      )


@torch.no_grad()
def post_stand_body_wrench(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  body_names: tuple[str, ...] = ("pelvis", "torso_link"),
  delay_steps: tuple[int, int] = (20, 60),
  duration_steps: tuple[int, int] = (10, 18),
  recovery_steps: int = 100,
  force_range: tuple[float, float] = (90.0, 190.0),
  torque_range: tuple[float, float] = (4.0, 14.0),
  curriculum_steps: int = 250_000,
) -> None:
  """Knock down each robot once after it reaches a stable standing stage.

  The event deliberately creates a second recovery within the same episode.
  This covers the state distribution produced by a real fall instead of only
  teleporting the robot into a static reset pose. Playback stays clean unless
  the existing auto-disturbances option explicitly enables this event.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  robot = env.scene["robot"]

  if not hasattr(env, "_v5_knockdown_body_ids"):
    body_ids = robot.find_bodies(list(body_names), preserve_order=True)[0]
    env._v5_knockdown_body_ids = body_ids  # type: ignore[attr-defined]
    num_bodies = len(body_ids)
    env._v5_knockdown_wait = torch.full(  # type: ignore[attr-defined]
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    env._v5_knockdown_active = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._v5_knockdown_recovery = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._v5_knockdown_done = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.bool, device=env.device
    )
    env._v5_knockdown_forces = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, num_bodies, 3, device=env.device
    )
    env._v5_knockdown_torques = torch.zeros_like(  # type: ignore[attr-defined]
      env._v5_knockdown_forces  # type: ignore[attr-defined]
    )

  wait = env._v5_knockdown_wait  # type: ignore[attr-defined]
  active = env._v5_knockdown_active  # type: ignore[attr-defined]
  recovery = env._v5_knockdown_recovery  # type: ignore[attr-defined]
  done = env._v5_knockdown_done  # type: ignore[attr-defined]
  forces = env._v5_knockdown_forces  # type: ignore[attr-defined]
  torques = env._v5_knockdown_torques  # type: ignore[attr-defined]
  body_ids = env._v5_knockdown_body_ids  # type: ignore[attr-defined]
  stage = getattr(env, "_v4_recovery_stage", None)
  if stage is None:
    return

  recovering = recovery[env_ids] > 0
  recovery[env_ids[recovering]] -= 1

  newly_standing = env_ids[
    (stage[env_ids] == 3)
    & (~done[env_ids])
    & (wait[env_ids] < 0)
    & (active[env_ids] <= 0)
  ]
  if newly_standing.numel() > 0:
    wait[newly_standing] = torch.randint(
      delay_steps[0],
      delay_steps[1] + 1,
      (newly_standing.numel(),),
      device=env.device,
    )

  waiting = env_ids[(wait[env_ids] > 0) & (~done[env_ids])]
  wait[waiting] -= 1
  start_ids = env_ids[(wait[env_ids] == 0) & (~done[env_ids]) & (active[env_ids] <= 0)]
  if start_ids.numel() > 0:
    progress = min(float(env.common_step_counter) / max(curriculum_steps, 1), 1.0)
    force_amp = force_range[0] + progress * (force_range[1] - force_range[0])
    torque_amp = torque_range[0] + progress * (torque_range[1] - torque_range[0])
    chosen = torch.randint(0, len(body_ids), (start_ids.numel(),), device=env.device)
    angle = torch.empty(start_ids.numel(), device=env.device).uniform_(
      -torch.pi, torch.pi
    )
    magnitude = torch.empty(start_ids.numel(), device=env.device).uniform_(
      0.75 * force_amp, force_amp
    )
    force_vec = torch.stack(
      (magnitude * torch.cos(angle), magnitude * torch.sin(angle), 0.0 * angle),
      dim=-1,
    )
    torque_vec = torch.empty_like(force_vec).uniform_(-torque_amp, torque_amp)
    torque_vec[:, 2] *= 0.35
    forces[start_ids] = 0.0
    torques[start_ids] = 0.0
    forces[start_ids, chosen] = force_vec
    torques[start_ids, chosen] = torque_vec
    active[start_ids] = torch.randint(
      duration_steps[0],
      duration_steps[1] + 1,
      (start_ids.numel(),),
      device=env.device,
    )
    recovery[start_ids] = active[start_ids] + recovery_steps
    done[start_ids] = True

  active_ids = env_ids[active[env_ids] > 0]
  inactive_ids = env_ids[active[env_ids] <= 0]
  forces[inactive_ids] = 0.0
  torques[inactive_ids] = 0.0
  robot.write_external_wrench_to_sim(
    forces[env_ids], torques[env_ids], env_ids=env_ids, body_ids=body_ids
  )
  if active_ids.numel() > 0:
    active[active_ids] -= 1


def _prime_static_smp_history(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  """Prime SMP history from the simulator state after a non-GSI reset."""
  robot = env.scene["robot"]
  origins = env.scene.env_origins[env_ids]
  buffer = env._smp_buffer  # type: ignore[attr-defined]
  window_size = buffer.window_size
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  root_pos = robot.data.root_link_pos_w[env_ids] - origins
  ee_pos = robot.data.body_link_pos_w[env_ids][:, ee_indexes] - origins[:, None, :]
  buffer.reset(
    env_ids,
    root_pos[:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_quat_w[env_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_lin_vel_w[env_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_ang_vel_w[env_ids][:, None, :].expand(-1, window_size, -1),
    ee_pos[:, None, :, :].expand(-1, window_size, -1, -1),
    robot.data.joint_pos[env_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.joint_vel[env_ids][:, None, :].expand(-1, window_size, -1),
  )


def _ensure_failure_state_buffer(env: ManagerBasedRlEnv, capacity: int) -> None:
  if hasattr(env, "_v6_failure_root"):
    return
  robot = env.scene["robot"]
  num_joints = robot.data.joint_pos.shape[1]
  env._v6_failure_root = torch.zeros(  # type: ignore[attr-defined]
    capacity, 13, device=env.device
  )
  env._v6_failure_joint_pos = torch.zeros(  # type: ignore[attr-defined]
    capacity, num_joints, device=env.device
  )
  env._v6_failure_joint_vel = torch.zeros_like(  # type: ignore[attr-defined]
    env._v6_failure_joint_pos  # type: ignore[attr-defined]
  )
  env._v6_failure_size = 0  # type: ignore[attr-defined]
  env._v6_failure_head = 0  # type: ignore[attr-defined]
  env._v6_failure_best = torch.zeros(  # type: ignore[attr-defined]
    env.num_envs, device=env.device
  )
  env._v6_failure_stagnant = torch.zeros(  # type: ignore[attr-defined]
    env.num_envs, dtype=torch.long, device=env.device
  )
  env._v6_failure_prev_stage = torch.zeros(  # type: ignore[attr-defined]
    env.num_envs, dtype=torch.long, device=env.device
  )


@torch.no_grad()
def record_failure_states(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  capacity: int = 8192,
  stagnation_steps: int = 75,
  progress_epsilon: float = 0.02,
  record_probability: float = 0.25,
  max_records_per_step: int = 64,
) -> None:
  """Store contact-rich states where recovery progress has stagnated.

  The ring is deliberately small and GPU-resident. It stores state snapshots,
  not trajectories, and is sampled only by the V6 replay reset event.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  _ensure_failure_state_buffer(env, capacity)
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[env_ids, head_idx, 2]
  upright = torch.clamp(-robot.data.projected_gravity_b[env_ids, 2], 0.0, 1.0)
  progress = 0.55 * torch.clamp((head_z - 0.35) / 0.85, 0.0, 1.0) + 0.45 * upright

  best = env._v6_failure_best  # type: ignore[attr-defined]
  stagnant = env._v6_failure_stagnant  # type: ignore[attr-defined]
  prev_stage = env._v6_failure_prev_stage  # type: ignore[attr-defined]
  stage = getattr(env, "_v4_recovery_stage", None)
  if stage is not None:
    # A post-stand fall starts a new progress attempt. Without this reset, the
    # previous standing maximum would make even a steadily improving second
    # recovery look stagnant for 1.5 seconds.
    new_fall = (prev_stage[env_ids] == 3) & (stage[env_ids] < 3)
    best[env_ids[new_fall]] = progress[new_fall]
    stagnant[env_ids[new_fall]] = 0
    prev_stage[env_ids] = stage[env_ids]
  improved = progress > best[env_ids] + progress_epsilon
  best[env_ids] = torch.where(improved, progress, best[env_ids])
  stagnant[env_ids] = torch.where(improved, 0, stagnant[env_ids] + 1)
  stable = (head_z > 1.12) & (upright > 0.85)
  stagnant[env_ids[stable]] = 0

  candidate_mask = (
    (stagnant[env_ids] >= stagnation_steps)
    & (head_z < 1.05)
    & (upright < 0.80)
    & (torch.rand(env_ids.numel(), device=env.device) < record_probability)
  )
  # Never persist a numerically corrupted state into the hard-state replay
  # buffer.  The raw MuJoCo state also covers task entities such as the plate.
  finite_state = torch.isfinite(env.sim.data.qpos[env_ids]).all(dim=-1)
  finite_state &= torch.isfinite(env.sim.data.qvel[env_ids]).all(dim=-1)
  candidate_mask &= finite_state
  record_ids = env_ids[candidate_mask][:max_records_per_step]
  if record_ids.numel() == 0:
    return

  count = record_ids.numel()
  slots = (
    torch.arange(count, device=env.device) + env._v6_failure_head  # type: ignore[attr-defined]
  ) % capacity
  root = torch.cat(
    (
      robot.data.root_link_pos_w[record_ids],
      robot.data.root_link_quat_w[record_ids],
      robot.data.root_link_lin_vel_w[record_ids],
      robot.data.root_link_ang_vel_w[record_ids],
    ),
    dim=-1,
  ).clone()
  root[:, :3] -= env.scene.env_origins[record_ids]
  env._v6_failure_root[slots] = root  # type: ignore[attr-defined]
  env._v6_failure_joint_pos[slots] = robot.data.joint_pos[record_ids]  # type: ignore[attr-defined]
  env._v6_failure_joint_vel[slots] = robot.data.joint_vel[record_ids]  # type: ignore[attr-defined]
  env._v6_failure_head = int(  # type: ignore[attr-defined]
    (env._v6_failure_head + count) % capacity  # type: ignore[attr-defined]
  )
  env._v6_failure_size = min(  # type: ignore[attr-defined]
    capacity,
    env._v6_failure_size + count,  # type: ignore[attr-defined]
  )
  stagnant[record_ids] = 0
  best[record_ids] = progress[candidate_mask][:max_records_per_step]


@torch.no_grad()
def failure_state_replay_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  replay_probability: float = 0.20,
  minimum_buffer_size: int = 256,
) -> None:
  """Replace a subset of resets with recently recorded hard recovery states."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  size = int(getattr(env, "_v6_failure_size", 0))
  if env_ids.numel() == 0 or size < minimum_buffer_size:
    return
  replay_ids = env_ids[
    torch.rand(env_ids.numel(), device=env.device) < replay_probability
  ]
  if replay_ids.numel() == 0:
    return

  sample = torch.randint(0, size, (replay_ids.numel(),), device=env.device)
  root = env._v6_failure_root[sample].clone()  # type: ignore[attr-defined]
  root[:, :3] += env.scene.env_origins[replay_ids]
  robot = env.scene["robot"]
  robot.write_root_state_to_sim(root, env_ids=replay_ids)
  robot.write_joint_state_to_sim(
    env._v6_failure_joint_pos[sample],  # type: ignore[attr-defined]
    env._v6_failure_joint_vel[sample],  # type: ignore[attr-defined]
    env_ids=replay_ids,
  )
  env.sim.forward()
  _prime_static_smp_history(env, replay_ids)
  if hasattr(env, "_robust_reset_type"):
    env._robust_reset_type[replay_ids] = 5  # type: ignore[attr-defined]
  reset_recovery_stage(env, replay_ids)


@torch.no_grad()
def stratified_post_stand_wrench(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  body_names: tuple[str, ...] = ("pelvis", "torso_link"),
  cohort_weights: tuple[float, float, float] = (0.25, 0.50, 0.25),
  delay_steps: tuple[int, int] = (20, 60),
  duration_steps: tuple[int, int] = (10, 18),
  recovery_steps: int = 100,
  standard_force_range: tuple[float, float] = (80.0, 170.0),
  intensive_force_range: tuple[float, float] = (120.0, 230.0),
  torque_range: tuple[float, float] = (4.0, 16.0),
  standard_max_pushes: int = 1,
  intensive_max_pushes: int = 3,
  curriculum_steps: int = 300_000,
) -> None:
  """Apply no, one, or repeated post-stand knockdowns by environment cohort."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  robot = env.scene["robot"]

  if not hasattr(env, "_v6_push_active"):
    weights = torch.tensor(cohort_weights, device=env.device, dtype=torch.float)
    if weights.shape != (3,) or torch.any(weights < 0) or weights.sum() <= 0:
      raise ValueError("cohort_weights must be three non-negative values")
    weights /= weights.sum()
    body_ids = robot.find_bodies(list(body_names), preserve_order=True)[0]
    env._v6_push_body_ids = body_ids  # type: ignore[attr-defined]
    env._v6_push_cohort_weights = weights  # type: ignore[attr-defined]
    env._v6_push_cohort = torch.multinomial(  # type: ignore[attr-defined]
      weights, env.num_envs, replacement=True
    )
    env._v6_push_wait = torch.full(  # type: ignore[attr-defined]
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    env._v6_push_active = torch.zeros_like(env._v6_push_wait)  # type: ignore[attr-defined]
    env._v6_push_recovery = torch.zeros_like(env._v6_push_wait)  # type: ignore[attr-defined]
    env._v6_push_count = torch.zeros_like(env._v6_push_wait)  # type: ignore[attr-defined]
    shape = (env.num_envs, len(body_ids), 3)
    env._v6_push_forces = torch.zeros(shape, device=env.device)  # type: ignore[attr-defined]
    env._v6_push_torques = torch.zeros(shape, device=env.device)  # type: ignore[attr-defined]

  stage = getattr(env, "_v4_recovery_stage", None)
  if stage is None:
    return
  cohort = env._v6_push_cohort  # type: ignore[attr-defined]
  wait = env._v6_push_wait  # type: ignore[attr-defined]
  active = env._v6_push_active  # type: ignore[attr-defined]
  recovery = env._v6_push_recovery  # type: ignore[attr-defined]
  count = env._v6_push_count  # type: ignore[attr-defined]
  forces = env._v6_push_forces  # type: ignore[attr-defined]
  torques = env._v6_push_torques  # type: ignore[attr-defined]
  body_ids = env._v6_push_body_ids  # type: ignore[attr-defined]

  recovering = recovery[env_ids] > 0
  recovery[env_ids[recovering]] -= 1
  fell = (wait[env_ids] == -2) & (stage[env_ids] < 3)
  wait[env_ids[fell]] = -1
  max_pushes = torch.zeros_like(count[env_ids])
  max_pushes = torch.where(
    cohort[env_ids] == 1, torch.full_like(max_pushes, standard_max_pushes), max_pushes
  )
  max_pushes = torch.where(
    cohort[env_ids] == 2, torch.full_like(max_pushes, intensive_max_pushes), max_pushes
  )
  newly_standing = env_ids[
    (stage[env_ids] == 3)
    & (wait[env_ids] == -1)
    & (active[env_ids] <= 0)
    & (count[env_ids] < max_pushes)
  ]
  if newly_standing.numel() > 0:
    wait[newly_standing] = torch.randint(
      delay_steps[0], delay_steps[1] + 1, (newly_standing.numel(),), device=env.device
    )
  waiting = env_ids[wait[env_ids] > 0]
  wait[waiting] -= 1
  start_ids = env_ids[(wait[env_ids] == 0) & (active[env_ids] <= 0)]
  if start_ids.numel() > 0:
    progress = min(float(env.common_step_counter) / max(curriculum_steps, 1), 1.0)
    std_amp = standard_force_range[0] + progress * (
      standard_force_range[1] - standard_force_range[0]
    )
    hard_amp = intensive_force_range[0] + progress * (
      intensive_force_range[1] - intensive_force_range[0]
    )
    amp = torch.where(
      cohort[start_ids] == 2,
      torch.full((start_ids.numel(),), hard_amp, device=env.device),
      torch.full((start_ids.numel(),), std_amp, device=env.device),
    )
    angle = torch.empty(start_ids.numel(), device=env.device).uniform_(
      -torch.pi, torch.pi
    )
    magnitude = amp * torch.empty_like(amp).uniform_(0.75, 1.0)
    force_vec = torch.stack(
      (
        magnitude * torch.cos(angle),
        magnitude * torch.sin(angle),
        torch.zeros_like(angle),
      ),
      dim=-1,
    )
    torque_amp = torque_range[0] + progress * (torque_range[1] - torque_range[0])
    torque_vec = torch.empty_like(force_vec).uniform_(-torque_amp, torque_amp)
    torque_vec[:, 2] *= 0.35
    chosen = torch.randint(0, len(body_ids), (start_ids.numel(),), device=env.device)
    forces[start_ids] = 0.0
    torques[start_ids] = 0.0
    forces[start_ids, chosen] = force_vec
    torques[start_ids, chosen] = torque_vec
    active[start_ids] = torch.randint(
      duration_steps[0], duration_steps[1] + 1, (start_ids.numel(),), device=env.device
    )
    recovery[start_ids] = active[start_ids] + recovery_steps
    count[start_ids] += 1

  active_ids = env_ids[active[env_ids] > 0]
  inactive_ids = env_ids[active[env_ids] <= 0]
  forces[inactive_ids] = 0.0
  torques[inactive_ids] = 0.0
  robot.write_external_wrench_to_sim(
    forces[env_ids], torques[env_ids], env_ids=env_ids, body_ids=body_ids
  )
  if active_ids.numel() > 0:
    active[active_ids] -= 1
    finished = active_ids[active[active_ids] == 0]
    wait[finished] = -2


_LAFAN_MILESTONE_IDS = {
  "kneeling_or_half_kneeling": 1,
  "crouched": 2,
  "standing": 3,
}


def _load_lafan_milestone_bank(
  env: ManagerBasedRlEnv,
  manifest_path: str,
  npz_dir: str,
  stage_names: tuple[str, ...],
) -> tuple[torch.Tensor, ...]:
  """Load approved route windows whose final frame lies in each stage span."""
  cache_key = (manifest_path, npz_dir, stage_names)
  if getattr(env, "_lafan_milestone_cache_key", None) == cache_key:
    return env._lafan_milestone_windows  # type: ignore[attr-defined]

  manifest_file = Path(manifest_path).expanduser().resolve()
  payload = json.loads(manifest_file.read_text())
  if not payload.get("approved_for_training", False):
    raise ValueError(f"LAFAN milestone manifest is not approved: {manifest_file}")
  input_fps = float(payload["input_fps"])
  requested = set(stage_names)
  if requested - _LAFAN_MILESTONE_IDS.keys():
    unknown = ", ".join(sorted(requested - _LAFAN_MILESTONE_IDS.keys()))
    raise ValueError(f"unknown LAFAN milestone stages: {unknown}")

  route_dir = Path(npz_dir).expanduser().resolve()
  by_stage: dict[str, list[np.ndarray]] = {name: [] for name in stage_names}
  for clip in payload["clips"]:
    npz_path = route_dir / Path(clip["output"]).with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as data:
      windows = data["windows"]
      output_fps = float(data["fps"][0])
      window_size = int(data["window_size"][0])
      for span in clip["stage_spans"]:
        name = str(span["name"])
        if name not in requested:
          continue
        start, end = (int(value) for value in span["frame_span"])
        output_start = round(start * output_fps / input_fps)
        output_end = round(end * output_fps / input_fps)
        # Window index i ends at interpolated frame i + window_size - 1.
        lo = max(0, output_start - window_size + 1)
        hi = min(len(windows), output_end - window_size + 1)
        if hi > lo:
          by_stage[name].append(windows[lo:hi].copy())

  bank: list[torch.Tensor] = []
  for name in stage_names:
    chunks = by_stage[name]
    if not chunks:
      raise ValueError(f"no LAFAN milestone windows found for stage {name!r}")
    values = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    bank.append(torch.from_numpy(values).to(env.device))

  result = tuple(bank)
  env._lafan_milestone_cache_key = cache_key  # type: ignore[attr-defined]
  env._lafan_milestone_windows = result  # type: ignore[attr-defined]
  return result


@torch.no_grad()
def lafan_milestone_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  probability: float = 0.20,
  manifest_path: str = "datasets/csv/getup_lafan_prone_routes_v7/manifest.json",
  npz_dir: str = "datasets/npz/getup_lafan_prone_routes_v7",
  stage_names: tuple[str, ...] = (
    "kneeling_or_half_kneeling",
    "crouched",
    "standing",
  ),
  stage_weights: tuple[float, ...] = (0.45, 0.35, 0.20),
) -> None:
  """Replace a reset subset with phase-balanced, reviewed LAFAN milestones.

  The complete feature window primes the SMP history while its final state seeds
  physics. The phase id is training-only state and never enters actor observations.
  This event must run after reset_recovery_stage so it can override the ordered
  stage without reading stale derived quantities after a simulator state write.
  """
  if not 0.0 <= probability <= 1.0:
    raise ValueError(f"probability must be in [0, 1], got {probability}")
  if len(stage_names) != len(stage_weights) or not stage_names:
    raise ValueError("stage_names and stage_weights must have equal non-zero length")
  weights = torch.tensor(stage_weights, dtype=torch.float, device=env.device)
  if torch.any(weights < 0.0) or weights.sum() <= 0.0:
    raise ValueError("stage_weights must be non-negative with positive sum")
  weights /= weights.sum()

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if not hasattr(env, "_lafan_milestone_stage"):
    env._lafan_milestone_stage = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
  milestone_stage = env._lafan_milestone_stage  # type: ignore[attr-defined]
  milestone_stage[env_ids] = 0
  chosen = torch.rand(env_ids.numel(), device=env.device) < probability
  chosen_ids = env_ids[chosen]
  if chosen_ids.numel() == 0:
    return

  bank = _load_lafan_milestone_bank(env, manifest_path, npz_dir, stage_names)
  slots = torch.multinomial(weights, chosen_ids.numel(), replacement=True)
  for slot, (name, windows) in enumerate(zip(stage_names, bank, strict=True)):
    stage_ids = chosen_ids[slots == slot]
    if stage_ids.numel() == 0:
      continue
    sample_ids = torch.randint(
      0, windows.shape[0], (stage_ids.numel(),), device=env.device
    )
    prime_sim_and_buffer(env, stage_ids, windows[sample_ids])
    stage = _LAFAN_MILESTONE_IDS[name]
    milestone_stage[stage_ids] = stage
    env._v4_recovery_stage[stage_ids] = stage  # type: ignore[attr-defined]
    env._v4_stage_hold[stage_ids] = 0  # type: ignore[attr-defined]
    if hasattr(env, "_robust_reset_type"):
      env._robust_reset_type[stage_ids] = 0  # type: ignore[attr-defined]
