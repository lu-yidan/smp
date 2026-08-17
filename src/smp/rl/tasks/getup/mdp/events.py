"""Reset events for the getup task."""

from __future__ import annotations

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

__all__ = [
  "apply_sustained_constraint",
  "failure_state_replay_reset",
  "mixed_fall_reset",
  "post_stand_body_wrench",
  "random_body_wrench",
  "record_failure_states",
  "reset_escape_obstacle",
  "reset_guided_escape_plate",
  "reset_guided_escape_plate_curriculum",
  "reset_recovery_stage",
  "reset_sustained_constraint",
  "reset_stand_counter",
  "stratified_post_stand_wrench",
  "update_escape_phase",
  "update_recovery_stage",
]


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
  target_body_name: str = "torso_link",
  eligible_reset_types: tuple[int, ...] | None = None,
  xy_offset_range: float = 0.015,
  body_origin_clearance: float = 0.26,
  align_to_body: bool = False,
  longitudinal_offset: float = 0.0,
  longitudinal_offset_curriculum: tuple[float, float] | None = None,
  lateral_offset_curriculum: tuple[float, float] | None = None,
  overlap_curriculum_steps: int = 0,
  crawl_ready_prone: bool = False,
  crawl_arm_noise: float = 0.0,
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
  active = torch.rand(n, device=env.device) < obstacle_probability
  if eligible_reset_types is not None:
    reset_type = getattr(env, "_robust_reset_type", None)
    if reset_type is None:
      raise RuntimeError("eligible_reset_types requires mixed_fall_reset state")
    eligible = torch.zeros(n, dtype=torch.bool, device=env.device)
    for reset_value in eligible_reset_types:
      eligible |= reset_type[env_ids] == reset_value
    active &= eligible

  active_ids = env_ids[active]
  if crawl_ready_prone and active_ids.numel() > 0:
    # A procedural prone reset is a rotated nominal stand; its hands are often
    # the highest collision geoms.  This symmetric pose places both hands on
    # the floor just outside the board edges so they can establish support.
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
    arm_ids, _ = robot.find_joints(arm_names, preserve_order=True)
    if len(arm_ids) != len(arm_names):
      raise ValueError("all crawl-ready arm joints must resolve exactly once")
    joint_pos = robot.data.joint_pos[active_ids].clone()
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
    arm_values = arm_pose[None, :].expand(active_ids.numel(), -1).clone()
    if crawl_arm_noise > 0.0:
      arm_values += torch.empty_like(arm_values).uniform_(
        -crawl_arm_noise, crawl_arm_noise
      )
    arm_local = torch.tensor(arm_ids, dtype=torch.long, device=env.device)
    joint_pos[:, arm_local] = arm_values
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=active_ids)
    env.sim.forward()

    # Put the lowest collision surface just above flat ground.  The robot no
    # longer falls away from a close board during the first control steps.
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
    root_state[:, 2] += env.scene.env_origins[active_ids, 2] + ground_clearance - lowest
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
    target_pos[:, :2] += longitudinal_offset * forward_xy
    lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=-1)
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
        target_pos[:, :2] += longitudinal_noise[:, None] * forward_xy
      if lateral_offset_curriculum is not None:
        amplitude = lateral_offset_curriculum[0] + progress * (
          lateral_offset_curriculum[1] - lateral_offset_curriculum[0]
        )
        lateral_noise = torch.empty(n, device=env.device).uniform_(
          -amplitude, amplitude
        )
        target_pos[:, :2] += lateral_noise[:, None] * lateral_xy
  target_pos[:, :2] += torch.empty(n, 2, device=env.device).uniform_(
    -xy_offset_range, xy_offset_range
  )
  if surface_gap is None:
    # Targeting only the torso centre recreated V2's bug whenever a hand, foot,
    # or head collision was higher.  This conservative envelope is independent
    # of which link happens to be uppermost in the sampled prone pose.
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
  max_initial_contact_head_height: float | None = None,
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
  first_contact_height[env_ids[first_contact]] = head_height[first_contact]
  late_contact = torch.zeros_like(contact)
  if max_initial_contact_head_height is not None:
    late_contact = first_contact & (head_height > max_initial_contact_head_height)
  setup_timeout = torch.zeros_like(contact)
  if max_wait_steps is not None:
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
  subset with supine, prone, left-side, or right-side poses, then re-primes the
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


def _head_height_and_upright(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  knee_ids = robot.find_joints(
    ["left_knee_joint", "right_knee_joint"], preserve_order=True
  )[0]
  head_z = robot.data.site_pos_w[env_ids, head_idx, 2]
  head_vz = robot.data.site_lin_vel_w[env_ids, head_idx, 2]
  upright = torch.clamp(-robot.data.projected_gravity_b[env_ids, 2], 0.0, 1.0)
  knee_flexion = robot.data.joint_pos[env_ids][:, knee_ids].mean(dim=-1)
  return head_z, head_vz, upright, knee_flexion


@torch.no_grad()
def reset_recovery_stage(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
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

  head_z, _, upright, knee_flexion = _head_height_and_upright(env, env_ids)
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


@torch.no_grad()
def update_recovery_stage(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  seated_hold_steps: int = 10,
  crouched_hold_steps: int = 10,
  standing_hold_steps: int = 25,
) -> None:
  """Advance recovery only after stable seated, crouched, and standing holds."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  if not hasattr(env, "_v4_recovery_stage"):
    reset_recovery_stage(env, env_ids)

  stage = env._v4_recovery_stage  # type: ignore[attr-defined]
  hold = env._v4_stage_hold  # type: ignore[attr-defined]
  head_z, head_vz, upright, knee_flexion = _head_height_and_upright(env, env_ids)
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
