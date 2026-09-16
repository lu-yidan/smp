"""Pinned V33 geometry/contact helpers, adapted for paired 93D experiments.
Source: baseline/v33-escape-model-95000-balanced. No legacy task imports.
"""
from __future__ import annotations
import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
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
def reset_guided_escape_plate(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  obstacle_probability: float = 0.90,
  active_mask: torch.Tensor | None = None,
  prepare_mask: torch.Tensor | None = None,
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
  active = (torch.rand(n, device=env.device) < obstacle_probability) if active_mask is None else active_mask[env_ids].clone()
  if eligible_reset_types is not None:
    reset_type = getattr(env, "_robust_reset_type", None)
    if reset_type is None:
      raise RuntimeError("eligible_reset_types requires mixed_fall_reset state")
    eligible = torch.zeros(n, dtype=torch.bool, device=env.device)
    for reset_value in eligible_reset_types:
      eligible |= reset_type[env_ids] == reset_value
    active &= eligible

  active_ids = env_ids[active if prepare_mask is None else prepare_mask[env_ids]]
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
    if torch.any(~torch.isfinite(surface_top) & active):
      raise RuntimeError("guided escape plate footprint overlaps no robot geometry")
    surface_top = torch.nan_to_num(surface_top, neginf=0.5)
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

