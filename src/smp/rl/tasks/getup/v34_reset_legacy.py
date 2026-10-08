"""Archived V34 reset functions; only SMP history adapter replaced for current API.
Source: smp-v38-unified/src/smp/rl/tasks/getup/mdp/events.py
Source SHA256: 8aa31c9b2dd34bce2f5441eb243fa57470798ac2dae0f7d64ce0131a674fa7c8
"""
from __future__ import annotations
import torch, numpy as np, mujoco, math
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, yaw_quat
from smp.rl.tasks.getup.natural_low_reset import prime_static_history as _prime_smp_history_from_current_state


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

@torch.no_grad()
def mixed_fall_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  procedural_probability: float = 0.5,
  root_height_range: tuple[float, float] = (0.48, 0.62),
  joint_noise: float = 0.12,
  joint_noise_levels: tuple[float, ...] | None = None,
  joint_noise_weights: tuple[float, ...] | None = None,
  joint_limit_margin: float = 0.0,
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

  if joint_noise < 0.0 or joint_limit_margin < 0.0:
    raise ValueError("joint noise and joint limit margin must be non-negative")
  noise_levels: torch.Tensor | None = None
  noise_weights: torch.Tensor | None = None
  if joint_noise_levels is not None:
    if joint_noise_weights is None or len(joint_noise_levels) != len(
      joint_noise_weights
    ):
      raise ValueError("joint noise levels and weights must have equal non-zero length")
    noise_levels = torch.tensor(
      joint_noise_levels, dtype=torch.float, device=env.device
    )
    noise_weights = torch.tensor(
      joint_noise_weights, dtype=torch.float, device=env.device
    )
    if (
      noise_levels.numel() == 0
      or torch.any(noise_levels < 0.0)
      or torch.any(noise_weights < 0.0)
      or noise_weights.sum() <= 0.0
    ):
      raise ValueError(
        "joint noise levels/weights must be non-negative with positive mass"
      )
    noise_weights /= noise_weights.sum()
  elif joint_noise_weights is not None:
    raise ValueError("joint noise weights require joint noise levels")

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
  orientation_offset = torch.zeros(n, 2, device=env.device)
  if orientation_noise > 0.0:
    # Noise on both axes creates continuous oblique front/back/side contacts.
    orientation_offset.uniform_(-orientation_noise, orientation_noise)
    roll += orientation_offset[:, 0]
    pitch += orientation_offset[:, 1]
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
  if noise_levels is None:
    sampled_joint_noise = torch.full((n,), joint_noise, device=env.device)
  else:
    assert noise_weights is not None
    sampled_joint_noise = noise_levels[
      torch.multinomial(noise_weights, n, replacement=True)
    ]
  joint_pos += (
    torch.empty_like(joint_pos).uniform_(-1.0, 1.0) * sampled_joint_noise[:, None]
  )
  joint_limits = getattr(robot.data, "soft_joint_pos_limits", None)
  if joint_limits is None:
    joint_limits = robot.data.joint_pos_limits
  joint_limits = joint_limits[fall_ids]
  lower = joint_limits[..., 0] + joint_limit_margin
  upper = joint_limits[..., 1] - joint_limit_margin
  if not bool(torch.all(lower <= upper)):
    raise RuntimeError("PHYSICAL_RESET_ALERT: joint limit margin collapsed a limit")
  joint_pos = torch.maximum(torch.minimum(joint_pos, upper), lower)
  joint_vel = torch.empty_like(joint_pos).uniform_(-0.1, 0.1)
  robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=fall_ids)
  env.sim.forward()

  if not hasattr(env, "_procedural_joint_noise_level"):
    env._procedural_joint_noise_level = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
    env._procedural_orientation_offset = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, device=env.device
    )
  env._procedural_joint_noise_level[fall_ids] = sampled_joint_noise  # type: ignore[attr-defined]
  env._procedural_orientation_offset[fall_ids] = orientation_offset.abs().amax(dim=-1)  # type: ignore[attr-defined]

  # A repeated static window is honest history for a newly placed pose and avoids
  # an artificial GSI-to-procedural discontinuity in the SMP score.
  _prime_smp_history_from_current_state(env, fall_ids)
  reset_types[fall_ids] = modes + 1
