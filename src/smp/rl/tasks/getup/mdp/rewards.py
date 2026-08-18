"""Reward components for the getup task: head-height + up-velocity.

Combined and SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from smp.rl.rewards import task_smp_product as _task_smp_product

__all__ = [
  "active_wrench_metric",
  "action_rate_rms_metric",
  "base_stationary_when_upright",
  "cached_product_score",
  "cached_raw_smp_score",
  "failure_buffer_fill_metric",
  "failure_replay_reset_metric",
  "ground_support_contact_metric",
  "prone_leg_splay_excess_l2",
  "prone_support_route",
  "recovery_initiation_progress",
  "cached_smp_score",
  "cached_task_score",
  "constraint_active_metric",
  "constraint_cohort_metric",
  "constraint_load_metric",
  "constraint_release_progress_metric",
  "crawl_with_hand_support",
  "escape_completion",
  "escape_contact_force_excess_l2",
  "escape_covered_geom_count_metric",
  "escape_best_covered_geom_count_metric",
  "escape_geometry_clearance_score",
  "escape_geometry_progress",
  "escape_gated_task_smp_product",
  "escape_object_displacement_metric",
  "escape_obstacle_episode_metric",
  "escape_phase_metric",
  "escape_separation_progress",
  "feet_stationary_when_upright",
  "head_vertical_speed_metric",
  "head_vertical_overspeed_l2",
  "hand_support_contact_metric",
  "hand_supported_escape_progress",
  "escape_invalid_contact_metric",
  "escape_invalid_setup_metric",
  "escape_first_contact_head_height_metric",
  "escape_hand_support_steps_metric",
  "escape_hand_supported_progress_metric",
  "escape_peak_contact_force_metric",
  "escape_peak_penetration_metric",
  "escape_planar_clearance_metric",
  "escape_plate_mass_metric",
  "joint_acc_rms_metric",
  "low_base_angular_velocity",
  "low_joint_velocity",
  "max_joint_speed_metric",
  "max_joint_power_metric",
  "max_joint_torque_metric",
  "mean_foot_speed_metric",
  "mean_knee_flexion_metric",
  "post_stand_knockdown_metric",
  "prone_reset_metric",
  "procedural_reset_metric",
  "quiet_stance_gate",
  "recovery_stage_complete_metric",
  "recovery_stage_metric",
  "smooth_action",
  "staged_action_acc_l2",
  "staged_action_rate_l2",
  "staged_joint_acc_l2",
  "staged_joint_power_excess_l2",
  "staged_joint_speed_excess_l2",
  "staged_joint_torques_l2",
  "staged_head_velocity_profile",
  "staged_recovery_pose",
  "stable_stand_metric",
  "supine_reset_metric",
  "track_head_height",
  "terrain_foot_slip_l2",
  "terrain_planar_displacement_l2",
  "terrain_stance_width_excess_l2",
  "track_head_velocity_profile",
  "upright_posture",
  "upward_velocity",
  "v6_active_wrench_metric",
  "v6_push_cohort_metric",
  "v6_push_count_metric",
]


def escape_gated_task_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  constrained_scale: float = 0.15,
) -> torch.Tensor:
  """Relax the upright/get-up objective while a physical route is blocked.

  SMP is still evaluated exactly once.  The gate is privileged reward shaping,
  not an actor observation; a later deployable adapter must infer it from
  proprioceptive and motor-response history.
  """
  product = _task_smp_product(
    env, task_terms=task_terms, fixed_timesteps=fixed_timesteps, ws=ws
  )
  phase = getattr(env, "_escape_phase", None)
  if phase is None:
    return product
  constrained = (phase == 1) | (phase == 2)
  scale = torch.where(
    constrained,
    torch.full_like(product, constrained_scale),
    torch.ones_like(product),
  )
  gated = product * scale
  env._smp_product_score = gated  # type: ignore[attr-defined]
  return gated


def _contact_found(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  found = env.scene[sensor_name].data.found
  if found is None:
    raise RuntimeError(f"{sensor_name} must expose the 'found' contact field")
  return found > 0


def ground_support_contact_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Fraction of configured support collision groups touching the terrain."""
  return _contact_found(env, sensor_name).float().mean(dim=-1)


def _procedural_prone_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  reset_type = getattr(env, "_robust_reset_type", None)
  if reset_type is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return reset_type == 1


def prone_support_route(
  env: ManagerBasedRlEnv,
  hand_sensor_name: str = "natural_hand_ground_contact",
  knee_sensor_name: str = "natural_knee_ground_contact",
  start_height: float = 0.28,
  waypoint_height: float = 0.62,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Reward a supported prone-to-kneeling route before the first waypoint.

  Hands are required for dense elevation credit. Knee/shin support raises the
  score but is not mandatory, allowing the demonstrated asymmetric kneeling
  variants. Once stage one is reached the term remains at one, so advancing is
  always preferable to holding a quadruped pose.
  """
  robot = env.scene["robot"]
  stage = _recovery_stage(env)
  prone = _procedural_prone_mask(env)
  hand = ground_support_contact_metric(env, hand_sensor_name)
  knee = ground_support_contact_metric(env, knee_sensor_name)
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  if relative_to_env_origin:
    head_z = head_z - env.scene.env_origins[:, 2]
  height = torch.clamp(
    (head_z - start_height) / max(waypoint_height - start_height, 1e-6),
    0.0,
    1.0,
  )
  supported_height = hand * height * (0.65 + 0.35 * knee)
  route = 0.20 * hand + 0.20 * hand * knee + 0.60 * supported_height
  route = torch.where(stage > 0, torch.ones_like(route), route)
  return prone.float() * route


def prone_leg_splay_excess_l2(
  env: ManagerBasedRlEnv,
  hip_roll_limit: float = 0.65,
  hip_yaw_limit: float = 0.75,
  max_head_height: float = 0.95,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Penalize extreme early-prone hip abduction/yaw without affecting other falls."""
  robot = env.scene["robot"]
  joint_ids = robot.find_joints(
    [
      "left_hip_roll_joint",
      "right_hip_roll_joint",
      "left_hip_yaw_joint",
      "right_hip_yaw_joint",
    ],
    preserve_order=True,
  )[0]
  hip = torch.abs(robot.data.joint_pos[:, joint_ids])
  roll_excess = torch.clamp(hip[:, :2] - hip_roll_limit, min=0.0)
  yaw_excess = torch.clamp(hip[:, 2:] - hip_yaw_limit, min=0.0)
  excess = torch.mean(torch.square(roll_excess), dim=-1)
  excess += torch.mean(torch.square(yaw_excess), dim=-1)
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  if relative_to_env_origin:
    head_z = head_z - env.scene.env_origins[:, 2]
  low = head_z < max_head_height
  early = _recovery_stage(env) <= 1
  return _procedural_prone_mask(env).float() * low.float() * early.float() * excess


def crawl_with_hand_support(
  env: ManagerBasedRlEnv,
  sensor_name: str = "hand_ground_contact",
  target_speed: float = 0.10,
  speed_scale: float = 55.0,
  max_head_height: float = 0.90,
) -> torch.Tensor:
  """Reward controlled low-pose translation while one or both hands support."""
  phase = getattr(env, "_escape_phase", None)
  if phase is None:
    return torch.zeros(env.num_envs, device=env.device)
  support_fraction = _contact_found(env, sensor_name).float().mean(dim=-1)
  robot = env.scene["robot"]
  speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
  speed_score = torch.exp(-speed_scale * torch.square(speed - target_speed))
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  low_pose = robot.data.site_pos_w[:, head_idx, 2] <= max_head_height
  return (phase == 2).float() * low_pose.float() * support_fraction * speed_score


def escape_separation_progress(
  env: ManagerBasedRlEnv,
  progress_scale: float = 0.025,
) -> torch.Tensor:
  """Reward only new robot-obstacle planar separation, not oscillation."""
  phase = getattr(env, "_escape_phase", None)
  delta = getattr(env, "_escape_separation_delta", None)
  if phase is None or delta is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (phase == 2).float() * torch.clamp(delta / progress_scale, 0.0, 1.0)


def hand_supported_escape_progress(
  env: ManagerBasedRlEnv,
  sensor_name: str = "hand_ground_contact",
  progress_scale: float = 0.025,
  max_head_height: float = 0.90,
) -> torch.Tensor:
  """Reward new separation only when a low robot is supported by its hands.

  Coupling support to monotonic progress prevents the V2 local optimum where the
  policy accumulated hand-contact/crawling reward while remaining under the plate.
  """
  phase = getattr(env, "_escape_phase", None)
  delta = getattr(env, "_escape_separation_delta", None)
  if phase is None or delta is None:
    return torch.zeros(env.num_envs, device=env.device)
  support_fraction = _contact_found(env, sensor_name).float().mean(dim=-1)
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  low_pose = robot.data.site_pos_w[:, head_idx, 2] <= max_head_height
  progress = torch.clamp(delta / progress_scale, 0.0, 1.0)
  return (phase == 2).float() * low_pose.float() * support_fraction * progress


def escape_geometry_progress(
  env: ManagerBasedRlEnv,
  sensor_name: str = "hand_ground_contact",
  coverage_scale: float = 0.025,
  clearance_scale: float = 0.02,
  max_head_height: float = 0.90,
) -> torch.Tensor:
  """Reward all-body footprint clearance gained while hand-supported.

  Unlike centre-distance progress, this term cannot be maximized while the
  head, torso, hand, or foot remains under a plate edge.
  """
  phase = getattr(env, "_escape_phase", None)
  coverage_delta = getattr(env, "_escape_coverage_delta", None)
  clearance_delta = getattr(env, "_escape_clearance_delta", None)
  if phase is None or coverage_delta is None or clearance_delta is None:
    return torch.zeros(env.num_envs, device=env.device)
  support_fraction = _contact_found(env, sensor_name).float().mean(dim=-1)
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  low_pose = robot.data.site_pos_w[:, head_idx, 2] <= max_head_height
  progress = torch.clamp(coverage_delta / coverage_scale, 0.0, 1.0)
  progress += 0.5 * torch.clamp(clearance_delta / clearance_scale, 0.0, 1.0)
  return (phase == 2).float() * low_pose.float() * support_fraction * progress


def escape_geometry_clearance_score(
  env: ManagerBasedRlEnv,
  target_clearance: float = 0.04,
) -> torch.Tensor:
  """Small dense score for reducing covered geoms and clearing the last edge."""
  phase = getattr(env, "_escape_phase", None)
  covered = getattr(env, "_escape_covered_geom_count", None)
  initial = getattr(env, "_escape_initial_covered_geom_count", None)
  clearance = getattr(env, "_escape_planar_clearance", None)
  if phase is None or covered is None or initial is None or clearance is None:
    return torch.zeros(env.num_envs, device=env.device)
  denominator = torch.clamp(initial.float(), min=1.0)
  uncovered_fraction = torch.clamp(
    1.0 - covered.float() / denominator, min=0.0, max=1.0
  )
  clearance_score = torch.clamp(clearance / target_clearance, 0.0, 1.0)
  active = (phase == 2) | (phase == 3)
  return active.float() * (0.85 * uncovered_fraction + 0.15 * clearance_score)


def escape_completion(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Persistent success signal after stable contact-free separation."""
  phase = getattr(env, "_escape_phase", None)
  if phase is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (phase == 3).float()


def escape_contact_force_excess_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str = "robot_obstacle_contact",
  force_limit: float = 300.0,
  force_scale: float = 300.0,
) -> torch.Tensor:
  """Penalize striking the plate rather than establishing controlled support."""
  phase = getattr(env, "_escape_phase", None)
  force = env.scene[sensor_name].data.force
  if phase is None or force is None:
    return torch.zeros(env.num_envs, device=env.device)
  peak = torch.linalg.vector_norm(force, dim=-1).amax(dim=-1)
  excess = torch.clamp(peak - force_limit, min=0.0) / max(force_scale, 1e-6)
  constrained = (phase == 1) | (phase == 2)
  return constrained.float() * torch.square(excess)


def _head_height(
  env: ManagerBasedRlEnv, relative_to_env_origin: bool = False
) -> torch.Tensor:
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  height = robot.data.site_pos_w[:, head_idx, 2]
  if relative_to_env_origin:
    height = height - env.scene.env_origins[:, 2]
  return height


def track_head_height(
  env: ManagerBasedRlEnv,
  target_height: float = 1.2,
  scale: float = 6.0,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Reward the ``head`` site reaching ``target_height``:
  ``exp(-scale·max(target_height − head_z, 0)²)`` (no penalty for overshoot).
  Needs the ``head`` site from ``getup_env_cfg.get_g1_spec_with_head``."""
  z = _head_height(env, relative_to_env_origin=relative_to_env_origin)
  shortfall = torch.clamp(z - target_height, max=0.0)
  return torch.exp(-scale * shortfall * shortfall)


def upward_velocity(
  env: ManagerBasedRlEnv,
  target_velocity: float = 0.25,
  head_height_threshold: float = 0.6,
  scale: float = 100.0,
) -> torch.Tensor:
  """Reward upward HEAD velocity below ``head_height_threshold`` (else ``1``):
  ``exp(-scale·max(target_velocity − head_vz, 0)²)``.  Uses the head site's world
  velocity (``site_lin_vel_w``, includes ω×r from torso pitch) so it drives the
  head, not the pelvis.  Needs ``getup_env_cfg.get_g1_spec_with_head``."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  shortfall = torch.clamp(head_vz - target_velocity, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(
    head_z < head_height_threshold,
    shaped,
    torch.ones_like(shaped),
  )


def track_head_velocity_profile(
  env: ManagerBasedRlEnv,
  start_height: float = 0.5,
  stop_height: float = 1.15,
  max_velocity: float = 0.15,
  speed_limit: float = 0.25,
  scale: float = 35.0,
  overspeed_scale: float = 80.0,
) -> torch.Tensor:
  """Track a height-dependent head velocity that tapers to zero near standing.

  Unlike ``upward_velocity``, this symmetrically penalizes both under- and
  over-speed motion, with an extra soft limit for violent vertical movement.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  height_span = max(stop_height - start_height, 1e-6)
  progress = torch.clamp((head_z - start_height) / height_span, 0.0, 1.0)
  target_vz = max_velocity * (1.0 - progress)
  excess_speed = torch.clamp(torch.abs(head_vz) - speed_limit, min=0.0)
  return torch.exp(-scale * torch.square(head_vz - target_vz)) * torch.exp(
    -overspeed_scale * torch.square(excess_speed)
  )


def _recovery_stage(env: ManagerBasedRlEnv) -> torch.Tensor:
  stage = getattr(env, "_v4_recovery_stage", None)
  if stage is None:
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  return stage


def _stage_multiplier(
  env: ManagerBasedRlEnv,
  multipliers: tuple[float, float, float, float],
) -> torch.Tensor:
  return env.scene["robot"].data.joint_pos.new_tensor(multipliers)[_recovery_stage(env)]


def staged_action_rate_l2(
  env: ManagerBasedRlEnv,
  multipliers: tuple[float, float, float, float] = (0.30, 0.60, 1.0, 1.0),
) -> torch.Tensor:
  """Action-rate cost relaxed while lying and restored near standing."""
  delta = env.action_manager.action - env.action_manager.prev_action
  return _stage_multiplier(env, multipliers) * torch.sum(delta * delta, dim=-1)


def staged_action_acc_l2(
  env: ManagerBasedRlEnv,
  multipliers: tuple[float, float, float, float] = (0.30, 0.60, 1.0, 1.0),
) -> torch.Tensor:
  """Action-acceleration cost relaxed only for low-pose recovery."""
  acc = (
    env.action_manager.action
    - 2.0 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return _stage_multiplier(env, multipliers) * torch.sum(acc * acc, dim=-1)


def staged_joint_acc_l2(
  env: ManagerBasedRlEnv,
  multipliers: tuple[float, float, float, float] = (0.35, 0.65, 1.0, 1.0),
) -> torch.Tensor:
  """Joint-acceleration cost with enough freedom to initiate a prone roll."""
  joint_acc = env.scene["robot"].data.joint_acc
  return _stage_multiplier(env, multipliers) * torch.sum(
    torch.square(joint_acc), dim=-1
  )


def staged_joint_torques_l2(
  env: ManagerBasedRlEnv,
  multipliers: tuple[float, float, float, float] = (0.50, 0.75, 1.0, 1.0),
) -> torch.Tensor:
  """Effort cost that does not over-constrain contact-rich low poses."""
  force = env.scene["robot"].data.actuator_force
  return _stage_multiplier(env, multipliers) * torch.sum(torch.square(force), dim=-1)


def staged_joint_speed_excess_l2(
  env: ManagerBasedRlEnv,
  speed_limits: tuple[float, float, float, float] = (6.0, 5.0, 4.0, 3.5),
) -> torch.Tensor:
  """Penalize only joint speed above a stage-specific soft safety limit."""
  speed = torch.abs(env.scene["robot"].data.joint_vel)
  limit = speed.new_tensor(speed_limits)[_recovery_stage(env), None]
  return torch.sum(torch.square(torch.clamp(speed - limit, min=0.0)), dim=-1)


def staged_joint_power_excess_l2(
  env: ManagerBasedRlEnv,
  power_limits: tuple[float, float, float, float] = (140.0, 110.0, 90.0, 75.0),
) -> torch.Tensor:
  """Penalize burst mechanical power without suppressing static support torque."""
  robot = env.scene["robot"]
  power = torch.abs(robot.data.actuator_force * robot.data.joint_vel)
  limit = power.new_tensor(power_limits)[_recovery_stage(env), None]
  return torch.mean(torch.square(torch.clamp(power - limit, min=0.0)), dim=-1)


def recovery_initiation_progress(
  env: ManagerBasedRlEnv, relative_to_env_origin: bool = False
) -> torch.Tensor:
  """Bounded, ungated progress reward that prevents a stay-down local optimum."""
  head_z = _head_height(env, relative_to_env_origin=relative_to_env_origin)
  upright = upright_posture(env, power=1.0)
  height_progress = torch.clamp((head_z - 0.35) / 0.27, 0.0, 1.0)
  upright_progress = torch.clamp(upright / 0.60, 0.0, 1.0)
  lying_progress = 0.55 * height_progress + 0.45 * upright_progress
  return torch.where(
    _recovery_stage(env) == 0, lying_progress, torch.ones_like(lying_progress)
  )


def staged_recovery_pose(
  env: ManagerBasedRlEnv,
  height_scale: float = 8.0,
  upright_scale: float = 6.0,
  knee_scale: float = 5.0,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Reward the current seated, crouched, or standing waypoint.

  Height, uprightness, and knee-flexion shortfalls are penalized. The stage
  event requires each waypoint to be held at low vertical speed before exposing
  the next target.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  knee_ids = robot.find_joints(
    ["left_knee_joint", "right_knee_joint"], preserve_order=True
  )[0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  if relative_to_env_origin:
    head_z = head_z - env.scene.env_origins[:, 2]
  knee_flexion = robot.data.joint_pos[:, knee_ids].mean(dim=-1)
  upright = upright_posture(env, power=1.0)
  stage = _recovery_stage(env)
  target_height = head_z.new_tensor((0.62, 0.86, 1.15, 1.15))[stage]
  target_upright = upright.new_tensor((0.60, 0.76, 0.90, 0.90))[stage]
  target_knee = knee_flexion.new_tensor((1.00, 0.80, 0.0, 0.0))[stage]
  height_shortfall = torch.clamp(target_height - head_z, min=0.0)
  upright_shortfall = torch.clamp(target_upright - upright, min=0.0)
  knee_shortfall = torch.clamp(target_knee - knee_flexion, min=0.0)
  return torch.exp(
    -height_scale * torch.square(height_shortfall)
    - upright_scale * torch.square(upright_shortfall)
    - knee_scale * torch.square(knee_shortfall)
  )


def staged_head_velocity_profile(
  env: ManagerBasedRlEnv,
  scale: float = 45.0,
  overspeed_scale: float = 140.0,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Track a deliberately slow vertical speed for each recovery waypoint."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  if relative_to_env_origin:
    head_z = head_z - env.scene.env_origins[:, 2]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  stage = _recovery_stage(env)
  target_height = head_z.new_tensor((0.62, 0.86, 1.15, 1.15))[stage]
  max_velocity = head_z.new_tensor((0.06, 0.08, 0.10, 0.0))[stage]
  speed_limit = head_z.new_tensor((0.16, 0.18, 0.18, 0.12))[stage]
  remaining = torch.clamp((target_height - head_z) / 0.20, 0.0, 1.0)
  target_vz = max_velocity * remaining
  excess_speed = torch.clamp(torch.abs(head_vz) - speed_limit, min=0.0)
  return torch.exp(-scale * torch.square(head_vz - target_vz)) * torch.exp(
    -overspeed_scale * torch.square(excess_speed)
  )


def head_vertical_overspeed_l2(
  env: ManagerBasedRlEnv,
  speed_limit: float = 0.20,
) -> torch.Tensor:
  """Penalize upward or downward head speed beyond a conservative soft limit."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  excess = torch.clamp(torch.abs(head_vz) - speed_limit, min=0.0)
  return torch.square(excess)


def upright_posture(env: ManagerBasedRlEnv, power: float = 2.0) -> torch.Tensor:
  """Return a smooth [0, 1] score for torso uprightness."""
  gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
  return torch.clamp(-gravity_z, 0.0, 1.0).pow(power)


def low_base_angular_velocity(
  env: ManagerBasedRlEnv, scale: float = 0.5
) -> torch.Tensor:
  """Reward low world-frame base angular velocity."""
  ang_vel = env.scene["robot"].data.root_link_ang_vel_w
  return torch.exp(-scale * torch.sum(ang_vel * ang_vel, dim=-1))


def low_joint_velocity(env: ManagerBasedRlEnv, scale: float = 0.02) -> torch.Tensor:
  """Reward low mean squared joint velocity."""
  joint_vel = env.scene["robot"].data.joint_vel
  return torch.exp(-scale * torch.mean(joint_vel * joint_vel, dim=-1))


def smooth_action(env: ManagerBasedRlEnv, scale: float = 2.0) -> torch.Tensor:
  """Reward small policy-action changes to suppress high-frequency shaking."""
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.exp(-scale * torch.mean(delta * delta, dim=-1))


def quiet_stance_gate(
  env: ManagerBasedRlEnv,
  head_height_start: float = 0.95,
  head_height_full: float = 1.15,
  upright_start: float = 0.70,
  upright_full: float = 0.90,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Smoothly activate quiet-standing objectives outside push recovery windows."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  if relative_to_env_origin:
    head_z = head_z - env.scene.env_origins[:, 2]
  upright = upright_posture(env, power=1.0)
  height_width = max(head_height_full - head_height_start, 1e-6)
  upright_width = max(upright_full - upright_start, 1e-6)
  height_gate = torch.clamp((head_z - head_height_start) / height_width, 0.0, 1.0)
  upright_gate = torch.clamp((upright - upright_start) / upright_width, 0.0, 1.0)
  recovery = getattr(env, "_robust_push_recovery", None)
  knockdown_recovery = getattr(env, "_v5_knockdown_recovery", None)
  if recovery is None and knockdown_recovery is None:
    quiet = torch.ones(env.num_envs, device=env.device)
  else:
    quiet = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if recovery is not None:
      quiet &= recovery <= 0
    if knockdown_recovery is not None:
      quiet &= knockdown_recovery <= 0
    quiet = quiet.float()
  return height_gate * upright_gate * quiet


def feet_stationary_when_upright(
  env: ManagerBasedRlEnv,
  site_names: tuple[str, ...] = ("left_foot", "right_foot"),
  scale: float = 20.0,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Reward low planar foot speed only when the robot should stand quietly."""
  robot = env.scene["robot"]
  site_ids = robot.find_sites(list(site_names), preserve_order=True)[0]
  gate = quiet_stance_gate(env, relative_to_env_origin=relative_to_env_origin)
  foot_vel_xy = robot.data.site_lin_vel_w[:, site_ids, :2]
  speed_sq = torch.mean(torch.sum(torch.square(foot_vel_xy), dim=-1), dim=-1)
  return gate * torch.exp(-scale * speed_sq)


def base_stationary_when_upright(
  env: ManagerBasedRlEnv,
  scale: float = 8.0,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Reward low horizontal base velocity during quiet standing."""
  gate = quiet_stance_gate(env, relative_to_env_origin=relative_to_env_origin)
  vel_xy_sq = torch.sum(
    torch.square(env.scene["robot"].data.root_link_lin_vel_w[:, :2]), dim=-1
  )
  return gate * torch.exp(-scale * vel_xy_sq)


def terrain_planar_displacement_l2(
  env: ManagerBasedRlEnv,
  free_radius: float = 0.40,
  stage_multipliers: tuple[float, float, float, float] = (0.25, 0.50, 1.0, 1.0),
) -> torch.Tensor:
  """Penalize rolling or stepping far from the terrain reset origin.

  A free radius preserves the short translations needed to turn and establish
  hand/foot support.  The cost grows only outside that radius and becomes fully
  active once the policy reaches crouch/stand stages.
  """
  root_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  displacement = torch.linalg.vector_norm(
    root_xy - env.scene.env_origins[:, :2], dim=-1
  )
  excess = torch.clamp(displacement - free_radius, min=0.0)
  return _stage_multiplier(env, stage_multipliers) * torch.square(excess)


def terrain_foot_slip_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_foot_ground_contact",
  site_names: tuple[str, str] = ("left_foot", "right_foot"),
) -> torch.Tensor:
  """Penalize planar foot velocity only while the corresponding foot contacts."""
  found = env.scene[sensor_name].data.found
  if found is None:
    raise RuntimeError(f"{sensor_name} must expose the 'found' contact field")
  flat_found = found.reshape(env.num_envs, -1) > 0
  split = max(flat_found.shape[1] // 2, 1)
  left = flat_found[:, :split].any(dim=-1)
  right = flat_found[:, split:].any(dim=-1)
  if flat_found.shape[1] == 1:
    right = left
  contact = torch.stack((left, right), dim=-1).float()
  robot = env.scene["robot"]
  site_ids = robot.find_sites(list(site_names), preserve_order=True)[0]
  speed_sq = torch.sum(torch.square(robot.data.site_lin_vel_w[:, site_ids, :2]), dim=-1)
  return torch.sum(contact * speed_sq, dim=-1)


def terrain_stance_width_excess_l2(
  env: ManagerBasedRlEnv,
  max_width: float = 0.65,
  site_names: tuple[str, str] = ("left_foot", "right_foot"),
) -> torch.Tensor:
  """Apply a mild wide-stance cost only after a terrain-relative stand."""
  robot = env.scene["robot"]
  site_ids = robot.find_sites(list(site_names), preserve_order=True)[0]
  feet_xy = robot.data.site_pos_w[:, site_ids, :2]
  width = torch.linalg.vector_norm(feet_xy[:, 0] - feet_xy[:, 1], dim=-1)
  excess = torch.clamp(width - max_width, min=0.0)
  gate = quiet_stance_gate(env, relative_to_env_origin=True)
  return gate * torch.square(excess)


def _cached_score(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  value = getattr(env, name, None)
  if value is None:
    return torch.zeros(env.num_envs, device=env.device)
  return value


def cached_task_score(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _cached_score(env, "_smp_task_score")


def cached_smp_score(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _cached_score(env, "_smp_score")


def cached_product_score(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _cached_score(env, "_smp_product_score")


def cached_raw_smp_score(env: ManagerBasedRlEnv, ws: float = 6.0) -> torch.Tensor:
  raw_err = getattr(env, "_smp_raw_err", None)
  if raw_err is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.exp(-ws * raw_err)


def procedural_reset_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  reset_type = getattr(env, "_robust_reset_type", None)
  if reset_type is None:
    return torch.zeros(env.num_envs, device=env.device)
  return ((reset_type >= 1) & (reset_type <= 4)).float()


def prone_reset_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one for the procedural prone reset mode."""
  reset_type = getattr(env, "_robust_reset_type", None)
  if reset_type is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (reset_type == 1).float()


def supine_reset_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one for the procedural supine reset mode."""
  reset_type = getattr(env, "_robust_reset_type", None)
  if reset_type is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (reset_type == 2).float()


def recovery_stage_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Normalized ordered recovery stage: lying=0 through standing=1."""
  return _recovery_stage(env).float() / 3.0


def recovery_stage_complete_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one after the stable standing waypoint has been held."""
  return (_recovery_stage(env) == 3).float()


def mean_knee_flexion_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean left/right knee flexion in radians."""
  robot = env.scene["robot"]
  knee_ids = robot.find_joints(
    ["left_knee_joint", "right_knee_joint"], preserve_order=True
  )[0]
  return robot.data.joint_pos[:, knee_ids].mean(dim=-1)


def stable_stand_metric(
  env: ManagerBasedRlEnv,
  head_height: float = 1.2,
  min_upright: float = 0.85,
  max_linear_speed: float = 0.5,
  max_angular_speed: float = 0.5,
  relative_to_env_origin: bool = False,
) -> torch.Tensor:
  """Return one only for a tall, upright, low-motion standing state."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  if relative_to_env_origin:
    head_z = head_z - env.scene.env_origins[:, 2]
  lin_speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)
  ang_speed = torch.linalg.norm(robot.data.root_link_ang_vel_w, dim=-1)
  upright = upright_posture(env, power=1.0)
  return (
    (head_z >= head_height)
    & (upright >= min_upright)
    & (lin_speed < max_linear_speed)
    & (ang_speed < max_angular_speed)
  ).float()


def active_wrench_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  active = getattr(env, "_robust_push_active", None)
  if active is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (active > 0).float()


def post_stand_knockdown_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one while the targeted V5 knockdown wrench is active."""
  active = getattr(env, "_v5_knockdown_active", None)
  if active is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (active > 0).float()


def head_vertical_speed_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Signed world-frame vertical speed of the head site."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  return robot.data.site_lin_vel_w[:, head_idx, 2]


def mean_foot_speed_metric(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Mean planar speed across the configured foot sites."""
  foot_vel_xy = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return torch.linalg.vector_norm(foot_vel_xy, dim=-1).mean(dim=-1)


def max_joint_speed_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Maximum absolute joint speed in each environment."""
  return torch.max(torch.abs(env.scene["robot"].data.joint_vel), dim=-1).values


def max_joint_torque_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Maximum absolute actuator torque in each environment."""
  force = env.scene["robot"].data.actuator_force
  return torch.max(torch.abs(force), dim=-1).values


def max_joint_power_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Maximum absolute mechanical joint power in each environment."""
  robot = env.scene["robot"]
  power = robot.data.actuator_force * robot.data.joint_vel
  return torch.max(torch.abs(power), dim=-1).values


def joint_acc_rms_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """RMS joint acceleration in each environment."""
  joint_acc = env.scene["robot"].data.joint_acc
  return torch.sqrt(torch.mean(torch.square(joint_acc), dim=-1))


def action_rate_rms_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """RMS change between consecutive policy actions."""
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sqrt(torch.mean(torch.square(delta), dim=-1))


def failure_replay_reset_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one for resets sampled from the V6 hard-state replay ring."""
  reset_type = getattr(env, "_robust_reset_type", None)
  if reset_type is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (reset_type == 5).float()


def failure_buffer_fill_metric(
  env: ManagerBasedRlEnv, capacity: int = 8192
) -> torch.Tensor:
  """Fraction of the V6 hard-state replay ring currently populated."""
  fill = float(getattr(env, "_v6_failure_size", 0)) / max(capacity, 1)
  return torch.full((env.num_envs,), fill, device=env.device)


def v6_active_wrench_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  active = getattr(env, "_v6_push_active", None)
  if active is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (active > 0).float()


def v6_push_cohort_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Normalized cohort: clean=0, standard=0.5, intensive=1."""
  cohort = getattr(env, "_v6_push_cohort", None)
  if cohort is None:
    return torch.zeros(env.num_envs, device=env.device)
  return cohort.float() / 2.0


def v6_push_count_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  count = getattr(env, "_v6_push_count", None)
  if count is None:
    return torch.zeros(env.num_envs, device=env.device)
  return count.float()


def constraint_active_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one while a sampled sustained constraint is being applied."""
  remaining = getattr(env, "_constraint_remaining", None)
  wait = getattr(env, "_constraint_wait", None)
  if remaining is None or wait is None:
    return torch.zeros(env.num_envs, device=env.device)
  return ((wait <= 0) & (remaining > 0)).float()


def constraint_cohort_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Normalized cohort: clean=0, trunk=0.5, limb=1."""
  cohort = getattr(env, "_constraint_cohort", None)
  if cohort is None:
    return torch.zeros(env.num_envs, device=env.device)
  return cohort.float() / 2.0


def constraint_load_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sampled downward load magnitude in newtons (zero for clean episodes)."""
  magnitude = getattr(env, "_constraint_force_magnitude", None)
  if magnitude is None:
    return torch.zeros(env.num_envs, device=env.device)
  return magnitude


def constraint_release_progress_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fraction of the sampled constraint duration that has elapsed."""
  remaining = getattr(env, "_constraint_remaining", None)
  duration = getattr(env, "_constraint_duration", None)
  cohort = getattr(env, "_constraint_cohort", None)
  if remaining is None or duration is None or cohort is None:
    return torch.zeros(env.num_envs, device=env.device)
  elapsed = 1.0 - remaining.float() / torch.clamp(duration.float(), min=1.0)
  return torch.where(cohort > 0, elapsed, torch.zeros_like(elapsed))


def escape_phase_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Normalized phase; invalid-contact episodes are reported as -1."""
  phase = getattr(env, "_escape_phase", None)
  if phase is None:
    return torch.zeros(env.num_envs, device=env.device)
  normalized = phase.float() / 3.0
  return torch.where(phase == 4, -torch.ones_like(normalized), normalized)


def escape_object_displacement_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Planar obstacle displacement from reset, in metres."""
  start = getattr(env, "_escape_start_obstacle_xy", None)
  if start is None:
    return torch.zeros(env.num_envs, device=env.device)
  current = env.scene["escape_obstacle"].data.root_link_pos_w[:, :2]
  return torch.linalg.vector_norm(current - start, dim=-1)


def escape_covered_geom_count_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Current number of robot collision geoms covered by the plate footprint."""
  count = getattr(env, "_escape_covered_geom_count", None)
  if count is None:
    return torch.zeros(env.num_envs, device=env.device)
  return count.float()


def escape_best_covered_geom_count_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Smallest covered collision-geom count achieved in the episode."""
  count = getattr(env, "_escape_best_covered_geom_count", None)
  if count is None:
    return torch.zeros(env.num_envs, device=env.device)
  return count.float()


def escape_planar_clearance_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Minimum all-body planar clearance outside the plate footprint, in metres."""
  clearance = getattr(env, "_escape_planar_clearance", None)
  if clearance is None:
    return torch.zeros(env.num_envs, device=env.device)
  return clearance


def escape_plate_mass_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-world guided-plate mass in kilograms."""
  obstacle = env.scene["escape_obstacle"]
  local_ids, _ = obstacle.find_bodies(["escape_plate"], preserve_order=True)
  if len(local_ids) != 1:
    return torch.zeros(env.num_envs, device=env.device)
  local = torch.tensor(local_ids, dtype=torch.long, device=env.device)
  body_id = obstacle.indexing.body_ids[local][0].long()
  return env.sim.model.body_mass[:, body_id]


def escape_obstacle_episode_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one for episodes initialized with the physical obstacle."""
  phase = getattr(env, "_escape_phase", None)
  if phase is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (phase > 0).float()


def escape_invalid_contact_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One when penetration/force exceeded the configured validity envelope."""
  invalid = getattr(env, "_escape_invalid_contact", None)
  if invalid is None:
    return torch.zeros(env.num_envs, device=env.device)
  return invalid.float()


def escape_invalid_setup_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One when the plate failed to load the robot promptly in a low pose."""
  invalid = getattr(env, "_escape_invalid_setup", None)
  if invalid is None:
    return torch.zeros(env.num_envs, device=env.device)
  return invalid.float()


def escape_first_contact_head_height_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Head height at first plate contact; -1 means contact never occurred."""
  height = getattr(env, "_escape_first_contact_head_height", None)
  if height is None:
    return -torch.ones(env.num_envs, device=env.device)
  return height


def escape_hand_support_steps_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Number of constrained control steps with at least one supporting hand."""
  steps = getattr(env, "_escape_hand_support_steps", None)
  if steps is None:
    return torch.zeros(env.num_envs, device=env.device)
  return steps.float()


def escape_hand_supported_progress_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """New planar separation accumulated while a hand supported on the ground."""
  progress = getattr(env, "_escape_hand_supported_progress", None)
  if progress is None:
    return torch.zeros(env.num_envs, device=env.device)
  return progress


def escape_peak_penetration_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Largest robot-plate penetration observed in the episode, in metres."""
  peak = getattr(env, "_escape_peak_penetration", None)
  if peak is None:
    return torch.zeros(env.num_envs, device=env.device)
  return peak


def escape_peak_contact_force_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Largest robot-plate contact-force norm observed in the episode, in newtons."""
  peak = getattr(env, "_escape_peak_contact_force", None)
  if peak is None:
    return torch.zeros(env.num_envs, device=env.device)
  return peak


def hand_support_contact_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str = "hand_ground_contact",
) -> torch.Tensor:
  """Fraction of left/right hands currently supporting on the ground."""
  return _contact_found(env, sensor_name).float().mean(dim=-1)
