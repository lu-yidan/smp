"""Reward components for the getup task: head-height + up-velocity.

Combined and SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

__all__ = [
  "active_wrench_metric",
  "action_rate_rms_metric",
  "base_stationary_when_upright",
  "cached_product_score",
  "cached_raw_smp_score",
  "failure_buffer_fill_metric",
  "failure_replay_reset_metric",
  "recovery_initiation_progress",
  "cached_smp_score",
  "cached_task_score",
  "feet_stationary_when_upright",
  "head_vertical_speed_metric",
  "head_vertical_overspeed_l2",
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
  "track_head_height",
  "track_head_velocity_profile",
  "upright_posture",
  "upward_velocity",
  "v6_active_wrench_metric",
  "v6_push_cohort_metric",
  "v6_push_count_metric",
]


def track_head_height(
  env: ManagerBasedRlEnv,
  target_height: float = 1.2,
  scale: float = 6.0,
) -> torch.Tensor:
  """Reward the ``head`` site reaching ``target_height``:
  ``exp(-scale·max(target_height − head_z, 0)²)`` (no penalty for overshoot).
  Needs the ``head`` site from ``getup_env_cfg.get_g1_spec_with_head``."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
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


def recovery_initiation_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Bounded, ungated progress reward that prevents a stay-down local optimum."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
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
) -> torch.Tensor:
  """Track a deliberately slow vertical speed for each recovery waypoint."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
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
) -> torch.Tensor:
  """Smoothly activate quiet-standing objectives outside push recovery windows."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
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
) -> torch.Tensor:
  """Reward low planar foot speed only when the robot should stand quietly."""
  robot = env.scene["robot"]
  site_ids = robot.find_sites(list(site_names), preserve_order=True)[0]
  gate = quiet_stance_gate(env)
  foot_vel_xy = robot.data.site_lin_vel_w[:, site_ids, :2]
  speed_sq = torch.mean(torch.sum(torch.square(foot_vel_xy), dim=-1), dim=-1)
  return gate * torch.exp(-scale * speed_sq)


def base_stationary_when_upright(
  env: ManagerBasedRlEnv,
  scale: float = 8.0,
) -> torch.Tensor:
  """Reward low horizontal base velocity during quiet standing."""
  gate = quiet_stance_gate(env)
  vel_xy_sq = torch.sum(
    torch.square(env.scene["robot"].data.root_link_lin_vel_w[:, :2]), dim=-1
  )
  return gate * torch.exp(-scale * vel_xy_sq)


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
) -> torch.Tensor:
  """Return one only for a tall, upright, low-motion standing state."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
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
