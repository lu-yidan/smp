"""Reward components for the getup task: head-height + up-velocity.

Combined and SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "active_wrench_metric",
  "cached_product_score",
  "cached_raw_smp_score",
  "cached_smp_score",
  "cached_task_score",
  "low_base_angular_velocity",
  "low_joint_velocity",
  "procedural_reset_metric",
  "smooth_action",
  "stable_stand_metric",
  "track_head_height",
  "upright_posture",
  "upward_velocity",
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
  return (reset_type > 0).float()


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
