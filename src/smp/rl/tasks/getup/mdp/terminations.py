"""Termination terms for the getup task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "invalid_escape_contact",
  "invalid_escape_episode",
  "smp_too_low",
  "stood_up",
  "terrain_patch_exit",
  "unstable_sim_state",
]


def unstable_sim_state(
  env: ManagerBasedRlEnv,
  max_abs_qpos: float = 1.0e3,
  max_abs_qvel: float = 1.0e3,
  max_abs_qacc: float = 1.0e6,
) -> torch.Tensor:
  """Reset only environments whose raw MuJoCo state has become unsafe.

  This term runs before observations are recomputed.  It therefore prevents a
  single rare contact explosion from putting NaNs into the actor observation
  and aborting an otherwise healthy many-environment training run.
  """
  data = env.sim.data
  invalid = ~torch.isfinite(data.qpos).all(dim=-1)
  invalid |= ~torch.isfinite(data.qvel).all(dim=-1)
  invalid |= ~torch.isfinite(data.qacc).all(dim=-1)
  invalid |= ~torch.isfinite(data.qacc_warmstart).all(dim=-1)
  invalid |= ~torch.isfinite(data.sensordata).all(dim=-1)
  invalid |= data.qpos.abs().amax(dim=-1) > max_abs_qpos
  invalid |= data.qvel.abs().amax(dim=-1) > max_abs_qvel
  invalid |= data.qacc.abs().amax(dim=-1) > max_abs_qacc
  invalid |= data.qacc_warmstart.abs().amax(dim=-1) > max_abs_qacc
  return invalid


def invalid_escape_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate samples whose obstacle contact is physically invalid."""
  invalid = getattr(env, "_escape_invalid_contact", None)
  if invalid is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return invalid


def invalid_escape_episode(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate either physically invalid contact or an invalid plate setup."""
  contact = getattr(env, "_escape_invalid_contact", None)
  setup = getattr(env, "_escape_invalid_setup", None)
  if contact is None and setup is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if contact is None:
    return setup
  if setup is None:
    return contact
  return contact | setup


def stood_up(
  env: ManagerBasedRlEnv,
  head_height: float = 1.2,
  max_speed: float = 0.5,
  hold_steps: int = 10,
  min_upright: float = 0.0,
  max_angular_speed: float = float("inf"),
  relative_to_env_origin: bool = False,
  max_origin_distance: float = float("inf"),
) -> torch.Tensor:
  """Truncate once a stable standing condition holds for consecutive steps.

  The optional upright and angular-speed checks let robust tasks reject tall but
  tilted or visibly shaking poses. Defaults preserve the baseline task behavior.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  if relative_to_env_origin:
    reference = getattr(env, "_terrain_reset_support_height", None)
    if reference is None:
      reference = env.scene.env_origins[:, 2]
    z = z - reference
  speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)
  angular_speed = torch.linalg.norm(robot.data.root_link_ang_vel_w, dim=-1)
  upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
  origin_distance = torch.linalg.vector_norm(
    robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=-1
  )
  is_standing = (
    (z >= head_height)
    & (speed < max_speed)
    & (upright >= min_upright)
    & (angular_speed < max_angular_speed)
    & (origin_distance <= max_origin_distance)
  )
  cnt = getattr(env, "_getup_stand_count", None)
  if cnt is None:
    cnt = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  cnt = torch.where(is_standing, cnt + 1, torch.zeros_like(cnt))
  env._getup_stand_count = cnt  # type: ignore[attr-defined]
  return cnt >= hold_steps


def terrain_patch_exit(
  env: ManagerBasedRlEnv,
  margin: float = 0.50,
) -> torch.Tensor:
  """Terminate after leaving the assigned terrain patch, not the whole grid."""
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  half_x = 0.5 * terrain.cfg.terrain_generator.size[0] - margin
  half_y = 0.5 * terrain.cfg.terrain_generator.size[1] - margin
  displacement = (
    env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()
  outside = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)
  return outside & (env.episode_length_buf > 2)


def smp_too_low(
  env: ManagerBasedRlEnv,
  threshold: float = 0.02,
  ws: float = 6.0,
  grace_steps: int = 15,
) -> torch.Tensor:
  """Terminate when the SMP score collapses (off-manifold): end if
  ``exp(-ws·env._smp_raw_err) < threshold`` past ``grace_steps``.  Uses the RAW MSE
  (stable absolute scale), so ``ws`` must match the reward's.  Kills the "violent
  get-up" shortcut — leaving the manifold drives the score to 0."""
  raw_err = getattr(env, "_smp_raw_err", None)
  if raw_err is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  raw_smp = torch.exp(-ws * raw_err)
  past_grace = env.episode_length_buf >= grace_steps
  return (raw_smp < threshold) & past_grace
