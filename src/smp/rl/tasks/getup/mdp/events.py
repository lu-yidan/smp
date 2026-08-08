"""Reset events for the getup task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

__all__ = [
  "mixed_fall_reset",
  "random_body_wrench",
  "reset_stand_counter",
]


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

  reset_types = getattr(env, "_robust_reset_type", None)
  if reset_types is None:
    reset_types = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._robust_reset_type = reset_types  # type: ignore[attr-defined]
  reset_types[env_ids] = 0

  # Clear any active wrench and reschedule the next push for reset environments.
  if hasattr(env, "_robust_push_active"):
    env._robust_push_active[env_ids] = 0  # type: ignore[attr-defined]
    env._robust_push_wait[env_ids] = torch.randint(  # type: ignore[attr-defined]
      50, 151, (env_ids.numel(),), device=env.device
    )
    env._robust_forces[env_ids] = 0.0  # type: ignore[attr-defined]
    env._robust_torques[env_ids] = 0.0  # type: ignore[attr-defined]

  choose = torch.rand(env_ids.numel(), device=env.device) < procedural_probability
  fall_ids = env_ids[choose]
  if fall_ids.numel() == 0:
    return

  robot = env.scene["robot"]
  n = fall_ids.numel()
  modes = torch.randint(0, 4, (n,), device=env.device)
  roll = torch.zeros(n, device=env.device)
  pitch = torch.zeros(n, device=env.device)
  roll = torch.where(modes == 2, torch.full_like(roll, torch.pi / 2), roll)
  roll = torch.where(modes == 3, torch.full_like(roll, -torch.pi / 2), roll)
  pitch = torch.where(modes == 0, torch.full_like(pitch, torch.pi / 2), pitch)
  pitch = torch.where(modes == 1, torch.full_like(pitch, -torch.pi / 2), pitch)
  yaw = torch.empty(n, device=env.device).uniform_(-torch.pi, torch.pi)

  default_root = robot.data.default_root_state[fall_ids].clone()
  delta_quat = quat_from_euler_xyz(roll, pitch, yaw)
  quat = quat_mul(default_root[:, 3:7], delta_quat)
  origins = env.scene.env_origins[fall_ids]
  xy = torch.empty(n, 2, device=env.device).uniform_(-0.1, 0.1)
  height = torch.empty(n, 1, device=env.device).uniform_(*root_height_range)
  pos = torch.cat([xy, height], dim=-1) + origins
  lin_vel = torch.empty(n, 3, device=env.device).uniform_(-0.1, 0.1)
  ang_vel = torch.empty(n, 3, device=env.device).uniform_(-0.2, 0.2)
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
  buffer = env._smp_buffer  # type: ignore[attr-defined]
  window_size = buffer.window_size
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  root_pos = robot.data.root_link_pos_w[fall_ids] - origins
  ee_pos = robot.data.body_link_pos_w[fall_ids][:, ee_indexes] - origins[:, None, :]
  buffer.reset(
    fall_ids,
    root_pos[:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_quat_w[fall_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_lin_vel_w[fall_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.root_link_ang_vel_w[fall_ids][:, None, :].expand(-1, window_size, -1),
    ee_pos[:, None, :, :].expand(-1, window_size, -1, -1),
    robot.data.joint_pos[fall_ids][:, None, :].expand(-1, window_size, -1),
    robot.data.joint_vel[fall_ids][:, None, :].expand(-1, window_size, -1),
  )
  reset_types[fall_ids] = modes + 1


@torch.no_grad()
def random_body_wrench(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  body_names: tuple[str, ...] = ("pelvis", "torso_link"),
  interval_steps: tuple[int, int] = (50, 150),
  duration_steps: tuple[int, int] = (5, 15),
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
    env._robust_forces = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, num_bodies, 3, device=env.device
    )
    env._robust_torques = torch.zeros_like(  # type: ignore[attr-defined]
      env._robust_forces  # type: ignore[attr-defined]
    )

  wait = env._robust_push_wait  # type: ignore[attr-defined]
  active = env._robust_push_active  # type: ignore[attr-defined]
  forces = env._robust_forces  # type: ignore[attr-defined]
  torques = env._robust_torques  # type: ignore[attr-defined]
  body_ids = env._robust_force_body_ids  # type: ignore[attr-defined]

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
    active[start_ids] = torch.randint(
      duration_steps[0],
      duration_steps[1] + 1,
      (start_ids.numel(),),
      device=env.device,
    )

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
