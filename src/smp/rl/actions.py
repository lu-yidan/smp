"""Deployable action transforms shared by SMP training and deployment.

The limiter in this module acts on joint-position targets at the policy rate.  It
does not use privileged state and therefore has an exact real-robot analogue.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

__all__ = [
  "RateLimitedJointPositionAction",
  "RateLimitedJointPositionActionCfg",
  "bounded_target_step",
]


def bounded_target_step(
  current_target: torch.Tensor,
  current_velocity: torch.Tensor,
  desired_target: torch.Tensor,
  *,
  dt: float,
  max_velocity: float,
  max_acceleration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Advance a position target with hard discrete velocity/acceleration bounds."""
  if dt <= 0.0 or max_velocity <= 0.0 or max_acceleration <= 0.0:
    raise ValueError("dt and target envelope limits must be positive")
  desired_velocity = torch.clamp(
    (desired_target - current_target) / dt,
    min=-max_velocity,
    max=max_velocity,
  )
  velocity_delta = torch.clamp(
    desired_velocity - current_velocity,
    min=-max_acceleration * dt,
    max=max_acceleration * dt,
  )
  next_velocity = torch.clamp(
    current_velocity + velocity_delta,
    min=-max_velocity,
    max=max_velocity,
  )
  return current_target + next_velocity * dt, next_velocity


class RateLimitedJointPositionAction(JointPositionAction):
  """Joint-position action with a stateful policy-rate target envelope."""

  cfg: RateLimitedJointPositionActionCfg

  def __init__(
    self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg=cfg, env=env)
    self._step_dt = float(env.step_dt)
    self._target_velocity = torch.zeros_like(self._processed_actions)
    self._target_acceleration = torch.zeros_like(self._processed_actions)
    self._desired_target = self._processed_actions.clone()
    self._limited_fraction = torch.zeros(self.num_envs, device=self.device)

  @property
  def target_velocity(self) -> torch.Tensor:
    return self._target_velocity

  @property
  def target_acceleration(self) -> torch.Tensor:
    return self._target_acceleration

  @property
  def desired_target(self) -> torch.Tensor:
    return self._desired_target

  @property
  def limited_fraction(self) -> torch.Tensor:
    return self._limited_fraction

  def process_actions(self, actions: torch.Tensor) -> None:
    previous_target = self._processed_actions.clone()
    previous_velocity = self._target_velocity.clone()
    super().process_actions(actions)
    self._desired_target[:] = self._processed_actions
    next_target, next_velocity = bounded_target_step(
      previous_target,
      previous_velocity,
      self._desired_target,
      dt=self._step_dt,
      max_velocity=self.cfg.max_target_velocity,
      max_acceleration=self.cfg.max_target_acceleration,
    )
    self._processed_actions[:] = next_target
    self._target_velocity[:] = next_velocity
    self._target_acceleration[:] = (
      next_velocity - previous_velocity
    ) / self._step_dt
    limited = torch.abs(next_target - self._desired_target) > 1.0e-6
    self._limited_fraction[:] = limited.float().mean(dim=-1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids=env_ids)
    # The first post-reset command begins at the measured pose rather than
    # jumping from the nominal standing offset.  Encoder bias is added here
    # because JointPositionAction subtracts it immediately before application.
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    reset_target = self._entity.data.joint_pos[:, self._target_ids] + encoder_bias
    self._processed_actions[env_ids] = reset_target[env_ids]
    self._desired_target[env_ids] = reset_target[env_ids]
    self._target_velocity[env_ids] = 0.0
    self._target_acceleration[env_ids] = 0.0
    self._limited_fraction[env_ids] = 0.0


@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
  """Configuration for a deployable joint-target velocity envelope."""

  max_target_velocity: float = 4.0
  max_target_acceleration: float = 30.0

  def build(self, env: ManagerBasedRlEnv) -> RateLimitedJointPositionAction:
    return RateLimitedJointPositionAction(self, env)
