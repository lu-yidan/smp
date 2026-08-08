"""Sparse-keyframe command built on MJLab's dense tracking motion loader."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


class SparseKeyframeCommand(MotionCommand):
  """Initialize from dense frames but expose the next sparse frame as the goal."""

  cfg: SparseKeyframeCommandCfg

  def __init__(self, cfg: SparseKeyframeCommandCfg, env):
    self._dense_reference = True
    super().__init__(cfg, env)
    with np.load(cfg.motion_file) as data:
      if "keyframe_indices" not in data:
        msg = f"{cfg.motion_file} does not contain keyframe_indices"
        raise ValueError(msg)
      keyframes = np.asarray(data["keyframe_indices"], dtype=np.int64)
    if keyframes.ndim != 1 or len(keyframes) < 2:
      msg = "keyframe_indices must be a one-dimensional array with at least two items"
      raise ValueError(msg)
    if np.any(np.diff(keyframes) <= 0):
      msg = "keyframe_indices must be strictly increasing"
      raise ValueError(msg)
    if keyframes[0] < 0 or keyframes[-1] >= self.motion.time_step_total:
      msg = (
        f"keyframe range [{keyframes[0]}, {keyframes[-1]}] is invalid for "
        f"{self.motion.time_step_total} motion frames"
      )
      raise ValueError(msg)
    self.keyframe_indices = torch.as_tensor(
      keyframes, dtype=torch.long, device=self.device
    )
    self.goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._dense_reference = False
    self._set_goal_steps()

  def _reference_steps(self) -> torch.Tensor:
    return self.time_steps if self._dense_reference else self.goal_steps

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self._reference_steps()]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self._reference_steps()]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self._reference_steps()]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self._reference_steps()]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self._reference_steps()]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self._reference_steps()]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  def _set_goal_steps(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    next_indices = torch.searchsorted(
      self.keyframe_indices, self.time_steps[ids], right=True
    )
    next_indices.clamp_(max=len(self.keyframe_indices) - 1)
    self.goal_steps[ids] = self.keyframe_indices[next_indices]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._dense_reference = True
    try:
      super()._resample_command(env_ids)
    finally:
      self._dense_reference = False
    self._set_goal_steps(env_ids)
    self.update_relative_body_poses()

  def _update_command(self) -> None:
    self.time_steps.add_(1)
    self.time_steps.clamp_(max=self.motion.time_step_total - 1)
    self._set_goal_steps()
    self.update_relative_body_poses()

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    self._dense_reference = True
    try:
      super().reset_to_frame(env_ids, frame)
    finally:
      self._dense_reference = False
    self._set_goal_steps(env_ids)
    self.update_relative_body_poses()


@dataclass(kw_only=True)
class SparseKeyframeCommandCfg(MotionCommandCfg):
  """Configuration that constructs :class:`SparseKeyframeCommand`."""

  def build(self, env) -> SparseKeyframeCommand:
    return SparseKeyframeCommand(self, env)
