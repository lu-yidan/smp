"""Sagittal symmetry transforms for G1 PPO augmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

# Legacy deploy frame includes a leading zero/estimated base linear velocity.
FRAME_DIM = 96
NO_LINEAR_VELOCITY_FRAME_DIM = 93
JOINT_DIM = 29


def _joint_mirror_spec(
  joint_names: tuple[str, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return the involutive left/right permutation and coordinate signs."""
  name_to_index = {name: i for i, name in enumerate(joint_names)}
  indices: list[int] = []
  signs: list[float] = []
  for name in joint_names:
    source = name
    if name.startswith("left_"):
      source = "right_" + name.removeprefix("left_")
    elif name.startswith("right_"):
      source = "left_" + name.removeprefix("right_")
    if source not in name_to_index:
      raise RuntimeError(f"missing mirrored joint for {name}: {source}")
    indices.append(name_to_index[source])
    signs.append(-1.0 if name.endswith(("_roll_joint", "_yaw_joint")) else 1.0)
  if len(indices) != JOINT_DIM or sorted(indices) != list(range(JOINT_DIM)):
    raise RuntimeError("G1 mirror mapping must be a 29-joint permutation")
  return (
    torch.tensor(indices, dtype=torch.long, device=device),
    torch.tensor(signs, dtype=torch.float, device=device),
  )


def mirror_g1_actor_tensor(
  obs: torch.Tensor,
  joint_indices: torch.Tensor,
  joint_signs: torch.Tensor,
) -> torch.Tensor:
  """Mirror term-wise flattened deployment observations for any history length."""
  obs_dim = obs.shape[-1]
  if obs_dim % FRAME_DIM == 0:
    frame_dim = FRAME_DIM
    vector_specs = (
      ((1.0, -1.0, 1.0), "linear velocity"),
      ((-1.0, 1.0, -1.0), "angular velocity"),
      ((1.0, -1.0, 1.0), "projected gravity"),
    )
  elif obs_dim % NO_LINEAR_VELOCITY_FRAME_DIM == 0:
    frame_dim = NO_LINEAR_VELOCITY_FRAME_DIM
    vector_specs = (
      ((-1.0, 1.0, -1.0), "angular velocity"),
      ((1.0, -1.0, 1.0), "projected gravity"),
    )
  else:
    raise ValueError(
      "expected a multiple of either "
      f"{FRAME_DIM} or {NO_LINEAR_VELOCITY_FRAME_DIM} observation dimensions, "
      f"got {obs_dim}"
    )
  history = obs_dim // frame_dim
  result = obs.clone()
  offset = 0
  # Linear velocity and gravity are polar vectors. Angular velocity is axial.
  for signs, _name in vector_specs:
    dim = 3
    size = history * dim
    term = obs[..., offset : offset + size].reshape(*obs.shape[:-1], history, dim)
    sign = torch.tensor(signs, dtype=obs.dtype, device=obs.device)
    result[..., offset : offset + size] = (term * sign).reshape(*obs.shape[:-1], size)
    offset += size

  # Joint position, joint velocity, and previous action share the same layout.
  for _ in range(3):
    size = history * JOINT_DIM
    term = obs[..., offset : offset + size].reshape(*obs.shape[:-1], history, JOINT_DIM)
    mirrored = term[..., joint_indices] * joint_signs
    result[..., offset : offset + size] = mirrored.reshape(*obs.shape[:-1], size)
    offset += size
  if offset != obs.shape[-1]:
    raise RuntimeError("G1 mirror did not consume every observation dimension")
  return result


def mirror_g1_actions(
  actions: torch.Tensor,
  joint_indices: torch.Tensor,
  joint_signs: torch.Tensor,
) -> torch.Tensor:
  if actions.shape[-1] != JOINT_DIM:
    raise ValueError(f"expected {JOINT_DIM} actions, got {actions.shape[-1]}")
  return actions[..., joint_indices] * joint_signs


def g1_sagittal_data_augmentation(
  env: VecEnv,
  obs: TensorDict | None,
  actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append original and sagittally mirrored samples for RSL-RL symmetry."""
  robot = env.unwrapped.scene["robot"]
  joint_indices, joint_signs = _joint_mirror_spec(robot.joint_names, env.device)

  augmented_obs = None
  if obs is not None:
    mirrored_obs = obs.clone()
    for key, value in obs.items():
      mirrored_obs[key] = mirror_g1_actor_tensor(value, joint_indices, joint_signs)
    augmented_obs = torch.cat((obs, mirrored_obs), dim=0)

  augmented_actions = None
  if actions is not None:
    mirrored_actions = mirror_g1_actions(actions, joint_indices, joint_signs)
    augmented_actions = torch.cat((actions, mirrored_actions), dim=0)

  return augmented_obs, augmented_actions


__all__ = [
  "FRAME_DIM",
  "JOINT_DIM",
  "NO_LINEAR_VELOCITY_FRAME_DIM",
  "g1_sagittal_data_augmentation",
  "mirror_g1_actions",
  "mirror_g1_actor_tensor",
]
