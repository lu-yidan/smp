"""Deterministic one-step action actor for diagnosing FIRM diffusion failures."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from smp.firm.action_diffusion import ACTION_DIM, JOINT_DIM, OBSERVATION_DIM


class _Encoder(nn.Module):
  def __init__(self, input_dim: int, output_dim: int) -> None:
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.SiLU(),
      nn.LayerNorm(output_dim),
    )

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    return self.network(value)


class FirmDeterministicActor(nn.Module):
  """Predict the next normalized expert action from state and relative goal."""

  def __init__(
    self,
    observation_dim: int = OBSERVATION_DIM,
    action_dim: int = ACTION_DIM,
    observation_latent_dim: int = 256,
    goal_latent_dim: int = 128,
    hidden_dims: tuple[int, ...] = (512, 512, 256),
  ) -> None:
    super().__init__()
    self.observation_dim = observation_dim
    self.action_dim = action_dim
    self.observation_encoder = _Encoder(observation_dim, observation_latent_dim)
    self.goal_encoder = _Encoder(JOINT_DIM, goal_latent_dim)
    layers: list[nn.Module] = []
    input_dim = observation_latent_dim + goal_latent_dim
    for hidden_dim in hidden_dims:
      layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU()))
      input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, action_dim))
    self.action_head = nn.Sequential(*layers)

  def forward(
    self,
    observation: torch.Tensor,
    current_joint_position: torch.Tensor,
    goal_joint_position: torch.Tensor,
  ) -> torch.Tensor:
    state = self.observation_encoder(observation)
    current_latent = self.goal_encoder(current_joint_position)
    goal_latent = self.goal_encoder(goal_joint_position)
    return self.action_head(torch.cat((state, goal_latent - current_latent), dim=-1))


def load_deterministic_actor_checkpoint(
  checkpoint_file: str | Path,
  device: str | torch.device,
) -> tuple[FirmDeterministicActor, dict[str, torch.Tensor], dict]:
  """Load a deterministic actor and its normalization tensors."""
  checkpoint_path = Path(checkpoint_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
  resolved_device = torch.device(device)
  payload = torch.load(
    checkpoint_path, map_location=resolved_device, weights_only=False
  )
  config = payload["config"]
  model = FirmDeterministicActor(
    observation_dim=int(
      config.get(
        "observation_dim", payload["normalization"]["observation_mean"].numel()
      )
    ),
    observation_latent_dim=int(config["observation_latent_dim"]),
    goal_latent_dim=int(config["goal_latent_dim"]),
    hidden_dims=tuple(config["hidden_dims"]),
  ).to(resolved_device)
  model.load_state_dict(payload["model"], strict=True)
  model.eval()
  normalization = {
    name: value.to(resolved_device) for name, value in payload["normalization"].items()
  }
  return model, normalization, payload
