"""Online keyframe-goal adapter for the FIRM reproduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from smp.firm.action_diffusion import (
  OBSERVATION_DIM,
  FirmRolloutWindowDataset,
)


class FirmGoalAdapterDataset(Dataset[dict[str, torch.Tensor]]):
  """Build episode-safe observation histories and current keyframe targets."""

  def __init__(
    self,
    manifest_files: tuple[str | Path, ...],
    history_steps: int = 50,
    successful_only: bool = True,
    verify_checksums: bool = True,
  ) -> None:
    if not manifest_files:
      raise ValueError("at least one rollout manifest is required")
    if history_steps <= 0:
      raise ValueError(f"history_steps must be positive, got {history_steps}")
    self.history_steps = history_steps
    self.sources: list[FirmRolloutWindowDataset] = []
    self.records: list[tuple[int, np.ndarray, int]] = []
    episode_group = 0
    sample_episode_groups: list[int] = []

    for source_index, manifest in enumerate(manifest_files):
      source = FirmRolloutWindowDataset(
        manifest,
        horizon=1,
        successful_only=successful_only,
        verify_checksums=verify_checksums,
      )
      self.sources.append(source)
      transition_ids = source.window_indices[:, 0]
      episode_ids = source.window_episode_ids
      for episode_id in np.unique(episode_ids).tolist():
        local = np.flatnonzero(episode_ids == episode_id)
        indices = transition_ids[local]
        steps = source.arrays["episode_step"][indices]
        order = np.argsort(steps)
        indices = indices[order]
        for position in range(len(indices)):
          self.records.append((source_index, indices, position))
          sample_episode_groups.append(episode_group)
        episode_group += 1

    if not self.records:
      raise ValueError("adapter dataset contains no transitions")
    self.sample_episode_groups = np.asarray(sample_episode_groups, dtype=np.int64)
    record_goals = np.asarray(
      [
        self.sources[source_index].arrays["goal"][episode_indices[position]]
        for source_index, episode_indices, position in self.records
      ],
      dtype=np.float32,
    )
    self._codebook_goals, self.sample_goal_indices = np.unique(
      record_goals, axis=0, return_inverse=True
    )

  def __len__(self) -> int:
    return len(self.records)

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    source_index, episode_indices, position = self.records[index]
    source = self.sources[source_index]
    start = max(0, position - self.history_steps + 1)
    history_ids = episode_indices[start : position + 1]
    history = source.arrays["observation"][history_ids]
    if len(history) < self.history_steps:
      padding = np.repeat(history[:1], self.history_steps - len(history), axis=0)
      history = np.concatenate([padding, history], axis=0)
    target_id = episode_indices[position]
    return {
      "observation_history": torch.from_numpy(history).float(),
      "goal": torch.from_numpy(source.arrays["goal"][target_id]).float(),
      "goal_index": torch.tensor(self.sample_goal_indices[index], dtype=torch.long),
      "episode_group": torch.tensor(
        self.sample_episode_groups[index], dtype=torch.long
      ),
    }

  def split_sample_indices(
    self, train_fraction: float, seed: int
  ) -> tuple[np.ndarray, np.ndarray]:
    """Split by source-qualified episode, preventing temporal leakage."""
    if not 0.0 < train_fraction < 1.0:
      raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    episodes = np.unique(self.sample_episode_groups)
    if len(episodes) < 2:
      raise ValueError("at least two episodes are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    count = min(max(1, round(len(episodes) * train_fraction)), len(episodes) - 1)
    train_episodes = episodes[:count]
    train = np.flatnonzero(np.isin(self.sample_episode_groups, train_episodes))
    validation = np.flatnonzero(
      ~np.isin(self.sample_episode_groups, train_episodes)
    )
    return train, validation

  def codebook_goals(self) -> torch.Tensor:
    """Return unique raw joint-position goals across all selected episodes."""
    goals = []
    for source_index, episode_indices, position in self.records:
      goals.append(
        self.sources[source_index].arrays["goal"][episode_indices[position]]
      )
    unique = np.unique(np.asarray(goals, dtype=np.float32), axis=0)
    return torch.from_numpy(unique)


class FirmGoalAdapter(nn.Module):
  """Three-layer temporal CNN matching the FIRM appendix specification."""

  def __init__(
    self,
    observation_dim: int = OBSERVATION_DIM,
    history_steps: int = 50,
    latent_dim: int = 64,
    channels: tuple[int, int, int] = (128, 256, 256),
  ) -> None:
    super().__init__()
    if len(channels) != 3 or any(channel <= 0 for channel in channels):
      raise ValueError("channels must contain three positive values")
    self.observation_dim = observation_dim
    self.history_steps = history_steps
    self.latent_dim = latent_dim
    self.channels = channels
    self.temporal = nn.Sequential(
      nn.Conv1d(observation_dim, channels[0], kernel_size=8, stride=4),
      nn.SiLU(),
      nn.Conv1d(channels[0], channels[1], kernel_size=5, stride=1),
      nn.SiLU(),
      nn.Conv1d(channels[1], channels[2], kernel_size=5, stride=1),
      nn.SiLU(),
    )
    with torch.no_grad():
      dummy = torch.zeros(1, observation_dim, history_steps)
      flattened_dim = int(self.temporal(dummy).numel())
    self.projection = nn.Linear(flattened_dim, latent_dim)

  def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
    if observation_history.shape[1:] != (
      self.history_steps,
      self.observation_dim,
    ):
      raise ValueError(
        "expected observation history "
        f"(*, {self.history_steps}, {self.observation_dim}), "
        f"got {tuple(observation_history.shape)}"
      )
    features = self.temporal(observation_history.transpose(1, 2)).flatten(1)
    return F.normalize(self.projection(features), dim=-1)


def normalize_adapter_history(
  observation_history: torch.Tensor,
  observation_mean: torch.Tensor,
  observation_std: torch.Tensor,
) -> torch.Tensor:
  """Apply the frozen action-policy observation normalization."""
  return (observation_history - observation_mean) / observation_std


def load_goal_adapter_checkpoint(
  checkpoint_file: str | Path,
  device: str | torch.device,
) -> tuple[FirmGoalAdapter, dict]:
  """Load a frozen adapter and its embedded keyframe codebook."""
  path = Path(checkpoint_file).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  resolved_device = torch.device(device)
  payload = torch.load(path, map_location=resolved_device, weights_only=False)
  config = payload["config"]
  model = FirmGoalAdapter(
    observation_dim=int(config["observation_dim"]),
    history_steps=int(config["history_steps"]),
    latent_dim=int(config["latent_dim"]),
    channels=tuple(int(value) for value in config["channels"]),
  ).to(resolved_device)
  model.load_state_dict(payload["model"], strict=True)
  model.eval()
  model.requires_grad_(False)
  return model, payload


@torch.no_grad()
def retrieve_adapter_goal(
  model: FirmGoalAdapter,
  observation_history: torch.Tensor,
  payload: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Retrieve raw joint goals by cosine nearest neighbour on the unit sphere."""
  normalized_history = normalize_adapter_history(
    observation_history,
    payload["observation_mean"],
    payload["observation_std"],
  )
  query = model(normalized_history)
  codebook_features = payload["codebook_features"]
  similarities = query @ codebook_features.transpose(0, 1)
  indexes = similarities.argmax(dim=-1)
  goals = payload["codebook_goals"][indexes]
  scores = similarities.gather(1, indexes[:, None]).squeeze(1)
  return goals, indexes, scores
