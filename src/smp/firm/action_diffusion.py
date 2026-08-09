"""Conditional action diffusion components for the FIRM reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from smp.firm.expert_runtime import sha256_file
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler

OBSERVATION_DIM = 90
JOINT_DIM = 29
ACTION_DIM = 29


class FirmRolloutWindowDataset(Dataset[dict[str, torch.Tensor]]):
  """Build constant-goal, episode-safe action horizons from rollout shards."""

  def __init__(
    self,
    manifest_file: str | Path,
    horizon: int = 12,
    successful_only: bool = True,
    verify_checksums: bool = True,
  ) -> None:
    if horizon <= 0:
      raise ValueError(f"horizon must be positive, got {horizon}")
    self.manifest_path = Path(manifest_file).expanduser().resolve()
    self.root = self.manifest_path.parent
    self.manifest = json.loads(self.manifest_path.read_text())
    self.horizon = horizon

    fields: dict[str, list[np.ndarray]] = {}
    for shard in self.manifest["shards"]:
      path = self.root / shard["file"]
      if verify_checksums and sha256_file(path) != shard["sha256"]:
        raise ValueError(f"checksum mismatch for {path}")
      with np.load(path) as data:
        for name in data.files:
          fields.setdefault(name, []).append(np.asarray(data[name]))
    self.arrays = {
      name: np.concatenate(parts, axis=0) for name, parts in fields.items()
    }
    expected = int(self.manifest["total_samples"])
    if any(len(array) != expected for array in self.arrays.values()):
      raise ValueError("rollout field lengths disagree with manifest total_samples")
    self._validate_shapes()

    allowed = None
    if successful_only:
      allowed = {
        int(record["episode_id"])
        for record in self.manifest["episode_records"]
        if record["success"]
      }

    windows: list[np.ndarray] = []
    window_episode_ids: list[int] = []
    episode_ids = np.unique(self.arrays["episode_id"])
    for episode_id in episode_ids.tolist():
      if allowed is not None and int(episode_id) not in allowed:
        continue
      indices = np.flatnonzero(self.arrays["episode_id"] == episode_id)
      order = np.argsort(self.arrays["episode_step"][indices])
      indices = indices[order]
      steps = self.arrays["episode_step"][indices]
      if not np.array_equal(steps, np.arange(len(steps), dtype=steps.dtype)):
        raise ValueError(f"episode {episode_id} has non-contiguous steps")
      for start in range(len(indices) - horizon + 1):
        window = indices[start : start + horizon]
        # One conditioning goal must remain valid over the entire action horizon.
        if np.all(
          self.arrays["goal_frame"][window] == self.arrays["goal_frame"][window[0]]
        ):
          windows.append(window)
          window_episode_ids.append(int(episode_id))

    if not windows:
      raise ValueError("rollout dataset contains no valid constant-goal windows")
    self.window_indices = np.stack(windows).astype(np.int64, copy=False)
    self.window_episode_ids = np.asarray(window_episode_ids, dtype=np.int64)

  def _validate_shapes(self) -> None:
    expected = {
      "observation": (OBSERVATION_DIM,),
      "goal": (JOINT_DIM,),
      "action": (ACTION_DIM,),
    }
    for name, tail in expected.items():
      if name not in self.arrays or self.arrays[name].shape[1:] != tail:
        actual = None if name not in self.arrays else self.arrays[name].shape
        raise ValueError(f"{name}: expected (*, {tail}), got {actual}")
    for name in ("episode_id", "episode_step", "goal_frame"):
      if name not in self.arrays or self.arrays[name].ndim != 1:
        raise ValueError(f"missing scalar rollout field {name}")

  def __len__(self) -> int:
    return len(self.window_indices)

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    window = self.window_indices[index]
    first = window[0]
    return {
      "observation": torch.from_numpy(self.arrays["observation"][first]).float(),
      "goal": torch.from_numpy(self.arrays["goal"][first]).float(),
      "actions": torch.from_numpy(self.arrays["action"][window]).float(),
      "episode_id": torch.tensor(self.window_episode_ids[index], dtype=torch.long),
    }

  def split_window_indices(
    self, train_fraction: float, seed: int
  ) -> tuple[np.ndarray, np.ndarray]:
    """Split by episode, preventing adjacent windows from leaking to validation."""
    if not 0.0 < train_fraction < 1.0:
      raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    episodes = np.unique(self.window_episode_ids)
    if len(episodes) < 2:
      raise ValueError("at least two successful episodes are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    count = min(max(1, round(len(episodes) * train_fraction)), len(episodes) - 1)
    train_episodes = set(episodes[:count].tolist())
    train = np.flatnonzero(
      np.fromiter(
        (episode in train_episodes for episode in self.window_episode_ids),
        dtype=bool,
        count=len(self.window_episode_ids),
      )
    )
    validation = np.flatnonzero(
      ~np.isin(np.arange(len(self.window_episode_ids)), train)
    )
    return train, validation

  def normalization_stats(self, window_ids: np.ndarray) -> dict[str, torch.Tensor]:
    """Compute train-only statistics without repeated central transitions."""
    windows = self.window_indices[window_ids]
    observation_ids = np.unique(windows[:, 0])
    action_ids = np.unique(windows.reshape(-1))
    observations = self.arrays["observation"][observation_ids].astype(np.float32)
    goals = self.arrays["goal"][observation_ids].astype(np.float32)
    actions = self.arrays["action"][action_ids].astype(np.float32)
    joints = np.concatenate([observations[:, 3:32], goals], axis=0)

    def stats(value: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
      mean = value.mean(axis=0)
      std = np.maximum(value.std(axis=0), 1.0e-5)
      return torch.from_numpy(mean), torch.from_numpy(std)

    observation_mean, observation_std = stats(observations)
    joint_mean, joint_std = stats(joints)
    action_mean, action_std = stats(actions)
    return {
      "observation_mean": observation_mean,
      "observation_std": observation_std,
      "joint_mean": joint_mean,
      "joint_std": joint_std,
      "action_mean": action_mean,
      "action_std": action_std,
    }


class _MlpEncoder(nn.Module):
  def __init__(self, input_dim: int, output_dim: int) -> None:
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(input_dim, 128),
      nn.SiLU(),
      nn.Linear(128, output_dim),
      nn.LayerNorm(output_dim),
    )

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    return self.network(value)


class FirmActionDiffusion(nn.Module):
  """12-step action denoiser conditioned on state and a relative goal latent."""

  def __init__(
    self,
    horizon: int = 12,
    observation_dim: int = OBSERVATION_DIM,
    action_dim: int = ACTION_DIM,
    goal_latent_dim: int = 64,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.horizon = horizon
    self.observation_dim = observation_dim
    self.action_dim = action_dim
    self.goal_encoder = _MlpEncoder(JOINT_DIM, goal_latent_dim)
    self.observation_encoder = _MlpEncoder(observation_dim, 128)
    self.condition_encoder = nn.Sequential(
      nn.Linear(128 + goal_latent_dim, d_model),
      nn.SiLU(),
      nn.LayerNorm(d_model),
    )
    self.denoiser = DiffusionDenoiser(
      feature_dim=action_dim,
      window_size=horizon,
      d_model=d_model,
      nhead=nhead,
      num_layers=num_layers,
      dropout=dropout,
      condition_dim=d_model,
    )

  def forward(
    self,
    noisy_actions: torch.Tensor,
    timesteps: torch.Tensor,
    observation: torch.Tensor,
    current_joint_position: torch.Tensor,
    goal_joint_position: torch.Tensor,
  ) -> torch.Tensor:
    if noisy_actions.shape[1:] != (self.horizon, self.action_dim):
      raise ValueError(
        f"expected noisy actions (*, {self.horizon}, {self.action_dim}), "
        f"got {tuple(noisy_actions.shape)}"
      )
    current_latent = self.goal_encoder(current_joint_position)
    goal_latent = self.goal_encoder(goal_joint_position)
    relative_goal = goal_latent - current_latent
    state = self.observation_encoder(observation)
    condition = self.condition_encoder(torch.cat([state, relative_goal], dim=-1))
    return self.denoiser(noisy_actions, timesteps, condition)


def load_action_diffusion_checkpoint(
  checkpoint_file: str | Path,
  device: str | torch.device,
  *,
  use_ema: bool = True,
) -> tuple[
  FirmActionDiffusion,
  DDPMScheduler,
  dict[str, torch.Tensor],
  dict,
]:
  """Load a trained action model, scheduler, and normalization tensors."""
  checkpoint_path = Path(checkpoint_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
  resolved_device = torch.device(device)
  payload = torch.load(
    checkpoint_path, map_location=resolved_device, weights_only=False
  )
  config = payload["config"]
  model = FirmActionDiffusion(
    horizon=int(config["horizon"]),
    goal_latent_dim=int(config["goal_latent_dim"]),
    d_model=int(config["d_model"]),
    nhead=int(config["nhead"]),
    num_layers=int(config["num_layers"]),
    dropout=float(config["dropout"]),
  ).to(resolved_device)
  state_name = "model_ema" if use_ema else "model"
  model.load_state_dict(payload[state_name], strict=True)
  model.eval()
  scheduler = DDPMScheduler(int(config["num_timesteps"])).to(resolved_device)
  normalization = {
    name: value.to(resolved_device) for name, value in payload["normalization"].items()
  }
  return model, scheduler, normalization, payload


def normalize_action_condition(
  observation: torch.Tensor,
  goal: torch.Tensor,
  statistics: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Normalize state, current joints, and goal with training-only statistics."""
  current_joint = observation[:, 3:32]
  normalized_observation = (observation - statistics["observation_mean"]) / statistics[
    "observation_std"
  ]
  normalized_current_joint = (current_joint - statistics["joint_mean"]) / statistics[
    "joint_std"
  ]
  normalized_goal = (goal - statistics["joint_mean"]) / statistics["joint_std"]
  return normalized_observation, normalized_current_joint, normalized_goal


@torch.no_grad()
def sample_action_horizon(
  model: FirmActionDiffusion,
  scheduler: DDPMScheduler,
  observation: torch.Tensor,
  current_joint: torch.Tensor,
  goal: torch.Tensor,
) -> torch.Tensor:
  """Run every ancestral DDPM step and return a normalized action horizon."""
  batch_size = observation.shape[0]
  actions = torch.randn(
    batch_size,
    model.horizon,
    model.action_dim,
    device=observation.device,
    dtype=observation.dtype,
  )
  for step in reversed(range(scheduler.num_timesteps)):
    timesteps = torch.full(
      (batch_size,), step, dtype=torch.long, device=observation.device
    )
    predicted_noise = model(actions, timesteps, observation, current_joint, goal)
    actions = scheduler.step(predicted_noise, actions, step)
  return actions


def denormalize_actions(
  normalized_actions: torch.Tensor,
  statistics: dict[str, torch.Tensor],
) -> torch.Tensor:
  """Map normalized action samples back to expert-policy action units."""
  return normalized_actions * statistics["action_std"] + statistics["action_mean"]
