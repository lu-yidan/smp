"""Load and execute the frozen 93-D deployable FIRM-R external reference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler

DEPLOYABLE_OBSERVATION_DIM = 93
JOINT_DIM = 29
ACTION_DIM = 29
JOINT_POSITION_SLICE = slice(6, 35)
SUPPORTED_HISTORY_STEPS = (1, 50)


def sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


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
  """Architecture-compatible inference copy frozen from FIRM commit cfa8572."""

  def __init__(
    self,
    *,
    horizon: int,
    observation_dim: int,
    goal_latent_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
  ) -> None:
    super().__init__()
    self.horizon = horizon
    self.observation_dim = observation_dim
    self.action_dim = ACTION_DIM
    self.goal_encoder = _MlpEncoder(JOINT_DIM, goal_latent_dim)
    self.observation_encoder = _MlpEncoder(observation_dim, 128)
    self.condition_encoder = nn.Sequential(
      nn.Linear(128 + goal_latent_dim, d_model),
      nn.SiLU(),
      nn.LayerNorm(d_model),
    )
    self.denoiser = DiffusionDenoiser(
      feature_dim=ACTION_DIM,
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
    current_latent = self.goal_encoder(current_joint_position)
    goal_latent = self.goal_encoder(goal_joint_position)
    state = self.observation_encoder(observation)
    condition = self.condition_encoder(
      torch.cat((state, goal_latent - current_latent), dim=-1)
    )
    return self.denoiser(noisy_actions, timesteps, condition)


class FirmGoalAdapter(nn.Module):
  """Architecture-compatible 1-frame or 50-frame causal goal adapter."""

  def __init__(
    self,
    *,
    observation_dim: int,
    history_steps: int,
    latent_dim: int,
    channels: tuple[int, int, int],
  ) -> None:
    super().__init__()
    self.observation_dim = observation_dim
    self.history_steps = history_steps
    self.latent_dim = latent_dim
    if history_steps >= 40:
      self.temporal = nn.Sequential(
        nn.Conv1d(observation_dim, channels[0], kernel_size=8, stride=4),
        nn.SiLU(),
        nn.Conv1d(channels[0], channels[1], kernel_size=5),
        nn.SiLU(),
        nn.Conv1d(channels[1], channels[2], kernel_size=5),
        nn.SiLU(),
      )
    else:
      self.temporal = nn.Sequential(
        nn.Conv1d(observation_dim, channels[0], kernel_size=1),
        nn.SiLU(),
        nn.Conv1d(channels[0], channels[1], kernel_size=1),
        nn.SiLU(),
        nn.Conv1d(channels[1], channels[2], kernel_size=1),
        nn.SiLU(),
      )
    with torch.no_grad():
      dummy = torch.zeros(1, observation_dim, history_steps)
      flattened_dim = int(self.temporal(dummy).numel())
    self.projection = nn.Linear(flattened_dim, latent_dim)

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    features = self.temporal(history.transpose(1, 2)).flatten(1)
    return F.normalize(self.projection(features), dim=-1)


def _checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = torch.load(path, map_location=device, weights_only=False)
  if not isinstance(payload, dict):
    raise TypeError(f"expected checkpoint dictionary: {path}")
  return payload


def _require_seed(config: Mapping[str, Any], expected_seed: int | None) -> int:
  seed = config.get("seed")
  if not isinstance(seed, int):
    raise ValueError("FIRM checkpoint has no integer training seed")
  if expected_seed is not None and seed != expected_seed:
    raise ValueError(f"FIRM checkpoint seed mismatch: {seed} != {expected_seed}")
  return seed


def _load_action(
  path: Path, device: torch.device, expected_seed: int | None
) -> tuple[
  FirmActionDiffusion,
  DDPMScheduler,
  dict[str, torch.Tensor],
  dict[str, Any],
  int,
]:
  payload = _checkpoint(path, device)
  config = payload.get("config", {})
  seed = _require_seed(config, expected_seed)
  observation_dim = int(config.get("observation_dim", -1))
  if observation_dim != DEPLOYABLE_OBSERVATION_DIM:
    raise ValueError(
      f"formal FIRM-R evaluation requires 93-D action input, got {observation_dim}"
    )
  model = FirmActionDiffusion(
    horizon=int(config["horizon"]),
    observation_dim=observation_dim,
    goal_latent_dim=int(config["goal_latent_dim"]),
    d_model=int(config["d_model"]),
    nhead=int(config["nhead"]),
    num_layers=int(config["num_layers"]),
    dropout=float(config["dropout"]),
  ).to(device)
  model.load_state_dict(payload["model_ema"], strict=True)
  model.eval().requires_grad_(False)
  scheduler = DDPMScheduler(int(config["num_timesteps"])).to(device)
  normalization = {
    name: value.to(device) for name, value in payload.get("normalization", {}).items()
  }
  expected_shapes = {
    "observation_mean": (DEPLOYABLE_OBSERVATION_DIM,),
    "observation_std": (DEPLOYABLE_OBSERVATION_DIM,),
    "joint_mean": (JOINT_DIM,),
    "joint_std": (JOINT_DIM,),
    "action_mean": (ACTION_DIM,),
    "action_std": (ACTION_DIM,),
  }
  for name, shape in expected_shapes.items():
    value = normalization.get(name)
    if value is None or tuple(value.shape) != shape or not torch.isfinite(value).all():
      raise ValueError(f"invalid FIRM normalization tensor {name}")
  return model, scheduler, normalization, payload, seed


def _load_adapter(
  path: Path,
  device: torch.device,
  action_path: Path,
  expected_seed: int | None,
  normalization: dict[str, torch.Tensor],
) -> tuple[FirmGoalAdapter, dict[str, Any], int]:
  payload = _checkpoint(path, device)
  config = payload.get("config", {})
  seed = _require_seed(config, expected_seed)
  observation_dim = int(config.get("observation_dim", -1))
  history_steps = int(config.get("history_steps", -1))
  if observation_dim != DEPLOYABLE_OBSERVATION_DIM:
    raise ValueError(
      f"formal FIRM-R evaluation requires 93-D adapter input, got {observation_dim}"
    )
  if history_steps not in SUPPORTED_HISTORY_STEPS:
    raise ValueError(f"unregistered FIRM adapter history: {history_steps}")
  artifacts = payload.get("artifacts", {})
  action_sha = sha256_file(action_path)
  if artifacts.get("action_checkpoint_sha256") != action_sha:
    raise ValueError("FIRM adapter/action checkpoint SHA-256 mismatch")
  for name in ("observation_mean", "observation_std"):
    value = payload.get(name)
    if value is None or not torch.allclose(value.to(device), normalization[name]):
      raise ValueError(f"FIRM adapter/action normalization mismatch: {name}")
  model = FirmGoalAdapter(
    observation_dim=observation_dim,
    history_steps=history_steps,
    latent_dim=int(config["latent_dim"]),
    channels=tuple(int(value) for value in config["channels"]),
  ).to(device)
  model.load_state_dict(payload["model"], strict=True)
  model.eval().requires_grad_(False)
  codebook_goals = payload.get("codebook_goals")
  codebook_features = payload.get("codebook_features")
  if (
    not isinstance(codebook_goals, torch.Tensor)
    or not isinstance(codebook_features, torch.Tensor)
    or codebook_goals.ndim != 2
    or tuple(codebook_goals.shape[1:]) != (JOINT_DIM,)
    or codebook_features.shape != (len(codebook_goals), model.latent_dim)
  ):
    raise ValueError("invalid FIRM adapter codebook")
  payload["codebook_goals"] = codebook_goals.to(device)
  payload["codebook_features"] = codebook_features.to(device)
  payload["observation_mean"] = payload["observation_mean"].to(device)
  payload["observation_std"] = payload["observation_std"].to(device)
  return model, payload, seed


class FirmDeployablePolicy:
  """Stateful receding-horizon policy using only causal 93-D actor observations."""

  def __init__(
    self,
    action_checkpoint: str | Path,
    adapter_checkpoint: str | Path,
    *,
    device: str | torch.device,
    expected_seed: int | None = None,
    goal_refresh_steps: int = 5,
    num_action_samples: int = 1,
  ) -> None:
    if goal_refresh_steps <= 0 or num_action_samples <= 0:
      raise ValueError("refresh steps and action samples must be positive")
    self.device = torch.device(device)
    self.action_path = Path(action_checkpoint).expanduser().resolve()
    self.adapter_path = Path(adapter_checkpoint).expanduser().resolve()
    self.action, self.scheduler, self.stats, _, action_seed = _load_action(
      self.action_path, self.device, expected_seed
    )
    self.adapter, self.adapter_payload, adapter_seed = _load_adapter(
      self.adapter_path,
      self.device,
      self.action_path,
      expected_seed,
      self.stats,
    )
    if action_seed != adapter_seed:
      raise ValueError("FIRM action and adapter training seeds differ")
    self.training_seed = action_seed
    self.goal_refresh_steps = goal_refresh_steps
    self.num_action_samples = num_action_samples
    self.history: torch.Tensor | None = None
    self.goal: torch.Tensor | None = None
    self.goal_index: torch.Tensor | None = None
    self.steps = 0
    self.retrieval_count = 0
    self.retrieval_switches = 0

  @property
  def observation_dim(self) -> int:
    return self.action.observation_dim

  @property
  def history_steps(self) -> int:
    return self.adapter.history_steps

  def _retrieve_goal(self) -> None:
    assert self.history is not None
    normalized = (
      self.history - self.adapter_payload["observation_mean"]
    ) / self.adapter_payload["observation_std"]
    query = self.adapter(normalized)
    similarities = query @ self.adapter_payload["codebook_features"].transpose(0, 1)
    index = similarities.argmax(dim=-1)
    if self.goal_index is not None:
      self.retrieval_switches += int((index != self.goal_index).sum())
    self.goal_index = index
    self.goal = self.adapter_payload["codebook_goals"][index]
    self.retrieval_count += len(index)

  def _sample(self, observation: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    current_joint = observation[:, JOINT_POSITION_SLICE]
    normalized_observation = (
      observation - self.stats["observation_mean"]
    ) / self.stats["observation_std"]
    normalized_current = (current_joint - self.stats["joint_mean"]) / self.stats[
      "joint_std"
    ]
    normalized_goal = (goal - self.stats["joint_mean"]) / self.stats["joint_std"]
    count = self.num_action_samples
    normalized_observation = normalized_observation.repeat_interleave(count, dim=0)
    normalized_current = normalized_current.repeat_interleave(count, dim=0)
    normalized_goal = normalized_goal.repeat_interleave(count, dim=0)
    actions = torch.randn(
      len(normalized_observation),
      self.action.horizon,
      ACTION_DIM,
      device=self.device,
      dtype=observation.dtype,
    )
    for timestep in reversed(range(self.scheduler.num_timesteps)):
      time = torch.full((len(actions),), timestep, dtype=torch.long, device=self.device)
      predicted_noise = self.action(
        actions,
        time,
        normalized_observation,
        normalized_current,
        normalized_goal,
      )
      actions = self.scheduler.step(predicted_noise, actions, timestep)
    first = actions[:, 0]
    if count > 1:
      first = first.view(len(observation), count, ACTION_DIM).mean(dim=1)
    return first * self.stats["action_std"] + self.stats["action_mean"]

  @torch.inference_mode()
  def __call__(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
    actor = observations["actor"]
    if actor.ndim != 2 or actor.shape[-1] != DEPLOYABLE_OBSERVATION_DIM:
      raise ValueError(
        f"FIRM-R requires exactly one causal 93-D actor frame, got {tuple(actor.shape)}"
      )
    actor = actor.to(self.device)
    if self.history is None:
      self.history = actor[:, None].expand(-1, self.history_steps, -1).clone()
    else:
      if len(self.history) != len(actor):
        raise ValueError("FIRM-R environment count changed during evaluation")
      self.history = torch.roll(self.history, shifts=-1, dims=1)
      self.history[:, -1] = actor
    if self.goal is None or self.steps % self.goal_refresh_steps == 0:
      self._retrieve_goal()
    assert self.goal is not None
    action = self._sample(actor, self.goal)
    self.steps += 1
    return action

  def metadata(self) -> dict[str, Any]:
    return {
      "method": "firm_r",
      "source_commit": "cfa8572f130dc32d05280b9592ce657c8b3a1b56",
      "training_seed": self.training_seed,
      "observation_dim": self.observation_dim,
      "history_steps": self.history_steps,
      "goal_refresh_steps": self.goal_refresh_steps,
      "num_action_samples": self.num_action_samples,
      "action_checkpoint": str(self.action_path),
      "action_checkpoint_sha256": sha256_file(self.action_path),
      "adapter_checkpoint": str(self.adapter_path),
      "adapter_checkpoint_sha256": sha256_file(self.adapter_path),
      "retrieval_count": self.retrieval_count,
      "retrieval_switches": self.retrieval_switches,
    }
