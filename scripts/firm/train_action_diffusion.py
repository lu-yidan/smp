"""Train the pilot FIRM 12-step goal-conditioned action diffusion model."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import ConcatDataset, DataLoader, Subset

from smp.firm.action_diffusion import (
  FirmActionDiffusion,
  FirmRolloutWindowDataset,
  joint_position_slice,
)
from smp.firm.expert_runtime import sha256_file
from smp.pretrain.scheduler import DDPMScheduler


@dataclass
class TrainActionDiffusionConfig:
  """Pilot action diffusion configuration."""

  manifest_file: str = (
    "datasets/firm/rollouts/c003_stage0_model_29999_pilot/manifest.json"
  )
  additional_manifest_files: tuple[str, ...] = ()
  """Extra rollout datasets mixed into warm-start refinement."""
  additional_dataset_repeats: tuple[int, ...] = ()
  """Training repeats for each additional dataset; empty means one each.

  Validation windows are never repeated.
  """
  horizon: int = 12
  successful_only: bool = True
  verify_checksums: bool = True
  train_fraction: float = 0.90

  initial_checkpoint_file: str | None = None
  """Optional action-diffusion checkpoint used to warm-start training."""
  initial_weights: str = "ema"
  """Warm-start from 'ema' or 'online' weights."""
  reuse_initial_normalization: bool = True

  d_model: int = 256
  nhead: int = 4
  num_layers: int = 4
  goal_latent_dim: int = 64
  dropout: float = 0.0
  num_timesteps: int = 50

  batch_size: int = 512
  num_epochs: int = 100
  learning_rate: float = 3.0e-4
  weight_decay: float = 1.0e-4
  max_grad_norm: float = 1.0
  ema_decay: float = 0.999

  log_interval: int = 1
  save_interval: int = 50
  log_dir: str = "logs/firm_action_diffusion"
  run_name: str = "firm_action_diffusion_c003_pilot"
  output_dir: str | None = None
  """Exact output directory for registered runs; must not already exist."""
  use_wandb: bool = True
  wandb_project: str = "smp"

  device: str = "cuda:0"
  seed: int = 42


class _Ema:
  def __init__(self, model: torch.nn.Module, decay: float) -> None:
    self.decay = decay
    self.shadow = copy.deepcopy(model).eval()
    for parameter in self.shadow.parameters():
      parameter.requires_grad_(False)

  @torch.no_grad()
  def update(self, model: torch.nn.Module) -> None:
    source = model.state_dict()
    target = self.shadow.state_dict()
    for name, source_value in source.items():
      target_value = target[name]
      if target_value.is_floating_point():
        target_value.mul_(self.decay).add_(
          source_value.detach(), alpha=1.0 - self.decay
        )
      else:
        target_value.copy_(source_value)


def _seed_everything(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _to_device(
  batch: dict[str, torch.Tensor],
  stats: dict[str, torch.Tensor],
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  observation = batch["observation"].to(device, non_blocking=True)
  goal = batch["goal"].to(device, non_blocking=True)
  actions = batch["actions"].to(device, non_blocking=True)
  current_joint = observation[:, joint_position_slice(observation.shape[-1])]

  observation = (observation - stats["observation_mean"]) / stats["observation_std"]
  current_joint = (current_joint - stats["joint_mean"]) / stats["joint_std"]
  goal = (goal - stats["joint_mean"]) / stats["joint_std"]
  actions = (actions - stats["action_mean"]) / stats["action_std"]
  return observation, current_joint, goal, actions


def _loss(
  model: FirmActionDiffusion,
  scheduler: DDPMScheduler,
  observation: torch.Tensor,
  current_joint: torch.Tensor,
  goal: torch.Tensor,
  actions: torch.Tensor,
) -> torch.Tensor:
  timesteps = scheduler.sample_timesteps(actions.shape[0], actions.device)
  noise = torch.randn_like(actions)
  noisy_actions = scheduler.add_noise(actions, noise, timesteps)
  prediction = model(
    noisy_actions,
    timesteps,
    observation,
    current_joint,
    goal,
  )
  return F.l1_loss(prediction, noise)


@torch.no_grad()
def _validate(
  model: FirmActionDiffusion,
  scheduler: DDPMScheduler,
  loader: DataLoader,
  stats: dict[str, torch.Tensor],
  device: torch.device,
) -> float:
  model.eval()
  total = torch.zeros((), device=device)
  count = 0
  for batch in loader:
    observation, current_joint, goal, actions = _to_device(batch, stats, device)
    total += _loss(model, scheduler, observation, current_joint, goal, actions).detach()
    count += 1
  return float((total / max(count, 1)).item())


def _save_checkpoint(
  path: Path,
  *,
  epoch: int,
  model: FirmActionDiffusion,
  ema: _Ema,
  optimizer: torch.optim.Optimizer | None,
  stats: dict[str, torch.Tensor],
  cfg: TrainActionDiffusionConfig,
  datasets: list[FirmRolloutWindowDataset],
  train_windows: int,
  validation_windows: int,
  validation_loss: float,
  initial_checkpoint: dict[str, Any] | None,
) -> None:
  payload: dict[str, Any] = {
    "format_version": 1,
    "epoch": epoch,
    "model": model.state_dict(),
    "model_ema": ema.shadow.state_dict(),
    "normalization": {name: value.cpu() for name, value in stats.items()},
    "config": {**asdict(cfg), "observation_dim": model.observation_dim},
    "manifest_file": str(datasets[0].manifest_path),
    "manifest_sha256": sha256_file(datasets[0].manifest_path),
    "manifests": [
      {
        "path": str(dataset.manifest_path),
        "sha256": sha256_file(dataset.manifest_path),
      }
      for dataset in datasets
    ],
    "train_windows": train_windows,
    "validation_windows": validation_windows,
    "validation_loss": validation_loss,
  }
  if initial_checkpoint is not None:
    payload["initial_checkpoint"] = initial_checkpoint
  if optimizer is not None:
    payload["optimizer"] = optimizer.state_dict()
  torch.save(payload, path)


def train(cfg: TrainActionDiffusionConfig) -> Path:
  """Train and persist the pilot conditional action denoiser."""
  if cfg.num_epochs <= 0 or cfg.batch_size <= 0:
    raise ValueError("num_epochs and batch_size must be positive")
  if cfg.save_interval <= 0 or cfg.log_interval <= 0:
    raise ValueError("save_interval and log_interval must be positive")
  _seed_everything(cfg.seed)
  device = torch.device(cfg.device)
  if cfg.initial_weights not in {"ema", "online"}:
    raise ValueError("initial_weights must be 'ema' or 'online'")

  initial_payload: dict[str, Any] | None = None
  initial_metadata: dict[str, Any] | None = None
  if cfg.initial_checkpoint_file is not None:
    initial_path = Path(cfg.initial_checkpoint_file).expanduser().resolve()
    if not initial_path.is_file():
      raise FileNotFoundError(initial_path)
    initial_payload = torch.load(
      initial_path,
      map_location=device,
      weights_only=False,
    )
    initial_metadata = {
      "path": str(initial_path),
      "sha256": sha256_file(initial_path),
      "epoch": int(initial_payload["epoch"]),
      "weights": cfg.initial_weights,
    }

  if cfg.additional_manifest_files and (
    initial_payload is None or not cfg.reuse_initial_normalization
  ):
    raise ValueError(
      "additional_manifest_files requires an initial checkpoint with "
      "reuse_initial_normalization enabled"
    )
  repeats = cfg.additional_dataset_repeats or (1,) * len(cfg.additional_manifest_files)
  if len(repeats) != len(cfg.additional_manifest_files):
    raise ValueError("additional_dataset_repeats must match additional_manifest_files")
  if any(repeat <= 0 for repeat in repeats):
    raise ValueError("additional_dataset_repeats must all be positive")

  manifest_files = (cfg.manifest_file, *cfg.additional_manifest_files)
  datasets = [
    FirmRolloutWindowDataset(
      manifest_file,
      horizon=cfg.horizon,
      successful_only=cfg.successful_only,
      verify_checksums=cfg.verify_checksums,
    )
    for manifest_file in manifest_files
  ]
  observation_dims = {dataset.observation_dim for dataset in datasets}
  if len(observation_dims) != 1:
    raise ValueError(
      f"all rollout datasets must share one layout, got {observation_dims}"
    )
  observation_dim = observation_dims.pop()
  splits = [
    dataset.split_window_indices(cfg.train_fraction, cfg.seed + index)
    for index, dataset in enumerate(datasets)
  ]
  train_parts = [Subset(datasets[0], splits[0][0].tolist())]
  for dataset, (train_ids, _), repeat in zip(
    datasets[1:], splits[1:], repeats, strict=True
  ):
    train_parts.extend([Subset(dataset, train_ids.tolist())] * repeat)
  validation_parts = [
    Subset(dataset, validation_ids.tolist())
    for dataset, (_, validation_ids) in zip(datasets, splits, strict=True)
  ]
  train_dataset = (
    train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
  )
  validation_dataset = (
    validation_parts[0]
    if len(validation_parts) == 1
    else ConcatDataset(validation_parts)
  )
  if initial_payload is not None and cfg.reuse_initial_normalization:
    statistics = {
      name: value.to(device) for name, value in initial_payload["normalization"].items()
    }
  else:
    statistics = {
      name: value.to(device)
      for name, value in datasets[0].normalization_stats(splits[0][0]).items()
    }
  pin_memory = device.type == "cuda"
  train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    pin_memory=pin_memory,
  )
  validation_loader = DataLoader(
    validation_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    pin_memory=pin_memory,
  )

  model = FirmActionDiffusion(
    horizon=cfg.horizon,
    goal_latent_dim=cfg.goal_latent_dim,
    observation_dim=observation_dim,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dropout=cfg.dropout,
  ).to(device)
  scheduler = DDPMScheduler(cfg.num_timesteps).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
  )
  ema = _Ema(model, cfg.ema_decay)
  if initial_payload is not None:
    state_name = "model_ema" if cfg.initial_weights == "ema" else "model"
    model.load_state_dict(initial_payload[state_name], strict=True)
    ema.shadow.load_state_dict(initial_payload[state_name], strict=True)
    print(
      f"[INFO] warm-started {state_name} from "
      f"{initial_metadata['path']} (epoch {initial_metadata['epoch']})"
    )
  parameter_count = sum(parameter.numel() for parameter in model.parameters())
  train_windows = len(train_dataset)
  validation_windows = len(validation_dataset)
  print(
    f"[INFO] datasets={len(datasets)}, train={train_windows}, "
    f"validation={validation_windows}"
  )
  for index, dataset in enumerate(datasets):
    repeat = 1 if index == 0 else repeats[index - 1]
    print(
      f"[INFO] dataset[{index}]={dataset.manifest_path}: {len(dataset)} windows, "
      f"{len(np.unique(dataset.window_episode_ids))} episodes, train_repeat={repeat}"
    )
  print(f"[INFO] conditional action denoiser parameters={parameter_count:,}")

  run_dir = (
    Path(cfg.output_dir)
    if cfg.output_dir is not None
    else Path(cfg.log_dir) / cfg.run_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  run_dir.mkdir(parents=True, exist_ok=False)

  wandb_run = None
  if cfg.use_wandb:
    import wandb

    wandb_run = wandb.init(
      project=cfg.wandb_project,
      name=cfg.run_name,
      config=asdict(cfg),
    )

  last_validation_loss = float("nan")
  for epoch in range(cfg.num_epochs):
    model.train()
    total = torch.zeros((), device=device)
    batches = 0
    for batch in train_loader:
      observation, current_joint, goal, actions = _to_device(batch, statistics, device)
      loss = _loss(model, scheduler, observation, current_joint, goal, actions)
      optimizer.zero_grad()
      loss.backward()
      if cfg.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
      optimizer.step()
      ema.update(model)
      total += loss.detach()
      batches += 1

    training_loss = float((total / max(batches, 1)).item())
    should_log = epoch % cfg.log_interval == 0 or epoch == cfg.num_epochs - 1
    if should_log:
      last_validation_loss = _validate(
        ema.shadow, scheduler, validation_loader, statistics, device
      )
      print(
        f"Epoch {epoch:03d}/{cfg.num_epochs - 1:03d} "
        f"train={training_loss:.6f} validation={last_validation_loss:.6f}"
      )
      if wandb_run is not None:
        wandb_run.log(
          {
            "epoch": epoch,
            "train/loss": training_loss,
            "validation/loss": last_validation_loss,
          }
        )

    should_save = (epoch + 1) % cfg.save_interval == 0 and epoch != cfg.num_epochs - 1
    if should_save:
      checkpoint_path = run_dir / f"checkpoint_{epoch:04d}.pt"
      _save_checkpoint(
        checkpoint_path,
        epoch=epoch,
        model=model,
        ema=ema,
        optimizer=None,
        stats=statistics,
        cfg=cfg,
        datasets=datasets,
        train_windows=train_windows,
        validation_windows=validation_windows,
        validation_loss=last_validation_loss,
        initial_checkpoint=initial_metadata,
      )
      if wandb_run is not None:
        wandb_run.save(str(checkpoint_path), base_path=str(run_dir))

  final_path = run_dir / "firm_action_diffusion.pt"
  _save_checkpoint(
    final_path,
    epoch=cfg.num_epochs - 1,
    model=model,
    ema=ema,
    optimizer=None,
    stats=statistics,
    cfg=cfg,
    datasets=datasets,
    train_windows=train_windows,
    validation_windows=validation_windows,
    validation_loss=last_validation_loss,
    initial_checkpoint=initial_metadata,
  )
  print(f"[INFO] final checkpoint: {final_path}")
  if wandb_run is not None:
    wandb_run.save(str(final_path), base_path=str(run_dir))
    wandb_run.finish()
  return final_path


def main() -> None:
  train(tyro.cli(TrainActionDiffusionConfig))


if __name__ == "__main__":
  main()
