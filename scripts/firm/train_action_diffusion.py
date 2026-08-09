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
from torch.utils.data import DataLoader, Subset

from smp.firm.action_diffusion import FirmActionDiffusion, FirmRolloutWindowDataset
from smp.firm.expert_runtime import sha256_file
from smp.pretrain.scheduler import DDPMScheduler


@dataclass
class TrainActionDiffusionConfig:
  """Pilot action diffusion configuration."""

  manifest_file: str = (
    "datasets/firm/rollouts/c003_stage0_model_29999_pilot/manifest.json"
  )
  horizon: int = 12
  successful_only: bool = True
  verify_checksums: bool = True
  train_fraction: float = 0.90

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
  current_joint = observation[:, 3:32]

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
  dataset: FirmRolloutWindowDataset,
  train_windows: int,
  validation_windows: int,
  validation_loss: float,
) -> None:
  payload: dict[str, Any] = {
    "format_version": 1,
    "epoch": epoch,
    "model": model.state_dict(),
    "model_ema": ema.shadow.state_dict(),
    "normalization": {name: value.cpu() for name, value in stats.items()},
    "config": asdict(cfg),
    "manifest_file": str(dataset.manifest_path),
    "manifest_sha256": sha256_file(dataset.manifest_path),
    "train_windows": train_windows,
    "validation_windows": validation_windows,
    "validation_loss": validation_loss,
  }
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

  dataset = FirmRolloutWindowDataset(
    cfg.manifest_file,
    horizon=cfg.horizon,
    successful_only=cfg.successful_only,
    verify_checksums=cfg.verify_checksums,
  )
  train_ids, validation_ids = dataset.split_window_indices(cfg.train_fraction, cfg.seed)
  statistics = {
    name: value.to(device)
    for name, value in dataset.normalization_stats(train_ids).items()
  }
  pin_memory = device.type == "cuda"
  train_loader = DataLoader(
    Subset(dataset, train_ids.tolist()),
    batch_size=cfg.batch_size,
    shuffle=True,
    pin_memory=pin_memory,
  )
  validation_loader = DataLoader(
    Subset(dataset, validation_ids.tolist()),
    batch_size=cfg.batch_size,
    shuffle=False,
    pin_memory=pin_memory,
  )

  model = FirmActionDiffusion(
    horizon=cfg.horizon,
    goal_latent_dim=cfg.goal_latent_dim,
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
  parameter_count = sum(parameter.numel() for parameter in model.parameters())
  print(
    f"[INFO] dataset={len(dataset)} windows, train={len(train_ids)}, "
    f"validation={len(validation_ids)}, episodes="
    f"{len(np.unique(dataset.window_episode_ids))}"
  )
  print(f"[INFO] conditional action denoiser parameters={parameter_count:,}")

  run_dir = (
    Path(cfg.log_dir) / cfg.run_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  run_dir.mkdir(parents=True, exist_ok=True)

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

    should_save = epoch % cfg.save_interval == 0 and epoch != cfg.num_epochs - 1
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
        dataset=dataset,
        train_windows=len(train_ids),
        validation_windows=len(validation_ids),
        validation_loss=last_validation_loss,
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
    dataset=dataset,
    train_windows=len(train_ids),
    validation_windows=len(validation_ids),
    validation_loss=last_validation_loss,
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
