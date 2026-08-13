"""Train a deterministic one-step actor as a FIRM action-model diagnostic."""

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
  FirmRolloutWindowDataset,
  normalize_action_condition,
)
from smp.firm.deterministic_actor import FirmDeterministicActor
from smp.firm.expert_runtime import sha256_file


@dataclass
class TrainDeterministicActorConfig:
  """One-step action-regression configuration."""

  manifest_file: str
  additional_manifest_files: tuple[str, ...] = ()
  additional_dataset_repeats: tuple[int, ...] = ()
  normalization_checkpoint_file: str | None = None
  successful_only: bool = True
  verify_checksums: bool = True
  train_fraction: float = 0.90

  observation_latent_dim: int = 256
  goal_latent_dim: int = 128
  hidden_dims: tuple[int, ...] = (512, 512, 256)

  batch_size: int = 1024
  num_epochs: int = 40
  learning_rate: float = 3.0e-4
  weight_decay: float = 1.0e-4
  max_grad_norm: float = 1.0
  loss_beta: float = 0.10

  log_interval: int = 1
  log_dir: str = "logs/firm_deterministic_actor"
  run_name: str = "firm_deterministic_actor_c003_v1"
  use_wandb: bool = True
  wandb_project: str = "smp"

  device: str = "cuda:0"
  seed: int = 42


def _seed_everything(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _load_statistics(
  cfg: TrainDeterministicActorConfig,
  dataset: FirmRolloutWindowDataset,
  train_ids: np.ndarray,
  device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
  if cfg.normalization_checkpoint_file is None:
    return (
      {
        name: value.to(device)
        for name, value in dataset.normalization_stats(train_ids).items()
      },
      None,
    )
  path = Path(cfg.normalization_checkpoint_file).expanduser().resolve()
  payload = torch.load(path, map_location="cpu", weights_only=False)
  return (
    {name: value.to(device) for name, value in payload["normalization"].items()},
    {"path": str(path), "sha256": sha256_file(path)},
  )


def _prepare(
  batch: dict[str, torch.Tensor],
  statistics: dict[str, torch.Tensor],
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  observation = batch["observation"].to(device, non_blocking=True)
  goal = batch["goal"].to(device, non_blocking=True)
  target = batch["actions"][:, 0].to(device, non_blocking=True)
  observation, current_joint, goal = normalize_action_condition(
    observation, goal, statistics
  )
  target = (target - statistics["action_mean"]) / statistics["action_std"]
  return observation, current_joint, goal, target


def _metrics(
  prediction: torch.Tensor,
  target: torch.Tensor,
  beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  loss = F.smooth_l1_loss(prediction, target, beta=beta)
  mae = F.l1_loss(prediction, target)
  rmse = torch.sqrt(F.mse_loss(prediction, target))
  return loss, mae, rmse


@torch.no_grad()
def _validate(
  model: FirmDeterministicActor,
  loader: DataLoader,
  statistics: dict[str, torch.Tensor],
  device: torch.device,
  beta: float,
) -> dict[str, float]:
  model.eval()
  totals = torch.zeros(3, device=device)
  batches = 0
  for batch in loader:
    observation, current_joint, goal, target = _prepare(batch, statistics, device)
    prediction = model(observation, current_joint, goal)
    totals += torch.stack(_metrics(prediction, target, beta))
    batches += 1
  totals /= max(batches, 1)
  return {
    "loss": float(totals[0]),
    "mae": float(totals[1]),
    "rmse": float(totals[2]),
  }


def _save(
  path: Path,
  *,
  epoch: int,
  model_state: dict[str, torch.Tensor],
  statistics: dict[str, torch.Tensor],
  cfg: TrainDeterministicActorConfig,
  datasets: list[FirmRolloutWindowDataset],
  train_samples: int,
  validation_samples: int,
  validation: dict[str, float],
  normalization_source: dict[str, Any] | None,
) -> None:
  torch.save(
    {
      "format_version": 1,
      "policy_type": "firm_deterministic_actor",
      "epoch": epoch,
      "model": model_state,
      "normalization": {
        name: value.detach().cpu() for name, value in statistics.items()
      },
      "config": {
        **asdict(cfg),
        "observation_dim": int(statistics["observation_mean"].numel()),
      },
      "manifests": [
        {
          "path": str(dataset.manifest_path),
          "sha256": sha256_file(dataset.manifest_path),
        }
        for dataset in datasets
      ],
      "train_samples": train_samples,
      "validation_samples": validation_samples,
      "validation": validation,
      "normalization_source": normalization_source,
    },
    path,
  )


def train(cfg: TrainDeterministicActorConfig) -> Path:
  """Train a direct one-step expert-action regressor."""
  if cfg.num_epochs <= 0 or cfg.batch_size <= 0:
    raise ValueError("num_epochs and batch_size must be positive")
  repeats = cfg.additional_dataset_repeats or (1,) * len(cfg.additional_manifest_files)
  if len(repeats) != len(cfg.additional_manifest_files):
    raise ValueError("additional_dataset_repeats must match additional manifests")
  if any(repeat <= 0 for repeat in repeats):
    raise ValueError("additional_dataset_repeats must be positive")

  _seed_everything(cfg.seed)
  device = torch.device(cfg.device)
  datasets = [
    FirmRolloutWindowDataset(
      manifest,
      horizon=1,
      successful_only=cfg.successful_only,
      verify_checksums=cfg.verify_checksums,
    )
    for manifest in (cfg.manifest_file, *cfg.additional_manifest_files)
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
  train_parts: list[Subset] = [Subset(datasets[0], splits[0][0].tolist())]
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
  statistics, normalization_source = _load_statistics(
    cfg, datasets[0], splits[0][0], device
  )
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

  model = FirmDeterministicActor(
    observation_dim=observation_dim,
    observation_latent_dim=cfg.observation_latent_dim,
    goal_latent_dim=cfg.goal_latent_dim,
    hidden_dims=cfg.hidden_dims,
  ).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
  )
  print(
    f"[INFO] datasets={len(datasets)}, train={len(train_dataset)}, "
    f"validation={len(validation_dataset)}, "
    f"parameters={sum(p.numel() for p in model.parameters()):,}"
  )
  for index, dataset in enumerate(datasets):
    repeat = 1 if index == 0 else repeats[index - 1]
    print(
      f"[INFO] dataset[{index}]={dataset.manifest_path}: {len(dataset)} samples, "
      f"train_repeat={repeat}"
    )

  run_dir = (
    Path(cfg.log_dir) / cfg.run_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  run_dir.mkdir(parents=True, exist_ok=True)
  wandb_run = None
  if cfg.use_wandb:
    import wandb

    wandb_run = wandb.init(
      project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg)
    )

  best_loss = float("inf")
  best_epoch = -1
  best_state: dict[str, torch.Tensor] | None = None
  best_validation: dict[str, float] = {}
  for epoch in range(cfg.num_epochs):
    model.train()
    totals = torch.zeros(3, device=device)
    batches = 0
    for batch in train_loader:
      observation, current_joint, goal, target = _prepare(batch, statistics, device)
      prediction = model(observation, current_joint, goal)
      loss, mae, rmse = _metrics(prediction, target, cfg.loss_beta)
      optimizer.zero_grad()
      loss.backward()
      if cfg.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
      optimizer.step()
      totals += torch.stack((loss.detach(), mae.detach(), rmse.detach()))
      batches += 1
    totals /= max(batches, 1)
    validation = _validate(model, validation_loader, statistics, device, cfg.loss_beta)
    if validation["loss"] < best_loss:
      best_loss = validation["loss"]
      best_epoch = epoch
      best_validation = validation
      best_state = copy.deepcopy(model.state_dict())
    if epoch % cfg.log_interval == 0 or epoch == cfg.num_epochs - 1:
      print(
        f"Epoch {epoch:03d}/{cfg.num_epochs - 1:03d} "
        f"train={float(totals[0]):.6f}/{float(totals[2]):.6f} "
        f"validation={validation['loss']:.6f}/{validation['rmse']:.6f} "
        f"best={best_loss:.6f}@{best_epoch}"
      )
    if wandb_run is not None:
      wandb_run.log(
        {
          "epoch": epoch,
          "train/loss": float(totals[0]),
          "train/mae": float(totals[1]),
          "train/rmse": float(totals[2]),
          "validation/loss": validation["loss"],
          "validation/mae": validation["mae"],
          "validation/rmse": validation["rmse"],
          "best/validation_loss": best_loss,
        }
      )

  assert best_state is not None
  final_path = run_dir / "firm_deterministic_actor.pt"
  _save(
    final_path,
    epoch=best_epoch,
    model_state=best_state,
    statistics=statistics,
    cfg=cfg,
    datasets=datasets,
    train_samples=len(train_dataset),
    validation_samples=len(validation_dataset),
    validation=best_validation,
    normalization_source=normalization_source,
  )
  print(f"[INFO] best checkpoint: {final_path} (epoch {best_epoch})")
  if wandb_run is not None:
    wandb_run.save(str(final_path), base_path=str(run_dir))
    wandb_run.finish()
  return final_path


def main() -> None:
  train(tyro.cli(TrainDeterministicActorConfig))


if __name__ == "__main__":
  main()
