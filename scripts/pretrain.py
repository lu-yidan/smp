"""Diffusion model pretraining loop."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import (
  ConcatDataset,
  DataLoader,
  Dataset,
  Subset,
  WeightedRandomSampler,
  random_split,
)

from smp.pretrain.dataset import MotionWindowDataset
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.pretrain_cfg import PretrainCfg
from smp.pretrain.scheduler import DDPMScheduler
from smp.utils import count_parameters, seed_everything


class _Ema:
  """Exponential moving average shadow of a model.

  Standard formula: ``θ_ema ← decay·θ_ema + (1−decay)·θ``, applied in-place
  over every entry of ``state_dict()`` (covers params and buffers).
  """

  def __init__(self, model: torch.nn.Module, decay: float) -> None:
    self.decay = decay
    self.shadow = copy.deepcopy(model)
    self.shadow.eval()
    for p in self.shadow.parameters():
      p.requires_grad_(False)

  @torch.no_grad()
  def update(self, model: torch.nn.Module) -> None:
    src = model.state_dict()
    dst = self.shadow.state_dict()
    for k, v_src in src.items():
      v_dst = dst[k]
      if v_dst.is_floating_point():
        v_dst.mul_(self.decay).add_(v_src.detach(), alpha=1.0 - self.decay)
      else:
        v_dst.copy_(v_src)


def _diffusion_loss(
  model: torch.nn.Module | DiffusionDenoiser,
  scheduler: DDPMScheduler,
  x_0: torch.Tensor,
  num_noise_samples: int,
) -> torch.Tensor:
  """DDPM ε-prediction L1 loss with multiple noise samples per data point.

  Each sample in the batch is paired with ``num_noise_samples`` random
  (timestep, noise) draws, giving lower-variance gradients than a single
  draw without the cost of exhausting all T timesteps.
  """
  B = x_0.shape[0]
  K = num_noise_samples
  # (B, W, F) → (B*K, W, F)
  x_0_exp = x_0[:, None].expand(B, K, *x_0.shape[1:]).reshape(B * K, *x_0.shape[1:])
  t = scheduler.sample_timesteps(B * K, x_0.device)
  noise = torch.randn_like(x_0_exp)
  x_t = scheduler.add_noise(x_0_exp, noise, t)
  return F.l1_loss(model(x_t, t), noise)


def _save_checkpoint(
  path: Path,
  epoch: int,
  model: DiffusionDenoiser,
  dataset: MotionWindowDataset,
  feature_dim: int,
  cfg: PretrainCfg,
  optimizer: torch.optim.Optimizer | None = None,
  ema: _Ema | None = None,
) -> None:
  data: dict[str, Any] = {
    "epoch": epoch,
    "model": model.state_dict(),
    "q_low": dataset.q_low,
    "q_high": dataset.q_high,
    "cfg": {
      **vars(cfg),
      "feature_dim": feature_dim,
      "window_size": dataset.window_size,
    },
  }
  if optimizer is not None:
    data["optimizer"] = optimizer.state_dict()
  if ema is not None:
    data["model_ema"] = ema.shadow.state_dict()
  torch.save(data, path)


def _split_dataset(
  dataset: Dataset[torch.Tensor],
  train_split: float,
  seed: int,
) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor]]:
  n_train = int(len(dataset) * train_split)
  n_val = len(dataset) - n_train
  if n_train == 0 or n_val == 0:
    raise ValueError(
      f"dataset split is empty: total={len(dataset)} train={n_train} val={n_val}"
    )
  generator = torch.Generator().manual_seed(seed)
  return random_split(dataset, [n_train, n_val], generator=generator)


def _split_route_dataset(
  dataset: MotionWindowDataset,
  train_split: float,
  seed: int,
) -> tuple[Subset[torch.Tensor], Subset[torch.Tensor], list[str], list[str]]:
  grouped_indices: dict[str, list[int]] = {}
  for stem, start, end in dataset.file_spans:
    base_stem = stem.removesuffix("__mirror")
    grouped_indices.setdefault(base_stem, []).extend(range(start, end))
  group_names = sorted(grouped_indices)
  if len(group_names) < 2:
    raise ValueError("route validation requires at least two independent motions")

  generator = torch.Generator().manual_seed(seed)
  permutation = torch.randperm(len(group_names), generator=generator).tolist()
  shuffled = [group_names[index] for index in permutation]
  n_train_groups = min(
    max(round(len(shuffled) * train_split), 1),
    len(shuffled) - 1,
  )
  train_groups = sorted(shuffled[:n_train_groups])
  val_groups = sorted(shuffled[n_train_groups:])
  train_indices = [index for group in train_groups for index in grouped_indices[group]]
  val_indices = [index for group in val_groups for index in grouped_indices[group]]
  return (
    Subset(dataset, train_indices),
    Subset(dataset, val_indices),
    train_groups,
    val_groups,
  )


def _build_loaders(
  general_dataset: MotionWindowDataset,
  route_dataset: MotionWindowDataset | None,
  cfg: PretrainCfg,
  pin_memory: bool,
) -> tuple[DataLoader[torch.Tensor], dict[str, DataLoader[torch.Tensor]]]:
  general_train, general_val = _split_dataset(
    general_dataset, cfg.train_split, cfg.seed
  )
  val_loaders = {
    "general": DataLoader(
      general_val,
      batch_size=cfg.batch_size,
      shuffle=False,
      pin_memory=pin_memory,
    )
  }

  if route_dataset is None:
    train_loader = DataLoader(
      general_train,
      batch_size=cfg.batch_size,
      shuffle=True,
      pin_memory=pin_memory,
    )
    print(
      f"General dataset: {len(general_dataset)} windows, "
      f"train={len(general_train)}, val={len(general_val)}"
    )
    return train_loader, val_loaders

  if (
    route_dataset.window_size != general_dataset.window_size
    or route_dataset.feature_dim != general_dataset.feature_dim
  ):
    raise ValueError("route and general datasets have incompatible window shapes")
  if not np.array_equal(
    route_dataset.q_low, general_dataset.q_low
  ) or not np.array_equal(route_dataset.q_high, general_dataset.q_high):
    raise ValueError("route and general datasets must use identical normalization")

  route_train, route_val, route_train_groups, route_val_groups = _split_route_dataset(
    route_dataset, cfg.train_split, cfg.seed + 1
  )
  combined_train = ConcatDataset([general_train, route_train])
  general_weight = (1.0 - cfg.route_train_fraction) / len(general_train)
  route_weight = cfg.route_train_fraction / len(route_train)
  sample_weights = torch.cat(
    [
      torch.full((len(general_train),), general_weight, dtype=torch.double),
      torch.full((len(route_train),), route_weight, dtype=torch.double),
    ]
  )
  samples_per_epoch = cfg.samples_per_epoch or len(combined_train)
  sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=samples_per_epoch,
    replacement=True,
    generator=torch.Generator().manual_seed(cfg.seed + 2),
  )
  train_loader = DataLoader(
    combined_train,
    batch_size=cfg.batch_size,
    sampler=sampler,
    pin_memory=pin_memory,
  )
  val_loaders["route"] = DataLoader(
    route_val,
    batch_size=cfg.batch_size,
    shuffle=False,
    pin_memory=pin_memory,
  )
  print(
    f"General dataset: {len(general_dataset)} windows, "
    f"train={len(general_train)}, val={len(general_val)}"
  )
  print(
    f"Route dataset: {len(route_dataset)} windows, "
    f"train={len(route_train)}, val={len(route_val)}"
  )
  print(
    f"Route groups: train={route_train_groups}, held_out_validation={route_val_groups}"
  )
  print(
    f"Weighted training: route_fraction={cfg.route_train_fraction:.3f}, "
    f"samples_per_epoch={samples_per_epoch}"
  )
  return train_loader, val_loaders


def _initialize_from_checkpoint(
  model: DiffusionDenoiser,
  dataset: MotionWindowDataset,
  cfg: PretrainCfg,
) -> None:
  if not cfg.init_checkpoint:
    return
  checkpoint_path = Path(cfg.init_checkpoint).expanduser().resolve()
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  checkpoint_cfg = checkpoint.get("cfg", {})
  expected = {
    "feature_dim": dataset.feature_dim,
    "window_size": dataset.window_size,
    "d_model": cfg.d_model,
    "nhead": cfg.nhead,
    "num_layers": cfg.num_layers,
  }
  mismatches = {
    key: (checkpoint_cfg.get(key), value)
    for key, value in expected.items()
    if checkpoint_cfg.get(key) != value
  }
  if mismatches:
    raise ValueError(f"initial checkpoint architecture mismatch: {mismatches}")
  if not np.array_equal(checkpoint["q_low"], dataset.q_low) or not np.array_equal(
    checkpoint["q_high"], dataset.q_high
  ):
    raise ValueError("initial checkpoint normalization differs from the dataset")

  state_key = "model_ema" if cfg.init_use_ema and "model_ema" in checkpoint else "model"
  model.load_state_dict(checkpoint[state_key], strict=True)
  print(
    f"Initialized from {checkpoint_path} [{state_key}] "
    f"at source epoch {checkpoint.get('epoch', 'unknown')}"
  )


def pretrain(cfg: PretrainCfg) -> Path:
  """Run diffusion pretraining."""
  seed_everything(cfg.seed)
  print(f"[INFO] seed={cfg.seed}")
  device = torch.device(cfg.device)

  dataset = MotionWindowDataset(cfg.data_dir, norm_stats_file=cfg.norm_stats_file)
  route_dataset = (
    MotionWindowDataset(cfg.route_data_dir, norm_stats_file=cfg.norm_stats_file)
    if cfg.route_data_dir
    else None
  )
  feature_dim = dataset.feature_dim
  window_size = dataset.window_size
  pin_memory = device.type == "cuda"
  train_loader, val_loaders = _build_loaders(dataset, route_dataset, cfg, pin_memory)
  print(f"Feature dim={feature_dim}, window size={window_size}")

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dropout=cfg.dropout,
  ).to(device)
  _initialize_from_checkpoint(model, dataset, cfg)
  scheduler = DDPMScheduler(
    num_timesteps=cfg.num_timesteps,
  ).to(device)
  print(f"Denoiser: {count_parameters(model):,} params")

  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )

  ema = _Ema(model, decay=cfg.ema_decay) if cfg.use_ema else None
  if ema is not None:
    print(f"EMA enabled (decay={cfg.ema_decay})")

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_dir = Path(cfg.log_dir) / cfg.name / timestamp
  save_dir.mkdir(parents=True, exist_ok=True)

  wandb_run = None
  if cfg.use_wandb:
    import wandb

    wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.name, config=vars(cfg))

  for epoch in range(cfg.num_epochs):
    model.train()
    epoch_loss = torch.zeros((), device=device)
    n_batches = 0

    for batch in train_loader:
      x_0 = batch.to(device, non_blocking=pin_memory)
      loss = _diffusion_loss(model, scheduler, x_0, cfg.num_noise_samples)

      optimizer.zero_grad()
      loss.backward()
      if cfg.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
      optimizer.step()
      if ema is not None:
        ema.update(model)

      epoch_loss += loss.detach()
      n_batches += 1

    avg_loss = (epoch_loss / max(n_batches, 1)).item()

    if epoch % cfg.log_interval == 0:
      eval_model = ema.shadow if ema is not None else model
      val_losses = {
        name: _validate(
          eval_model, scheduler, loader, device, pin_memory, cfg.num_noise_samples
        )
        for name, loader in val_loaders.items()
      }
      mixed_val_loss = (
        (1.0 - cfg.route_train_fraction) * val_losses["general"]
        + cfg.route_train_fraction * val_losses["route"]
        if "route" in val_losses
        else val_losses["general"]
      )
      metrics = {
        "epoch": epoch,
        "train/loss": avg_loss,
        "val/loss": mixed_val_loss,
        **{f"val/loss_{name}": loss for name, loss in val_losses.items()},
      }
      val_text = " ".join(f"val_{name}={loss:.6f}" for name, loss in val_losses.items())
      print(f"Epoch {epoch:4d} | train={avg_loss:.6f} | {val_text}")
      if wandb_run is not None:
        wandb_run.log(metrics)

    if epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs - 1:
      ckpt_path = save_dir / f"checkpoint_{epoch:05d}.pt"
      _save_checkpoint(
        ckpt_path, epoch, model, dataset, feature_dim, cfg, optimizer, ema
      )
      if wandb_run is not None:
        wandb_run.save(str(ckpt_path), base_path=str(save_dir))

  final_path = save_dir / "pretrained.pt"
  _save_checkpoint(
    final_path, cfg.num_epochs, model, dataset, feature_dim, cfg, ema=ema
  )
  print(f"Saved final checkpoint to {final_path}")

  if wandb_run is not None:
    wandb_run.save(str(final_path), base_path=str(save_dir))
    wandb_run.finish()

  return final_path


@torch.no_grad()
def _validate(
  model: torch.nn.Module | DiffusionDenoiser,
  scheduler: DDPMScheduler,
  val_loader: DataLoader[torch.Tensor],
  device: torch.device,
  pin_memory: bool,
  num_noise_samples: int,
) -> float:
  model.eval()
  total = torch.zeros((), device=device)
  n = 0
  for batch in val_loader:
    x_0 = batch.to(device, non_blocking=pin_memory)
    total += _diffusion_loss(model, scheduler, x_0, num_noise_samples)
    n += 1
  return (total / max(n, 1)).item()


if __name__ == "__main__":
  pretrain(tyro.cli(PretrainCfg))
