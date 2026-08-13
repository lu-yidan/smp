"""Train the FIRM online keyframe-goal adapter."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from smp.firm.action_diffusion import load_action_diffusion_checkpoint
from smp.firm.expert_runtime import sha256_file
from smp.firm.goal_adapter import FirmGoalAdapter, FirmGoalAdapterDataset


@dataclass(frozen=True)
class TrainGoalAdapterConfig:
  action_checkpoint_file: str
  manifest_file: str
  additional_manifest_files: tuple[str, ...] = ()
  history_steps: int = 50
  channels: tuple[int, int, int] = (128, 256, 256)
  train_fraction: float = 0.9
  successful_only: bool = True
  balance_goal_sampling: bool = True
  batch_size: int = 1024
  num_epochs: int = 20
  learning_rate: float = 3e-4
  weight_decay: float = 1e-4
  seed: int = 42
  num_workers: int = 4
  log_interval: int = 1
  output_root: str = "logs/firm_goal_adapter"
  run_name: str = "firm_goal_adapter"
  use_wandb: bool = True
  wandb_project: str = "smp"
  device: str = "cuda"


def _seed_everything(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _target_latent(
  action_model,
  goal: torch.Tensor,
  joint_mean: torch.Tensor,
  joint_std: torch.Tensor,
) -> torch.Tensor:
  normalized_goal = (goal - joint_mean) / joint_std
  return F.normalize(action_model.goal_encoder(normalized_goal), dim=-1)


def _evaluate(
  adapter: FirmGoalAdapter,
  action_model,
  loader: DataLoader,
  statistics: dict[str, torch.Tensor],
  codebook_features: torch.Tensor,
  device: torch.device,
) -> tuple[float, float, float]:
  adapter.eval()
  loss_sum = 0.0
  correct = 0
  retrieval_correct = 0
  count = 0
  with torch.no_grad():
    for batch in loader:
      history = batch["observation_history"].to(device)
      goal = batch["goal"].to(device)
      goal_index = batch["goal_index"].to(device)
      history = (
        history - statistics["observation_mean"]
      ) / statistics["observation_std"]
      target = _target_latent(
        action_model,
        goal,
        statistics["joint_mean"],
        statistics["joint_std"],
      )
      predicted = adapter(history)
      cosine = torch.sum(predicted * target, dim=-1)
      loss_sum += float((1.0 - cosine).sum())
      correct += int((cosine >= 0.95).sum())
      count += len(history)
      retrieved = (predicted @ codebook_features.transpose(0, 1)).argmax(dim=-1)
      retrieval_correct += int((retrieved == goal_index).sum())
  return (
    loss_sum / max(count, 1),
    correct / max(count, 1),
    retrieval_correct / max(count, 1),
  )


def train(cfg: TrainGoalAdapterConfig) -> Path:
  if cfg.num_epochs <= 0 or cfg.batch_size <= 0:
    raise ValueError("num_epochs and batch_size must be positive")
  _seed_everything(cfg.seed)
  device = torch.device(cfg.device)
  manifests = (cfg.manifest_file, *cfg.additional_manifest_files)
  dataset = FirmGoalAdapterDataset(
    manifests,
    history_steps=cfg.history_steps,
    successful_only=cfg.successful_only,
  )
  train_ids, validation_ids = dataset.split_sample_indices(
    cfg.train_fraction, cfg.seed
  )
  generator = torch.Generator().manual_seed(cfg.seed)
  train_sampler = None
  if cfg.balance_goal_sampling:
    train_goal_indices = dataset.sample_goal_indices[train_ids]
    class_counts = np.bincount(
      train_goal_indices, minlength=len(dataset.codebook_goals())
    )
    sample_weights = 1.0 / class_counts[train_goal_indices]
    train_sampler = WeightedRandomSampler(
      torch.from_numpy(sample_weights),
      num_samples=len(train_ids),
      replacement=True,
      generator=generator,
    )
  train_loader = DataLoader(
    Subset(dataset, train_ids.tolist()),
    batch_size=cfg.batch_size,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    generator=generator,
    num_workers=cfg.num_workers,
    pin_memory=device.type == "cuda",
  )
  validation_loader = DataLoader(
    Subset(dataset, validation_ids.tolist()),
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=device.type == "cuda",
  )

  action_model, _, statistics, action_payload = load_action_diffusion_checkpoint(
    cfg.action_checkpoint_file,
    device,
    use_ema=True,
  )
  action_model.requires_grad_(False)
  with torch.no_grad():
    latent_dim = action_model.goal_encoder(
      torch.zeros(1, 29, device=device)
    ).shape[-1]
  codebook_goals = dataset.codebook_goals().to(device)
  with torch.no_grad():
    codebook_features = _target_latent(
      action_model,
      codebook_goals,
      statistics["joint_mean"],
      statistics["joint_std"],
    )
  adapter = FirmGoalAdapter(
    history_steps=cfg.history_steps,
    latent_dim=int(latent_dim),
    channels=cfg.channels,
  ).to(device)
  optimizer = torch.optim.AdamW(
    adapter.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
  )

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  output_dir = Path(cfg.output_root) / cfg.run_name / timestamp
  output_dir.mkdir(parents=True, exist_ok=False)
  wandb_run = None
  if cfg.use_wandb:
    import wandb

    wandb_run = wandb.init(
      project=cfg.wandb_project,
      name=cfg.run_name,
      config=asdict(cfg),
    )

  print(
    f"[INFO] adapter samples: train={len(train_ids)} "
    f"validation={len(validation_ids)} episodes="
    f"{len(np.unique(dataset.sample_episode_groups))}"
  )
  for epoch in range(cfg.num_epochs):
    adapter.train()
    train_loss_sum = 0.0
    train_count = 0
    for batch in train_loader:
      history = batch["observation_history"].to(device, non_blocking=True)
      goal = batch["goal"].to(device, non_blocking=True)
      history = (
        history - statistics["observation_mean"]
      ) / statistics["observation_std"]
      with torch.no_grad():
        target = _target_latent(
          action_model,
          goal,
          statistics["joint_mean"],
          statistics["joint_std"],
        )
      predicted = adapter(history)
      loss = (1.0 - torch.sum(predicted * target, dim=-1)).mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss_sum += float(loss.detach()) * len(history)
      train_count += len(history)

    validation_loss, validation_cos095, validation_retrieval_accuracy = _evaluate(
      adapter,
      action_model,
      validation_loader,
      statistics,
      codebook_features,
      device,
    )
    train_loss = train_loss_sum / max(train_count, 1)
    if epoch % cfg.log_interval == 0 or epoch == cfg.num_epochs - 1:
      print(
        f"Epoch {epoch:03d} | train={train_loss:.6f} "
        f"validation={validation_loss:.6f} "
        f"val_cos>=0.95={validation_cos095:.4f} "
        f"val_retrieval={validation_retrieval_accuracy:.4f}"
      )
    if wandb_run is not None:
      wandb_run.log(
        {
          "epoch": epoch,
          "train/loss": train_loss,
          "validation/loss": validation_loss,
          "validation/cosine_ge_0.95": validation_cos095,
          "validation/retrieval_accuracy": validation_retrieval_accuracy,
        }
      )

  codebook_goals = dataset.codebook_goals().to(device)
  with torch.no_grad():
    codebook_features = _target_latent(
      action_model,
      codebook_goals,
      statistics["joint_mean"],
      statistics["joint_std"],
    )
  checkpoint_path = output_dir / "firm_goal_adapter.pt"
  payload = {
    "format_version": 1,
    "epoch": cfg.num_epochs - 1,
    "model": adapter.state_dict(),
    "config": {
      **asdict(cfg),
      "observation_dim": adapter.observation_dim,
      "latent_dim": adapter.latent_dim,
    },
    "observation_mean": statistics["observation_mean"],
    "observation_std": statistics["observation_std"],
    "codebook_goals": codebook_goals,
    "codebook_features": codebook_features,
    "artifacts": {
      "action_checkpoint": str(Path(cfg.action_checkpoint_file).resolve()),
      "action_checkpoint_sha256": sha256_file(cfg.action_checkpoint_file),
      "action_checkpoint_epoch": int(action_payload["epoch"]),
      "manifest_sha256": {
        str(Path(path).resolve()): sha256_file(path) for path in manifests
      },
    },
    "dataset": {
      "train_samples": int(len(train_ids)),
      "validation_samples": int(len(validation_ids)),
      "episodes": int(len(np.unique(dataset.sample_episode_groups))),
      "codebook_entries": int(len(codebook_goals)),
    },
  }
  torch.save(payload, checkpoint_path)
  metadata_path = output_dir / "metadata.json"
  metadata_path.write_text(
    json.dumps(
      {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset": payload["dataset"],
        "artifacts": payload["artifacts"],
      },
      indent=2,
    )
    + "\n"
  )
  print(f"[INFO] saved {checkpoint_path} sha256={sha256_file(checkpoint_path)}")
  if wandb_run is not None:
    wandb_run.save(str(checkpoint_path), base_path=str(output_dir))
    wandb_run.save(str(metadata_path), base_path=str(output_dir))
    wandb_run.finish()
  return checkpoint_path


def main() -> None:
  train(tyro.cli(TrainGoalAdapterConfig))


if __name__ == "__main__":
  main()
