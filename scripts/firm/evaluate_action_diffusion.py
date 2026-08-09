"""Evaluate full DDPM action sampling against held-out expert windows."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader, Subset

from smp.firm.action_diffusion import (
  FirmRolloutWindowDataset,
  denormalize_actions,
  load_action_diffusion_checkpoint,
  normalize_action_condition,
  sample_action_horizon,
)
from smp.firm.expert_runtime import sha256_file


@dataclass(frozen=True)
class EvaluateActionDiffusionConfig:
  """Held-out conditional action sampling configuration."""

  checkpoint_file: str
  manifest_file: str
  batch_size: int = 256
  max_windows: int = 0
  """Zero evaluates every held-out window; otherwise sample this many."""
  use_ema: bool = True
  successful_only: bool = True
  verify_checksums: bool = True
  device: str = "cuda:0"
  seed: int = 42
  output_file: str | None = None


def _rms(sum_squared: torch.Tensor, count: int) -> float:
  return float(torch.sqrt(sum_squared / max(count, 1)).item())


def evaluate(cfg: EvaluateActionDiffusionConfig) -> dict:
  """Sample action horizons and compare them with held-out expert sequences."""
  if cfg.batch_size <= 0 or cfg.max_windows < 0:
    raise ValueError("batch_size must be positive and max_windows non-negative")
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)

  device = torch.device(cfg.device)
  model, scheduler, statistics, checkpoint = load_action_diffusion_checkpoint(
    cfg.checkpoint_file,
    device,
    use_ema=cfg.use_ema,
  )
  train_cfg = checkpoint["config"]
  dataset = FirmRolloutWindowDataset(
    cfg.manifest_file,
    horizon=int(train_cfg["horizon"]),
    successful_only=cfg.successful_only,
    verify_checksums=cfg.verify_checksums,
  )
  _, validation_ids = dataset.split_window_indices(
    float(train_cfg["train_fraction"]), int(train_cfg["seed"])
  )
  rng = np.random.default_rng(cfg.seed)
  rng.shuffle(validation_ids)
  if cfg.max_windows > 0:
    validation_ids = validation_ids[: cfg.max_windows]
  loader = DataLoader(
    Subset(dataset, validation_ids.tolist()),
    batch_size=cfg.batch_size,
    shuffle=False,
    pin_memory=device.type == "cuda",
  )

  horizon = int(train_cfg["horizon"])
  action_dim = model.action_dim
  total_squared_error = torch.zeros((), device=device)
  total_absolute_error = torch.zeros((), device=device)
  normalized_squared_error = torch.zeros((), device=device)
  first_squared_error = torch.zeros((), device=device)
  first_absolute_error = torch.zeros((), device=device)
  per_step_squared_error = torch.zeros(horizon, device=device)
  target_squared = torch.zeros((), device=device)
  predicted_squared = torch.zeros((), device=device)
  target_rate_squared = torch.zeros((), device=device)
  predicted_rate_squared = torch.zeros((), device=device)
  window_rmse: list[torch.Tensor] = []
  first_rmse: list[torch.Tensor] = []
  processed = 0
  nonfinite_windows = 0
  start_time = time.perf_counter()

  for batch_index, batch in enumerate(loader):
    observation = batch["observation"].to(device, non_blocking=True)
    goal = batch["goal"].to(device, non_blocking=True)
    target = batch["actions"].to(device, non_blocking=True)
    normalized_observation, current_joint, normalized_goal = normalize_action_condition(
      observation, goal, statistics
    )
    normalized_prediction = sample_action_horizon(
      model,
      scheduler,
      normalized_observation,
      current_joint,
      normalized_goal,
    )
    prediction = denormalize_actions(normalized_prediction, statistics)
    normalized_target = (target - statistics["action_mean"]) / statistics["action_std"]

    finite = torch.isfinite(prediction).all(dim=-1).all(dim=-1)
    nonfinite_windows += int((~finite).sum().item())
    if not finite.all():
      prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
      normalized_prediction = torch.nan_to_num(
        normalized_prediction, nan=0.0, posinf=0.0, neginf=0.0
      )

    error = prediction - target
    normalized_error = normalized_prediction - normalized_target
    batch_size = target.shape[0]
    processed += batch_size
    total_squared_error += torch.square(error).sum()
    total_absolute_error += error.abs().sum()
    normalized_squared_error += torch.square(normalized_error).sum()
    first_squared_error += torch.square(error[:, 0]).sum()
    first_absolute_error += error[:, 0].abs().sum()
    per_step_squared_error += torch.square(error).sum(dim=(0, 2))
    target_squared += torch.square(target).sum()
    predicted_squared += torch.square(prediction).sum()

    previous_action = observation[:, 61:90]
    target_with_previous = torch.cat([previous_action[:, None], target], dim=1)
    prediction_with_previous = torch.cat([previous_action[:, None], prediction], dim=1)
    target_rate_squared += torch.square(torch.diff(target_with_previous, dim=1)).sum()
    predicted_rate_squared += torch.square(
      torch.diff(prediction_with_previous, dim=1)
    ).sum()
    window_rmse.append(torch.sqrt(torch.mean(torch.square(error), dim=(1, 2))).cpu())
    first_rmse.append(torch.sqrt(torch.mean(torch.square(error[:, 0]), dim=1)).cpu())

    if batch_index % 10 == 0:
      print(
        f"[INFO] batches={batch_index + 1}/{len(loader)} "
        f"windows={processed}/{len(validation_ids)}"
      )

  elapsed = time.perf_counter() - start_time
  action_elements = processed * horizon * action_dim
  first_elements = processed * action_dim
  rate_elements = action_elements
  per_step_count = processed * action_dim
  window_rmse_tensor = torch.cat(window_rmse).float()
  first_rmse_tensor = torch.cat(first_rmse).float()
  per_step_rmse = torch.sqrt(per_step_squared_error / max(per_step_count, 1))

  metrics = {
    "windows": processed,
    "nonfinite_windows": nonfinite_windows,
    "finite_window_rate": 1.0 - nonfinite_windows / max(processed, 1),
    "sampling_seconds": elapsed,
    "windows_per_second": processed / max(elapsed, 1.0e-9),
    "horizon_action_rmse": _rms(total_squared_error, action_elements),
    "horizon_action_mae": float(
      (total_absolute_error / max(action_elements, 1)).item()
    ),
    "normalized_horizon_rmse": _rms(normalized_squared_error, action_elements),
    "first_action_rmse": _rms(first_squared_error, first_elements),
    "first_action_mae": float((first_absolute_error / max(first_elements, 1)).item()),
    "first_action_window_rmse_p50": float(
      torch.quantile(first_rmse_tensor, 0.50).item()
    ),
    "first_action_window_rmse_p95": float(
      torch.quantile(first_rmse_tensor, 0.95).item()
    ),
    "horizon_window_rmse_p50": float(torch.quantile(window_rmse_tensor, 0.50).item()),
    "horizon_window_rmse_p95": float(torch.quantile(window_rmse_tensor, 0.95).item()),
    "target_action_rms": _rms(target_squared, action_elements),
    "predicted_action_rms": _rms(predicted_squared, action_elements),
    "target_action_rate_rms": _rms(target_rate_squared, rate_elements),
    "predicted_action_rate_rms": _rms(predicted_rate_squared, rate_elements),
    "per_step_action_rmse": per_step_rmse.cpu().tolist(),
  }
  result = {
    "format_version": 1,
    "config": asdict(cfg),
    "checkpoint": {
      "path": str(Path(cfg.checkpoint_file).expanduser().resolve()),
      "sha256": sha256_file(cfg.checkpoint_file),
      "epoch": int(checkpoint["epoch"]),
      "training_validation_loss": float(checkpoint["validation_loss"]),
      "manifest_sha256": checkpoint["manifest_sha256"],
      "weights": "ema" if cfg.use_ema else "online",
    },
    "dataset": {
      "manifest": str(Path(cfg.manifest_file).expanduser().resolve()),
      "manifest_sha256": sha256_file(cfg.manifest_file),
      "all_valid_windows": len(dataset),
      "held_out_windows": len(validation_ids),
    },
    "metrics": metrics,
  }
  print(json.dumps(metrics, indent=2))
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[INFO] Evaluation written to {output_path}")
  return result


def main() -> None:
  evaluate(tyro.cli(EvaluateActionDiffusionConfig))


if __name__ == "__main__":
  main()
