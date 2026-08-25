"""Prepare a V3.3 checkpoint for low-exploration deployable fine-tuning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass(frozen=True)
class ConvertCfg:
  input: Path
  output: Path
  action_std: float = 0.15


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def main(cfg: ConvertCfg) -> None:
  if not 0.0 < cfg.action_std <= 1.0:
    raise ValueError("action_std must be in (0, 1]")
  if cfg.output.exists():
    raise FileExistsError(f"Refusing to overwrite existing checkpoint: {cfg.output}")

  checkpoint = torch.load(cfg.input, map_location="cpu", weights_only=False)
  actor_state = checkpoint.get("actor_state_dict")
  if not isinstance(actor_state, dict):
    raise KeyError("Expected actor_state_dict in checkpoint")
  std_key = "distribution.std_param"
  if std_key not in actor_state:
    raise KeyError(f"Expected {std_key} in actor_state_dict")

  original_std = actor_state[std_key]
  if original_std.ndim != 1:
    raise ValueError(f"Unexpected action std shape: {tuple(original_std.shape)}")
  actor_state[std_key] = torch.full_like(original_std, cfg.action_std)

  infos = checkpoint.get("infos") or {}
  infos["deploy_seed"] = {
    "source_checkpoint": str(cfg.input),
    "source_sha256": sha256(cfg.input),
    "action_std_before": float(original_std.mean()),
    "action_std_after": cfg.action_std,
    "deterministic_actor_weights_changed": False,
  }
  checkpoint["infos"] = infos

  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint, cfg.output)
  print(f"source_sha256={infos['deploy_seed']['source_sha256']}")
  print(f"output_sha256={sha256(cfg.output)}")
  print(f"action_std={cfg.action_std}")


if __name__ == "__main__":
  main(tyro.cli(ConvertCfg))
