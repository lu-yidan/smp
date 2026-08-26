"""Adapt a 96-D deploy actor checkpoint to V3.8 term-wise history layout."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

TERM_DIMS = (3, 3, 3, 29, 29, 29)
DEFAULT_HISTORY_LENGTH = 4


def _repeat_term_statistics(value: torch.Tensor, history_length: int) -> torch.Tensor:
  if value.shape != (1, sum(TERM_DIMS)):
    raise ValueError(f"unexpected actor normalizer shape: {tuple(value.shape)}")
  chunks = []
  start = 0
  for width in TERM_DIMS:
    term = value[:, start : start + width]
    chunks.append(term.repeat(1, history_length))
    start += width
  return torch.cat(chunks, dim=-1)


def _expand_first_layer(value: torch.Tensor, history_length: int) -> torch.Tensor:
  if value.ndim != 2 or value.shape[1] != sum(TERM_DIMS):
    raise ValueError(f"unexpected actor first-layer shape: {tuple(value.shape)}")
  expanded = value.new_zeros(value.shape[0], sum(TERM_DIMS) * history_length)
  old_start = 0
  new_start = 0
  for width in TERM_DIMS:
    latest = new_start + (history_length - 1) * width
    expanded[:, latest : latest + width] = value[:, old_start : old_start + width]
    old_start += width
    new_start += history_length * width
  return expanded


def adapt_checkpoint(
  source: Path,
  output: Path,
  history_length: int,
  learning_rate: float,
  force: bool = False,
) -> None:
  if history_length < 2:
    raise ValueError("history_length must be at least two")
  if output.exists() and not force:
    raise FileExistsError(f"output already exists: {output}")

  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  actor = deepcopy(checkpoint["actor_state_dict"])
  if actor["mlp.0.weight"].shape[1] != sum(TERM_DIMS):
    raise ValueError("source actor is not the expected 96-D deploy policy")

  for key in (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
  ):
    actor[key] = _repeat_term_statistics(actor[key], history_length)
  actor["mlp.0.weight"] = _expand_first_layer(actor["mlp.0.weight"], history_length)
  checkpoint["actor_state_dict"] = actor

  # Preserve optimizer parameter ordering while discarding shape-dependent
  # moments. Loading this checkpoint therefore starts a clean V3.8 optimizer.
  optimizer = deepcopy(checkpoint["optimizer_state_dict"])
  optimizer["state"] = {}
  for group in optimizer["param_groups"]:
    group["lr"] = learning_rate
    if "initial_lr" in group:
      group["initial_lr"] = learning_rate
  checkpoint["optimizer_state_dict"] = optimizer
  checkpoint["iter"] = 0
  infos = deepcopy(checkpoint.get("infos") or {})
  infos["env_state"] = {"common_step_counter": 0}
  infos["actor_history_adapter"] = {
    "source": str(source),
    "source_actor_dim": sum(TERM_DIMS),
    "history_length": history_length,
    "target_actor_dim": sum(TERM_DIMS) * history_length,
    "layout": "term-wise oldest-to-newest; source weights on latest frame",
  }
  checkpoint["infos"] = infos

  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint, output)
  print(f"adapted {source} -> {output}")
  print(
    f"actor input: {sum(TERM_DIMS)} -> "
    f"{sum(TERM_DIMS) * history_length}; optimizer reset"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--history-length", type=int, default=DEFAULT_HISTORY_LENGTH)
  parser.add_argument("--learning-rate", type=float, default=5.0e-6)
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args()
  adapt_checkpoint(
    args.source,
    args.output,
    args.history_length,
    args.learning_rate,
    args.force,
  )


if __name__ == "__main__":
  main()
