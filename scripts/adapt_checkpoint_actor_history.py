"""Expand a deploy actor checkpoint while preserving its history behavior."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

TERM_DIMS = (3, 3, 3, 29, 29, 29)
DEFAULT_HISTORY_LENGTH = 4


def _infer_history_length(actor_input_dim: int) -> int:
  single_frame_dim = sum(TERM_DIMS)
  if actor_input_dim % single_frame_dim != 0:
    raise ValueError(f"unexpected actor input dimension: {actor_input_dim}")
  return actor_input_dim // single_frame_dim


def _expand_term_statistics(
  value: torch.Tensor, source_history_length: int, target_history_length: int
) -> torch.Tensor:
  expected = sum(TERM_DIMS) * source_history_length
  if value.shape != (1, expected):
    raise ValueError(f"unexpected actor normalizer shape: {tuple(value.shape)}")
  chunks: list[torch.Tensor] = []
  source_start = 0
  for width in TERM_DIMS:
    source_width = width * source_history_length
    source = value[:, source_start : source_start + source_width].reshape(
      1, source_history_length, width
    )
    # Older slots have zero actor weights initially, but copying the oldest
    # source statistics gives them a numerically sensible starting scale once
    # PPO begins learning temporal features.
    target = source[:, :1, :].expand(1, target_history_length, width).clone()
    target[:, -source_history_length:, :] = source
    chunks.append(target.reshape(1, target_history_length * width))
    source_start += source_width
  return torch.cat(chunks, dim=-1)


def _expand_first_layer(
  value: torch.Tensor, source_history_length: int, target_history_length: int
) -> torch.Tensor:
  expected = sum(TERM_DIMS) * source_history_length
  if value.ndim != 2 or value.shape[1] != expected:
    raise ValueError(f"unexpected actor first-layer shape: {tuple(value.shape)}")
  expanded = value.new_zeros(value.shape[0], sum(TERM_DIMS) * target_history_length)
  source_start = 0
  target_start = 0
  for width in TERM_DIMS:
    source_width = source_history_length * width
    target_width = target_history_length * width
    aligned_start = target_start + target_width - source_width
    expanded[:, aligned_start : aligned_start + source_width] = value[
      :, source_start : source_start + source_width
    ]
    source_start += source_width
    target_start += target_width
  return expanded


def _verify_first_layer_equivalence(
  source_actor: dict[str, torch.Tensor],
  target_actor: dict[str, torch.Tensor],
  source_history_length: int,
  target_history_length: int,
) -> float:
  source_dim = sum(TERM_DIMS) * source_history_length
  target_dim = sum(TERM_DIMS) * target_history_length
  generator = torch.Generator(device="cpu").manual_seed(382)
  source_obs = torch.randn(32, source_dim, generator=generator)
  target_obs = torch.randn(32, target_dim, generator=generator)
  source_start = 0
  target_start = 0
  for width in TERM_DIMS:
    source_width = source_history_length * width
    target_width = target_history_length * width
    target_obs[
      :, target_start + target_width - source_width : target_start + target_width
    ] = source_obs[:, source_start : source_start + source_width]
    source_start += source_width
    target_start += target_width
  source_normalized = (source_obs - source_actor["obs_normalizer._mean"]) / torch.clamp(
    source_actor["obs_normalizer._std"], min=1e-8
  )
  target_normalized = (target_obs - target_actor["obs_normalizer._mean"]) / torch.clamp(
    target_actor["obs_normalizer._std"], min=1e-8
  )
  source_output = torch.nn.functional.linear(
    source_normalized, source_actor["mlp.0.weight"], source_actor["mlp.0.bias"]
  )
  target_output = torch.nn.functional.linear(
    target_normalized, target_actor["mlp.0.weight"], target_actor["mlp.0.bias"]
  )
  error = float((source_output - target_output).abs().max())
  torch.testing.assert_close(source_output, target_output, rtol=2e-5, atol=2e-5)
  return error


def adapt_checkpoint(
  source: Path,
  output: Path,
  history_length: int,
  learning_rate: float,
  force: bool = False,
) -> None:
  if output.exists() and not force:
    raise FileExistsError(f"output already exists: {output}")

  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  source_actor = checkpoint["actor_state_dict"]
  source_history_length = _infer_history_length(source_actor["mlp.0.weight"].shape[1])
  if history_length < source_history_length:
    raise ValueError(
      "target history length must not be shorter than source history length; "
      f"got {source_history_length} -> {history_length}"
    )
  actor = deepcopy(source_actor)

  for key in (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
  ):
    actor[key] = _expand_term_statistics(
      actor[key], source_history_length, history_length
    )
  actor["mlp.0.weight"] = _expand_first_layer(
    actor["mlp.0.weight"], source_history_length, history_length
  )
  equivalence_error = _verify_first_layer_equivalence(
    source_actor, actor, source_history_length, history_length
  )
  checkpoint["actor_state_dict"] = actor

  # Preserve optimizer parameter ordering while discarding shape-dependent
  # moments. Loading this checkpoint therefore starts a clean optimizer.
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
    "source_history_length": source_history_length,
    "target_history_length": history_length,
    "source_actor_dim": sum(TERM_DIMS) * source_history_length,
    "target_actor_dim": sum(TERM_DIMS) * history_length,
    "layout": "term-wise oldest-to-newest; source history right-aligned",
    "first_layer_equivalence_max_abs_error": equivalence_error,
  }
  checkpoint["infos"] = infos

  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint, output)
  print(f"adapted {source} -> {output}")
  print(
    f"actor input: {sum(TERM_DIMS) * source_history_length} -> "
    f"{sum(TERM_DIMS) * history_length}; optimizer reset"
  )
  print(f"first-layer equivalence max abs error: {equivalence_error:.3e}")


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
