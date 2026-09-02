"""Project the frozen V3.4 96D actor to a deployable 93D warm start."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

_EXPECTED_SOURCE_SHA256 = (
  "fa54ac58f09a1a0ed0b46f96fb920f18de20422190c9ee92207f3080a3cbe393"
)


@dataclass(frozen=True)
class ProjectionCfg:
  source: Path = Path("run_control/v34_93d_control/source/model_98000.pt")
  output: Path = Path(
    "run_control/v34_93d_control/projected/model_98000_zero_velocity_93d.pt"
  )
  manifest: Path = Path("run_control/v34_93d_control/projected/projection_manifest.json")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _finite_tensor_inventory(value: Any) -> tuple[int, int]:
  tensors: list[torch.Tensor] = []

  def collect(item: Any) -> None:
    if torch.is_tensor(item):
      tensors.append(item)
    elif isinstance(item, dict):
      for child in item.values():
        collect(child)
    elif isinstance(item, (tuple, list)):
      for child in item:
        collect(child)

  collect(value)
  if not tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
    raise RuntimeError("V34_93D_PROJECTION_ALERT: checkpoint contains non-finite tensors")
  return len(tensors), sum(tensor.numel() for tensor in tensors)


def project_checkpoint(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  """Remove base_lin_vel while preserving the policy at raw velocity zero."""
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  if checkpoint.get("iter") != 98000:
    raise ValueError("V34 source embedded iteration is not 98000")
  actor = checkpoint.get("actor_state_dict")
  critic = checkpoint.get("critic_state_dict")
  if not isinstance(actor, dict) or not isinstance(critic, dict):
    raise ValueError("V34 source lacks actor or critic state")
  weight = actor.get("mlp.0.weight")
  bias = actor.get("mlp.0.bias")
  mean = actor.get("obs_normalizer._mean")
  std = actor.get("obs_normalizer._std")
  if (
    not torch.is_tensor(weight)
    or tuple(weight.shape) != (512, 96)
    or not torch.is_tensor(bias)
    or tuple(bias.shape) != (512,)
    or not torch.is_tensor(mean)
    or tuple(mean.shape) != (1, 96)
    or not torch.is_tensor(std)
    or tuple(std.shape) != (1, 96)
  ):
    raise ValueError("unexpected V34 actor architecture or normalizer")
  if tuple(critic["mlp.0.weight"].shape) != (512, 960):
    raise ValueError("unexpected V34 critic input dimension")

  # The removed actor term is the first three entries.  Fold its contribution
  # at raw base linear velocity == 0 into the first-layer bias.  The projected
  # network therefore exactly matches the 96D policy under the old deploy-time
  # zero-velocity convention before it starts adapting to velocity-free input.
  normalized_zero_velocity = (torch.zeros_like(mean[:, :3]) - mean[:, :3]) / std[
    :, :3
  ].clamp_min(1.0e-8)
  removed_contribution = torch.matmul(
    weight[:, :3], normalized_zero_velocity.reshape(3)
  )

  projected = copy.deepcopy(checkpoint)
  projected_actor = projected["actor_state_dict"]
  projected_actor["mlp.0.weight"] = weight[:, 3:].clone()
  projected_actor["mlp.0.bias"] = bias + removed_contribution
  for key in (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
  ):
    value = actor.get(key)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 96):
      raise ValueError(f"unexpected V34 actor normalizer tensor {key}")
    projected_actor[key] = value[:, 3:].clone()

  # An optimizer over the old 96D first layer is structurally invalid.  Keep
  # the checkpoint key but empty it so accidental continuation fails loudly;
  # the registered warm-start runner explicitly creates a fresh optimizer and
  # resets iteration and environment steps.
  projected["optimizer_state_dict"] = {}
  projected["infos"] = {
    "projection": {
      "schema_version": 1,
      "source_iteration": 98000,
      "removed_actor_term": "base_lin_vel",
      "removed_actor_indices": [0, 1, 2],
      "bias_reference_raw_base_lin_vel": [0.0, 0.0, 0.0],
      "critic_unchanged": True,
      "optimizer_usable": False,
    }
  }

  projected_weight = projected_actor["mlp.0.weight"]
  projected_bias = projected_actor["mlp.0.bias"]
  generator = torch.Generator().manual_seed(20260902)
  raw_rest = torch.randn(257, 93, generator=generator)
  raw_96 = torch.cat((torch.zeros(257, 3), raw_rest), dim=1)
  old_normalized = (raw_96 - mean) / std.clamp_min(1.0e-8)
  new_mean = projected_actor["obs_normalizer._mean"]
  new_std = projected_actor["obs_normalizer._std"]
  new_normalized = (raw_rest - new_mean) / new_std.clamp_min(1.0e-8)
  old_first = old_normalized @ weight.T + bias
  new_first = new_normalized @ projected_weight.T + projected_bias
  max_abs_error = float((old_first - new_first).abs().max())
  if max_abs_error > 2.0e-5:
    raise RuntimeError(
      "V34_93D_PROJECTION_ALERT: zero-velocity equivalence failed "
      f"({max_abs_error})"
    )
  tensor_count, tensor_elements = _finite_tensor_inventory(
    {
      "actor_state_dict": projected["actor_state_dict"],
      "critic_state_dict": projected["critic_state_dict"],
    }
  )
  audit = {
    "schema_version": 1,
    "status": "PROJECTED_SOURCE_READY_NOT_PERFORMANCE_EVIDENCE",
    "source_checkpoint_sha256": _EXPECTED_SOURCE_SHA256,
    "source_iteration": 98000,
    "source_actor_input_dim": 96,
    "projected_actor_input_dim": 93,
    "critic_input_dim": 960,
    "removed_actor_term": "base_lin_vel",
    "removed_actor_indices": [0, 1, 2],
    "bias_reference_raw_base_lin_vel": [0.0, 0.0, 0.0],
    "zero_velocity_first_layer_max_abs_error": max_abs_error,
    "optimizer_discarded": True,
    "tensor_count": tensor_count,
    "tensor_elements": tensor_elements,
    "all_tensors_finite": True,
  }
  return projected, audit


def run(cfg: ProjectionCfg) -> dict[str, Any]:
  source = cfg.source.resolve()
  output = cfg.output.resolve()
  manifest = cfg.manifest.resolve()
  if not source.is_file() or _sha256(source) != _EXPECTED_SOURCE_SHA256:
    raise RuntimeError("V34_93D_SOURCE_ALERT: source missing or SHA-256 drifted")
  if output.exists() or manifest.exists():
    raise FileExistsError("refusing to overwrite an existing projection artifact")
  projected, audit = project_checkpoint(source)
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_suffix(output.suffix + ".tmp")
  torch.save(projected, temporary)
  temporary.replace(output)
  audit.update(
    {
      "source_checkpoint": str(source),
      "projected_checkpoint": str(output),
      "projected_checkpoint_sha256": _sha256(output),
    }
  )
  manifest.parent.mkdir(parents=True, exist_ok=True)
  temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
  temporary_manifest.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
  temporary_manifest.replace(manifest)
  return audit


def main(cfg: ProjectionCfg) -> None:
  print(json.dumps(run(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(ProjectionCfg))
