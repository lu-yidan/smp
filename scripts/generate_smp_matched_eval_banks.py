"""Generate disjoint, SHA-locked held-out reset banks for Tier-A evaluation."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401
from build_smp_causal_manifest import _ARMS
from generate_smp_matched_reset_bank import _capture_batch
from smp.rl.tasks.getup.mdp.events import _validate_matched_reset_bank_payload

_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")
_MODE_WEIGHTS = {
  "prone": (1.0, 0.0, 0.0, 0.0),
  "supine": (0.0, 1.0, 0.0, 0.0),
  "left_side": (0.0, 0.0, 1.0, 0.0),
  "right_side": (0.0, 0.0, 0.0, 1.0),
}
_DEFAULT_PROCEDURAL_PARAMS = {
  "root_height_range": (0.48, 0.62),
  "joint_noise": 0.12,
  "orientation_noise": 0.0,
  "root_xy_range": 0.1,
  "root_linear_velocity": 0.1,
  "root_angular_velocity": 0.2,
}


@dataclass(frozen=True)
class EvalBankCfg:
  promotion: Path
  training_bank: Path = Path("run_control/ral_baselines/matched_reset_bank.pt")
  training_bank_manifest: Path = Path(
    "run_control/ral_baselines/matched_reset_bank.json"
  )
  registry: Path = Path("docs/ral_baseline_registry.json")
  output_dir: Path = Path("run_control/ral_baselines/held_out_eval_banks")
  manifest: Path = Path("run_control/ral_baselines/held_out_eval_banks.json")
  num_states_per_mode: int = 512
  seed: int = 20260829
  device: str = "cuda:6"
  run: bool = False


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _gpu_processes() -> list[str]:
  result = subprocess.run(
    ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"),
    check=True,
    capture_output=True,
    text=True,
  )
  return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_commit() -> str:
  result = subprocess.run(
    ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
  )
  return result.stdout.strip()


def _verify_source(path_value: Any, digest: Any, label: str) -> Path:
  if not isinstance(path_value, str) or not isinstance(digest, str):
    raise ValueError(f"source lacks {label} provenance")
  path = Path(path_value)
  if not path.is_file() or _sha256(path) != digest:
    raise ValueError(f"{label} source changed: {path}")
  return path.resolve()


def build_plan(cfg: EvalBankCfg) -> dict[str, Any]:
  promotion = _load_json(cfg.promotion)
  if promotion.get("status") != "PROMOTE_TP_SPECIALISTS":
    raise ValueError("flat promotion does not authorize held-out bank generation")
  _verify_source(
    promotion.get("aggregate"), promotion.get("aggregate_sha256"), "aggregate"
  )
  _verify_source(
    promotion.get("confirmation_manifest_index"),
    promotion.get("confirmation_manifest_index_sha256"),
    "confirmation manifest",
  )
  _verify_source(
    promotion.get("protocol"), promotion.get("protocol_sha256"), "protocol"
  )
  training_manifest = _load_json(cfg.training_bank_manifest)
  if (
    training_manifest.get("status") != "READY"
    or training_manifest.get("promotion_id") != promotion.get("promotion_id")
    or training_manifest.get("bank") != str(cfg.training_bank.resolve())
    or training_manifest.get("bank_sha256") != _sha256(cfg.training_bank)
  ):
    raise ValueError("training bank is not READY for the same flat promotion")
  registry = _load_json(cfg.registry)
  contract = registry.get("held_out_evaluation_banks", {})
  if (
    contract.get("status") != "preregistered"
    or tuple(contract.get("modes", ())) != _MODES
    or contract.get("generation_seed") != cfg.seed
    or contract.get("num_states_per_mode") != cfg.num_states_per_mode
    or contract.get("training_bank_disjoint_required") is not True
  ):
    raise ValueError("held-out generation request differs from frozen registry")

  selected_arm = promotion.get("selected_arm")
  catalog = {arm["name"]: arm for arm in _ARMS}
  if selected_arm not in catalog:
    raise ValueError(f"unknown selected arm: {selected_arm}")
  env_cfg = load_env_cfg(catalog[selected_arm]["task"])
  prior_path = Path(env_cfg.events["init_smp_state"].params["ckpt_path"])
  prior = torch.load(prior_path, map_location="cpu", weights_only=False)
  prior_cfg = prior.get("cfg", {})
  if prior_cfg.get("window_size") != 10 or prior_cfg.get("feature_dim") != 59:
    raise ValueError("selected prior differs from the frozen 10x59 contract")
  mixed = env_cfg.events.get("mixed_fall_reset")
  procedural_params = dict(_DEFAULT_PROCEDURAL_PARAMS)
  if mixed is not None:
    procedural_params.update(
      {
        key: value
        for key, value in mixed.params.items()
        if key not in ("procedural_probability", "mode_weights")
      }
    )
  material = {
    "schema_version": 1,
    "selected_arm": selected_arm,
    "task": catalog[selected_arm]["task"],
    "prior": str(prior_path.resolve()),
    "prior_sha256": _sha256(prior_path),
    "promotion": str(cfg.promotion.resolve()),
    "promotion_sha256": _sha256(cfg.promotion),
    "promotion_id": promotion["promotion_id"],
    "training_bank": str(cfg.training_bank.resolve()),
    "training_bank_sha256": _sha256(cfg.training_bank),
    "training_bank_manifest": str(cfg.training_bank_manifest.resolve()),
    "training_bank_manifest_sha256": _sha256(cfg.training_bank_manifest),
    "registry": str(cfg.registry.resolve()),
    "registry_sha256": _sha256(cfg.registry),
    "generator_code_commit": _git_commit(),
    "generator_code_sha256": _sha256(Path(__file__).resolve()),
    "capture_code_sha256": _sha256(
      Path(__file__).with_name("generate_smp_matched_reset_bank.py")
    ),
    "generation_seed": cfg.seed,
    "num_states_per_mode": cfg.num_states_per_mode,
    "modes": list(_MODES),
    "procedural_params": procedural_params,
    "window_size": 10,
    "feature_dim": 59,
  }
  stable = {
    key: value for key, value in material.items() if key != "generator_code_commit"
  }
  material["plan_id"] = hashlib.sha256(
    json.dumps(stable, sort_keys=True).encode()
  ).hexdigest()
  return material


def _mode_params(plan: dict[str, Any], mode: str) -> dict[str, Any]:
  if mode == "native_gsi":
    return {"procedural_probability": 0.0}
  return {
    **plan["procedural_params"],
    "procedural_probability": 1.0,
    "mode_weights": _MODE_WEIGHTS[mode],
  }


def _validate_mode(
  payload: dict[str, torch.Tensor], mode: str, count: int
) -> list[int]:
  _validate_matched_reset_bank_payload(payload, count)
  counts = torch.bincount(payload["reset_type"].long(), minlength=5).tolist()
  expected_type = 0 if mode == "native_gsi" else _MODES.index(mode)
  expected = [0] * 5
  expected[expected_type] = count
  if counts != expected:
    raise ValueError(f"held-out {mode} reset types drifted: {counts} != {expected}")
  return counts


def _state_matrix(payload: dict[str, torch.Tensor]) -> torch.Tensor:
  return torch.cat(
    (
      payload["root_state"].float(),
      payload["joint_pos"].float(),
      payload["joint_vel"].float(),
    ),
    dim=-1,
  ).contiguous()


def _row_hashes(matrix: torch.Tensor) -> torch.Tensor:
  bits = matrix.view(torch.int32).to(torch.int64)
  hashes = torch.full((matrix.shape[0],), 1469598103934665603, dtype=torch.int64)
  for column in range(bits.shape[1]):
    hashes = (hashes ^ bits[:, column]) * 1099511628211
  return hashes


def _exact_overlap_count(
  training: dict[str, torch.Tensor], evaluations: dict[str, dict[str, torch.Tensor]]
) -> int:
  training_matrix = _state_matrix(training)
  eval_matrix = torch.cat([_state_matrix(evaluations[mode]) for mode in _MODES])
  if torch.unique(eval_matrix, dim=0).shape[0] != eval_matrix.shape[0]:
    raise ValueError("held-out evaluation banks contain duplicate exact states")
  training_hashes = _row_hashes(training_matrix)
  eval_hashes = _row_hashes(eval_matrix)
  sorted_hashes, order = torch.sort(training_hashes)
  overlap = 0
  for index, value in enumerate(eval_hashes):
    left = int(torch.searchsorted(sorted_hashes, value, right=False))
    right = int(torch.searchsorted(sorted_hashes, value, right=True))
    for candidate in order[left:right]:
      if torch.equal(training_matrix[int(candidate)], eval_matrix[index]):
        overlap += 1
  return overlap


def _validate_existing(cfg: EvalBankCfg, plan: dict[str, Any]) -> dict[str, Any] | None:
  paths = [cfg.output_dir / f"{mode}.pt" for mode in _MODES]
  exists = [cfg.manifest.is_file(), *(path.is_file() for path in paths)]
  if not any(exists):
    return None
  if not all(exists):
    raise ValueError("partial held-out evaluation-bank artifacts exist")
  manifest = _load_json(cfg.manifest)
  if manifest.get("status") != "READY" or manifest.get("plan_id") != plan["plan_id"]:
    raise ValueError("existing held-out bank manifest conflicts with frozen plan")
  for mode, path in zip(_MODES, paths, strict=True):
    row = manifest.get("banks", {}).get(mode, {})
    if row.get("path") != str(path.resolve()) or row.get("sha256") != _sha256(path):
      raise ValueError(f"existing held-out {mode} bank changed")
  if manifest.get("exact_training_overlap_count") != 0:
    raise ValueError("existing held-out bank did not prove training disjointness")
  return manifest


def generate(cfg: EvalBankCfg) -> dict[str, Any]:
  plan = build_plan(cfg)
  existing = _validate_existing(cfg, plan)
  if existing is not None:
    return existing
  if not cfg.run:
    return {**plan, "status": "PLANNED"}
  if _gpu_processes():
    raise RuntimeError(
      "refusing held-out bank generation while a GPU process is active"
    )

  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg(plan["task"])
  env_cfg.scene.num_envs = cfg.num_states_per_mode
  env_cfg.seed = cfg.seed
  env_cfg.events.pop("gsi_refresh", None)
  env_cfg.events.pop("push_robot", None)
  env_cfg.events["init_smp_state"].params.update(
    {"compile_model": False, "gsi_buffer_size": 1, "gsi_batch_size": 1}
  )
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  payloads = {}
  try:
    for mode in _MODES:
      payload = _capture_batch(env, cfg.num_states_per_mode, _mode_params(plan, mode))
      _validate_mode(payload, mode, cfg.num_states_per_mode)
      payloads[mode] = payload
  finally:
    env.close()

  training = torch.load(cfg.training_bank, map_location="cpu", weights_only=True)
  if not isinstance(training, dict):
    raise ValueError("training reset bank must contain tensors")
  _validate_matched_reset_bank_payload(training)
  overlap = _exact_overlap_count(training, payloads)
  del training
  gc.collect()
  if overlap:
    raise ValueError(f"held-out banks overlap {overlap} exact training states")

  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  banks = {}
  for mode in _MODES:
    path = cfg.output_dir / f"{mode}.pt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payloads[mode], temporary)
    temporary.replace(path)
    banks[mode] = {
      "path": str(path.resolve()),
      "sha256": _sha256(path),
      "num_states": cfg.num_states_per_mode,
      "reset_type_counts": torch.bincount(
        payloads[mode]["reset_type"].long(), minlength=5
      ).tolist(),
      "tensor_shapes": {
        name: list(tensor.shape) for name, tensor in payloads[mode].items()
      },
    }
  manifest = {
    **plan,
    "status": "READY",
    "banks": banks,
    "exact_training_overlap_count": overlap,
    "disjoint_check_fields": ["root_state", "joint_pos", "joint_vel"],
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "claim_boundary": (
      "Held-out banks are evaluation inputs; their existence is not policy "
      "performance evidence."
    ),
  }
  _atomic_json(cfg.manifest, manifest)
  return manifest


def main(cfg: EvalBankCfg) -> None:
  result = generate(cfg)
  print(f"{result['status']}: {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(EvalBankCfg))
