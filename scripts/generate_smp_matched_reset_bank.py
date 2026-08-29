"""Generate the SHA-locked shared reset-state bank for Tier-A baselines."""

from __future__ import annotations

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
from smp.rl.events import _ddpm_sample, prime_sim_and_buffer
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.mdp.events import _validate_matched_reset_bank_payload


@dataclass(frozen=True)
class ResetBankCfg:
  promotion: Path
  output: Path = Path("run_control/ral_baselines/matched_reset_bank.pt")
  manifest: Path = Path("run_control/ral_baselines/matched_reset_bank.json")
  registry_template: Path = Path("docs/ral_baseline_registry.json")
  runtime_registry: Path = Path("run_control/ral_baselines/registry.json")
  num_states: int = 262144
  batch_size: int = 4096
  seed: int = 20260920
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


def _verify_source(path_value: Any, hash_value: Any, label: str) -> None:
  if not isinstance(path_value, str) or not isinstance(hash_value, str):
    raise ValueError(f"promotion lacks {label} provenance")
  path = Path(path_value)
  if not path.is_file() or _sha256(path) != hash_value:
    raise ValueError(f"promotion {label} source changed: {path}")


def build_plan(cfg: ResetBankCfg) -> dict[str, Any]:
  promotion = _load_json(cfg.promotion)
  if promotion.get("status") != "PROMOTE_TP_SPECIALISTS":
    raise ValueError("flat policy has not passed three-seed promotion")
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
  selected_arm = promotion.get("selected_arm")
  catalog = {arm["name"]: arm for arm in _ARMS}
  if selected_arm not in catalog:
    raise ValueError(f"unknown selected arm: {selected_arm}")
  registry = _load_json(cfg.registry_template)
  bank_contract = registry.get("shared_reset_bank", {})
  if (
    bank_contract.get("status") != "missing"
    or bank_contract.get("num_states") != cfg.num_states
    or bank_contract.get("generation_seed") != cfg.seed
  ):
    raise ValueError("runtime request differs from frozen reset-bank contract")
  env_cfg = load_env_cfg(catalog[selected_arm]["task"])
  prior_path = Path(env_cfg.events["init_smp_state"].params["ckpt_path"])
  if not prior_path.is_file():
    raise FileNotFoundError(prior_path)
  prior_checkpoint = torch.load(prior_path, map_location="cpu", weights_only=False)
  prior_cfg = prior_checkpoint.get("cfg", {})
  if prior_cfg.get("window_size") != 10 or prior_cfg.get("feature_dim") != 59:
    raise ValueError("selected prior differs from the frozen 10x59 history contract")
  mixed = env_cfg.events.get("mixed_fall_reset")
  mixed_params = dict(mixed.params) if mixed is not None else {}
  procedural_probability = float(mixed_params.get("procedural_probability", 0.0))
  material = {
    "schema_version": 1,
    "selected_arm": selected_arm,
    "task": catalog[selected_arm]["task"],
    "prior": str(prior_path.resolve()),
    "prior_sha256": _sha256(prior_path),
    "promotion": str(cfg.promotion.resolve()),
    "promotion_sha256": _sha256(cfg.promotion),
    "promotion_id": promotion.get("promotion_id"),
    "registry_template": str(cfg.registry_template.resolve()),
    "registry_template_sha256": _sha256(cfg.registry_template),
    "generator_code_commit": _git_commit(),
    "generator_code_sha256": _sha256(Path(__file__).resolve()),
    "num_states": cfg.num_states,
    "batch_size": cfg.batch_size,
    "generation_seed": cfg.seed,
    "procedural_probability": procedural_probability,
    "procedural_params": mixed_params,
    "window_size": 10,
    "feature_dim": 59,
  }
  stable_material = {
    key: value for key, value in material.items() if key != "generator_code_commit"
  }
  material["plan_id"] = hashlib.sha256(
    json.dumps(stable_material, sort_keys=True).encode()
  ).hexdigest()
  return material


def _capture_batch(env: ManagerBasedRlEnv, count: int, mixed_params: dict[str, Any]):
  env_ids = torch.arange(count, device=env.device)
  windows = _ddpm_sample(env, count)
  prime_sim_and_buffer(env, env_ids, windows)
  if mixed_params:
    mdp.mixed_fall_reset(env, env_ids, **mixed_params)
  robot = env.scene["robot"]
  origins = env.scene.env_origins[env_ids]
  root_state = torch.cat(
    [
      robot.data.root_link_pos_w[env_ids] - origins,
      robot.data.root_link_quat_w[env_ids],
      robot.data.root_link_lin_vel_w[env_ids],
      robot.data.root_link_ang_vel_w[env_ids],
    ],
    dim=-1,
  )
  reset_type = getattr(env, "_robust_reset_type", None)
  if reset_type is None:
    reset_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  return {
    "root_state": root_state.cpu(),
    "joint_pos": robot.data.joint_pos[env_ids].cpu(),
    "joint_vel": robot.data.joint_vel[env_ids].cpu(),
    "smp_window": env._smp_buffer.compute_features()[env_ids].cpu(),  # type: ignore[attr-defined]
    "reset_type": reset_type[env_ids].to(torch.int8).cpu(),
  }


def _validate_reset_distribution(
  reset_type: torch.Tensor,
  procedural_probability: float,
  mode_weights: tuple[float, float, float, float] | None,
) -> list[int]:
  counts = torch.bincount(reset_type.long(), minlength=5).tolist()
  total = int(reset_type.numel())
  observed_procedural = sum(counts[1:]) / total
  if abs(observed_procedural - procedural_probability) > 0.005:
    raise ValueError(
      "matched bank procedural share differs from the frozen reset distribution"
    )
  if procedural_probability == 0.0:
    return counts
  expected = torch.tensor(mode_weights or (1.0, 1.0, 1.0, 1.0), dtype=torch.float)
  expected /= expected.sum()
  observed = torch.tensor(counts[1:], dtype=torch.float) / max(1, sum(counts[1:]))
  if torch.any(torch.abs(observed - expected) > 0.01):
    raise ValueError("matched bank procedural pose balance drifted")
  return counts


def _materialize_runtime_registry(
  template_path: Path,
  output_path: Path,
  bank_path: Path,
  bank_sha256: str,
  manifest_path: Path,
  manifest_sha256: str,
) -> None:
  registry = _load_json(template_path)
  registry["shared_reset_bank"].update(
    {
      "status": "ready",
      "result_path": str(bank_path.resolve()),
      "sha256": bank_sha256,
      "manifest_path": str(manifest_path.resolve()),
      "manifest_sha256": manifest_sha256,
    }
  )
  promotion_bound_methods = {
    "task_only_ppo",
    "original_product_smp",
    "proposed_smp_recovery",
  }
  for method in registry.get("methods", ()):
    blockers = [
      blocker
      for blocker in method.get("blocked_on", ())
      if blocker != "matched_reset_bank"
      and not (
        blocker == "confirmed_flat_arm" and method.get("id") in promotion_bound_methods
      )
    ]
    method["blocked_on"] = blockers
    if not blockers and method.get("implementation"):
      method["status"] = "ready_for_training"
  if output_path.exists():
    if _load_json(output_path) != registry:
      raise ValueError("existing runtime baseline registry conflicts with reset bank")
    return
  _atomic_json(output_path, registry)


def generate(cfg: ResetBankCfg) -> dict[str, Any]:
  plan = build_plan(cfg)
  if not cfg.run:
    return {**plan, "status": "PLANNED"}
  if _gpu_processes():
    raise RuntimeError("refusing reset-bank generation while a GPU process is active")
  if cfg.num_states <= 0 or cfg.batch_size <= 0 or cfg.batch_size > cfg.num_states:
    raise ValueError("invalid reset-bank size or batch size")
  if cfg.output.exists() or cfg.manifest.exists():
    if not cfg.output.is_file() or not cfg.manifest.is_file():
      raise ValueError("partial reset-bank artifact exists")
    manifest = _load_json(cfg.manifest)
    if manifest.get("plan_id") != plan["plan_id"] or _sha256(
      cfg.output
    ) != manifest.get("bank_sha256"):
      raise ValueError("existing reset bank conflicts with frozen plan")
    _materialize_runtime_registry(
      cfg.registry_template,
      cfg.runtime_registry,
      cfg.output,
      manifest["bank_sha256"],
      cfg.manifest,
      _sha256(cfg.manifest),
    )
    return manifest

  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg(plan["task"])
  env_cfg.scene.num_envs = cfg.batch_size
  env_cfg.seed = cfg.seed
  env_cfg.events.pop("gsi_refresh", None)
  env_cfg.events.pop("push_robot", None)
  env_cfg.events["init_smp_state"].params.update(
    {"compile_model": False, "gsi_buffer_size": 1, "gsi_batch_size": 1}
  )
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  chunks: dict[str, list[torch.Tensor]] = {
    "root_state": [],
    "joint_pos": [],
    "joint_vel": [],
    "smp_window": [],
    "reset_type": [],
  }
  try:
    mixed = env_cfg.events.get("mixed_fall_reset")
    mixed_params = dict(mixed.params) if mixed is not None else {}
    for start in range(0, cfg.num_states, cfg.batch_size):
      count = min(cfg.batch_size, cfg.num_states - start)
      batch = _capture_batch(env, count, mixed_params)
      for name, tensor in batch.items():
        chunks[name].append(tensor)
  finally:
    env.close()
  payload = {name: torch.cat(parts, dim=0) for name, parts in chunks.items()}
  _validate_matched_reset_bank_payload(payload, cfg.num_states)
  reset_counts = _validate_reset_distribution(
    payload["reset_type"],
    plan["procedural_probability"],
    plan["procedural_params"].get("mode_weights"),
  )
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
  torch.save(payload, temporary)
  temporary.replace(cfg.output)
  manifest = {
    **plan,
    "status": "READY",
    "bank": str(cfg.output.resolve()),
    "bank_sha256": _sha256(cfg.output),
    "reset_type_counts": reset_counts,
    "tensor_shapes": {name: list(tensor.shape) for name, tensor in payload.items()},
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "claim_boundary": "A reset bank is protocol input, not policy performance evidence.",
  }
  _atomic_json(cfg.manifest, manifest)
  _materialize_runtime_registry(
    cfg.registry_template,
    cfg.runtime_registry,
    cfg.output,
    manifest["bank_sha256"],
    cfg.manifest,
    _sha256(cfg.manifest),
  )
  return manifest


def main(cfg: ResetBankCfg) -> None:
  result = generate(cfg)
  print(f"{result['status']}: {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(ResetBankCfg))
