"""Bind immutable native checkpoints to the matched held-out evaluation banks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")
_GATES = (8000, 15000, 25000, 29999)
_SEEDS = (20260901, 20260902, 20260903)
_METHODS = ("task_only_ppo", "original_product_smp", "proposed_smp_recovery")


@dataclass(frozen=True)
class EvalBindingCfg:
  checkpoint_index: Path
  eval_bank_manifest: Path
  output_dir: Path = Path("run_control/ral_baselines/formal_manifests")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
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


def _validate_eval_banks(path: Path) -> dict[str, Any]:
  manifest = _load(path)
  if (
    manifest.get("status") != "READY"
    or tuple(manifest.get("modes", ())) != _MODES
    or manifest.get("generation_seed") != 20260829
    or manifest.get("num_states_per_mode") != 512
    or manifest.get("exact_training_overlap_count") != 0
    or not isinstance(manifest.get("training_bank_sha256"), str)
  ):
    raise ValueError("held-out bank manifest violates the frozen protocol")
  for index, mode in enumerate(_MODES):
    row = manifest.get("banks", {}).get(mode, {})
    bank = Path(row.get("path", ""))
    expected_counts = [0] * 5
    expected_counts[index] = 512
    if (
      not bank.is_file()
      or row.get("sha256") != _sha256(bank)
      or row.get("num_states") != 512
      or row.get("reset_type_counts") != expected_counts
    ):
      raise ValueError(f"held-out {mode} bank changed or has invalid provenance")
  return manifest


def _validate_checkpoint_index(
  path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  index = _load(path)
  if (
    index.get("status") != "CHECKPOINTS_READY_EVALUATION_BANK_BLOCKED"
    or tuple(index.get("policy_seeds", ())) != _SEEDS
    or tuple(index.get("checkpoint_steps", ())) != _GATES
    or tuple(index.get("methods", ())) != tuple(sorted(_METHODS))
  ):
    raise ValueError("native checkpoint index violates the frozen protocol")
  rows = index.get("manifests")
  if not isinstance(rows, list) or len(rows) != 12:
    raise ValueError("native checkpoint index must contain twelve manifests")
  pairs = {(int(row["policy_seed"]), int(row["checkpoint_step"])) for row in rows}
  if pairs != {(seed, gate) for seed in _SEEDS for gate in _GATES}:
    raise ValueError("native checkpoint index lacks the full seed/gate factorial")
  for row in rows:
    manifest = Path(row["path"])
    if not manifest.is_file() or row.get("sha256") != _sha256(manifest):
      raise ValueError(f"native checkpoint manifest changed: {manifest}")
  return index, rows


def _bind_one(
  source_path: Path,
  source_sha: str,
  eval_path: Path,
  eval_sha: str,
  eval_banks: dict[str, Any],
) -> dict[str, Any]:
  source = _load(source_path)
  protocol = source.get("evaluation_protocol", {})
  if (
    source.get("evaluation_status") != "BLOCKED_ON_MATCHED_HELD_OUT_RESET_BANK"
    or tuple(protocol.get("reset_modes", ())) != _MODES
    or protocol.get("num_envs") != 512
    or protocol.get("steps") != 500
    or protocol.get("evaluation_seed") != 20260829
    or source.get("training_reset_bank_sha256")
    != eval_banks.get("training_bank_sha256")
    or source.get("promotion_id") != eval_banks.get("promotion_id")
  ):
    raise ValueError(f"checkpoint/evaluation bank lineage differs: {source_path}")
  runs = source.get("runs")
  if not isinstance(runs, list) or tuple(
    sorted(run.get("name") for run in runs)
  ) != tuple(sorted(_METHODS)):
    raise ValueError(f"checkpoint manifest has an invalid method set: {source_path}")
  for run in runs:
    checkpoint = Path(run["checkpoint"])
    if not checkpoint.is_file() or run.get("checkpoint_sha256") != _sha256(checkpoint):
      raise ValueError(f"native checkpoint changed: {checkpoint}")

  stable = {
    **{
      key: value
      for key, value in source.items()
      if key not in ("generated_at_utc", "claim_boundary", "manifest_id")
    },
    "evaluation_status": "READY_WITH_MATCHED_HELD_OUT_RESET_BANK",
    "checkpoint_manifest": str(source_path.resolve()),
    "checkpoint_manifest_sha256": source_sha,
    "checkpoint_manifest_id": source["manifest_id"],
    "matched_eval_manifest": str(eval_path.resolve()),
    "matched_eval_manifest_sha256": eval_sha,
    "matched_eval_plan_id": eval_banks["plan_id"],
    "matched_eval_exact_training_overlap_count": 0,
  }
  stable["binding_id"] = hashlib.sha256(
    json.dumps(stable, sort_keys=True).encode()
  ).hexdigest()
  return {
    **stable,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "claim_boundary": (
      "This binding authorizes the frozen matrix but is not performance evidence."
    ),
  }


def build(cfg: EvalBindingCfg) -> tuple[dict[str, Any], dict[tuple[int, int], dict]]:
  index, rows = _validate_checkpoint_index(cfg.checkpoint_index)
  eval_banks = _validate_eval_banks(cfg.eval_bank_manifest)
  eval_sha = _sha256(cfg.eval_bank_manifest)
  payloads = {}
  for row in rows:
    seed = int(row["policy_seed"])
    gate = int(row["checkpoint_step"])
    source = Path(row["path"])
    payloads[(seed, gate)] = _bind_one(
      source, row["sha256"], cfg.eval_bank_manifest, eval_sha, eval_banks
    )
  material = {
    "schema_version": 1,
    "status": "READY",
    "checkpoint_index": str(cfg.checkpoint_index.resolve()),
    "checkpoint_index_sha256": _sha256(cfg.checkpoint_index),
    "checkpoint_index_id": index["index_id"],
    "matched_eval_manifest": str(cfg.eval_bank_manifest.resolve()),
    "matched_eval_manifest_sha256": eval_sha,
    "matched_eval_plan_id": eval_banks["plan_id"],
    "policy_seeds": list(_SEEDS),
    "checkpoint_steps": list(_GATES),
    "methods": list(_METHODS),
    "binding_ids": sorted(payload["binding_id"] for payload in payloads.values()),
  }
  material["index_id"] = hashlib.sha256(
    json.dumps(material, sort_keys=True).encode()
  ).hexdigest()
  return material, payloads


def write_bindings(cfg: EvalBindingCfg) -> dict[str, Any]:
  index, payloads = build(cfg)
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  index_path = cfg.output_dir / "index.json"
  existing_index = _load(index_path) if index_path.exists() else None
  if existing_index is not None and existing_index.get("index_id") != index["index_id"]:
    raise ValueError(f"existing formal baseline index conflicts: {index_path}")
  existing_rows = {
    (int(row["policy_seed"]), int(row["checkpoint_step"])): row
    for row in (existing_index or {}).get("manifests", ())
  }
  artifacts = set(cfg.output_dir.glob("gate_*_seed_*.json"))
  if existing_index is None and artifacts:
    raise ValueError("partial formal baseline bindings exist without an index")
  rows = []
  for (seed, gate), payload in sorted(payloads.items()):
    path = cfg.output_dir / f"gate_{gate}_seed_{seed}.json"
    if path.exists():
      existing = _load(path)
      frozen = existing_rows.get((seed, gate))
      if (
        existing.get("binding_id") != payload["binding_id"]
        or frozen is None
        or frozen.get("path") != str(path.resolve())
        or frozen.get("sha256") != _sha256(path)
      ):
        raise ValueError(f"formal baseline binding changed after indexing: {path}")
    else:
      if existing_index is not None:
        raise ValueError(f"indexed formal baseline binding is missing: {path}")
      _atomic_json(path, payload)
    rows.append(
      {
        "policy_seed": seed,
        "checkpoint_step": gate,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
      }
    )
  index["manifests"] = rows
  if existing_index is not None:
    if existing_index.get("manifests") != rows:
      raise ValueError(f"formal baseline index rows conflict: {index_path}")
  else:
    _atomic_json(index_path, index)
  return index


def main(cfg: EvalBindingCfg) -> None:
  result = write_bindings(cfg)
  print(f"{result['status']}: {len(result['manifests'])} formal manifests")


if __name__ == "__main__":
  main(tyro.cli(EvalBindingCfg))
