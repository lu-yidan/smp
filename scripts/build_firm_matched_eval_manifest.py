"""Bind completed deployable FIRM-R artifacts to the frozen held-out reset bank."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro


@dataclass(frozen=True)
class BuildCfg:
  firm_protocol: Path
  firm_state: Path
  matched_eval_manifest: Path
  matched_eval_manifest_sha256: str
  output: Path
  task: str = "Smp-Getup-Matched-TaskOnly-G1"
  goal_refresh_steps: int = 5
  num_action_samples: int = 1


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def _checkpoint(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  value = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(value, dict):
    raise TypeError(f"expected checkpoint dictionary: {path}")
  return value


def _validate_held_out(path: Path, expected_sha256: str) -> dict[str, Any]:
  if len(expected_sha256) != 64 or _sha256(path) != expected_sha256:
    raise ValueError("matched held-out manifest SHA-256 mismatch")
  manifest = _json(path)
  modes = ("native_gsi", "prone", "supine", "left_side", "right_side")
  if (
    manifest.get("status") != "READY"
    or manifest.get("generation_seed") != 20260829
    or manifest.get("num_states_per_mode") != 512
    or tuple(manifest.get("modes", ())) != modes
    or manifest.get("exact_training_overlap_count") != 0
  ):
    raise ValueError("matched held-out manifest violates the frozen protocol")
  for mode in modes:
    row = manifest.get("banks", {}).get(mode, {})
    bank = Path(row.get("path", ""))
    if (
      not bank.is_file()
      or not isinstance(row.get("sha256"), str)
      or _sha256(bank) != row["sha256"]
      or row.get("num_states") != 512
    ):
      raise ValueError(f"matched held-out bank changed: {mode}")
  return manifest


def build(cfg: BuildCfg) -> dict[str, Any]:
  if cfg.goal_refresh_steps <= 0 or cfg.num_action_samples <= 0:
    raise ValueError("FIRM inference parameters must be positive")
  protocol = _json(cfg.firm_protocol)
  state = _json(cfg.firm_state)
  protocol_sha = _sha256(cfg.firm_protocol)
  if protocol.get("status") != "FROZEN_BEFORE_DATA_COLLECTION":
    raise ValueError("FIRM protocol is not frozen")
  if protocol.get("tier_a_eligible") is not False:
    raise ValueError("FIRM external reference cannot be relabeled Tier-A")
  if (
    state.get("status") != "READY_FOR_MATCHED_EVAL_ADAPTER"
    or state.get("protocol_sha256") != protocol_sha
  ):
    raise ValueError("FIRM artifacts are not complete under the frozen protocol")
  held_out = _validate_held_out(
    cfg.matched_eval_manifest, cfg.matched_eval_manifest_sha256
  )
  outputs = protocol["outputs"]
  variants = protocol["adapter_variants"]
  seeds = protocol["replicate_seeds"]
  runs: list[dict[str, Any]] = []
  for seed in seeds:
    action = Path(outputs["action_root"]) / f"seed_{seed}/firm_action_diffusion.pt"
    action_payload = _checkpoint(action)
    action_config = action_payload.get("config", {})
    if action_config.get("seed") != seed or action_config.get("observation_dim") != 93:
      raise ValueError(f"FIRM action seed/layout mismatch: {action}")
    action_sha = _sha256(action)
    for variant in variants:
      name = variant["name"]
      history_steps = variant["history_steps"]
      adapter = (
        Path(outputs["adapter_root"]) / name / f"seed_{seed}/firm_goal_adapter.pt"
      )
      adapter_payload = _checkpoint(adapter)
      adapter_config = adapter_payload.get("config", {})
      if (
        adapter_config.get("seed") != seed
        or adapter_config.get("observation_dim") != 93
        or adapter_config.get("history_steps") != history_steps
        or adapter_payload.get("artifacts", {}).get("action_checkpoint_sha256")
        != action_sha
      ):
        raise ValueError(f"FIRM adapter lineage mismatch: {adapter}")
      runs.append(
        {
          "name": f"{name}_seed_{seed}",
          "method": name,
          "policy_kind": "firm_r",
          "task": cfg.task,
          "policy_seed": seed,
          "checkpoint": str(action.resolve()),
          "checkpoint_sha256": action_sha,
          "firm_adapter_checkpoint": str(adapter.resolve()),
          "firm_adapter_checkpoint_sha256": _sha256(adapter),
          "firm_history_steps": history_steps,
          "firm_goal_refresh_steps": cfg.goal_refresh_steps,
          "firm_num_action_samples": cfg.num_action_samples,
          "reporting_class": variant["reporting_class"],
        }
      )
  result = {
    "schema_version": 1,
    "evaluation_status": "READY_WITH_MATCHED_HELD_OUT_RESET_BANK",
    "comparison_class": "external_reference",
    "tier_a_eligible": False,
    "eligibility_note": protocol["eligibility_note"],
    "firm_source_commit": "cfa8572f130dc32d05280b9592ce657c8b3a1b56",
    "firm_protocol": str(cfg.firm_protocol.resolve()),
    "firm_protocol_sha256": protocol_sha,
    "firm_state": str(cfg.firm_state.resolve()),
    "firm_state_sha256": _sha256(cfg.firm_state),
    "matched_eval_manifest": str(cfg.matched_eval_manifest.resolve()),
    "matched_eval_manifest_sha256": cfg.matched_eval_manifest_sha256,
    "matched_eval_training_bank_sha256": held_out["training_bank_sha256"],
    "matched_eval_exact_training_overlap_count": 0,
    "evaluation_seed": 20260829,
    "num_envs": 512,
    "steps": 500,
    "runs": runs,
  }
  encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if cfg.output.exists():
    if cfg.output.read_text() != encoded:
      raise FileExistsError(f"immutable FIRM evaluation manifest changed: {cfg.output}")
    return result
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
  temporary.write_text(encoded)
  temporary.replace(cfg.output)
  return result


def main() -> None:
  result = build(tyro.cli(BuildCfg))
  print(
    "FIRM_MATCHED_EVAL_MANIFEST_READY: "
    f"{len(result['runs'])} runs, external_reference=True"
  )


if __name__ == "__main__":
  main()
