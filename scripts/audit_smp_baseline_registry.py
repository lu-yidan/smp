"""Fail-closed audit of the frozen RA-L Tier-A baseline registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

EXPECTED_METHODS = {
  "task_only_ppo",
  "original_product_smp",
  "proposed_smp_recovery",
  "firm_r_deployable",
  "recovery_tracking",
}
EXPECTED_METHOD_PRIOR_CONTRACT = {
  "task_only_ppo": (False, False),
  "original_product_smp": (True, True),
  "proposed_smp_recovery": (True, True),
  "firm_r_deployable": (False, False),
  "recovery_tracking": (False, False),
}
EXPECTED_FIELDS = (
  "base_angular_velocity",
  "projected_gravity",
  "joint_position",
  "joint_velocity",
  "previous_action",
)
EXPECTED_FORBIDDEN_FIELDS = {
  "true_base_linear_velocity",
  "terrain_label",
  "terrain_heightmap",
  "plate_pose",
  "plate_contact",
  "reset_family",
  "reference_identity",
  "future_reference",
  "simulator_contact_force",
}
EXPECTED_MODES = (
  "native_gsi",
  "prone",
  "supine",
  "left_side",
  "right_side",
)
VALID_METHOD_STATUSES = {"blocked", "ready_for_training", "complete"}


@dataclass(frozen=True)
class AuditCfg:
  registry: Path = Path("docs/ral_baseline_registry.json")
  output: Path = Path("run_control/ral_baselines/readiness.json")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _repo_root(registry_path: Path) -> Path:
  resolved = registry_path.resolve()
  for candidate in resolved.parents:
    if (candidate / "pyproject.toml").is_file():
      return candidate
  return resolved.parent.parent


def _validate_reset_bank(bank: dict[str, Any], repo_root: Path) -> bool:
  _require(bank.get("num_states") == 262144, "reset bank size drifted")
  _require(bank.get("generation_seed") == 20260920, "reset bank seed drifted")
  _require(
    bank.get("source") == "selected_flat_arm_exact_gsi_procedural_mixture",
    "reset bank source drifted",
  )
  _require(
    bank.get("actor_exposes_reset_family") is False,
    "reset family must remain hidden from the actor",
  )
  _require(
    bank.get("state_fields") == ["root_state", "joint_pos", "joint_vel", "reset_type"],
    "reset bank state fields drifted",
  )
  _require(
    bank.get("smp_history") == {"window_size": 10, "feature_dim": 59},
    "reset bank history contract drifted",
  )
  status = bank.get("status")
  _require(status in {"missing", "ready"}, "invalid reset bank status")
  if status == "missing":
    _require(bank.get("result_path") is None, "missing bank has a result path")
    _require(bank.get("sha256") is None, "missing bank has a hash")
    _require(bank.get("manifest_path") is None, "missing bank has a manifest")
    _require(bank.get("manifest_sha256") is None, "missing bank has a manifest hash")
    return False
  target = bank.get("result_path")
  expected_hash = bank.get("sha256")
  _require(isinstance(target, str) and target, "ready bank lacks result path")
  _require(
    isinstance(expected_hash, str) and len(expected_hash) == 64,
    "ready bank lacks SHA-256",
  )
  path = Path(target)
  path = path if path.is_absolute() else repo_root / path
  _require(path.is_file() and path.stat().st_size > 0, "reset bank is missing")
  _require(_sha256(path) == expected_hash, "reset bank SHA-256 mismatch")
  manifest_target = bank.get("manifest_path")
  manifest_hash = bank.get("manifest_sha256")
  _require(
    isinstance(manifest_target, str) and manifest_target,
    "ready bank lacks manifest path",
  )
  _require(
    isinstance(manifest_hash, str) and len(manifest_hash) == 64,
    "ready bank lacks manifest SHA-256",
  )
  manifest_path = Path(manifest_target)
  manifest_path = (
    manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
  )
  _require(manifest_path.is_file(), "reset bank manifest is missing")
  _require(_sha256(manifest_path) == manifest_hash, "reset bank manifest changed")
  manifest = json.loads(manifest_path.read_text())
  _require(manifest.get("status") == "READY", "reset bank manifest is not READY")
  _require(manifest.get("bank_sha256") == expected_hash, "manifest bank hash mismatch")
  _require(manifest.get("num_states") == 262144, "manifest bank size drifted")
  _require(
    manifest.get("tensor_shapes", {}).get("smp_window") == [262144, 10, 59],
    "manifest SMP history shape drifted",
  )
  return True


def audit(registry: dict[str, Any], registry_path: Path) -> dict[str, Any]:
  _require(registry.get("schema_version") == 1, "unsupported registry schema")
  _require(
    registry.get("protocol_id") == "smp-ral-tier-a-baselines-v1",
    "unexpected baseline protocol id",
  )
  _require(registry.get("frozen_before_selection") is True, "protocol not frozen")
  observation = registry.get("actor_observation", {})
  _require(observation.get("dimension") == 93, "actor dimension must be 93")
  _require(observation.get("history_frames") == 1, "actor must use one frame")
  _require(
    tuple(observation.get("fields", ())) == EXPECTED_FIELDS, "actor fields drifted"
  )
  _require(
    set(observation.get("forbidden_fields", ())) == EXPECTED_FORBIDDEN_FIELDS,
    "forbidden actor fields drifted",
  )

  budget = registry.get("training_budget", {})
  _require(budget.get("policy_seeds") == [20260901, 20260902, 20260903], "seed drift")
  _require(budget.get("num_envs") == 4096, "environment budget drifted")
  _require(budget.get("transitions_per_env_per_update") == 24, "rollout budget drifted")
  _require(budget.get("max_updates") == 30000, "update budget drifted")
  _require(budget.get("save_interval") == 1000, "save interval drifted")
  _require(budget.get("checkpoint_gates") == [8000, 15000, 25000, 29999], "gate drift")

  evaluation = registry.get("evaluation", {})
  _require(evaluation.get("schema_version") == 2, "evaluation schema drifted")
  _require(evaluation.get("seed") == 20260829, "evaluation seed drifted")
  _require(evaluation.get("num_envs_per_mode") == 512, "evaluation size drifted")
  _require(evaluation.get("num_steps") == 500, "evaluation horizon drifted")
  _require(
    tuple(evaluation.get("modes", ())) == EXPECTED_MODES, "evaluation modes drifted"
  )
  _require(
    evaluation.get("success_definition") == "stable_standing_hold_25_steps",
    "success definition drifted",
  )
  paired = registry.get("paired_analysis", {})
  _require(
    paired.get("sampling_unit") == "independently_trained_policy_seed",
    "paired sampling unit drifted",
  )
  _require(paired.get("bootstrap_replicates") == 20000, "paired bootstrap drifted")
  _require(paired.get("bootstrap_seed") == 20260829, "paired seed drifted")
  _require(
    paired.get("stable_gates") == [15000, 25000, 29999],
    "paired stable gates drifted",
  )
  _require(paired.get("final_gate") == 29999, "paired final gate drifted")
  _require(
    paired.get("late_regression_reference_gate") == 25000,
    "paired regression reference drifted",
  )
  _require(paired.get("max_late_regression") == 0.10, "paired regression drifted")
  _require(
    paired.get("success_thresholds")
    == {
      "native_gsi": 0.95,
      "fixed_macro": 0.80,
      "fixed_worst": 0.60,
      "finite_action_rate": 1.0,
    },
    "paired success thresholds drifted",
  )
  _require(
    paired.get("noninferiority_margin") == 0.05,
    "paired noninferiority margin drifted",
  )
  _require(
    paired.get("primary_contrast")
    == "proposed_smp_recovery_minus_original_product_smp",
    "paired primary contrast drifted",
  )
  _require(
    paired.get("support_rule")
    == "primary_fixed_worst_ci_low_gt_0_and_primary_macro_ci_low_ge_minus_margin_and_task_only_worst_ci_low_ge_minus_margin_and_proposed_stable",
    "paired support rule drifted",
  )

  repo_root = _repo_root(registry_path)
  bank_ready = _validate_reset_bank(registry.get("shared_reset_bank", {}), repo_root)
  held_out = registry.get("held_out_evaluation_banks", {})
  _require(held_out.get("status") == "preregistered", "held-out bank status drifted")
  _require(
    held_out.get("generation_seed") == 20260829,
    "held-out bank generation seed drifted",
  )
  _require(
    held_out.get("num_states_per_mode") == 512,
    "held-out bank size drifted",
  )
  _require(
    tuple(held_out.get("modes", ())) == EXPECTED_MODES,
    "held-out bank modes drifted",
  )
  _require(
    held_out.get("source") == "selected_flat_arm_gsi_and_frozen_procedural_poses",
    "held-out bank source drifted",
  )
  _require(
    held_out.get("training_bank_disjoint_required") is True,
    "held-out bank must remain disjoint from training",
  )
  _require(
    held_out.get("result_manifest") is None and held_out.get("manifest_sha256") is None,
    "preregistered held-out bank cannot contain mutable result fields",
  )
  methods = registry.get("methods")
  _require(isinstance(methods, list), "methods must be a list")
  ids = [method.get("id") for method in methods]
  _require(len(ids) == len(set(ids)), "duplicate baseline method id")
  _require(set(ids) == EXPECTED_METHODS, "baseline method set drifted")
  method_reports = []
  for method in methods:
    method_id = method["id"]
    _require(method.get("tier") == "A", f"{method_id} is not Tier A")
    _require(
      method.get("observation_id") == observation.get("id"),
      f"{method_id} observation contract drifted",
    )
    _require(
      method.get("uses_runtime_motion_prior") is False,
      f"{method_id} uses runtime prior",
    )
    expected_objective, expected_termination = EXPECTED_METHOD_PRIOR_CONTRACT[method_id]
    _require(
      method.get("uses_motion_prior_objective") is expected_objective,
      f"{method_id} motion-prior objective contract drifted",
    )
    _require(
      method.get("uses_motion_prior_termination") is expected_termination,
      f"{method_id} motion-prior termination contract drifted",
    )
    _require(
      method.get("actor_extra_inputs") == [], f"{method_id} exposes extra actor inputs"
    )
    status = method.get("status")
    _require(status in VALID_METHOD_STATUSES, f"{method_id} has invalid status")
    blocked_on = method.get("blocked_on", [])
    _require(isinstance(blocked_on, list), f"{method_id} blocked_on must be a list")
    implementation = method.get("implementation", {})
    _require(
      isinstance(implementation, dict), f"{method_id} implementation must be an object"
    )
    if status != "blocked":
      _require(bank_ready, f"{method_id} cannot be ready before reset bank")
      _require(not blocked_on, f"{method_id} still has blockers")
      _require(bool(implementation), f"{method_id} lacks implementation provenance")
      for label, target in implementation.items():
        _require(isinstance(target, str) and target, f"{method_id} invalid {label}")
        path = Path(target)
        path = path if path.is_absolute() else repo_root / path
        _require(path.exists(), f"{method_id} missing implementation {label}: {path}")
    if status == "complete":
      result_path = method.get("result_path")
      _require(
        isinstance(result_path, str) and result_path, f"{method_id} lacks result"
      )
      result = Path(result_path)
      result = result if result.is_absolute() else repo_root / result
      _require(
        result.is_file() and result.stat().st_size > 0, f"{method_id} result missing"
      )
    else:
      _require(method.get("result_path") is None, f"{method_id} has premature result")
    method_reports.append({"id": method_id, "status": status, "blocked_on": blocked_on})

  statuses = {item["status"] for item in method_reports}
  if statuses == {"complete"}:
    status = "BASELINES_COMPLETE"
  elif statuses <= {"ready_for_training", "complete"}:
    status = "BASELINES_READY_FOR_TRAINING"
  else:
    status = "BASELINES_BLOCKED"
  return {
    "status": status,
    "protocol_id": registry["protocol_id"],
    "reset_bank_ready": bank_ready,
    "methods": method_reports,
    "interpretation_limit": (
      "Registry validity and readiness are not baseline performance evidence."
    ),
  }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def main(cfg: AuditCfg) -> None:
  registry = json.loads(cfg.registry.read_text())
  report = audit(registry, cfg.registry)
  _atomic_write(cfg.output, report)
  print(f"{report['status']}: reset_bank_ready={report['reset_bank_ready']}")


if __name__ == "__main__":
  main(tyro.cli(AuditCfg))
