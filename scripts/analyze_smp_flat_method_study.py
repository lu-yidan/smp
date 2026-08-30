"""Audit and aggregate the frozen fresh-seed flat method study."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_PROTOCOL_SHA256 = "6ca241aa3bfb303084de8eac4f1cd6e02a4728ef5969a632dc7ba2b54750e0e0"
_PLAN_ID = "3e8a7aee720f4ac408b3d91eb912b9ace14c20dfd0ca4e82c8d8d2c9185ceb28"
_STUDY_ID = "smp-flat-procedural-coverage-v1"
_ARMS = ("a6_replication_control", "a8_balanced_bridge")
_PROPOSED = "a8_balanced_bridge"
_CONTROL = "a6_replication_control"
_SEEDS = (20261001, 20261002, 20261003)
_GATES = (8000, 15000, 25000, 29999)
_STABILITY_GATES = (15000, 25000, 29999)
_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")
_FIXED_MODES = _MODES[1:]
_EVAL_SEED = 20261010
_NUM_ENVS = 512
_STEPS = 500
_SCHEMA = 2
_FAILURE_SCHEMA = 1
_BOOTSTRAP_REPLICATES = 20000
_BOOTSTRAP_SEED = 20261010
_FAILURE_CODEBOOK = {
  "0": "success",
  "1": "nonfinite_action",
  "2": "invalid_dynamics",
  "3": "terrain_patch_exit",
  "4": "invalid_escape_setup",
  "5": "invalid_escape_contact",
  "6": "plate_not_escaped",
  "7": "never_reached_head_height",
  "8": "never_upright_while_high",
  "9": "never_low_linear_speed_while_upright",
  "10": "never_low_angular_speed_while_pose_stable",
  "11": "strict_candidate_hold_too_short",
}
_PER_ENV_GATE_ARRAYS = (
  "strict_first_step",
  "strict_failure_reason_code",
  "head_height_gate_first_step",
  "upright_while_high_gate_first_step",
  "low_linear_speed_gate_first_step",
  "low_angular_speed_gate_first_step",
  "strict_candidate_first_step",
  "strict_candidate_max_hold_steps",
  "finite_action",
  "invalid_dynamics",
)
_SAFETY_MAX = (
  "contact_foot_slip_p95_m_s",
  "post_success_root_drift_p95_m",
  "secondary_fall_rate_after_success",
  "foot_separation_at_success_p95_m",
  "action_delta_rms_p95",
  "action_second_difference_rms_p95",
  "max_power_mean_w",
  "max_joint_speed_p95_rad_s",
)


@dataclass(frozen=True)
class FlatMethodAnalysisCfg:
  manifest_index: Path = Path(
    "run_control/flat_method_study_v1_eval/manifests/index.json"
  )
  evaluation_root: Path = Path("run_control/flat_method_study_v1_eval/formal")
  protocol: Path = Path("docs/ral_flat_method_study_v1.json")
  output_json: Path = Path(
    "run_control/flat_method_study_v1_eval/flat_method_analysis.json"
  )
  output_markdown: Path = Path(
    "run_control/flat_method_study_v1_eval/flat_method_analysis.md"
  )


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


def _finite(value: Any, name: str) -> float:
  number = float(value)
  if not math.isfinite(number):
    raise ValueError(f"nonfinite {name}: {value}")
  return number


def _quantile(values: list[float], q: float) -> float:
  ordered = sorted(values)
  if not ordered:
    raise ValueError("cannot take quantile of empty values")
  index = q * (len(ordered) - 1)
  lower = int(index)
  upper = min(lower + 1, len(ordered) - 1)
  fraction = index - lower
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_paired(values: list[float], rng: random.Random) -> dict[str, Any]:
  if len(values) != len(_SEEDS):
    raise ValueError("paired analysis requires exactly three policy seeds")
  draws = []
  for _ in range(_BOOTSTRAP_REPLICATES):
    draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
  return {
    "mean": sum(values) / len(values),
    "ci95_low": _quantile(draws, 0.025),
    "ci95_high": _quantile(draws, 0.975),
    "paired_policy_seed_values": dict(zip(map(str, _SEEDS), values, strict=True)),
  }


def _validate_protocol(path: Path) -> dict[str, Any]:
  protocol = _load(path)
  if _sha256(path) != _PROTOCOL_SHA256:
    raise ValueError("flat method protocol SHA-256 drifted")
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != _STUDY_ID
  ):
    raise ValueError("flat method protocol identity drifted")
  return protocol


def _validate_index(path: Path) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
  index = _load(path)
  if (
    index.get("schema_version") != 1
    or index.get("status") != "READY_FOR_FROZEN_EVALUATION"
    or index.get("study_id") != _STUDY_ID
    or index.get("launch_plan_id") != _PLAN_ID
    or index.get("protocol_sha256") != _PROTOCOL_SHA256
    or index.get("arms") != list(_ARMS)
    or index.get("policy_seeds") != list(_SEEDS)
    or index.get("checkpoint_steps") != list(_GATES)
    or index.get("checkpoint_entry_count") != 24
  ):
    raise ValueError("flat method manifest index drifted")
  stable_index = {
    key: value for key, value in index.items() if key not in {"index_id", "manifests"}
  }
  expected_index_id = hashlib.sha256(
    json.dumps(stable_index, sort_keys=True).encode()
  ).hexdigest()
  if index.get("index_id") != expected_index_id:
    raise ValueError("flat method manifest index ID drifted")
  rows = index.get("manifests")
  if not isinstance(rows, list) or len(rows) != 12:
    raise ValueError("flat method manifest index must contain twelve rows")
  by_key: dict[tuple[int, int], dict[str, Any]] = {}
  for row in rows:
    key = (int(row.get("policy_seed")), int(row.get("checkpoint_step")))
    manifest_path = Path(row.get("path", ""))
    if key in by_key or not manifest_path.is_file() or _sha256(manifest_path) != row.get("sha256"):
      raise ValueError(f"flat method manifest index row changed: {key}")
    by_key[key] = row
  if set(by_key) != {(seed, gate) for seed in _SEEDS for gate in _GATES}:
    raise ValueError("flat method manifest index factorial is incomplete")
  return index, by_key


def _validate_result(
  path: Path,
  manifest: dict[str, Any],
  runs: dict[str, dict[str, Any]],
  seed: int,
  gate: int,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
  result = _load(path)
  arm_matches = [arm for arm in _ARMS if path.name.startswith(f"{arm}__")]
  if len(arm_matches) != 1:
    raise ValueError(f"cannot identify arm from raw result: {path}")
  arm = arm_matches[0]
  run = runs[arm]
  mode = str(result.get("reset_mode"))
  expected = {
    "evaluation_schema_version": _SCHEMA,
    "task": run["task"],
    "checkpoint_path": str(Path(run["checkpoint"]).resolve()),
    "checkpoint_sha256": run["checkpoint_sha256"],
    "policy_seed": seed,
    "seed": _EVAL_SEED,
    "num_envs": _NUM_ENVS,
    "steps": _STEPS,
  }
  for name, value in expected.items():
    if result.get(name) != value:
      raise ValueError(f"{path} {name} drifted: {result.get(name)} != {value}")
  if mode not in _MODES:
    raise ValueError(f"{path} has an unexpected reset mode: {mode}")
  diagnosis = result.get("strict_failure_diagnosis")
  if not isinstance(diagnosis, dict) or diagnosis.get("schema_version") != _FAILURE_SCHEMA:
    raise ValueError(f"{path} lacks failure telemetry schema 1")
  if diagnosis.get("does_not_change_strict_success") is not True:
    raise ValueError(f"{path} failure telemetry changed strict success")
  if diagnosis.get("reason_codebook") != _FAILURE_CODEBOOK:
    raise ValueError(f"{path} failure reason codebook drifted")
  counts = diagnosis.get("reason_counts")
  if not isinstance(counts, dict) or set(counts) != set(_FAILURE_CODEBOOK):
    raise ValueError(f"{path} failure reason counts are incomplete")
  if sum(int(value) for value in counts.values()) != _NUM_ENVS:
    raise ValueError(f"{path} failure reason counts do not sum to {_NUM_ENVS}")
  strict_successes = int(result.get("strict_successes", -1))
  if int(counts["0"]) != strict_successes:
    raise ValueError(f"{path} success reason count differs from strict successes")
  per_env = result.get("per_env")
  if not isinstance(per_env, dict):
    raise ValueError(f"{path} lacks per-environment telemetry")
  for name, values in per_env.items():
    if not isinstance(values, list) or len(values) != _NUM_ENVS:
      raise ValueError(f"{path} per_env.{name} does not contain {_NUM_ENVS} rows")
  for name in _PER_ENV_GATE_ARRAYS:
    if name not in per_env:
      raise ValueError(f"{path} lacks per_env.{name}")
  observed_counts = Counter(str(int(code)) for code in per_env["strict_failure_reason_code"])
  if {code: observed_counts.get(code, 0) for code in _FAILURE_CODEBOOK} != {
    code: int(counts[code]) for code in _FAILURE_CODEBOOK
  }:
    raise ValueError(f"{path} per-environment failure codes disagree with counts")
  if sum(bool(value) for value in per_env["finite_action"]) != _NUM_ENVS:
    raise ValueError(f"{path} has a nonfinite action rollout")
  if any(bool(value) for value in per_env["invalid_dynamics"]):
    raise ValueError(f"{path} has invalid dynamics")
  rate = strict_successes / _NUM_ENVS
  if not math.isclose(_finite(result.get("strict_success_rate"), "strict success"), rate):
    raise ValueError(f"{path} strict success rate disagrees with counts")
  if _finite(result.get("finite_action_rate"), "finite action rate") != 1.0:
    raise ValueError(f"{path} finite action rate is not one")
  if _finite(result.get("invalid_dynamics_rate"), "invalid dynamics rate") != 0.0:
    raise ValueError(f"{path} invalid dynamics rate is not zero")
  return arm, mode, result, {
    "path": str(path.resolve()),
    "sha256": _sha256(path),
    "arm": arm,
    "reset_mode": mode,
  }


def _arm_metrics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
  if set(rows) != set(_MODES):
    raise ValueError(f"arm does not contain exactly the five frozen modes: {set(rows)}")
  modes = {mode: int(rows[mode]["strict_successes"]) / _NUM_ENVS for mode in _MODES}
  fixed = [modes[mode] for mode in _FIXED_MODES]
  failure_counts = {
    code: sum(int(rows[mode]["strict_failure_diagnosis"]["reason_counts"][code]) for mode in _MODES)
    for code in _FAILURE_CODEBOOK
  }
  return {
    "modes": modes,
    "gsi": modes["native_gsi"],
    "fixed_macro": sum(fixed) / len(fixed),
    "fixed_worst": min(fixed),
    "finite_action_rate_min": min(float(row["finite_action_rate"]) for row in rows.values()),
    "invalid_dynamics_rate_max": max(float(row["invalid_dynamics_rate"]) for row in rows.values()),
    "safety_pareto": {
      name: max(float(row[name]) for row in rows.values()) for name in _SAFETY_MAX
    },
    "failure_reason_counts": failure_counts,
  }


def _audit_matrix(
  cfg: FlatMethodAnalysisCfg,
  manifest_row: dict[str, Any],
  seed: int,
  gate: int,
) -> dict[str, Any]:
  manifest_path = Path(manifest_row["path"])
  manifest = _load(manifest_path)
  if (
    manifest.get("status") != "READY_FOR_FROZEN_EVALUATION"
    or manifest.get("study_id") != _STUDY_ID
    or manifest.get("launch_plan_id") != _PLAN_ID
    or manifest.get("protocol_sha256") != _PROTOCOL_SHA256
    or manifest.get("checkpoint_step") != gate
    or manifest.get("policy_seed") != seed
  ):
    raise ValueError(f"flat method manifest identity drifted: seed {seed} gate {gate}")
  stable_manifest = {
    key: value for key, value in manifest.items()
    if key not in {"manifest_id", "generated_at_utc", "claim_boundary"}
  }
  if manifest.get("manifest_id") != hashlib.sha256(
    json.dumps(stable_manifest, sort_keys=True).encode()
  ).hexdigest():
    raise ValueError(f"flat method manifest ID drifted: seed {seed} gate {gate}")
  manifest_runs = manifest.get("runs")
  if not isinstance(manifest_runs, list) or len(manifest_runs) != 2:
    raise ValueError("flat method manifest must contain both arms")
  runs = {str(run["name"]): run for run in manifest_runs}
  if set(runs) != set(_ARMS):
    raise ValueError("flat method manifest arm set drifted")
  matrix_dir = cfg.evaluation_root / f"gate_{gate}" / f"seed_{seed}"
  complete = _load(matrix_dir / "_COMPLETE.json")
  expected_complete = {
    "evaluation_schema_version": _SCHEMA,
    "manifest": str(manifest_path.resolve()),
    "result_count": 10,
    "modes": list(_MODES),
    "eval_seeds": [_EVAL_SEED],
    "num_envs": _NUM_ENVS,
    "steps": _STEPS,
  }
  for name, value in expected_complete.items():
    if complete.get(name) != value:
      raise ValueError(f"matrix seed {seed} gate {gate} completion {name} drifted")
  summary_path = matrix_dir / "summary.json"
  summary = _load(summary_path)
  if summary.get("metadata", {}).get("manifest_id") != manifest.get("manifest_id"):
    raise ValueError(f"matrix seed {seed} gate {gate} summary lineage drifted")
  summary_rows = summary.get("evaluations")
  if not isinstance(summary_rows, list) or len(summary_rows) != 10:
    raise ValueError(f"matrix seed {seed} gate {gate} summary is incomplete")
  raw_paths = sorted(
    path for path in matrix_dir.glob("*.json")
    if path.name not in {"_COMPLETE.json", "summary.json"}
  )
  if len(raw_paths) != 10:
    raise ValueError(f"matrix seed {seed} gate {gate} must contain ten raw results")
  by_arm: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in _ARMS}
  raw_evidence = []
  raw_by_key = {}
  for path in raw_paths:
    arm, mode, result, evidence = _validate_result(path, manifest, runs, seed, gate)
    if mode in by_arm[arm]:
      raise ValueError(f"duplicate result for {arm}/{mode} seed {seed} gate {gate}")
    by_arm[arm][mode] = result
    raw_evidence.append(evidence)
  summary_keys = {(row.get("arm"), row.get("reset_mode")) for row in summary_rows}
  raw_by_key = {
    (arm, mode): result
    for arm, modes in by_arm.items()
    for mode, result in modes.items()
  }
  if summary_keys != {(arm, mode) for arm in _ARMS for mode in _MODES}:
    raise ValueError(f"matrix seed {seed} gate {gate} summary factorial drifted")
  summary_fields = (
    "evaluation_schema_version",
    "task",
    "checkpoint_path",
    "checkpoint_sha256",
    "policy_seed",
    "seed",
    "num_envs",
    "steps",
    "strict_successes",
    "strict_success_rate",
    "finite_action_rate",
    "invalid_dynamics_rate",
  )
  for row in summary_rows:
    raw = raw_by_key[(row["arm"], row["reset_mode"])]
    if any(row.get(name) != raw.get(name) for name in summary_fields):
      raise ValueError(
        f"matrix seed {seed} gate {gate} summary disagrees with raw result: "
        f"{row['arm']}/{row['reset_mode']}"
      )
  return {
    "policy_seed": seed,
    "checkpoint_step": gate,
    "manifest": str(manifest_path.resolve()),
    "manifest_sha256": manifest_row["sha256"],
    "complete_sha256": _sha256(matrix_dir / "_COMPLETE.json"),
    "summary_sha256": _sha256(summary_path),
    "raw_result_count": len(raw_paths),
    "raw_results": sorted(raw_evidence, key=lambda row: (row["arm"], row["reset_mode"])),
    "arms": {arm: _arm_metrics(by_arm[arm]) for arm in _ARMS},
  }


def analyze(cfg: FlatMethodAnalysisCfg) -> dict[str, Any]:
  protocol = _validate_protocol(cfg.protocol)
  index, manifests = _validate_index(cfg.manifest_index)
  matrices = []
  metrics: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}
  for gate in _GATES:
    metrics[gate] = {}
    for seed in _SEEDS:
      matrix = _audit_matrix(cfg, manifests[(seed, gate)], seed, gate)
      matrices.append(matrix)
      metrics[gate][seed] = matrix["arms"]

  gate_aggregates: dict[str, Any] = {}
  for gate in _GATES:
    arms = {}
    for arm in _ARMS:
      seed_rows = {seed: metrics[gate][seed][arm] for seed in _SEEDS}
      arms[arm] = {
        "policy_seeds": {str(seed): seed_rows[seed] for seed in _SEEDS},
        "gsi_mean": sum(row["gsi"] for row in seed_rows.values()) / len(_SEEDS),
        "fixed_macro_mean": sum(row["fixed_macro"] for row in seed_rows.values()) / len(_SEEDS),
        "fixed_macro_min_seed": min(row["fixed_macro"] for row in seed_rows.values()),
        "fixed_worst_mean": sum(row["fixed_worst"] for row in seed_rows.values()) / len(_SEEDS),
        "fixed_worst_min_seed": min(row["fixed_worst"] for row in seed_rows.values()),
        "finite_action_rate_min": min(row["finite_action_rate_min"] for row in seed_rows.values()),
        "invalid_dynamics_rate_max": max(row["invalid_dynamics_rate_max"] for row in seed_rows.values()),
        "safety_pareto": {
          name: {
            "policy_seed_values": {
              str(seed): seed_rows[seed]["safety_pareto"][name] for seed in _SEEDS
            },
            "mean": sum(seed_rows[seed]["safety_pareto"][name] for seed in _SEEDS)
            / len(_SEEDS),
            "worst": max(seed_rows[seed]["safety_pareto"][name] for seed in _SEEDS),
          }
          for name in _SAFETY_MAX
        },
      }
    gate_aggregates[str(gate)] = {"arms": arms}

  absolute = protocol["frozen_absolute_promotion_rule"]
  absolute_gate_results = {}
  for gate in _STABILITY_GATES:
    row = gate_aggregates[str(gate)]["arms"][_PROPOSED]
    checks = {
      "gsi_mean": row["gsi_mean"] >= float(absolute["gsi_mean_min"]),
      "fixed_macro_mean": row["fixed_macro_mean"] >= float(absolute["fixed_macro_mean_min"]),
      "fixed_macro_min_seed": row["fixed_macro_min_seed"] >= float(absolute["fixed_macro_min_seed"]),
      "fixed_worst_mean": row["fixed_worst_mean"] >= float(absolute["fixed_worst_mean_min"]),
      "fixed_worst_min_seed": row["fixed_worst_min_seed"] >= float(absolute["fixed_worst_min_seed"]),
      "finite_actions": row["finite_action_rate_min"] == float(absolute["finite_action_rate_required"]),
      "valid_dynamics": row["invalid_dynamics_rate_max"] == float(absolute["invalid_dynamics_rate_required"]),
    }
    absolute_gate_results[str(gate)] = {"checks": checks, "pass": all(checks.values())}

  regression_rows = {}
  max_regression = float(absolute["max_25k_to_final_regression"])
  for seed in _SEEDS:
    before = metrics[25000][seed][_PROPOSED]
    final = metrics[29999][seed][_PROPOSED]
    changes = {
      name: final[name] - before[name] for name in ("gsi", "fixed_macro", "fixed_worst")
    }
    regression_rows[str(seed)] = {
      "final_minus_25k": changes,
      "pass": all(value >= -max_regression for value in changes.values()),
    }
  regression_pass = all(row["pass"] for row in regression_rows.values())

  rng = random.Random(_BOOTSTRAP_SEED)
  paired = {}
  for name in ("fixed_worst", "fixed_macro", "gsi"):
    effects = [
      metrics[29999][seed][_PROPOSED][name] - metrics[29999][seed][_CONTROL][name]
      for seed in _SEEDS
    ]
    paired[name] = _bootstrap_paired(effects, rng)
  safety_paired = {}
  for name in _SAFETY_MAX:
    effects = [
      metrics[29999][seed][_PROPOSED]["safety_pareto"][name]
      - metrics[29999][seed][_CONTROL]["safety_pareto"][name]
      for seed in _SEEDS
    ]
    safety_paired[name] = {
      **_bootstrap_paired(effects, rng),
      "direction": "negative_favors_a8",
      "role": "reported_pareto_not_a_promotion_gate",
    }
  paired_checks = {
    "fixed_worst_superiority": paired["fixed_worst"]["ci95_low"] > 0.0,
    "fixed_macro_noninferiority": paired["fixed_macro"]["ci95_low"] >= -float(
      protocol["paired_causal_analysis"]["fixed_macro_noninferiority_margin"]
    ),
    "gsi_noninferiority": paired["gsi"]["ci95_low"] >= -float(
      protocol["paired_causal_analysis"]["gsi_noninferiority_margin"]
    ),
  }
  absolute_pass = all(row["pass"] for row in absolute_gate_results.values())
  paired_pass = all(paired_checks.values())
  promote = absolute_pass and regression_pass and paired_pass
  return {
    "schema_version": 1,
    "status": "PROMOTE_TP_SPECIALISTS" if promote else "FLAT_METHOD_COMPLETE_NO_PROMOTION",
    "study_id": _STUDY_ID,
    "claim_boundary": (
      "This fresh-seed flat study can only promote A8 into the existing T/P specialist "
      "workflow. It is not terrain, plate, unified, hardware, baseline, or RAL evidence."
    ),
    "protocol": str(cfg.protocol.resolve()),
    "protocol_sha256": _PROTOCOL_SHA256,
    "manifest_index": str(cfg.manifest_index.resolve()),
    "manifest_index_sha256": _sha256(cfg.manifest_index),
    "manifest_index_id": index["index_id"],
    "evaluation_contract": {
      "policy_seeds": list(_SEEDS),
      "checkpoint_steps": list(_GATES),
      "reset_modes": list(_MODES),
      "evaluation_seed": _EVAL_SEED,
      "num_envs": _NUM_ENVS,
      "steps": _STEPS,
      "evaluation_schema_version": _SCHEMA,
      "matrix_count": 12,
      "raw_result_count": 120,
      "per_environment_rollout_count": 120 * _NUM_ENVS,
      "sampling_unit": "independently_trained_policy_seed",
    },
    "matrix_audits": matrices,
    "gate_aggregates": gate_aggregates,
    "absolute_promotion_gate": {
      "thresholds": absolute,
      "gates": absolute_gate_results,
      "pass": absolute_pass,
    },
    "late_regression_gate": {
      "maximum_allowed_25k_to_final_regression": max_regression,
      "policy_seeds": regression_rows,
      "pass": regression_pass,
    },
    "paired_causal_analysis": {
      "pairing_key": "policy_seed",
      "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
      "bootstrap_seed": _BOOTSTRAP_SEED,
      "effects": paired,
      "safety_pareto_effects": safety_paired,
      "checks": paired_checks,
      "pass": paired_pass,
    },
    "promotion": {
      "eligible_arm": _PROPOSED,
      "control_arm_promotion_eligible": False,
      "decision": "PROMOTE_TP_SPECIALISTS" if promote else "NO_PROMOTION",
      "pass": promote,
    },
    "limitations": [
      "Three policy seeds provide coarse policy-seed uncertainty.",
      "Rollout environments are never counted as independent training replicates.",
      "A complete null or unsafe result is retained without threshold changes or selective reruns.",
    ],
  }


def _markdown(result: dict[str, Any]) -> str:
  lines = [
    "# Fresh-seed flat method study",
    "",
    f"Decision: **{result['status']}**",
    "",
    "| Gate | Arm | GSI mean | Fixed macro | Fixed macro min seed | Fixed worst | Fixed worst min seed |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
  ]
  for gate in _GATES:
    for arm in _ARMS:
      row = result["gate_aggregates"][str(gate)]["arms"][arm]
      lines.append(
        f"| {gate} | {arm} | {row['gsi_mean']:.1%} | {row['fixed_macro_mean']:.1%} | "
        f"{row['fixed_macro_min_seed']:.1%} | {row['fixed_worst_mean']:.1%} | "
        f"{row['fixed_worst_min_seed']:.1%} |"
      )
  lines.extend(("", "## Paired final effects (A8 - A6)", ""))
  for name, effect in result["paired_causal_analysis"]["effects"].items():
    lines.append(
      f"- {name}: {effect['mean']:+.3f} "
      f"[{effect['ci95_low']:+.3f}, {effect['ci95_high']:+.3f}]"
    )
  lines.extend(
    (
      "",
      "The policy seed is the sampling unit. This result is not terrain, plate, "
      "unified, hardware, baseline, or RAL evidence.",
      "",
    )
  )
  return "\n".join(lines)


def _write_immutable(path: Path, content: str) -> None:
  if path.exists():
    if path.read_text() != content:
      raise ValueError(f"existing flat method analysis conflicts: {path}")
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def write_analysis(cfg: FlatMethodAnalysisCfg) -> dict[str, Any]:
  result = analyze(cfg)
  _write_immutable(cfg.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
  _write_immutable(cfg.output_markdown, _markdown(result))
  return result


def main(cfg: FlatMethodAnalysisCfg) -> None:
  result = write_analysis(cfg)
  print(
    f"{result['status']}: matrices={result['evaluation_contract']['matrix_count']} "
    f"raw_results={result['evaluation_contract']['raw_result_count']}"
  )


if __name__ == "__main__":
  main(tyro.cli(FlatMethodAnalysisCfg))
