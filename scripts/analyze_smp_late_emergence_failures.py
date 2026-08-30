"""Diagnose a complete negative SMP late-emergence follow-up without retuning."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")
_FIXED_MODES = _MODES[1:]
_EVALUATION_SCHEMA_VERSION = 2
_DYNAMIC_METRICS = (
  "contact_foot_slip_p95_m_s",
  "root_planar_excursion_p95_m",
  "max_root_linear_speed_mean_m_s",
  "max_root_angular_speed_mean_rad_s",
  "max_joint_speed_p95_rad_s",
  "max_power_mean_w",
  "action_delta_rms_p95",
  "action_second_difference_rms_p95",
)
_PER_ENV_REQUIRED = (
  "strict_first_step",
  "initial_head_z_m",
  "root_planar_excursion_m",
  "contact_foot_slip_m_s",
  "max_joint_speed_rad_s",
  "max_power_w",
  "action_delta_rms",
  "action_second_difference_rms",
)


@dataclass(frozen=True)
class DiagnosticCfg:
  evidence_dir: Path
  promotion: Path
  aggregate: Path
  output_json: Path
  output_markdown: Path | None = None
  expected_policy_seeds: tuple[int, ...] = (20260901, 20260902, 20260903)
  expected_arms: tuple[str, ...] = ("a1_v7_gsi", "a6_f2s2_mix_bridge")
  evaluation_seed: int = 20260829
  num_envs: int = 512
  steps: int = 500


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise ValueError(f"required artifact is missing: {path}")
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"{path} must contain a JSON object")
  return payload


def _quantile(values: list[float], q: float) -> float | None:
  finite = sorted(value for value in values if math.isfinite(value))
  if not finite:
    return None
  index = q * (len(finite) - 1)
  lower = int(index)
  upper = min(lower + 1, len(finite) - 1)
  fraction = index - lower
  return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def _stats(values: list[float]) -> dict[str, float | int | None]:
  finite = [value for value in values if math.isfinite(value)]
  return {
    "count": len(finite),
    "mean": (sum(finite) / len(finite) if finite else None),
    "median": (statistics.median(finite) if finite else None),
    "p05": _quantile(finite, 0.05),
    "p95": _quantile(finite, 0.95),
  }


def _float_list(values: Any, *, path: Path, key: str, count: int) -> list[float]:
  if not isinstance(values, list) or len(values) != count:
    raise ValueError(f"{path} per_env.{key} must contain {count} values")
  output = []
  for value in values:
    try:
      number = float(value)
    except (TypeError, ValueError) as error:
      raise ValueError(f"{path} per_env.{key} contains a non-numeric value") from error
    if not math.isfinite(number):
      raise ValueError(f"{path} per_env.{key} contains NaN/Inf")
    output.append(number)
  return output


def _validate_cell(
  path: Path,
  payload: dict[str, Any],
  *,
  arm: str,
  mode: str,
  policy_seed: int,
  cfg: DiagnosticCfg,
) -> None:
  expected = {
    "evaluation_schema_version": _EVALUATION_SCHEMA_VERSION,
    "seed": cfg.evaluation_seed,
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "policy_seed": policy_seed,
    "reset_mode": mode,
    "actor_observation_dim": 93,
    "checkpoint": "model_29999.pt",
  }
  for key, value in expected.items():
    if payload.get(key) != value:
      raise ValueError(f"{path} {key}={payload.get(key)!r}, expected {value!r}")
  if payload.get("finite_action_rate") != 1.0:
    raise ValueError(f"{path} has non-finite actions")
  success_count = sum(
    int(step) >= 0 for step in payload.get("per_env", {}).get("strict_first_step", [])
  )
  if payload.get("strict_successes") != success_count:
    raise ValueError(f"{path} strict success count does not match per-env data")
  expected_rate = success_count / cfg.num_envs
  if payload.get("strict_success_rate") != expected_rate:
    raise ValueError(f"{path} strict success rate does not match per-env data")
  task = str(payload.get("task", ""))
  if arm not in path.name or not task:
    raise ValueError(f"{path} does not bind arm {arm} to a task")
  per_env = payload.get("per_env")
  if not isinstance(per_env, dict):
    raise ValueError(f"{path} has no per_env object")
  for key in _PER_ENV_REQUIRED:
    _float_list(per_env.get(key), path=path, key=key, count=cfg.num_envs)


def _validate_completion(path: Path, payload: dict[str, Any], cfg: DiagnosticCfg) -> None:
  expected = {
    "evaluation_schema_version": _EVALUATION_SCHEMA_VERSION,
    "eval_seeds": [cfg.evaluation_seed],
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "modes": list(_MODES),
    "result_count": len(cfg.expected_arms) * len(_MODES),
  }
  for key, value in expected.items():
    if payload.get(key) != value:
      raise ValueError(
        f"{path} completion {key}={payload.get(key)!r}, expected {value!r}"
      )
  devices = payload.get("devices")
  if not isinstance(devices, list) or not devices:
    raise ValueError(f"{path} completion marker has no evaluation devices")
  if not isinstance(payload.get("manifest"), str) or not payload["manifest"]:
    raise ValueError(f"{path} completion marker has no manifest binding")


def _cell_diagnostic(path: Path, payload: dict[str, Any], cfg: DiagnosticCfg) -> dict[str, Any]:
  per_env = payload["per_env"]
  first_step = _float_list(
    per_env["strict_first_step"], path=path, key="strict_first_step", count=cfg.num_envs
  )
  success = [step >= 0 for step in first_step]
  initial_head = _float_list(
    per_env["initial_head_z_m"], path=path, key="initial_head_z_m", count=cfg.num_envs
  )
  result = {
    "path": str(path.resolve()),
    "sha256": _sha256(path),
    "strict_successes": int(payload["strict_successes"]),
    "num_envs": cfg.num_envs,
    "strict_success_rate": float(payload["strict_success_rate"]),
    "initial_head_z_m": {
      "all": _stats(initial_head),
      "success": _stats([value for value, passed in zip(initial_head, success) if passed]),
      "failure": _stats([value for value, passed in zip(initial_head, success) if not passed]),
    },
    "strict_first_step_success_only": _stats(
      [step for step in first_step if step >= 0]
    ),
  }
  for metric in _DYNAMIC_METRICS:
    value = payload.get(metric)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
      raise ValueError(f"{path} has invalid {metric}")
    result[metric] = float(value)
  return result


def _pattern(rates: list[float]) -> str:
  if max(rates) == 0.0:
    return "ALL_SEEDS_ZERO_SUCCESS"
  if min(rates) == 0.0:
    return "SEED_COLLAPSE_TO_ZERO"
  if max(rates) - min(rates) >= 0.25:
    return "HIGH_SEED_SPREAD"
  return "CONSISTENT_NONZERO_SUCCESS"


def _arm_diagnostic(
  arm: str,
  cells: dict[int, dict[str, dict[str, Any]]],
  seeds: tuple[int, ...],
) -> dict[str, Any]:
  seed_level = []
  for seed in seeds:
    modes = cells[seed]
    rates = {mode: modes[mode]["strict_success_rate"] for mode in _MODES}
    fixed = [rates[mode] for mode in _FIXED_MODES]
    seed_level.append(
      {
        "policy_seed": seed,
        "mode_success_rates": rates,
        "gsi": rates["native_gsi"],
        "fixed_macro": sum(fixed) / len(fixed),
        "fixed_worst": min(fixed),
        "dynamic_worst": {
          metric: max(modes[mode][metric] for mode in _MODES)
          for metric in _DYNAMIC_METRICS
        },
      }
    )
  mode_patterns = {}
  for mode in _MODES:
    rates = [cells[seed][mode]["strict_success_rate"] for seed in seeds]
    mode_patterns[mode] = {
      "policy_seed_values": rates,
      "mean": sum(rates) / len(rates),
      "minimum": min(rates),
      "maximum": max(rates),
      "spread": max(rates) - min(rates),
      "pattern": _pattern(rates),
    }
  return {
    "arm": arm,
    "seed_level": seed_level,
    "mode_patterns": mode_patterns,
    "all_seed_fixed_worst_zero": all(row["fixed_worst"] == 0.0 for row in seed_level),
    "has_seed_collapse": any(
      item["pattern"] == "SEED_COLLAPSE_TO_ZERO" for item in mode_patterns.values()
    ),
    "cells": {str(seed): cells[seed] for seed in seeds},
  }


def analyze(cfg: DiagnosticCfg) -> dict[str, Any]:
  if len(cfg.expected_policy_seeds) < 3:
    raise ValueError("diagnosis requires at least three independent policy seeds")
  if len(set(cfg.expected_policy_seeds)) != len(cfg.expected_policy_seeds):
    raise ValueError("expected policy seeds must be unique")
  promotion = _load_json(cfg.promotion)
  aggregate = _load_json(cfg.aggregate)
  if promotion.get("status") != "NO_PROMOTION" or promotion.get("selected_arm") is not None:
    raise ValueError("late-emergence diagnosis is only valid for terminal NO_PROMOTION")
  if aggregate.get("status") != "MINIMUM_POLICY_SEEDS_MET":
    raise ValueError("policy-seed aggregate is not complete")
  if tuple(aggregate.get("policy_seeds", ())) != tuple(sorted(cfg.expected_policy_seeds)):
    raise ValueError("aggregate policy seeds do not match the frozen diagnosis protocol")
  if promotion.get("aggregate_sha256") != _sha256(cfg.aggregate):
    raise ValueError("promotion does not bind the supplied aggregate SHA")

  all_cells: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
  source_artifacts = []
  for seed in cfg.expected_policy_seeds:
    marker_path = cfg.evidence_dir / f"seed_{seed}" / "_COMPLETE.json"
    complete = _load_json(marker_path)
    _validate_completion(marker_path, complete, cfg)
    source_artifacts.append(
      {
        "kind": "completion_marker",
        "path": str(marker_path.resolve()),
        "sha256": _sha256(marker_path),
      }
    )

  for arm in cfg.expected_arms:
    arm_cells: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in cfg.expected_policy_seeds:
      seed_dir = cfg.evidence_dir / f"seed_{seed}"
      mode_cells = {}
      for mode in _MODES:
        path = seed_dir / (
          f"{arm}__model_29999__{mode}__eval{cfg.evaluation_seed}.json"
        )
        payload = _load_json(path)
        _validate_cell(
          path,
          payload,
          arm=arm,
          mode=mode,
          policy_seed=seed,
          cfg=cfg,
        )
        mode_cells[mode] = _cell_diagnostic(path, payload, cfg)
      arm_cells[seed] = mode_cells
    all_cells[arm] = arm_cells

  arms = {
    arm: _arm_diagnostic(arm, all_cells[arm], cfg.expected_policy_seeds)
    for arm in cfg.expected_arms
  }
  fixed_zero = [arm for arm, item in arms.items() if item["all_seed_fixed_worst_zero"]]
  seed_collapse = [arm for arm, item in arms.items() if item["has_seed_collapse"]]
  expected_cells = len(cfg.expected_policy_seeds) * len(cfg.expected_arms) * len(_MODES)
  return {
    "schema_version": 1,
    "status": "DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_RETRAINING",
    "study_status": "COMPLETE_NO_PROMOTION",
    "expected_cell_count": expected_cells,
    "validated_cell_count": expected_cells,
    "policy_seeds": list(cfg.expected_policy_seeds),
    "arms": arms,
    "dominant_failure_structure": {
      "arms_with_zero_fixed_worst_for_every_seed": fixed_zero,
      "arms_with_at_least_one_pose_collapsing_to_zero_by_seed": seed_collapse,
      "interpretation": (
        "The retained result is dominated by fixed-pose coverage and training-seed "
        "stability failure, not by missing files, invalid physics, or non-finite actions."
      ),
    },
    "telemetry_limit": (
      "Schema-2 preserves per-environment success, initial state, excursion, action, and "
      "dynamic loads, but not the stepwise head-height/upright traces or terminal failure "
      "reason needed to distinguish never-rises from rises-but-fails-the-25-step-hold."
    ),
    "next_study_requirements": [
      "Preserve the original and late-emergence NO_PROMOTION artifacts and thresholds.",
      "Treat fixed-pose coverage and across-seed stability as primary design objectives in any new preregistered study.",
      "Add stepwise gate telemetry or a frozen failure-reason state machine before claiming a mechanism for zero-success cells.",
      "Use fresh training and evaluation seeds for a new study; do not selectively rerun these 30 cells.",
      "Do not trade higher GSI for unbounded slip, power, qvel, root motion, or action variation; report the full Pareto vector.",
    ],
    "automatic_action": "STOP_NO_SAFE_AUTOMATIC_RETRAINING",
    "claim_boundary": (
      "This post-hoc diagnostic summarizes an already complete frozen null result. It does "
      "not authorize threshold changes, selective reruns, T/P, baselines, hardware, or RA-L claims."
    ),
    "promotion": {
      "path": str(cfg.promotion.resolve()),
      "sha256": _sha256(cfg.promotion),
      "promotion_id": promotion.get("promotion_id"),
    },
    "aggregate": {
      "path": str(cfg.aggregate.resolve()),
      "sha256": _sha256(cfg.aggregate),
    },
    "source_artifacts": source_artifacts,
  }


def _markdown(result: dict[str, Any]) -> str:
  lines = [
    "# SMP late-emergence failure diagnosis",
    "",
    f"Status: **{result['status']}**",
    "",
    "| Arm | Mode | Seed success rates | Pattern |",
    "| --- | --- | --- | --- |",
  ]
  for arm, arm_data in result["arms"].items():
    for mode, data in arm_data["mode_patterns"].items():
      rates = ", ".join(f"{value:.3f}" for value in data["policy_seed_values"])
      lines.append(f"| {arm} | {mode} | {rates} | {data['pattern']} |")
  lines.extend(
    (
      "",
      "## Interpretation boundary",
      "",
      result["telemetry_limit"],
      "",
      result["claim_boundary"],
      "",
    )
  )
  return "\n".join(lines)


def _write_immutable(path: Path, content: str) -> None:
  if path.exists():
    if path.read_text() != content:
      raise ValueError(f"immutable output already exists with different content: {path}")
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def write_diagnostic(cfg: DiagnosticCfg) -> dict[str, Any]:
  result = analyze(cfg)
  output_markdown = cfg.output_markdown or cfg.output_json.with_suffix(".md")
  _write_immutable(
    cfg.output_json,
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
  )
  _write_immutable(output_markdown, _markdown(result))
  return result


def main(cfg: DiagnosticCfg) -> None:
  result = write_diagnostic(cfg)
  print(
    f"{result['status']}: validated_cells={result['validated_cell_count']} "
    f"automatic_action={result['automatic_action']}"
  )


if __name__ == "__main__":
  main(tyro.cli(DiagnosticCfg))
