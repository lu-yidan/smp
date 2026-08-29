"""Validate and summarize the preregistered real-G1 recovery trial ledger."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_POSE_COUNTS = {
  "prone": 15,
  "supine": 15,
  "left_side": 15,
  "right_side": 15,
  "random_fall_state": 20,
}
_FIXED_POSES = ("prone", "supine", "left_side", "right_side")
_BOOLEAN_FIELDS = (
  "valid_initialization",
  "success",
  "first_stand",
  "secondary_fall",
  "safety_abort",
  "human_contact",
  "tether_assist",
)
_SAFETY_FIELDS = (
  "max_abs_joint_velocity_rad_s",
  "max_abs_tau_est_nm",
  "max_abs_tau_cmd_est_nm",
  "max_abs_imu_angular_velocity_rad_s",
  "action_delta_rms",
  "action_second_difference_rms",
)
_PROVENANCE_FIELDS = (
  "block_id",
  "randomization_seed",
  "policy_seed",
  "checkpoint_sha256",
  "onnx_sha256",
  "deploy_git_commit",
  "logger_schema_version",
  "robot_id",
  "condition",
  "surface",
)
_REQUIRED_FIELDS = {
  "trial_id",
  "order_index",
  "deploy_repository_dirty",
  "recovery_time_s",
  "failure_class",
  "log_bin_path",
  "log_json_path",
  "video_uri",
  "operator_id",
  "policy_start_time_utc",
  "initial_pose",
  *_BOOLEAN_FIELDS,
  *_SAFETY_FIELDS,
  *_PROVENANCE_FIELDS,
}
_FAILURE_CLASSES = {
  "no_progress",
  "repeated_struggle",
  "pelvis_slip",
  "leg_flailing",
  "rearward_fall",
  "lateral_fall",
  "small_step_instability",
  "joint_or_torque_limit",
  "operator_abort",
  "estimator_fault",
  "invalid_initialization",
  "other",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class HardwareAnalysisCfg:
  trials: Path
  output_json: Path
  output_markdown: Path | None = None
  require_complete: bool = True
  require_clean_deploy: bool = True
  required_logger_schema: int = 2
  expected_condition: str = "flat_core"


def _boolean(value: str, field: str, trial_id: str) -> bool:
  normalized = value.strip().lower()
  if normalized in {"1", "true", "yes"}:
    return True
  if normalized in {"0", "false", "no"}:
    return False
  raise ValueError(f"{trial_id}: {field} must be true or false, got {value!r}")


def _number(value: str, field: str, trial_id: str) -> float:
  try:
    result = float(value)
  except ValueError as error:
    raise ValueError(f"{trial_id}: invalid {field}: {value!r}") from error
  if not math.isfinite(result):
    raise ValueError(f"{trial_id}: non-finite {field}: {value!r}")
  return result


def _resolve(path: str, ledger: Path) -> Path:
  candidate = Path(path)
  return candidate if candidate.is_absolute() else ledger.parent / candidate


def _wilson(successes: int, total: int) -> dict[str, float | int]:
  if total <= 0:
    return {
      "successes": successes,
      "total": total,
      "rate": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0,
    }
  z = 1.959963984540054
  rate = successes / total
  denominator = 1.0 + z**2 / total
  center = (rate + z**2 / (2.0 * total)) / denominator
  radius = (
    z * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) / denominator
  )
  return {
    "successes": successes,
    "total": total,
    "rate": rate,
    "ci95_low": max(0.0, center - radius),
    "ci95_high": min(1.0, center + radius),
  }


def _quantile(values: list[float], fraction: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  position = fraction * (len(ordered) - 1)
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  weight = position - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int | None]:
  return {
    "count": len(values),
    "median": statistics.median(values) if values else None,
    "p90": _quantile(values, 0.90),
    "p95": _quantile(values, 0.95),
    "max": max(values) if values else None,
  }


def _load_rows(cfg: HardwareAnalysisCfg) -> tuple[list[dict[str, Any]], dict[str, str]]:
  with cfg.trials.open(newline="") as stream:
    reader = csv.DictReader(stream)
    missing = _REQUIRED_FIELDS - set(reader.fieldnames or ())
    if missing:
      raise ValueError(f"trial ledger is missing columns: {sorted(missing)}")
    raw_rows = list(reader)
  if not raw_rows:
    raise ValueError("trial ledger contains no rows")

  seen_trials: set[str] = set()
  seen_orders: set[int] = set()
  rows = []
  for raw in raw_rows:
    trial_id = raw["trial_id"].strip()
    if not trial_id or trial_id in seen_trials:
      raise ValueError(f"missing or duplicate trial_id: {trial_id!r}")
    seen_trials.add(trial_id)
    order = int(raw["order_index"])
    if order in seen_orders:
      raise ValueError(f"duplicate order_index: {order}")
    seen_orders.add(order)
    row: dict[str, Any] = dict(raw)
    row["trial_id"] = trial_id
    row["order_index"] = order
    for field in _BOOLEAN_FIELDS:
      row[field] = _boolean(raw[field], field, trial_id)
    row["deploy_repository_dirty"] = _boolean(
      raw["deploy_repository_dirty"], "deploy_repository_dirty", trial_id
    )
    for field in _SAFETY_FIELDS:
      if raw[field].strip():
        row[field] = _number(raw[field], field, trial_id)
        if row[field] < 0.0:
          raise ValueError(f"{trial_id}: {field} must be nonnegative")
      elif row["valid_initialization"]:
        raise ValueError(f"{trial_id}: valid trial requires {field}")
      else:
        row[field] = None
    if raw["recovery_time_s"].strip():
      row["recovery_time_s"] = _number(
        raw["recovery_time_s"], "recovery_time_s", trial_id
      )
    else:
      row["recovery_time_s"] = None
    if row["success"] and (
      not row["first_stand"]
      or row["secondary_fall"]
      or row["safety_abort"]
      or row["human_contact"]
      or row["tether_assist"]
    ):
      raise ValueError(
        f"{trial_id}: success conflicts with intervention or failure fields"
      )
    if row["first_stand"]:
      recovery_time = row["recovery_time_s"]
      if recovery_time is None or not 0.0 <= recovery_time <= 10.0:
        raise ValueError(f"{trial_id}: first stand requires recovery_time_s in [0, 10]")
    failure_class = raw["failure_class"].strip()
    if row["valid_initialization"] and not row["success"]:
      if failure_class not in _FAILURE_CLASSES:
        raise ValueError(f"{trial_id}: invalid or missing failure_class")
    elif row["success"] and failure_class:
      raise ValueError(f"{trial_id}: successful trial cannot have failure_class")
    if not raw["video_uri"].strip():
      raise ValueError(f"{trial_id}: video_uri is required")
    rows.append(row)
  if seen_orders != set(range(len(rows))):
    raise ValueError("order_index must contain every integer from 0 through N-1")

  provenance: dict[str, str] = {}
  for field in _PROVENANCE_FIELDS:
    values = {str(row[field]).strip() for row in rows if row["valid_initialization"]}
    if len(values) != 1:
      raise ValueError(f"valid trials must share one {field}, got {sorted(values)}")
    provenance[field] = values.pop()
  if provenance["condition"] != cfg.expected_condition:
    raise ValueError(
      f"expected condition {cfg.expected_condition!r}, got {provenance['condition']!r}"
    )
  if int(provenance["logger_schema_version"]) < cfg.required_logger_schema:
    raise ValueError("logger schema is too old for quantitative hardware evidence")
  if not _SHA256.fullmatch(provenance["checkpoint_sha256"]):
    raise ValueError("checkpoint_sha256 must contain 64 lowercase hexadecimal digits")
  if not _SHA256.fullmatch(provenance["onnx_sha256"]):
    raise ValueError("onnx_sha256 must contain 64 lowercase hexadecimal digits")
  if not _COMMIT.fullmatch(provenance["deploy_git_commit"]):
    raise ValueError("deploy_git_commit is not a valid commit identifier")
  if cfg.require_clean_deploy and any(
    row["deploy_repository_dirty"] for row in rows if row["valid_initialization"]
  ):
    raise ValueError("final evidence contains a dirty deployment worktree")

  for row in rows:
    if not row["valid_initialization"]:
      continue
    binary = _resolve(row["log_bin_path"], cfg.trials)
    metadata = _resolve(row["log_json_path"], cfg.trials)
    if not binary.is_file() or binary.stat().st_size == 0:
      raise FileNotFoundError(binary)
    if not metadata.is_file():
      raise FileNotFoundError(metadata)
    meta = json.loads(metadata.read_text())
    if int(meta.get("logger_schema_version", 0)) < cfg.required_logger_schema:
      raise ValueError(f"{row['trial_id']}: log metadata has an old logger schema")
    if meta.get("deploy_git_commit") != row["deploy_git_commit"]:
      raise ValueError(
        f"{row['trial_id']}: deployment commit does not match log metadata"
      )
    if bool(meta.get("deploy_repository_dirty")) != row["deploy_repository_dirty"]:
      raise ValueError(
        f"{row['trial_id']}: dirty-worktree flag does not match metadata"
      )
    if int(meta.get("total_steps", 0)) <= 0:
      raise ValueError(f"{row['trial_id']}: log metadata reports no samples")
  return rows, provenance


def analyze(cfg: HardwareAnalysisCfg) -> dict[str, Any]:
  rows, provenance = _load_rows(cfg)
  valid = [row for row in rows if row["valid_initialization"]]
  counts = Counter(row["initial_pose"] for row in valid)
  unknown = set(counts) - set(_POSE_COUNTS)
  if unknown:
    raise ValueError(f"unknown initial pose strata: {sorted(unknown)}")
  complete = all(counts[pose] == expected for pose, expected in _POSE_COUNTS.items())
  if cfg.require_complete and not complete:
    raise ValueError(
      f"incomplete preregistered matrix: got {dict(counts)}, expected {_POSE_COUNTS}"
    )

  by_pose = {}
  for pose in _POSE_COUNTS:
    group = [row for row in valid if row["initial_pose"] == pose]
    success = sum(row["success"] for row in group)
    first_stand = sum(row["first_stand"] for row in group)
    secondary = sum(row["secondary_fall"] for row in group)
    aborts = sum(row["safety_abort"] for row in group)
    recovery = [row["recovery_time_s"] for row in group if row["success"]]
    by_pose[pose] = {
      "success": _wilson(success, len(group)),
      "first_stand": _wilson(first_stand, len(group)),
      "secondary_fall_rate": secondary / len(group) if group else 0.0,
      "safety_abort_rate": aborts / len(group) if group else 0.0,
      "recovery_time_s": _distribution(recovery),
    }

  overall_success = sum(row["success"] for row in valid)
  fixed_rates = [float(by_pose[pose]["success"]["rate"]) for pose in _FIXED_POSES]
  safety = {
    field: _distribution([float(row[field]) for row in valid])
    for field in _SAFETY_FIELDS
  }
  failures = Counter(
    row["failure_class"] or "unclassified" for row in valid if not row["success"]
  )
  return {
    "status": "COMPLETE" if complete else "VALID_PARTIAL",
    "protocol": "real_g1_flat_core_v1",
    "ledger": str(cfg.trials.resolve()),
    "provenance": provenance,
    "raw_row_count": len(rows),
    "valid_trial_count": len(valid),
    "invalid_initialization_count": len(rows) - len(valid),
    "pose_counts": dict(counts),
    "overall_success": _wilson(overall_success, len(valid)),
    "fixed_pose_macro_success": sum(fixed_rates) / len(fixed_rates),
    "fixed_pose_worst_success": min(fixed_rates),
    "by_pose": by_pose,
    "safety": safety,
    "failure_counts": dict(sorted(failures.items())),
  }


def _markdown(result: dict[str, Any]) -> str:
  lines = [
    "# Real-G1 recovery trial analysis",
    "",
    f"Status: **{result['status']}**",
    "",
    f"Valid trials: {result['valid_trial_count']}",
    "",
    "| Pose | Success | 95% CI | First stand | Secondary fall | Safety abort | Median time |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for pose, metrics in result["by_pose"].items():
    success = metrics["success"]
    first = metrics["first_stand"]
    median = metrics["recovery_time_s"]["median"]
    median_text = "n/a" if median is None else f"{median:.2f} s"
    lines.append(
      f"| {pose} | {success['successes']}/{success['total']} ({success['rate']:.1%}) | "
      f"[{success['ci95_low']:.1%}, {success['ci95_high']:.1%}] | "
      f"{first['rate']:.1%} | {metrics['secondary_fall_rate']:.1%} | "
      f"{metrics['safety_abort_rate']:.1%} | {median_text} |"
    )
  lines.extend(
    (
      "",
      f"Fixed-pose macro success: {result['fixed_pose_macro_success']:.1%}",
      "",
      f"Fixed-pose worst success: {result['fixed_pose_worst_success']:.1%}",
      "",
      "Safety values are distributions from logger schema 2; `tau_cmd_est` is a PD-command estimate, not a calibrated torque-sensor measurement.",
      "",
    )
  )
  return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def main(cfg: HardwareAnalysisCfg) -> None:
  result = analyze(cfg)
  markdown = cfg.output_markdown or cfg.output_json.with_suffix(".md")
  _atomic_write(cfg.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
  _atomic_write(markdown, _markdown(result))
  print(
    f"{result['status']}: {result['overall_success']['successes']}/"
    f"{result['overall_success']['total']} successful trials"
  )


if __name__ == "__main__":
  main(tyro.cli(HardwareAnalysisCfg))
