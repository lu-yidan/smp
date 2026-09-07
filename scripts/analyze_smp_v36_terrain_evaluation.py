"""Audit and aggregate the frozen V36 final terrain diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


_PROTOCOL_SHA256 = "3db053546142fbb223126d65c8d14084fda7ebdb120d77b6b7b95fcda1620b4e"
_PER_ENV_KEYS = (
  "strict_success",
  "strict_success_step",
  "failure_reason",
  "first_head_height_step",
  "first_upright_step",
  "first_linear_speed_settled_step",
  "first_angular_speed_settled_step",
  "first_strict_candidate_step",
  "longest_stable_stand_hold_steps",
  "secondary_fall",
  "terrain_exit",
  "invalid_dynamics",
  "terrain_reset_contact_valid",
  "terrain_reset_refinement_steps",
  "foot_slip_mean_m_s",
  "max_planar_displacement_m",
  "max_stance_width_m",
  "action_first_difference_mean_l2",
  "action_second_difference_mean_l2",
  "max_joint_speed_rad_s",
  "max_torque_nm",
  "max_power_w",
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> Any:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  with path.open(encoding="utf-8") as stream:
    return [json.loads(line) for line in stream if line.strip()]


def _mean(values: list[float]) -> float:
  return statistics.fmean(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  index = q * (len(ordered) - 1)
  lower = int(index)
  upper = min(lower + 1, len(ordered) - 1)
  alpha = index - lower
  return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _audit_policy(
  policy: dict[str, Any], matrix: dict[str, Any], result_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  arm = policy["arm"]
  seed = int(policy["seed"])
  path = result_dir / f"{arm}_{seed}.jsonl"
  rows = _load_jsonl(path)
  expected_cells = int(matrix["cells_per_policy"])
  if len(rows) != expected_cells:
    raise RuntimeError(f"{path}: expected {expected_cells} rows, found {len(rows)}")

  expected_terrains = set(matrix["terrain_types"])
  expected_poses = set(matrix["reset_poses"])
  cells: set[tuple[str, str]] = set()
  all_values: dict[str, list[float]] = {
    "foot_slip_mean_m_s": [],
    "max_planar_displacement_m": [],
    "max_stance_width_m": [],
    "action_first_difference_mean_l2": [],
    "action_second_difference_mean_l2": [],
    "max_joint_speed_rad_s": [],
    "max_torque_nm": [],
    "max_power_w": [],
  }
  success_by_pose: dict[str, list[float]] = {pose: [] for pose in expected_poses}
  success_by_terrain: dict[str, list[float]] = {
    terrain: [] for terrain in expected_terrains
  }
  failure_counts: Counter[str] = Counter()
  successes = 0
  secondary_falls = 0
  total_envs = 0
  recovery_times: list[float] = []

  for row in rows:
    if row.get("schema_version") != 2:
      raise RuntimeError(f"{path}: every row must be schema version 2")
    if row.get("checkpoint_sha256") != policy["checkpoint_sha256"]:
      raise RuntimeError(f"{path}: checkpoint hash drift")
    if row.get("protocol_sha256") != _PROTOCOL_SHA256:
      raise RuntimeError(f"{path}: protocol hash drift")
    if row.get("task") != policy["task"]:
      raise RuntimeError(f"{path}: task drift")
    if row.get("seed") != matrix["eval_seed"]:
      raise RuntimeError(f"{path}: evaluation seed drift")
    if row.get("num_envs") != matrix["num_envs_per_cell"]:
      raise RuntimeError(f"{path}: environment count drift")
    if row.get("steps") != matrix["steps_per_cell"]:
      raise RuntimeError(f"{path}: rollout length drift")
    if row.get("terrain_level") != matrix["terrain_level"]:
      raise RuntimeError(f"{path}: terrain level drift")
    terrain = row.get("terrain_type")
    pose = row.get("reset_mode")
    if terrain not in expected_terrains or pose not in expected_poses:
      raise RuntimeError(f"{path}: unregistered terrain/pose cell")
    cell = (terrain, pose)
    if cell in cells:
      raise RuntimeError(f"{path}: duplicate cell {cell}")
    cells.add(cell)

    per_env = row.get("per_env")
    if not isinstance(per_env, dict):
      raise RuntimeError(f"{path}: per_env telemetry missing")
    num_envs = int(row["num_envs"])
    for key in _PER_ENV_KEYS:
      if not isinstance(per_env.get(key), list) or len(per_env[key]) != num_envs:
        raise RuntimeError(f"{path}: invalid per_env array {key}")
    reasons = Counter(per_env["failure_reason"])
    diagnosis = row.get("strict_failure_diagnosis", {})
    if diagnosis.get("schema_version") != 1:
      raise RuntimeError(f"{path}: failure diagnosis schema drift")
    if reasons != Counter(diagnosis.get("reason_counts", {})):
      raise RuntimeError(f"{path}: failure reason counts disagree")
    cell_successes = sum(bool(value) for value in per_env["strict_success"])
    if reasons["success"] != cell_successes or cell_successes != row["success"]:
      raise RuntimeError(f"{path}: strict success accounting disagrees")
    if not all(bool(value) for value in per_env["terrain_reset_contact_valid"]):
      raise RuntimeError(f"{path}: invalid reset contact reached policy evaluation")

    successes += cell_successes
    secondary_falls += sum(bool(value) for value in per_env["secondary_fall"])
    total_envs += num_envs
    failure_counts.update(reasons)
    rate = float(row["success_rate"])
    success_by_pose[pose].append(rate)
    success_by_terrain[terrain].append(rate)
    valid_recovery = [
      float(step) * 0.02
      for step in per_env["strict_success_step"]
      if int(step) >= 0
    ]
    recovery_times.extend(valid_recovery)
    for key in all_values:
      all_values[key].extend(float(value) for value in per_env[key])

  expected_pairs = {
    (terrain, pose) for terrain in expected_terrains for pose in expected_poses
  }
  if cells != expected_pairs:
    raise RuntimeError(f"{path}: matrix is incomplete")

  pose_rates = {pose: _mean(values) for pose, values in success_by_pose.items()}
  terrain_rates = {
    terrain: _mean(values) for terrain, values in success_by_terrain.items()
  }
  row_rates = [float(row["success_rate"]) for row in rows]
  summary = {
    "arm": arm,
    "policy_seed": seed,
    "wandb_run_id": policy["wandb_run_id"],
    "checkpoint_sha256": policy["checkpoint_sha256"],
    "cells": len(rows),
    "rollouts": total_envs,
    "strict_successes": successes,
    "macro_success_rate": _mean(row_rates),
    "worst_cell_success_rate": min(row_rates),
    "success_rate_by_pose": pose_rates,
    "worst_pose_success_rate": min(pose_rates.values()),
    "success_rate_by_terrain": terrain_rates,
    "worst_terrain_success_rate": min(terrain_rates.values()),
    "recovery_time_median_s": statistics.median(recovery_times)
    if recovery_times
    else -1.0,
    "recovery_time_p90_s": _quantile(recovery_times, 0.90)
    if recovery_times
    else -1.0,
    "secondary_fall_rate": secondary_falls / total_envs,
    "failure_reason_counts": dict(sorted(failure_counts.items())),
    "invalid_dynamics_rate": failure_counts["invalid_dynamics"] / total_envs,
    "terrain_exit_rate": failure_counts["terrain_exit"] / total_envs,
    "terrain_reset_contact_valid_rate": 1.0,
  }
  for key, values in all_values.items():
    summary[f"{key}_mean"] = _mean(values)
    summary[f"{key}_p95"] = _quantile(values, 0.95)
  return summary, rows


def _paired_effects(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
  lookup = {(row["arm"], row["policy_seed"]): row for row in summaries}
  effects = []
  for seed in sorted({int(row["policy_seed"]) for row in summaries}):
    for left, right, label in (
      ("S", "M", "S_minus_M"),
      ("SD", "MD", "SD_minus_MD"),
      ("MD", "M", "MD_minus_M"),
      ("SD", "S", "SD_minus_S"),
    ):
      a = lookup[(left, seed)]
      b = lookup[(right, seed)]
      effects.append(
        {
          "comparison": label,
          "policy_seed": seed,
          "macro_success_delta": a["macro_success_rate"]
          - b["macro_success_rate"],
          "worst_pose_success_delta": a["worst_pose_success_rate"]
          - b["worst_pose_success_rate"],
          "max_torque_mean_delta_nm": a["max_torque_nm_mean"]
          - b["max_torque_nm_mean"],
          "max_power_mean_delta_w": a["max_power_w_mean"]
          - b["max_power_w_mean"],
          "action_second_difference_mean_delta": a[
            "action_second_difference_mean_l2_mean"
          ]
          - b["action_second_difference_mean_l2_mean"],
        }
      )
  return effects


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--protocol",
    type=Path,
    default=Path("docs/v36_93d_safe_terrain_evaluation_v1.json"),
  )
  parser.add_argument(
    "--result-dir",
    type=Path,
    default=Path("run_control/v36_93d_safe_terrain_eval_v1/results"),
  )
  parser.add_argument(
    "--output-json",
    type=Path,
    default=Path("run_control/v36_93d_safe_terrain_eval_v1/analysis.json"),
  )
  parser.add_argument(
    "--output-markdown",
    type=Path,
    default=Path("run_control/v36_93d_safe_terrain_eval_v1/analysis.md"),
  )
  args = parser.parse_args()

  if _sha256(args.protocol) != _PROTOCOL_SHA256:
    raise RuntimeError("V36 evaluation protocol hash drift")
  protocol = _load_json(args.protocol)
  matrix = protocol["matrix"]
  summaries = []
  for policy in protocol["policies"]:
    summary, _ = _audit_policy(policy, matrix, args.result_dir)
    summaries.append(summary)
  if sum(int(row["cells"]) for row in summaries) != matrix["total_cells"]:
    raise RuntimeError("V36 evaluation total-cell count is incomplete")

  result = {
    "schema_version": 1,
    "status": "V36_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_PROMOTION",
    "protocol_sha256": _PROTOCOL_SHA256,
    "policies": summaries,
    "paired_effects": _paired_effects(summaries),
    "claim_boundary": protocol["claim_boundary"],
  }
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )

  lines = [
    "# V36 final terrain diagnostic",
    "",
    "This is engineering evidence only; it is not a promotion or hardware "
    "authorization.",
    "",
    "| arm | seed | macro | worst pose | worst terrain | torque mean | "
    "power mean | action d2 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in summaries:
    lines.append(
      "| {arm} | {policy_seed} | {macro_success_rate:.3f} | "
      "{worst_pose_success_rate:.3f} | {worst_terrain_success_rate:.3f} | "
      "{max_torque_nm_mean:.1f} | {max_power_w_mean:.1f} | "
      "{action_second_difference_mean_l2_mean:.3f} |".format(**row)
    )
  args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
  main()
