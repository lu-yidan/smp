"""Audit and summarize the frozen V39 support-exit evaluation."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from run_smp_v39_support_exit_eval import _read_rows

_GATES = (0, 2000, 4000, 8000, 11999)
_SEEDS = (20262201, 20262202, 20262203)
_METRICS = (
  "foot_slip",
  "root_drift",
  "max_stance_width_m",
  "stance_width_mean_m",
  "bilateral_support_fraction",
  "minimum_foot_load_share_mean",
  "hand_support_fraction",
  "knee_support_fraction",
  "leg_asymmetry_mean_rad",
  "trap_dwell_steps",
  "action_first_difference",
  "action_second_difference",
  "peak_joint_speed",
  "peak_joint_torque",
  "peak_joint_power",
  "max_head_z",
  "max_upright",
)


def _mean(values: list[float]) -> float:
  return statistics.fmean(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
  ordered = sorted(values)
  if not ordered:
    return 0.0
  point = q * (len(ordered) - 1)
  low = int(point)
  high = min(low + 1, len(ordered) - 1)
  alpha = point - low
  return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


def _summarize(job: dict[str, Any]) -> dict[str, Any]:
  rows = _read_rows(job)
  pose = {row["reset_mode"]: float(row["success_rate"]) for row in rows}
  values = {
    key: [float(value) for row in rows for value in row["per_env"][key]]
    for key in _METRICS
  }
  recovery = [
    float(step) * 0.02
    for row in rows
    for step in row["per_env"]["strict_success_step"]
    if int(step) >= 0
  ]
  total = sum(int(row["num_envs"]) for row in rows)
  failure_counts: dict[str, int] = {}
  for row in rows:
    for reason, count in row["strict_failure_diagnosis"]["reason_counts"].items():
      failure_counts[reason] = failure_counts.get(reason, 0) + int(count)
  summary = {
    "arm": job["arm"],
    "policy_seed": job["policy_seed"],
    "gate": job["gate"],
    "wandb_run_id": job["wandb_run_id"],
    "checkpoint": job["checkpoint"],
    "checkpoint_sha256": job["checkpoint_sha256"],
    "cells": len(rows),
    "rollouts": total,
    "strict_successes": failure_counts.get("success", 0),
    "macro_success_rate": _mean(list(pose.values())),
    "worst_pose_success_rate": min(pose.values()),
    "success_rate_by_pose": pose,
    "recovery_time_median_s": statistics.median(recovery) if recovery else -1.0,
    "recovery_time_p90_s": _quantile(recovery, 0.90) if recovery else -1.0,
    "secondary_fall_rate": sum(
      bool(value) for row in rows for value in row["per_env"]["secondary_fall"]
    )
    / total,
    "finite_action_rate": sum(
      bool(value) for row in rows for value in row["per_env"]["finite_action"]
    )
    / total,
    "invalid_dynamics_rate": failure_counts.get("invalid_dynamics", 0) / total,
    "failure_reason_counts": dict(sorted(failure_counts.items())),
  }
  for key, metric_values in values.items():
    summary[key + "_mean"] = _mean(metric_values)
    summary[key + "_p95"] = _quantile(metric_values, 0.95)
  return summary


def _paired(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
  lookup = {
    (row["arm"], int(row["policy_seed"]), int(row["gate"])): row for row in summaries
  }
  effects = []
  for seed in _SEEDS:
    for gate in _GATES:
      a = lookup[("A", seed, gate)]
      b = lookup[("B", seed, gate)]
      effects.append(
        {
          "comparison": "B_minus_A",
          "policy_seed": seed,
          "gate": gate,
          "macro_success_delta": b["macro_success_rate"] - a["macro_success_rate"],
          "worst_pose_success_delta": (
            b["worst_pose_success_rate"] - a["worst_pose_success_rate"]
          ),
          "prone_success_delta": (
            b["success_rate_by_pose"]["prone"] - a["success_rate_by_pose"]["prone"]
          ),
          "post_roll_success_delta": (
            b["success_rate_by_pose"]["post_roll_supine_crossed"]
            - a["success_rate_by_pose"]["post_roll_supine_crossed"]
          ),
          "seated_trap_success_delta": (
            b["success_rate_by_pose"]["synthetic_seated_trap"]
            - a["success_rate_by_pose"]["synthetic_seated_trap"]
          ),
          "bilateral_support_delta": (
            b["bilateral_support_fraction_mean"] - a["bilateral_support_fraction_mean"]
          ),
          "hand_support_delta": (
            b["hand_support_fraction_mean"] - a["hand_support_fraction_mean"]
          ),
          "knee_support_delta": (
            b["knee_support_fraction_mean"] - a["knee_support_fraction_mean"]
          ),
          "leg_asymmetry_delta_rad": (
            b["leg_asymmetry_mean_rad_mean"] - a["leg_asymmetry_mean_rad_mean"]
          ),
          "action_d2_delta": (
            b["action_second_difference_mean"] - a["action_second_difference_mean"]
          ),
          "peak_torque_delta_nm": (
            b["peak_joint_torque_mean"] - a["peak_joint_torque_mean"]
          ),
          "peak_power_delta_w": (
            b["peak_joint_power_mean"] - a["peak_joint_power_mean"]
          ),
        }
      )
  return effects


def _bootstrap_final(effects: list[dict[str, Any]]) -> dict[str, Any]:
  final = [row for row in effects if row["gate"] == 11999]
  rng = random.Random(20262210)
  keys = tuple(key for key in final[0] if key.endswith("_delta") or "_delta_" in key)
  result: dict[str, Any] = {
    "sampling_unit": "matched_policy_seed",
    "policy_seeds": [row["policy_seed"] for row in final],
    "repetitions": 20000,
    "seed": 20262210,
  }
  for key in keys:
    metric_values = [float(row[key]) for row in final]
    draws = [
      _mean([metric_values[rng.randrange(len(metric_values))] for _ in metric_values])
      for _ in range(20000)
    ]
    result[key] = {
      "mean": _mean(metric_values),
      "ci95": [_quantile(draws, 0.025), _quantile(draws, 0.975)],
    }
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--control-dir",
    type=Path,
    default=Path("run_control/v39_93d_support_exit_v1/evaluation"),
  )
  args = parser.parse_args()
  plan = json.loads((args.control_dir / "immutable_plan.json").read_text())
  summaries = [_summarize(job) for job in plan["jobs"]]
  if len(summaries) != 30 or sum(row["cells"] for row in summaries) != 180:
    raise RuntimeError("V39_EVAL_ALERT: expected 30 matrices and 180 cells")
  effects = _paired(summaries)
  result = {
    "schema_version": 1,
    "status": "V39_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_HARDWARE_AUTHORIZATION",
    "plan_id": plan["plan_id"],
    "protocol_sha256": plan["protocol_sha256"],
    "matrices": len(summaries),
    "cells": 180,
    "rollouts": 180 * 256,
    "policies": summaries,
    "paired_effects": effects,
    "final_paired_bootstrap": _bootstrap_final(effects),
    "claim_boundary": plan["claim_boundary"],
  }
  (args.control_dir / "analysis.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
  )
  final = [row for row in summaries if row["gate"] == 11999]
  lines = [
    "# V39 support-exit diagnostic",
    "",
    "Engineering diagnostic only; not RAL evidence or hardware authorization.",
    "",
    "| arm | seed | macro | worst | prone | post-roll | seated | bilateral | "
    "hand | knee | action d2 | torque | power |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in final:
    lines.append(
      "| {arm} | {policy_seed} | {macro_success_rate:.3f} | "
      "{worst_pose_success_rate:.3f} | {prone:.3f} | {post_roll:.3f} | "
      "{seated:.3f} | {bilateral_support_fraction_mean:.3f} | "
      "{hand_support_fraction_mean:.3f} | {knee_support_fraction_mean:.3f} | "
      "{action_second_difference_mean:.3f} | {peak_joint_torque_mean:.1f} | "
      "{peak_joint_power_mean:.1f} |".format(
        prone=row["success_rate_by_pose"]["prone"],
        post_roll=row["success_rate_by_pose"]["post_roll_supine_crossed"],
        seated=row["success_rate_by_pose"]["synthetic_seated_trap"],
        **row,
      )
    )
  (args.control_dir / "analysis.md").write_text("\n".join(lines) + "\n")
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
