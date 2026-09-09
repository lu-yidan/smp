"""Audit and summarize the frozen V37 support-transition evaluation."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from run_smp_v37_support_transition_eval import _read_rows


_METRICS = (
  "foot_slip_mean_m_s",
  "max_planar_displacement_m",
  "max_stance_width_m",
  "stance_width_mean_m",
  "bilateral_foot_support_fraction",
  "minimum_foot_load_share_mean",
  "leg_asymmetry_mean_rad",
  "trap_dwell_steps",
  "action_first_difference_mean_l2",
  "action_second_difference_mean_l2",
  "max_joint_speed_rad_s",
  "max_torque_nm",
  "max_power_w",
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
  all_values = {
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
    "wandb_run_id": job.get("wandb_run_id"),
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
    ) / total,
    "finite_action_rate": sum(
      bool(value) for row in rows for value in row["per_env"]["finite_action"]
    ) / total,
    "invalid_dynamics_rate": failure_counts.get("invalid_dynamics", 0) / total,
    "failure_reason_counts": dict(sorted(failure_counts.items())),
  }
  for key, values in all_values.items():
    summary[key + "_mean"] = _mean(values)
    summary[key + "_p95"] = _quantile(values, 0.95)
  return summary


def _paired(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
  lookup = {
    (row["arm"], int(row["policy_seed"]), int(row["gate"])): row
    for row in summaries
  }
  effects = []
  for seed in (20262001, 20262002, 20262003):
    for gate in (0, 500, 1000, 2000, 3500, 5000, 5999):
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
          "trap_success_delta": (
            b["success_rate_by_pose"]["synthetic_seated_trap"]
            - a["success_rate_by_pose"]["synthetic_seated_trap"]
          ),
          "bilateral_support_delta": (
            b["bilateral_foot_support_fraction_mean"]
            - a["bilateral_foot_support_fraction_mean"]
          ),
          "leg_asymmetry_delta_rad": (
            b["leg_asymmetry_mean_rad_mean"] - a["leg_asymmetry_mean_rad_mean"]
          ),
          "action_d2_delta": (
            b["action_second_difference_mean_l2_mean"]
            - a["action_second_difference_mean_l2_mean"]
          ),
          "peak_torque_delta_nm": (
            b["max_torque_nm_mean"] - a["max_torque_nm_mean"]
          ),
          "peak_power_delta_w": b["max_power_w_mean"] - a["max_power_w_mean"],
        }
      )
  return effects


def _bootstrap_final(effects: list[dict[str, Any]]) -> dict[str, Any]:
  final = [row for row in effects if row["gate"] == 5999]
  rng = random.Random(20262010)
  keys = (
    "macro_success_delta",
    "worst_pose_success_delta",
    "trap_success_delta",
    "bilateral_support_delta",
    "leg_asymmetry_delta_rad",
    "action_d2_delta",
    "peak_torque_delta_nm",
    "peak_power_delta_w",
  )
  result: dict[str, Any] = {
    "sampling_unit": "matched_policy_seed",
    "policy_seeds": [row["policy_seed"] for row in final],
    "repetitions": 20000,
    "seed": 20262010,
  }
  for key in keys:
    values = [float(row[key]) for row in final]
    draws = [
      _mean([values[rng.randrange(len(values))] for _ in values])
      for _ in range(20000)
    ]
    result[key] = {
      "mean": _mean(values),
      "ci95": [_quantile(draws, 0.025), _quantile(draws, 0.975)],
    }
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--control-dir",
    type=Path,
    default=Path("run_control/v37_93d_support_transition_v1/evaluation"),
  )
  args = parser.parse_args()
  plan = json.loads((args.control_dir / "immutable_plan.json").read_text())
  summaries = [_summarize(job) for job in plan["jobs"]]
  if len(summaries) != 42 or sum(row["cells"] for row in summaries) != 210:
    raise RuntimeError("V37_EVAL_ALERT: expected 42 matrices and 210 cells")
  effects = _paired(summaries)
  result = {
    "schema_version": 1,
    "status": "V37_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_HARDWARE_AUTHORIZATION",
    "plan_id": plan["plan_id"],
    "protocol_sha256": plan["protocol_sha256"],
    "matrices": len(summaries),
    "cells": 210,
    "rollouts": 210 * 256,
    "policies": summaries,
    "paired_effects": effects,
    "final_paired_bootstrap": _bootstrap_final(effects),
    "claim_boundary": plan["claim_boundary"],
  }
  output = args.control_dir / "analysis.json"
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  final = [row for row in summaries if row["gate"] == 5999]
  lines = [
    "# V37 support-transition diagnostic",
    "",
    "Engineering diagnostic only; this is not RAL evidence or hardware authorization.",
    "",
    "| arm | seed | macro | worst | trap | bilateral | stance | leg asym | "
    "action d2 | torque | power |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in final:
    lines.append(
      "| {arm} | {policy_seed} | {macro_success_rate:.3f} | "
      "{worst_pose_success_rate:.3f} | {trap:.3f} | "
      "{bilateral_foot_support_fraction_mean:.3f} | "
      "{stance_width_mean_m_mean:.3f} | {leg_asymmetry_mean_rad_mean:.3f} | "
      "{action_second_difference_mean_l2_mean:.3f} | {max_torque_nm_mean:.1f} | "
      "{max_power_w_mean:.1f} |".format(
        trap=row["success_rate_by_pose"]["synthetic_seated_trap"], **row
      )
    )
  (args.control_dir / "analysis.md").write_text("\n".join(lines) + "\n")
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
