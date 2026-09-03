"""Audit and summarize the matched V34 96D-versus-93D evaluation matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_EXPECTED_PROTOCOL_SHA256 = (
  "1f212abf7e627ac031fcca6637e69b5f43c2c94e6f703bfbc8de6e676734f52e"
)
_EXPECTED_PLAN_ID = "6ea8233a7f1dab9d0439979696caa6f55dc84edbec62ccfaad3f936d55473c16"
_EXPECTED_EVAL_COMMIT = "f02c0718295267c05831f86911ed93bbf8d6400f"


@dataclass(frozen=True)
class AnalysisCfg:
  control_dir: Path = Path("run_control/v34_93d_control/evaluation")
  output_json: Path = Path("run_control/v34_93d_control/evaluation/aggregate.json")
  output_markdown: Path = Path("run_control/v34_93d_control/evaluation/analysis.md")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _atomic_text(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(text)
  temporary.replace(path)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
  return sum(float(row[key]) for row in rows) / len(rows)


def analyze(cfg: AnalysisCfg) -> dict[str, Any]:
  control = cfg.control_dir.resolve()
  plan = _json(control / "immutable_plan.json")
  marker = _json(control / "active_evaluation.json")
  if (
    plan.get("plan_id") != _EXPECTED_PLAN_ID
    or marker.get("plan_id") != _EXPECTED_PLAN_ID
    or plan.get("protocol_sha256") != _EXPECTED_PROTOCOL_SHA256
    or marker.get("protocol_sha256") != _EXPECTED_PROTOCOL_SHA256
    or plan.get("code_commit") != _EXPECTED_EVAL_COMMIT
    or marker.get("code_commit") != _EXPECTED_EVAL_COMMIT
    or plan.get("total_cells") != 14
    or _sha256(Path(plan["protocol"])) != _EXPECTED_PROTOCOL_SHA256
  ):
    raise RuntimeError("V34_93D_EVAL_ALERT: plan, marker, or protocol drifted")
  rows: list[dict[str, Any]] = []
  seen: set[str] = set()
  for job in plan["jobs"]:
    result_path = Path(job["result"])
    log_path = Path(job["log"])
    if not result_path.is_file() or not log_path.is_file():
      raise RuntimeError(f"V34_93D_EVAL_ALERT: incomplete cell {job['cell_id']}")
    row = _json(result_path)
    if (
      row.get("status") != "CELL_COMPLETE"
      or row.get("cell_id") != job["cell_id"]
      or row.get("cell_id") in seen
      or row.get("plan_id") != _EXPECTED_PLAN_ID
      or row.get("protocol_sha256") != _EXPECTED_PROTOCOL_SHA256
      or row.get("checkpoint_sha256") != job["checkpoint_sha256"]
      or row.get("code_commit") != _EXPECTED_EVAL_COMMIT
      or row.get("num_envs") != 512
      or row.get("active") != 512
      or row.get("steps") != 1000
      or row.get("seed") != 20261710
      or row.get("reset_pose") != job["reset_pose"]
      or row.get("log_sha256") != _sha256(log_path)
    ):
      raise RuntimeError(f"V34_93D_EVAL_ALERT: invalid cell {job['cell_id']}")
    seen.add(row["cell_id"])
    rows.append(row)
  expected_workers = {row["physical_device"] for row in marker["workers"]}
  for device in expected_workers:
    worker = _json(control / "workers" / f"gpu{device}.json")
    if worker.get("status") != "COMPLETE" or (
      worker.get("completed_jobs") != worker.get("total_jobs")
    ):
      raise RuntimeError(f"V34_93D_EVAL_ALERT: worker {device} is incomplete")

  keys = [("v34_96d", 98000)] + [
    ("v34_93d", gate) for gate in (0, 1000, 3000, 6000, 9000, 11999)
  ]
  summaries: list[dict[str, Any]] = []
  for label, gate in keys:
    group = [row for row in rows if row["label"] == label and row["gate"] == gate]
    group.sort(key=lambda row: ("prone", "supine").index(row["reset_pose"]))
    if [row["reset_pose"] for row in group] != ["prone", "supine"]:
      raise RuntimeError(f"V34_93D_EVAL_ALERT: pose strata incomplete for {label} {gate}")
    success = [float(row["escape_and_stand_rate"]) for row in group]
    summaries.append(
      {
        "label": label,
        "gate": gate,
        "prone_success": success[0],
        "supine_success": success[1],
        "macro_success": sum(success) / 2.0,
        "worst_pose_success": min(success),
        "invalid_macro": _mean(group, "invalid_rate"),
        "setup_invalid_macro": _mean(group, "setup_invalid_rate"),
        "stable_foot_separation_macro_m": _mean(
          group, "stable_foot_separation_median_m"
        ),
        "wide_stance_rate_macro": _mean(group, "wide_stance_rate_at_stable"),
        "joint_speed_p95_worst_rad_s": max(
          float(row["max_joint_speed_p95_rad_s"]) for row in group
        ),
        "mean_torque_macro_nm": _mean(group, "max_torque_mean_nm"),
        "mean_power_macro_w": _mean(group, "max_power_mean_w"),
        "escape_time_macro_s": _mean(group, "escape_time_median_s"),
        "stable_stand_time_macro_s": _mean(group, "stable_stand_time_median_s"),
      }
    )
  baseline = summaries[0]
  for row in summaries[1:]:
    row["delta_vs_v34_96d"] = {
      "macro_success_pp": 100.0
      * (row["macro_success"] - baseline["macro_success"]),
      "worst_pose_success_pp": 100.0
      * (row["worst_pose_success"] - baseline["worst_pose_success"]),
      "invalid_macro_pp": 100.0
      * (row["invalid_macro"] - baseline["invalid_macro"]),
      "joint_speed_p95_worst_rad_s": row["joint_speed_p95_worst_rad_s"]
      - baseline["joint_speed_p95_worst_rad_s"],
      "mean_torque_macro_nm": row["mean_torque_macro_nm"]
      - baseline["mean_torque_macro_nm"],
      "mean_power_macro_w": row["mean_power_macro_w"]
      - baseline["mean_power_macro_w"],
    }
  best_recovery = max(
    summaries[1:], key=lambda row: (row["worst_pose_success"], row["macro_success"])
  )
  aggregate = {
    "schema_version": 1,
    "status": "ANALYSIS_COMPLETE_SIMULATION_REVIEW_ONLY",
    "automatic_action": "NO_HARDWARE_AUTHORIZATION",
    "plan_id": _EXPECTED_PLAN_ID,
    "protocol_sha256": _EXPECTED_PROTOCOL_SHA256,
    "total_cells": len(rows),
    "total_rollouts": sum(int(row["active"]) for row in rows),
    "summaries": summaries,
    "descriptive_best_recovery_gate": best_recovery["gate"],
    "descriptive_recommendation": (
      "REVIEW_GATE_6000_IN_MUJOCO; torque and power remain above the 96D control, "
      "so this is not a hardware recommendation"
    ),
    "limitations": [
      "one historical 96D policy and one projected 93D fine-tuning seed",
      "aggregate cell outputs do not retain per-environment paired outcomes",
      "no A14 action envelope in this comparison",
      "simulation result only; no hardware authorization",
    ],
  }
  output_json = cfg.output_json.resolve()
  _atomic_text(output_json, json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
  lines = [
    "# V34 96D versus projected 93D matched evaluation",
    "",
    "Status: `ANALYSIS_COMPLETE_SIMULATION_REVIEW_ONLY`.",
    "",
    "| policy | gate | prone | supine | macro | worst | invalid | speed p95 worst | torque | power |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in summaries:
    lines.append(
      "| {label} | {gate} | {prone_success:.2%} | {supine_success:.2%} | "
      "{macro_success:.2%} | {worst_pose_success:.2%} | {invalid_macro:.2%} | "
      "{joint_speed_p95_worst_rad_s:.2f} rad/s | {mean_torque_macro_nm:.2f} Nm | "
      "{mean_power_macro_w:.2f} W |".format(**row)
    )
  lines.extend(
    [
      "",
      "Gate 6000 is the descriptive recovery candidate, not a hardware candidate. "
      "It improves both pose success and invalidity, but its mean torque and power "
      "are above the 96D control.",
      "",
    ]
  )
  _atomic_text(cfg.output_markdown.resolve(), "\n".join(lines))
  return aggregate


def main(cfg: AnalysisCfg) -> None:
  result = analyze(cfg)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(AnalysisCfg))
