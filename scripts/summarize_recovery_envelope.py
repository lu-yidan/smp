"""Summarize trial-level constrained-recovery results into an envelope table.

Input is deliberately simulator-agnostic so the same analysis is used for
MuJoCo and hardware trials. Required columns are documented in
``docs/constrained_recovery.md`` and checked before any aggregation.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REQUIRED_COLUMNS = {
  "pose_bin",
  "terrain",
  "constraint_body",
  "load_n",
  "duration_s",
  "success",
  "recovery_time_s",
  "stall_time_s",
  "max_joint_speed",
  "max_joint_torque",
  "max_joint_power",
}


def _mean(values: list[float]) -> float:
  finite = [value for value in values if math.isfinite(value)]
  return sum(finite) / len(finite) if finite else math.nan


def _parse_bool(value: str) -> float:
  normalized = value.strip().lower()
  if normalized in {"1", "true", "yes", "success"}:
    return 1.0
  if normalized in {"0", "false", "no", "failure"}:
    return 0.0
  raise ValueError(f"invalid success value: {value!r}")


def summarize(input_path: Path, output_path: Path) -> None:
  with input_path.open(newline="", encoding="utf-8") as input_file:
    reader = csv.DictReader(input_file)
    missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
    if missing:
      raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in reader:
      key = (row["pose_bin"], row["terrain"], row["constraint_body"])
      groups[key].append(row)

  fieldnames = [
    "pose_bin",
    "terrain",
    "constraint_body",
    "num_trials",
    "success_rate",
    "mean_load_n",
    "mean_duration_s",
    "mean_recovery_time_success_s",
    "mean_stall_time_s",
    "mean_max_joint_speed",
    "mean_max_joint_torque",
    "mean_max_joint_power",
  ]
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    for key in sorted(groups):
      rows = groups[key]
      success = [_parse_bool(row["success"]) for row in rows]
      successful_times = [
        float(row["recovery_time_s"])
        for row, passed in zip(rows, success, strict=True)
        if passed > 0.5
      ]
      writer.writerow(
        {
          "pose_bin": key[0],
          "terrain": key[1],
          "constraint_body": key[2],
          "num_trials": len(rows),
          "success_rate": f"{_mean(success):.6f}",
          "mean_load_n": f"{_mean([float(r['load_n']) for r in rows]):.6f}",
          "mean_duration_s": f"{_mean([float(r['duration_s']) for r in rows]):.6f}",
          "mean_recovery_time_success_s": f"{_mean(successful_times):.6f}",
          "mean_stall_time_s": f"{_mean([float(r['stall_time_s']) for r in rows]):.6f}",
          "mean_max_joint_speed": f"{_mean([float(r['max_joint_speed']) for r in rows]):.6f}",
          "mean_max_joint_torque": f"{_mean([float(r['max_joint_torque']) for r in rows]):.6f}",
          "mean_max_joint_power": f"{_mean([float(r['max_joint_power']) for r in rows]):.6f}",
        }
      )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_csv", type=Path, help="trial-level result CSV")
  parser.add_argument("output_csv", type=Path, help="stratified summary CSV")
  args = parser.parse_args()
  summarize(args.input_csv, args.output_csv)


if __name__ == "__main__":
  main()
