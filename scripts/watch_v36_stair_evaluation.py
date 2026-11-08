"""Evaluate a finite training job's baseline and saved checkpoints on one GPU."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--training-pid", type=int, required=True)
  parser.add_argument("--training-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = args.output_dir / "summary.jsonl"
  jobs = [
    (
      "projected_baseline",
      Path("artifacts/v36_projected/model_102000_zero_velocity_93d.pt"),
    )
  ]
  jobs += [
    (f"update_{i:05d}", args.training_dir / f"model_{i}.pt")
    for i in [*range(1000, 12000, 1000), 11999]
  ]
  for name, checkpoint in jobs:
    result_path = args.output_dir / f"{name}.jsonl"
    if result_path.exists():
      raise FileExistsError(result_path)
    while not (checkpoint.exists() and time.time() - checkpoint.stat().st_mtime > 15):
      cmd_path = Path(f"/proc/{args.training_pid}/cmdline")
      alive = (
        cmd_path.exists() and str(args.training_dir).encode() in cmd_path.read_bytes()
      )
      if not alive:
        if checkpoint.exists():
          time.sleep(20)
          break
        print("TRAINING_ENDED_BEFORE_CHECKPOINT", checkpoint, flush=True)
        return
      time.sleep(30)
    cmd = [
      sys.executable,
      "-u",
      "scripts/evaluate_v36_93d.py",
      "--task",
      "Smp-Getup-Stairs-V36-93D-G1",
      "--checkpoint",
      str(checkpoint),
      "--terrain-types",
      "stairs",
      "--levels",
      "0",
      "1",
      "2",
      "3",
      "--reset-modes",
      "prone",
      "supine",
      "left_side",
      "right_side",
      "--num-envs",
      str(args.num_envs),
      "--output",
      str(result_path),
    ]
    print("EVALUATING", name, checkpoint, flush=True)
    with (args.output_dir / f"{name}.log").open("x") as log:
      result = subprocess.run(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
      )
    summary = {
      "name": name,
      "checkpoint": str(checkpoint),
      "returncode": result.returncode,
    }
    if result.returncode == 0:
      rows = [json.loads(x) for x in result_path.read_text().splitlines()]
      assert len(rows) == 16, len(rows)
      summary.update(
        n=sum(row["num_envs"] for row in rows),
        sr3=sum(row["sr3_success"] for row in rows)
        / sum(row["num_envs"] for row in rows),
        local_sr3=sum(row["local_sr3"] * row["num_envs"] for row in rows)
        / sum(row["num_envs"] for row in rows),
        per_case=str(result_path),
      )
    else:
      summary["status"] = "evaluation_failed_see_log"
    with summary_path.open("a") as out:
      out.write(json.dumps(summary) + "\n")
    print("EVAL_RESULT", json.dumps(summary), flush=True)


if __name__ == "__main__":
  main()
