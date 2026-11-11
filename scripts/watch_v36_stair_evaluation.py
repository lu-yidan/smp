"""Evaluate a finite training job's baseline and saved checkpoints on one GPU."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def significantly_regressed(current, best):
  """Require a large drop in both recovery and local recovery, not small noise."""
  return (
    current["sr3"] < 0.75 * best["sr3"]
    and current["local_sr3"] < 0.75 * best["local_sr3"]
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--training-pid", type=int, required=True)
  parser.add_argument("--training-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--task", default="Smp-Getup-Stairs-V36-93D-G1")
  parser.add_argument("--stop-on-regression", action="store_true")
  parser.add_argument(
    "--baseline",
    type=Path,
    default=Path("artifacts/v36_projected/model_102000_zero_velocity_93d.pt"),
  )
  parser.add_argument(
    "--checkpoints", type=int, nargs="+", default=[*range(1000, 12000, 1000), 11999]
  )
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = args.output_dir / "summary.jsonl"
  best = None
  regression_streak = 0
  jobs = [
    (
      "parent_baseline",
      args.baseline,
    )
  ]
  jobs += [
    (f"update_{i:05d}", args.training_dir / f"model_{i}.pt") for i in args.checkpoints
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
      args.task,
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
      if all("terrain_retained_sr3_success" in row for row in rows):
        summary["terrain_retained_sr3"] = (
          sum(row["terrain_retained_sr3_success"] for row in rows) / summary["n"]
        )
    else:
      summary["status"] = "evaluation_failed_see_log"
    with summary_path.open("a") as out:
      out.write(json.dumps(summary) + "\n")
    print("EVAL_RESULT", json.dumps(summary), flush=True)
    if result.returncode == 0:
      if best is None or summary["sr3"] > best["sr3"]:
        best = summary
        regression_streak = 0
      elif significantly_regressed(summary, best):
        regression_streak += 1
      else:
        regression_streak = 0
      if args.stop_on_regression and regression_streak >= 2:
        cmd_path = Path(f"/proc/{args.training_pid}/cmdline")
        cmdline = cmd_path.read_bytes() if cmd_path.exists() else b""
        matches = (
          b"scripts/train_v36_93d.py" in cmdline
          and str(args.training_dir).encode() in cmdline
        )
        guard = {
          "reason": "Two consecutive checkpoints lost >25% relative SR3 and local SR3 versus best-SR3 checkpoint",
          "best": best,
          "current": summary,
          "pid": args.training_pid,
          "signal_sent": False,
        }
        guard_path = args.output_dir / "regression_guard.json"
        guard_path.write_text(json.dumps(guard, indent=2))
        if matches:
          os.kill(args.training_pid, signal.SIGINT)
          guard["signal_sent"] = True
          guard_path.write_text(json.dumps(guard, indent=2))
        print("REGRESSION_GUARD", json.dumps(guard), flush=True)
        return
    else:
      regression_streak = 0


if __name__ == "__main__":
  main()
