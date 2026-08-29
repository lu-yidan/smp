"""Create a machine-readable health snapshot without ranking policy quality."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

_ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
_FATAL_PATTERNS = {
  "traceback": "Traceback",
  "cuda_oom": "CUDA out of memory",
  "segmentation_fault": "Segmentation fault",
}
_METRICS = {
  "steps_per_second": "Steps per second",
  "iteration_time_s": "Iteration time",
  "mean_reward": "Mean reward",
  "mean_action_std": "Mean action std",
  "task_score": "Episode_Metrics/task_score",
  "smp_score": "Episode_Metrics/smp_score",
  "max_joint_speed": "Episode_Metrics/max_joint_speed",
  "max_joint_torque": "Episode_Metrics/max_joint_torque",
  "max_joint_power": "Episode_Metrics/max_joint_power",
  "stood_up_termination": "Episode_Termination/stood_up",
}


@dataclass(frozen=True)
class HealthCfg:
  control_dir: Path
  output: Path
  expected_jobs: int = 8
  stale_seconds: float = 300.0
  throughput_fraction_alert: float = 0.50


def _metric(block: str, label: str) -> float | None:
  match = re.search(re.escape(label) + r":\s*" + _NUMBER, block)
  return float(match.group(1)) if match else None


def _inspect_log(path: Path, now: float, stale_seconds: float) -> dict[str, Any]:
  text = path.read_text(errors="replace")
  matches = list(_ITERATION.finditer(text))
  fatal = [name for name, pattern in _FATAL_PATTERNS.items() if pattern in text]
  age = max(0.0, now - path.stat().st_mtime)
  alerts = [f"fatal:{name}" for name in fatal]
  if not matches:
    alerts.append("missing_iteration")
    block = text[-20000:]
    iteration = None
    maximum = None
  else:
    latest = matches[-1]
    block = text[latest.start() :]
    iteration = int(latest.group(1))
    maximum = int(latest.group(2))
  completed = (
    iteration is not None and maximum is not None and iteration >= maximum - 1
  )
  metrics = {name: _metric(block, label) for name, label in _METRICS.items()}
  nonfinite = [
    name for name, value in metrics.items() if value is not None and not math.isfinite(value)
  ]
  alerts.extend(f"nonfinite:{name}" for name in nonfinite)
  for name, label in _METRICS.items():
    if re.search(
      re.escape(label) + r":\s*(?:nan|[-+]?inf(?:inity)?)\b",
      block,
      flags=re.IGNORECASE,
    ):
      alerts.append(f"nonfinite:{name}")
  alerts = list(dict.fromkeys(alerts))
  if age > stale_seconds and not completed:
    alerts.append("stale_log")
  return {
    "log": path.name,
    "iteration": iteration,
    "max_iteration": maximum,
    "completed": completed,
    "progress": iteration / maximum if iteration is not None and maximum else None,
    "mtime_age_s": age,
    "metrics": metrics,
    "alerts": alerts,
  }


def inspect(cfg: HealthCfg, now: float | None = None) -> dict[str, Any]:
  timestamp = time.time() if now is None else now
  logs = sorted(cfg.control_dir.glob("gpu*.log"))
  jobs = [_inspect_log(path, timestamp, cfg.stale_seconds) for path in logs]
  throughputs = [
    float(job["metrics"]["steps_per_second"])
    for job in jobs
    if job["metrics"]["steps_per_second"] is not None
  ]
  median_throughput = statistics.median(throughputs) if throughputs else None
  if median_throughput and median_throughput > 0.0:
    for job in jobs:
      throughput = job["metrics"]["steps_per_second"]
      if not job["completed"] and throughput is not None and throughput < (
        cfg.throughput_fraction_alert * median_throughput
      ):
        job["alerts"].append("low_relative_throughput")

  global_alerts = []
  if len(jobs) != cfg.expected_jobs:
    global_alerts.append(
      f"expected_{cfg.expected_jobs}_logs_but_found_{len(jobs)}"
    )
  unhealthy = [job["log"] for job in jobs if job["alerts"]]
  if unhealthy:
    global_alerts.append("unhealthy_jobs:" + ",".join(unhealthy))
  return {
    "observed_at_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
    "control_dir": str(cfg.control_dir.resolve()),
    "expected_jobs": cfg.expected_jobs,
    "observed_jobs": len(jobs),
    "median_steps_per_second": median_throughput,
    "healthy": not global_alerts,
    "global_alerts": global_alerts,
    "jobs": jobs,
    "interpretation_limit": (
      "This snapshot detects execution health only. Reward and training metrics "
      "must not be used to rank policies without frozen evaluation."
    ),
  }


def main(cfg: HealthCfg) -> None:
  snapshot = inspect(cfg)
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
  temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
  temporary.replace(cfg.output)
  status = "HEALTHY" if snapshot["healthy"] else "ALERT"
  print(f"{status}: {snapshot['observed_jobs']}/{cfg.expected_jobs} jobs")
  if snapshot["global_alerts"]:
    print("\n".join(snapshot["global_alerts"]))


if __name__ == "__main__":
  main(tyro.cli(HealthCfg))
