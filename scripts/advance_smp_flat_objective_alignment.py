"""Read-only, fail-closed health monitor for flat objective-alignment training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from launch_smp_flat_objective_alignment import (
  _ARM_ORDER,
  _DEVICES,
  _POLICY_SEEDS,
  _PROTOCOL_SHA256,
  _RESERVED_DEVICES,
  _atomic_json,
  _git,
  _load_json,
  _validate_protocol,
)

_EXPERIMENTS = {
  "a6_replication_control": "smp_scratch_a6_f2s2_mix_bridge_g1",
  "a9_objective_aligned": "smp_scratch_a9_f2s2_objective_aligned_g1",
}
_GATES = (8000, 15000, 25000, 29999)
_ERROR_PATTERN = re.compile(
  r"traceback|cuda out of memory|outofmemoryerror|fatal|segmentation fault|"
  r"(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)",
  re.IGNORECASE | re.MULTILINE,
)
_ITERATION_PATTERN = re.compile(r"Learning iteration\s+(\d+)\s*/\s*30000")
_THROUGHPUT_PATTERN = re.compile(
  r"Steps per second:\s*([0-9][0-9,]*)|([0-9][0-9,]*)\s+steps/s",
  re.IGNORECASE,
)
_WANDB_PATTERN = re.compile(r"wandb\.ai/[^\s]+/runs/([A-Za-z0-9]+)")
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt")


@dataclass(frozen=True)
class FlatObjectiveAdvanceCfg:
  protocol: Path = Path("docs/ral_flat_objective_alignment_v1.json")
  training_control_dir: Path = Path("run_control/flat_objective_alignment_v1_training")
  state: Path = Path("run_control/automation_state/flat_objective_alignment_latest.json")
  logs_root: Path = Path("logs/rsl_rl")
  stale_log_minutes: float = 45.0


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _tail(path: Path, maximum_bytes: int = 8 * 1024 * 1024) -> str:
  if not path.is_file():
    return ""
  with path.open("rb") as stream:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - maximum_bytes))
    return stream.read().decode(errors="replace")


def _discover_run(logs_root: Path, arm: str, run_name: str) -> Path | None:
  experiment = logs_root / _EXPERIMENTS[arm]
  if not experiment.is_dir():
    return None
  matches = sorted(
    (path for path in experiment.iterdir() if path.is_dir() and path.name.endswith(run_name)),
    key=lambda path: path.stat().st_mtime,
  )
  if len(matches) > 1:
    raise ValueError(f"multiple run directories match immutable run name {run_name}")
  return matches[0] if matches else None


def _latest_checkpoint_iteration(run_dir: Path | None) -> int | None:
  if run_dir is None:
    return None
  values = []
  for path in run_dir.glob("model_*.pt"):
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)
    if match and path.is_file():
      values.append(int(match.group(1)))
  return max(values) if values else None


def _validate_launch(launch: dict[str, Any]) -> None:
  if (
    launch.get("schema_version") != 1
    or launch.get("status") != "LAUNCHED"
    or launch.get("study_id") != "smp-flat-objective-alignment-v1"
    or launch.get("protocol_sha256") != _PROTOCOL_SHA256
    or launch.get("policy_seeds") != list(_POLICY_SEEDS)
    or launch.get("devices") != list(_DEVICES)
    or launch.get("reserved_idle_devices") != list(_RESERVED_DEVICES)
    or launch.get("random_actor_critic_and_normalizers") is not True
    or launch.get("actor_observation_dim") != 93
    or launch.get("actor_history_steps") != 1
    or launch.get("critic_observation_dim") != 960
  ):
    raise ValueError("immutable launch manifest header drifted")
  jobs = launch.get("jobs")
  if not isinstance(jobs, list) or len(jobs) != 6:
    raise ValueError("immutable launch job count drifted")
  expected = [
    (arm, seed, gpu)
    for gpu, (arm, seed) in zip(
      _DEVICES,
      ((arm, seed) for arm in _ARM_ORDER for seed in _POLICY_SEEDS),
      strict=True,
    )
  ]
  for job, (arm, seed, gpu) in zip(jobs, expected, strict=True):
    command = job.get("command", [])
    if (
      job.get("arm") != arm
      or job.get("policy_seed") != seed
      or job.get("environment_seed") != seed
      or job.get("gpu") != gpu
      or not isinstance(job.get("pid"), int)
      or "--checkpoint" in command
      or "--resume" in command
      or str(seed) not in command
    ):
      raise ValueError(f"immutable launch job drifted: {arm}/{seed}")
  material = {
    "protocol_sha256": launch["protocol_sha256"],
    "code_commit": launch["code_commit"],
    "jobs": [{key: value for key, value in job.items() if key != "pid"} for job in jobs],
    "reserved_idle_devices": launch["reserved_idle_devices"],
  }
  expected_plan = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  if launch.get("plan_id") != expected_plan:
    raise ValueError("immutable launch plan hash drifted")


def _health(cfg: FlatObjectiveAdvanceCfg, launch: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
  rows = []
  for job in launch["jobs"]:
    log = Path(job["log"])
    text = _tail(log)
    iterations = _ITERATION_PATTERN.findall(text)
    throughputs = [first or second for first, second in _THROUGHPUT_PATTERN.findall(text)]
    wandb = _WANDB_PATTERN.findall(text)
    mtime = datetime.fromtimestamp(log.stat().st_mtime, timezone.utc) if log.is_file() else None
    age = (now - mtime).total_seconds() / 60.0 if mtime else None
    run_dir = _discover_run(cfg.logs_root, str(job["arm"]), str(job["run_name"]))
    latest_log = max(map(int, iterations)) if iterations else None
    latest_checkpoint = _latest_checkpoint_iteration(run_dir)
    progress = [value for value in (latest_log, latest_checkpoint) if value is not None]
    rows.append({
      "arm": job["arm"],
      "policy_seed": int(job["policy_seed"]),
      "gpu": int(job["gpu"]),
      "pid": int(job["pid"]),
      "process_alive": _pid_alive(int(job["pid"])),
      "log": str(log.resolve()),
      "log_exists": log.is_file(),
      "log_mtime_utc": mtime.isoformat() if mtime else None,
      "log_age_minutes": age,
      "latest_iteration": latest_log,
      "latest_checkpoint_iteration": latest_checkpoint,
      "progress_iteration_lower_bound": max(progress) if progress else None,
      "latest_throughput_steps_s": int(throughputs[-1].replace(",", "")) if throughputs else None,
      "wandb_run_id": wandb[-1] if wandb else None,
      "error_match": (match.group(0) if (match := _ERROR_PATTERN.search(text)) else None),
      "run_dir": str(run_dir.resolve()) if run_dir else None,
      "final_checkpoint_present": bool(run_dir and (run_dir / "model_29999.pt").is_file()),
      "gate_checkpoints_present": {
        str(gate): bool(run_dir and (run_dir / f"model_{gate}.pt").is_file())
        for gate in _GATES
      },
    })
  return rows


def advance(cfg: FlatObjectiveAdvanceCfg) -> dict[str, Any]:
  repo_root = Path(__file__).resolve().parents[1]
  now = datetime.now(timezone.utc)
  state: dict[str, Any] = {
    "schema_version": 1,
    "observed_at_utc": now.isoformat(),
    "code_commit": _git(repo_root, "rev-parse", "HEAD"),
    "study_id": "smp-flat-objective-alignment-v1",
    "protocol_sha256": _PROTOCOL_SHA256,
  }
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo_root / cfg.protocol
  _validate_protocol(protocol_path, repo_root)
  launch_path = cfg.training_control_dir / "launch_manifest.json"
  if not launch_path.is_file():
    return {**state, "status": "WAITING_FOR_FROZEN_TRAINING_LAUNCH"}
  launch = _load_json(launch_path)
  _validate_launch(launch)
  try:
    _git(repo_root, "merge-base", "--is-ancestor", str(launch["code_commit"]), "HEAD")
  except subprocess.CalledProcessError as error:
    raise ValueError("current code no longer descends from frozen launch commit") from error
  rows = _health(cfg, launch, now)
  state["plan_id"] = launch["plan_id"]
  state["training"] = {
    "job_count": len(rows),
    "alive_count": sum(row["process_alive"] for row in rows),
    "final_checkpoint_count": sum(row["final_checkpoint_present"] for row in rows),
    "jobs": rows,
  }
  errors = [row for row in rows if row["error_match"]]
  dead_incomplete = [row for row in rows if not row["process_alive"] and not row["final_checkpoint_present"]]
  stale = [
    row for row in rows
    if row["process_alive"] and (
      not row["log_exists"] or row["log_age_minutes"] is None
      or row["log_age_minutes"] > cfg.stale_log_minutes
    )
  ]
  if errors or dead_incomplete or stale:
    return {
      **state,
      "status": "FLAT_OBJECTIVE_TRAINING_ALERT",
      "alert": {
        "error_jobs": [row["pid"] for row in errors],
        "dead_incomplete_jobs": [row["pid"] for row in dead_incomplete],
        "stale_jobs": [row["pid"] for row in stale],
        "automatic_restart_forbidden": True,
      },
    }
  if not all(row["final_checkpoint_present"] for row in rows):
    return {**state, "status": "FLAT_OBJECTIVE_TRAINING_ACTIVE"}
  if any(row["process_alive"] for row in rows):
    return {**state, "status": "FLAT_OBJECTIVE_TRAINING_FINALIZING"}
  return {
    **state,
    "status": "FLAT_OBJECTIVE_CHECKPOINTS_READY_FOR_MANIFESTS",
    "next_action": "Build and hash-lock 12 immutable A6/A9 manifests before frozen evaluation.",
  }


def main(cfg: FlatObjectiveAdvanceCfg) -> None:
  try:
    state = advance(cfg)
  except Exception as error:
    state = {
      "schema_version": 1,
      "observed_at_utc": datetime.now(timezone.utc).isoformat(),
      "study_id": "smp-flat-objective-alignment-v1",
      "protocol_sha256": _PROTOCOL_SHA256,
      "status": "FLAT_OBJECTIVE_MONITOR_ALERT",
      "error": f"{type(error).__name__}: {error}",
      "automatic_restart_forbidden": True,
    }
  _atomic_json(cfg.state, state)
  print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(FlatObjectiveAdvanceCfg))
