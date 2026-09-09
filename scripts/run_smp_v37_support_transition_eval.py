"""Fail-closed launcher for the frozen V37 support-transition matrix."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import tyro

_PROTOCOL_SHA256 = "d15d9d8e03e0b1d01d191f1c224c9201a176e103231d8975bea8ad07c90e1ab3"
_TRAINING_PROTOCOL_SHA256 = (
  "834331f230ec9a6232a59b2698d26baaf4ec7ca8d03bd4354270d12d69d55005"
)
_LAUNCH_PLAN_ID = "c83e1464527d1c6fba65d3345f3a1f71d5522bd8e933ee1476f55a13b5eb4f17"
_MINIMUM_COMMIT = "7cb279c0a285840abd039c0c995de900f96b3c2e"
_POSES = ("prone", "supine", "left_side", "right_side", "synthetic_seated_trap")
_GATES = (0, 500, 1000, 2000, 3500, 5000, 5999)
_SEEDS = (20262001, 20262002, 20262003)
_REASONS = (
  "success",
  "invalid_reset_contact",
  "invalid_dynamics",
  "terrain_exit",
  "head_height_not_reached",
  "upright_not_reached",
  "linear_speed_not_settled",
  "angular_speed_not_settled",
  "head_vertical_speed_not_settled",
  "strict_candidate_not_held",
)
_PER_ENV = (
  "strict_success",
  "strict_success_step",
  "failure_reason",
  "first_head_height_step",
  "first_upright_step",
  "first_head_vertical_speed_settled_step",
  "first_strict_candidate_step",
  "longest_stable_stand_hold_steps",
  "secondary_fall",
  "terrain_exit",
  "invalid_dynamics",
  "terrain_reset_contact_valid",
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
  "finite_action",
)
_GIB = 1024**3


@dataclass(frozen=True)
class V37EvalCfg:
  protocol: Path = Path("docs/v37_93d_support_transition_evaluation_v1.json")
  training_control: Path = Path(
    "run_control/v37_93d_support_transition_v1/training"
  )
  control_dir: Path = Path("run_control/v37_93d_support_transition_v1/evaluation")
  launch: bool = False
  worker_device: int | None = None


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise RuntimeError(f"V37_EVAL_ALERT: expected JSON object in {path}")
  return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ("git", *args), cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _tensors(value: Any) -> list[torch.Tensor]:
  if torch.is_tensor(value):
    return [value]
  if isinstance(value, dict):
    return [tensor for child in value.values() for tensor in _tensors(child)]
  if isinstance(value, (tuple, list)):
    return [tensor for child in value for tensor in _tensors(child)]
  return []


def _checkpoint(path: Path, gate: int) -> dict[str, Any]:
  if not path.is_file():
    raise RuntimeError(f"V37_MANIFEST_ALERT: missing checkpoint {path}")
  value = torch.load(path, map_location="cpu", weights_only=False)
  actor = value.get("actor_state_dict")
  critic = value.get("critic_state_dict")
  tensors = _tensors(value)
  if (
    value.get("iter") != gate
    or not isinstance(actor, dict)
    or not isinstance(critic, dict)
    or tuple(actor["mlp.0.weight"].shape) != (512, 93)
    or tuple(actor["obs_normalizer._mean"].shape) != (1, 93)
    or tuple(critic["mlp.0.weight"].shape) != (512, 960)
    or not tensors
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError(f"V37_MANIFEST_ALERT: checkpoint integrity failed {path}")
  return {
    "sha256": _sha256(path),
    "embedded_iteration": gate,
    "actor_input_dim": 93,
    "critic_input_dim": 960,
    "tensor_count": len(tensors),
    "tensor_elements": sum(tensor.numel() for tensor in tensors),
    "all_tensors_finite": True,
  }


def _preflight(repo: Path) -> dict[str, Any]:
  if _git(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("V37_EVAL_ALERT: tracked worktree is dirty")
  usage = shutil.disk_usage(repo)
  stats = os.statvfs(repo)
  free_gib = usage.free / _GIB
  inode_free = stats.f_favail / stats.f_files if stats.f_files else 0.0
  if free_gib < 100.0 or inode_free < 0.10:
    raise RuntimeError(
      f"DISK_SPACE_ALERT: free_gib={free_gib:.1f} inode_free={inode_free:.3f}"
    )
  active = subprocess.run(
    (
      "nvidia-smi",
      "--query-compute-apps=gpu_uuid,pid,process_name",
      "--format=csv,noheader",
    ),
    check=True,
    capture_output=True,
    text=True,
  ).stdout.splitlines()
  active = [line.strip() for line in active if line.strip()]
  if active:
    raise RuntimeError(f"V37_EVAL_ALERT: GPU compute is active: {active}")
  return {
    "free_gib": free_gib,
    "inode_free_fraction": inode_free,
    "gpu_compute_processes": [],
  }


def _validate_protocol(path: Path) -> dict[str, Any]:
  if not path.is_file() or _sha256(path) != _PROTOCOL_SHA256:
    raise RuntimeError("V37_EVAL_ALERT: evaluation protocol hash drift")
  value = _json(path)
  matrix = value.get("matrix", {})
  strict = value.get("strict_success", {})
  if (
    value.get("status") != "PREREGISTERED_READY_FOR_EVALUATION"
    or value.get("training_protocol", {}).get("sha256")
    != _TRAINING_PROTOCOL_SHA256
    or value.get("training_protocol", {}).get("launch_plan_id") != _LAUNCH_PLAN_ID
    or tuple(value.get("policy_seeds", ())) != _SEEDS
    or tuple(value.get("checkpoint_gates", ())) != _GATES
    or tuple(matrix.get("reset_poses", ())) != _POSES
    or matrix.get("total_cells") != 210
    or matrix.get("num_envs_per_cell") != 256
    or matrix.get("steps_per_cell") != 750
    or matrix.get("eval_seed") != 20262010
    or strict.get("head_height_above_local_support_m") != 1.08
    or strict.get("upright_min") != 0.85
    or strict.get("absolute_head_vertical_speed_max_m_s") != 0.12
    or strict.get("consecutive_hold_steps") != 100
    or tuple(value.get("failure_reason_codebook", ())) != _REASONS
    or tuple(value.get("required_per_environment_telemetry", ())) != _PER_ENV
  ):
    raise RuntimeError("V37_EVAL_ALERT: evaluation protocol content drift")
  return value


def _find_run(repo: Path, job: dict[str, Any]) -> Path:
  roots = list(
    (repo / "logs/rsl_rl" / job["experiment"]).glob("*_" + job["run_name"])
  )
  if len(roots) != 1:
    raise RuntimeError(f"V37_MANIFEST_ALERT: expected one run for {job['run_name']}")
  return roots[0]


def _seed_from_yaml(path: Path) -> int:
  matches = re.findall(r"(?m)^seed:\s*(\d+)\s*$", path.read_text())
  if len(matches) != 1:
    raise RuntimeError(f"V37_MANIFEST_ALERT: expected one root seed in {path}")
  return int(matches[0])


def build_plan(cfg: V37EvalCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo / cfg.protocol
  protocol = _validate_protocol(protocol_path)
  training_control = (
    cfg.training_control
    if cfg.training_control.is_absolute()
    else repo / cfg.training_control
  )
  launched = _json(training_control / "launch_manifest.json")
  if (
    launched.get("plan_id") != _LAUNCH_PLAN_ID
    or launched.get("code_commit") != _MINIMUM_COMMIT
    or len(launched.get("jobs", ())) != 6
  ):
    raise RuntimeError("V37_MANIFEST_ALERT: training launch lineage drift")
  commit = _git(repo, "rev-parse", "HEAD")
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  jobs = []
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  for launch_job in launched["jobs"]:
    run_dir = _find_run(repo, launch_job)
    if (
      _seed_from_yaml(run_dir / "params/agent.yaml") != launch_job["seed"]
      or _seed_from_yaml(run_dir / "params/env.yaml") != launch_job["seed"]
    ):
      raise RuntimeError(f"V37_MANIFEST_ALERT: seed drift in {run_dir}")
    for gate in _GATES:
      checkpoint = run_dir / f"model_{gate}.pt"
      integrity = _checkpoint(checkpoint, gate)
      matrix_id = f"{launch_job['arm']}_seed{launch_job['seed']}_gate{gate}"
      output = control / "results" / f"{matrix_id}.jsonl"
      command = [
        str(repo / ".venv/bin/python"),
        "scripts/evaluate_terrain_recovery.py",
        "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", integrity["sha256"],
        "--protocol", str(protocol_path),
        "--protocol-sha256", _PROTOCOL_SHA256,
        "--task", launch_job["task"],
        "--terrain-types", "flat",
        "--levels", "0",
        "--reset-modes", *_POSES,
        "--num-envs", "256",
        "--steps", "750",
        "--seed", "20262010",
        "--device", "cuda:0",
        "--output", str(output),
        "--stand-head-height-m", "1.08",
        "--stand-min-upright", "0.85",
        "--stand-max-linear-speed-m-s", "1000000000",
        "--stand-max-angular-speed-rad-s", "1000000000",
        "--stand-max-abs-head-vertical-speed-m-s", "0.12",
        "--stable-hold-steps", "100",
      ]
      jobs.append(
        {
          "matrix_id": matrix_id,
          "arm": launch_job["arm"],
          "policy_seed": launch_job["seed"],
          "gate": gate,
          "task": launch_job["task"],
          "wandb_run_id": launch_job.get("wandb_id"),
          "run_dir": str(run_dir),
          "checkpoint": str(checkpoint),
          "checkpoint_sha256": integrity["sha256"],
          "checkpoint_integrity": integrity,
          "agent_yaml_sha256": _sha256(run_dir / "params/agent.yaml"),
          "env_yaml_sha256": _sha256(run_dir / "params/env.yaml"),
          "physical_device": len(jobs) % 8,
          "output": str(output),
          "log": str(control / "logs" / f"{matrix_id}.log"),
          "command": command,
        }
      )
  material = {
    "protocol_sha256": _PROTOCOL_SHA256,
    "training_plan_id": _LAUNCH_PLAN_ID,
    "code_commit": commit,
    "jobs": [
      {
        "matrix_id": job["matrix_id"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "physical_device": job["physical_device"],
        "command": job["command"],
      }
      for job in jobs
    ],
  }
  return {
    "schema_version": 1,
    "status": "PLANNED",
    "plan_id": hashlib.sha256(
      json.dumps(material, sort_keys=True).encode()
    ).hexdigest(),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "protocol": str(protocol_path),
    "protocol_sha256": _PROTOCOL_SHA256,
    "training_plan_id": _LAUNCH_PLAN_ID,
    "code_commit": commit,
    "total_matrices": len(jobs),
    "total_cells": len(jobs) * len(_POSES),
    "jobs": jobs,
    "claim_boundary": protocol["claim_boundary"],
  }


def _read_rows(job: dict[str, Any]) -> list[dict[str, Any]]:
  path = Path(job["output"])
  rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  if len(rows) != len(_POSES):
    raise RuntimeError(f"V37_EVAL_ALERT: incomplete matrix {path}")
  seen = set()
  for row in rows:
    pose = row.get("reset_mode")
    per_env = row.get("per_env")
    diagnosis = row.get("strict_failure_diagnosis", {})
    if (
      row.get("schema_version") != 2
      or row.get("checkpoint_sha256") != job["checkpoint_sha256"]
      or row.get("protocol_sha256") != _PROTOCOL_SHA256
      or row.get("task") != job["task"]
      or row.get("terrain_type") != "flat"
      or row.get("terrain_level") != 0
      or pose not in _POSES
      or pose in seen
      or row.get("seed") != 20262010
      or row.get("num_envs") != 256
      or row.get("steps") != 750
      or row.get("strict_success_hold_steps") != 100
      or not isinstance(per_env, dict)
      or diagnosis.get("schema_version") != 1
    ):
      raise RuntimeError(f"V37_EVAL_ALERT: matrix metadata drift {path}")
    seen.add(pose)
    for key in _PER_ENV:
      if not isinstance(per_env.get(key), list) or len(per_env[key]) != 256:
        raise RuntimeError(f"V37_EVAL_ALERT: invalid {key} in {path}")
    counts = {reason: per_env["failure_reason"].count(reason) for reason in _REASONS}
    if (
      sum(counts.values()) != 256
      or counts != diagnosis.get("reason_counts")
      or counts["success"] != row.get("success")
      or counts["success"] != sum(bool(x) for x in per_env["strict_success"])
      or not all(bool(x) for x in per_env["terrain_reset_contact_valid"])
    ):
      raise RuntimeError(f"V37_EVAL_ALERT: failure telemetry mismatch {path}")
  if seen != set(_POSES):
    raise RuntimeError(f"V37_EVAL_ALERT: pose matrix incomplete {path}")
  return rows


def _worker(cfg: V37EvalCfg) -> None:
  repo = Path(__file__).resolve().parents[1]
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  plan = _json(control / "immutable_plan.json")
  device = int(cfg.worker_device)  # type: ignore[arg-type]
  assigned = [job for job in plan["jobs"] if job["physical_device"] == device]
  state_path = control / "workers" / f"gpu{device}.json"
  state = {
    "status": "RUNNING",
    "plan_id": plan["plan_id"],
    "physical_device": device,
    "pid": os.getpid(),
    "total_matrices": len(assigned),
    "completed_matrices": 0,
    "current_matrix": None,
  }
  _atomic_json(state_path, state)
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = str(device)
  env["PYTHONUNBUFFERED"] = "1"
  for job in assigned:
    output = Path(job["output"])
    if output.exists():
      raise RuntimeError(f"V37_EVAL_ALERT: refusing to overwrite {output}")
    state["current_matrix"] = job["matrix_id"]
    state["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, state)
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
      result = subprocess.run(
        job["command"], cwd=repo, env=env, stdout=stream,
        stderr=subprocess.STDOUT, text=True
      )
    if result.returncode != 0:
      state.update({"status": "V37_EVAL_ALERT", "returncode": result.returncode})
      _atomic_json(state_path, state)
      raise SystemExit(result.returncode)
    _read_rows(job)
    state["completed_matrices"] += 1
  state.update(
    {
      "status": "COMPLETE",
      "current_matrix": None,
      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
  )
  _atomic_json(state_path, state)


def _status(control: Path, plan: dict[str, Any]) -> dict[str, Any]:
  complete = []
  for job in plan["jobs"]:
    path = Path(job["output"])
    if path.exists():
      _read_rows(job)
      complete.append(job["matrix_id"])
  active_path = control / "active_evaluation.json"
  active = _json(active_path) if active_path.exists() else None
  if len(complete) == len(plan["jobs"]):
    return {
      "status": "V37_EVALUATION_COMPLETE_READY_FOR_ANALYSIS",
      "plan_id": plan["plan_id"],
      "completed_matrices": len(complete),
      "completed_cells": len(complete) * len(_POSES),
    }
  if active is not None:
    dead = [pid for pid in active.get("worker_pids", []) if not _pid_alive(pid)]
    if dead:
      return {
        "status": "V37_EVAL_ALERT",
        "reason": "worker died before the frozen matrix was complete",
        "dead_pids": dead,
        "completed_matrices": len(complete),
      }
    return {
      "status": "V37_EVALUATION_RUNNING",
      "plan_id": plan["plan_id"],
      "completed_matrices": len(complete),
      "completed_cells": len(complete) * len(_POSES),
      "worker_pids": active.get("worker_pids", []),
    }
  if complete:
    return {
      "status": "V37_EVAL_ALERT",
      "reason": "partial artifacts exist without an active evaluator",
      "completed_matrices": len(complete),
    }
  return {"status": "V37_EVALUATION_READY", "completed_matrices": 0}


def main(cfg: V37EvalCfg) -> None:
  repo = Path(__file__).resolve().parents[1]
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  if cfg.worker_device is not None:
    _worker(cfg)
    return
  plan_path = control / "immutable_plan.json"
  if plan_path.exists():
    plan = _json(plan_path)
    rebuilt = build_plan(cfg)
    if plan.get("plan_id") != rebuilt.get("plan_id"):
      raise RuntimeError("V37_EVAL_ALERT: immutable plan drift")
  else:
    if control.exists() and any(control.iterdir()):
      raise RuntimeError("V37_EVAL_ALERT: partial control directory without plan")
    plan = build_plan(cfg)
    _atomic_json(plan_path, plan)
  status = _status(control, plan)
  if not cfg.launch or status["status"] != "V37_EVALUATION_READY":
    print(json.dumps(status, sort_keys=True))
    return
  resources = _preflight(repo)
  worker_pids = []
  for device in range(8):
    log = control / "workers" / f"gpu{device}_launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("x")
    proc = subprocess.Popen(
      [
        str(repo / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--protocol", str(cfg.protocol),
        "--training-control", str(cfg.training_control),
        "--control-dir", str(cfg.control_dir),
        "--worker-device", str(device),
      ],
      cwd=repo,
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
    stream.close()
    worker_pids.append(proc.pid)
  active = {
    "status": "RUNNING",
    "plan_id": plan["plan_id"],
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "worker_pids": worker_pids,
    "resources": resources,
  }
  _atomic_json(control / "active_evaluation.json", active)
  print(json.dumps({"status": "V37_EVALUATION_LAUNCHED", **active}, sort_keys=True))


if __name__ == "__main__":
  tyro.cli(main)
