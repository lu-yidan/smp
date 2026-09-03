"""Launch the preregistered V34 96D-versus-93D escape evaluation matrix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import tyro

_PROTOCOL_SHA256 = "1f212abf7e627ac031fcca6637e69b5f43c2c94e6f703bfbc8de6e676734f52e"
_MINIMUM_COMMIT = "bb505f5483ca824a8ea8a7d489076a8c3f8afeb4"
_GIB = 1024**3


@dataclass(frozen=True)
class V34NinetyThreeDimEvalCfg:
  protocol: Path = Path("docs/ral_v34_93d_evaluation_v1.json")
  control_dir: Path = Path("run_control/v34_93d_control/evaluation")
  launch: bool = False
  worker_device: int | None = None


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def _checkpoint_integrity(
  path: Path, sha256: str, iteration: int, actor_dim: int
) -> dict[str, Any]:
  if not path.is_file() or _sha256(path) != sha256:
    raise RuntimeError(f"V34_93D_EVAL_ALERT: checkpoint missing or drifted: {path}")
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  actor = checkpoint["actor_state_dict"]
  critic = checkpoint["critic_state_dict"]
  tensors: list[torch.Tensor] = []

  def collect(item: Any) -> None:
    if torch.is_tensor(item):
      tensors.append(item)
    elif isinstance(item, dict):
      for child in item.values():
        collect(child)
    elif isinstance(item, (tuple, list)):
      for child in item:
        collect(child)

  collect(checkpoint)
  if (
    checkpoint.get("iter") != iteration
    or tuple(actor["mlp.0.weight"].shape) != (512, actor_dim)
    or tuple(actor["obs_normalizer._mean"].shape) != (1, actor_dim)
    or tuple(critic["mlp.0.weight"].shape) != (512, 960)
    or not tensors
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError(f"V34_93D_EVAL_ALERT: checkpoint integrity failed: {path}")
  return {
    "embedded_iteration": iteration,
    "actor_input_dim": actor_dim,
    "critic_input_dim": 960,
    "tensor_count": len(tensors),
    "tensor_elements": sum(tensor.numel() for tensor in tensors),
    "all_tensors_finite": True,
  }


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
  digest = _sha256(path) if path.is_file() else ""
  if digest != _PROTOCOL_SHA256:
    raise ValueError("V34 93D evaluation protocol SHA-256 mismatch")
  protocol = _json(path)
  matrix = protocol.get("matrix", {})
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_EVALUATION"
    or protocol.get("study_id") != "smp-v34-93d-matched-evaluation-v1"
    or matrix.get("total_cells") != 14
    or matrix.get("reset_poses") != ["prone", "supine"]
    or matrix.get("num_envs_per_cell") != 512
    or matrix.get("steps_per_environment") != 1000
    or matrix.get("evaluation_seed") != 20261710
    or matrix.get("physical_devices") != list(range(8))
  ):
    raise ValueError("V34 93D evaluation protocol drifted")
  if list(protocol["treatment"]["gates"]) != [
    "0",
    "1000",
    "3000",
    "6000",
    "9000",
    "11999",
  ]:
    raise ValueError("V34 93D evaluation gate order drifted")
  return protocol, digest


def _command(repo: Path, job: dict[str, Any], matrix: dict[str, Any]) -> list[str]:
  return [
    str(repo / ".venv/bin/python"),
    "scripts/evaluate_escape_checkpoint.py",
    "--checkpoint",
    job["checkpoint"],
    "--task",
    job["task"],
    "--num-envs",
    str(matrix["num_envs_per_cell"]),
    "--steps",
    str(matrix["steps_per_environment"]),
    "--seed",
    str(matrix["evaluation_seed"]),
    "--device",
    "cuda:0",
    "--plate-mass-kg",
    str(matrix["plate_mass_kg"]),
    "--plate-length-m",
    str(matrix["plate_length_m"]),
    "--plate-width-m",
    str(matrix["plate_width_m"]),
    "--plate-thickness-m",
    str(matrix["plate_thickness_m"]),
    "--plate-friction",
    str(matrix["plate_friction"]),
    "--reset-pose",
    job["reset_pose"],
    "--longitudinal-offset-m",
    str(matrix["longitudinal_offset_m"]),
    "--lateral-offset-m",
    str(matrix["lateral_offset_m"]),
    "--longitudinal-jitter-m",
    str(matrix["longitudinal_jitter_m"]),
    "--lateral-jitter-m",
    str(matrix["lateral_jitter_m"]),
    "--xy-jitter-m",
    str(matrix["xy_jitter_m"]),
    "--stable-hold-steps",
    str(matrix["stable_hold_steps"]),
    "--stand-head-height-m",
    str(matrix["stand_head_height_m"]),
    "--stand-min-upright",
    str(matrix["stand_min_upright"]),
    "--stand-max-linear-speed-m-s",
    str(matrix["stand_max_linear_speed_m_s"]),
    "--stand-max-angular-speed-rad-s",
    str(matrix["stand_max_angular_speed_rad_s"]),
    "--wide-stance-threshold-m",
    str(matrix["wide_stance_threshold_m"]),
  ]


def build_plan(cfg: V34NinetyThreeDimEvalCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo / cfg.protocol
  protocol, protocol_sha = _load_protocol(protocol_path)
  commit = _git(repo, "rev-parse", "HEAD")
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  matrix = protocol["matrix"]
  specs: list[dict[str, Any]] = []
  control = protocol["control"]
  control_path = (repo / control["checkpoint"]).resolve()
  control_integrity = _checkpoint_integrity(
    control_path,
    control["checkpoint_sha256"],
    control["checkpoint_iteration"],
    control["actor_input_dim"],
  )
  for pose in matrix["reset_poses"]:
    specs.append(
      {
        "label": control["label"],
        "gate": 98000,
        "task": control["task"],
        "checkpoint": str(control_path),
        "checkpoint_sha256": control["checkpoint_sha256"],
        "checkpoint_integrity": control_integrity,
        "reset_pose": pose,
      }
    )
  treatment = protocol["treatment"]
  run_dir = (repo / treatment["run_dir"]).resolve()
  for gate_text, digest in treatment["gates"].items():
    gate = int(gate_text)
    checkpoint = run_dir / f"model_{gate}.pt"
    integrity = _checkpoint_integrity(checkpoint, digest, gate, 93)
    for pose in matrix["reset_poses"]:
      specs.append(
        {
          "label": treatment["label"],
          "gate": gate,
          "task": treatment["task"],
          "checkpoint": str(checkpoint),
          "checkpoint_sha256": digest,
          "checkpoint_integrity": integrity,
          "reset_pose": pose,
        }
      )
  jobs = []
  control_dir = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  for index, spec in enumerate(specs):
    device = matrix["physical_devices"][index % len(matrix["physical_devices"])]
    cell_id = f"{spec['label']}_gate{spec['gate']}_{spec['reset_pose']}"
    job = {
      **spec,
      "cell_index": index,
      "cell_id": cell_id,
      "physical_device": device,
      "result": str(control_dir / "results" / f"{cell_id}.json"),
      "log": str(control_dir / "logs" / f"{cell_id}.log"),
    }
    job["command"] = _command(repo, job, matrix)
    jobs.append(job)
  material = {
    "protocol_sha256": protocol_sha,
    "code_commit": commit,
    "jobs": [
      {
        "cell_id": job["cell_id"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "command": job["command"],
        "physical_device": job["physical_device"],
      }
      for job in jobs
    ],
  }
  plan_id = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  return {
    "schema_version": 1,
    "status": "PLANNED",
    "study_id": protocol["study_id"],
    "plan_id": plan_id,
    "protocol": str(protocol_path.resolve()),
    "protocol_sha256": protocol_sha,
    "code_commit": commit,
    "total_cells": len(jobs),
    "matrix": matrix,
    "jobs": jobs,
    "claim_boundary": protocol["claim_boundary"],
  }


def _validate_result(job: dict[str, Any], protocol_sha: str) -> dict[str, Any]:
  result_path = Path(job["result"])
  result = _json(result_path)
  if (
    result.get("cell_id") != job["cell_id"]
    or result.get("protocol_sha256") != protocol_sha
    or result.get("checkpoint_sha256") != job["checkpoint_sha256"]
    or result.get("reset_pose") != job["reset_pose"]
    or result.get("num_envs") != 512
    or result.get("steps") != 1000
    or result.get("seed") != 20261710
    or result.get("active") != 512
  ):
    raise RuntimeError(f"V34_93D_EVAL_ALERT: invalid result {result_path}")
  return result


def _worker(cfg: V34NinetyThreeDimEvalCfg) -> None:
  repo = Path(__file__).resolve().parents[1]
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  plan = _json(control / "immutable_plan.json")
  device = int(cfg.worker_device)  # type: ignore[arg-type]
  jobs = [job for job in plan["jobs"] if job["physical_device"] == device]
  progress_path = control / "workers" / f"gpu{device}.json"
  progress = {
    "schema_version": 1,
    "status": "RUNNING",
    "plan_id": plan["plan_id"],
    "physical_device": device,
    "pid": os.getpid(),
    "total_jobs": len(jobs),
    "completed_jobs": 0,
    "current_cell": None,
  }
  _atomic_json(progress_path, progress)
  environment = os.environ.copy()
  environment["CUDA_VISIBLE_DEVICES"] = str(device)
  environment["PYTHONUNBUFFERED"] = "1"
  for index, job in enumerate(jobs):
    result_path = Path(job["result"])
    if result_path.exists():
      raise RuntimeError(
        f"V34_93D_EVAL_ALERT: refusing to overwrite {result_path}"
      )
    progress["current_cell"] = job["cell_id"]
    progress["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(progress_path, progress)
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
      completed = subprocess.run(
        job["command"],
        cwd=repo,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
      )
    if completed.returncode != 0:
      progress["status"] = "FAILED"
      progress["returncode"] = completed.returncode
      progress["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
      _atomic_json(progress_path, progress)
      raise RuntimeError(f"V34_93D_EVAL_ALERT: failed cell {job['cell_id']}")
    marker = "ESCAPE_EVAL_JSON="
    rows = [
      line[len(marker) :]
      for line in log_path.read_text(errors="replace").splitlines()
      if line.startswith(marker)
    ]
    if len(rows) != 1:
      raise RuntimeError(f"V34_93D_EVAL_ALERT: malformed cell {job['cell_id']}")
    result = json.loads(rows[0])
    result.update(
      {
        "schema_version": 1,
        "status": "CELL_COMPLETE",
        "cell_id": job["cell_id"],
        "cell_index": job["cell_index"],
        "label": job["label"],
        "gate": job["gate"],
        "task": job["task"],
        "checkpoint_path": job["checkpoint"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "plan_id": plan["plan_id"],
        "code_commit": plan["code_commit"],
        "physical_device": device,
        "log_sha256": _sha256(log_path),
      }
    )
    _atomic_json(result_path, result)
    _validate_result(job, plan["protocol_sha256"])
    progress["completed_jobs"] = index + 1
    progress["current_cell"] = None
    progress["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(progress_path, progress)
  progress["status"] = "COMPLETE"
  progress["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(progress_path, progress)


def _disk_preflight(repo: Path) -> dict[str, float]:
  usage = shutil.disk_usage(repo)
  stats = os.statvfs(repo)
  free_gib = usage.free / _GIB
  inode_free = stats.f_favail / stats.f_files if stats.f_files else 0.0
  if free_gib < 100.0 or inode_free < 0.10:
    raise RuntimeError(
      f"DISK_SPACE_ALERT: free_gib={free_gib:.1f}, inode_free_fraction={inode_free:.3f}"
    )
  return {"free_gib": free_gib, "inode_free_fraction": inode_free}


def launch_matrix(cfg: V34NinetyThreeDimEvalCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  marker_path = control / "active_evaluation.json"
  if marker_path.exists():
    marker = _json(marker_path)
    if marker.get("status") != "LAUNCHED":
      raise RuntimeError("V34_93D_EVAL_ALERT: invalid active marker")
    plan = _json(control / "immutable_plan.json")
    completed = sum(Path(job["result"]).is_file() for job in plan["jobs"])
    dead = [row for row in marker["workers"] if not _pid_alive(int(row["pid"]))]
    if dead and completed != plan["total_cells"]:
      raise RuntimeError(
        f"V34_93D_EVAL_ALERT: workers exited with {completed}/{plan['total_cells']} cells"
      )
    return {**marker, "completed_cells": completed}
  plan = build_plan(cfg)
  if not cfg.launch:
    return plan
  if _git(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing V34 93D evaluation from tracked-dirty worktree")
  processes = subprocess.run(
    (
      "nvidia-smi",
      "--query-compute-apps=pid,gpu_uuid,process_name",
      "--format=csv,noheader",
    ),
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  if processes:
    raise RuntimeError(f"V34_93D_EVAL_ALERT: GPU compute is active: {processes}")
  plan["resource_preflight"] = _disk_preflight(repo)
  plan["status"] = "IMMUTABLE_PLAN"
  plan_path = control / "immutable_plan.json"
  _atomic_json(plan_path, plan)
  workers = []
  for device in plan["matrix"]["physical_devices"]:
    worker_log = control / "workers" / f"gpu{device}.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
      sys.executable,
      str(Path(__file__).resolve()),
      "--protocol",
      str(Path(plan["protocol"])),
      "--control-dir",
      str(control),
      "--worker-device",
      str(device),
    ]
    with worker_log.open("w") as stream:
      process = subprocess.Popen(
        command,
        cwd=repo,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
    workers.append(
      {"physical_device": device, "pid": process.pid, "log": str(worker_log)}
    )
  marker = {
    "schema_version": 1,
    "status": "LAUNCHED",
    "plan_id": plan["plan_id"],
    "protocol_sha256": plan["protocol_sha256"],
    "code_commit": plan["code_commit"],
    "total_cells": plan["total_cells"],
    "workers": workers,
    "launched_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(marker_path, marker)
  return marker


def main(cfg: V34NinetyThreeDimEvalCfg) -> None:
  if cfg.worker_device is not None:
    _worker(cfg)
    return
  print(json.dumps(launch_matrix(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(V34NinetyThreeDimEvalCfg))
