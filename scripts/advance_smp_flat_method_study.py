"""Fail-closed automation for the preregistered flat method study."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from analyze_smp_flat_method_study import (
  _GATES,
  _PLAN_ID,
  _PROTOCOL_SHA256,
  _SEEDS,
  FlatMethodAnalysisCfg,
  _audit_matrix,
  _validate_index,
  _validate_protocol,
  write_analysis,
)
from build_smp_flat_method_manifests import (
  _ARMS,
  FlatMethodManifestCfg,
  _discover_run,
  _validate_launch,
  write_manifests,
)

_MINIMUM_COMMIT = "5a12f08"
_ERROR_PATTERN = re.compile(
  r"traceback|cuda out of memory|outofmemoryerror|"
  r"(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)|fatal|segmentation fault",
  re.IGNORECASE | re.MULTILINE,
)
_ITERATION_PATTERN = re.compile(r"Learning iteration\s+(\d+)\s*/\s*30000")
_THROUGHPUT_PATTERN = re.compile(r"([0-9][0-9,]*)\s+steps/s")
_WANDB_PATTERN = re.compile(r"wandb\.ai/[^\s]+/runs/([A-Za-z0-9]+)")
_GIB = 1024**3


@dataclass(frozen=True)
class FlatMethodAdvanceCfg:
  protocol: Path = Path("docs/ral_flat_method_study_v1.json")
  training_control_dir: Path = Path("run_control/flat_method_study_v1_training")
  manifest_dir: Path = Path("run_control/flat_method_study_v1_eval/manifests")
  evaluation_root: Path = Path("run_control/flat_method_study_v1_eval/formal")
  analysis_json: Path = Path(
    "run_control/flat_method_study_v1_eval/flat_method_analysis.json"
  )
  analysis_markdown: Path = Path(
    "run_control/flat_method_study_v1_eval/flat_method_analysis.md"
  )
  state: Path = Path("run_control/automation_state/flat_method_study_latest.json")
  logs_root: Path = Path("logs/rsl_rl")
  devices: tuple[str, ...] = tuple(f"cuda:{index}" for index in range(8))
  launch_evaluations_when_ready: bool = False
  stale_log_minutes: float = 45.0


def _load(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _gpu_processes() -> list[str]:
  result = subprocess.run(
    (
      "nvidia-smi",
      "--query-compute-apps=pid,gpu_uuid,process_name",
      "--format=csv,noheader",
    ),
    check=True,
    capture_output=True,
    text=True,
  )
  return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git(repo_root: Path, *args: str) -> str:
  return subprocess.run(
    ("git", *args),
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def _disk_preflight(path: Path) -> dict[str, float]:
  usage = shutil.disk_usage(path)
  stats = os.statvfs(path)
  inode_free_fraction = stats.f_favail / stats.f_files if stats.f_files else 0.0
  free_gib = usage.free / _GIB
  if free_gib < 100.0 or inode_free_fraction < 0.10:
    raise RuntimeError(
      f"DISK_SPACE_ALERT: free_gib={free_gib:.1f}, "
      f"inode_free_fraction={inode_free_fraction:.3f}"
    )
  return {"free_gib": free_gib, "inode_free_fraction": inode_free_fraction}


def _tail(path: Path, maximum_bytes: int = 8 * 1024 * 1024) -> str:
  if not path.is_file():
    return ""
  with path.open("rb") as stream:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - maximum_bytes))
    return stream.read().decode(errors="replace")


def _training_health(
  cfg: FlatMethodAdvanceCfg, launch: dict[str, Any], now: datetime
) -> list[dict[str, Any]]:
  rows = []
  for job in launch["jobs"]:
    pid = int(job["pid"])
    alive = _pid_alive(pid)
    log = Path(job["log"])
    text = _tail(log)
    iteration_matches = _ITERATION_PATTERN.findall(text)
    throughput_matches = _THROUGHPUT_PATTERN.findall(text)
    wandb_matches = _WANDB_PATTERN.findall(text)
    mtime = datetime.fromtimestamp(log.stat().st_mtime, timezone.utc) if log.is_file() else None
    age_minutes = (now - mtime).total_seconds() / 60.0 if mtime else None
    error = _ERROR_PATTERN.search(text)
    run_dir = None
    final_checkpoint = None
    try:
      run_dir = _discover_run(
        cfg.logs_root,
        _ARMS[str(job["arm"])]["experiment"],
        str(job["run_name"]),
      )
      final_checkpoint = run_dir / "model_29999.pt"
    except (FileNotFoundError, ValueError):
      pass
    rows.append(
      {
        "arm": job["arm"],
        "policy_seed": int(job["policy_seed"]),
        "gpu": int(job["gpu"]),
        "pid": pid,
        "process_alive": alive,
        "log": str(log.resolve()),
        "log_exists": log.is_file(),
        "log_mtime_utc": mtime.isoformat() if mtime else None,
        "log_age_minutes": age_minutes,
        "latest_iteration": max(map(int, iteration_matches)) if iteration_matches else None,
        "latest_throughput_steps_s": (
          int(throughput_matches[-1].replace(",", "")) if throughput_matches else None
        ),
        "wandb_run_id": wandb_matches[-1] if wandb_matches else None,
        "error_match": error.group(0) if error else None,
        "run_dir": str(run_dir) if run_dir else None,
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
        "final_checkpoint_present": bool(final_checkpoint and final_checkpoint.is_file()),
        "gate_checkpoints_present": {
          str(gate): bool(run_dir and (run_dir / f"model_{gate}.pt").is_file())
          for gate in _GATES
        },
      }
    )
  return rows


def _analysis_cfg(cfg: FlatMethodAdvanceCfg) -> FlatMethodAnalysisCfg:
  return FlatMethodAnalysisCfg(
    manifest_index=cfg.manifest_dir / "index.json",
    evaluation_root=cfg.evaluation_root,
    protocol=cfg.protocol,
    output_json=cfg.analysis_json,
    output_markdown=cfg.analysis_markdown,
  )


def _launch_matrix(
  cfg: FlatMethodAdvanceCfg,
  manifest: Path,
  seed: int,
  gate: int,
  output_dir: Path,
  preflight: dict[str, float],
) -> dict[str, Any]:
  runner = Path(__file__).with_name("run_smp_frozen_eval_matrix.py").resolve()
  output_dir.mkdir(parents=True)
  command = [
    sys.executable,
    str(runner),
    "--manifest",
    str(manifest.resolve()),
    "--output-dir",
    str(output_dir.resolve()),
    "--devices",
    *cfg.devices,
    "--modes",
    "native_gsi",
    "prone",
    "supine",
    "left_side",
    "right_side",
    "--eval-seeds",
    "20261010",
    "--num-envs",
    "512",
    "--steps",
    "500",
    "--include-per-env",
  ]
  log = output_dir / "evaluation.log"
  with log.open("a") as stream:
    process = subprocess.Popen(
      command,
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  marker = {
    "schema_version": 1,
    "status": "ACTIVE",
    "plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
    "policy_seed": seed,
    "checkpoint_step": gate,
    "manifest": str(manifest.resolve()),
    "output_dir": str(output_dir.resolve()),
    "log": str(log.resolve()),
    "command": command,
    "devices": list(cfg.devices),
    "pid": process.pid,
    "attempt": 1,
    "resource_preflight": preflight,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(cfg.evaluation_root / "active_evaluation.json", marker)
  return marker


def _base_state(repo_root: Path, now: datetime) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "observed_at_utc": now.isoformat(),
    "code_commit": _git(repo_root, "rev-parse", "HEAD"),
    "study_id": "smp-flat-procedural-coverage-v1",
    "plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
  }


def advance(cfg: FlatMethodAdvanceCfg) -> dict[str, Any]:
  repo_root = Path(__file__).resolve().parents[1]
  now = datetime.now(timezone.utc)
  state = _base_state(repo_root, now)
  _git(repo_root, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, "HEAD")
  _validate_protocol(cfg.protocol)
  launch_path = cfg.training_control_dir / "launch_manifest.json"
  if not launch_path.is_file():
    return {**state, "status": "WAITING_FOR_FROZEN_TRAINING_LAUNCH"}
  launch, _ = _validate_launch(
    FlatMethodManifestCfg(
      launch_manifest=launch_path,
      output_dir=cfg.manifest_dir,
      protocol=cfg.protocol,
      logs_root=cfg.logs_root,
    )
  )
  health = _training_health(cfg, launch, now)
  state["training"] = {
    "job_count": len(health),
    "alive_count": sum(row["process_alive"] for row in health),
    "final_checkpoint_count": sum(row["final_checkpoint_present"] for row in health),
    "jobs": health,
  }
  errors = [row for row in health if row["error_match"]]
  dead_incomplete = [
    row for row in health if not row["process_alive"] and not row["final_checkpoint_present"]
  ]
  stale = [
    row for row in health
    if row["process_alive"]
    and (not row["log_exists"] or row["log_age_minutes"] is None
         or row["log_age_minutes"] > cfg.stale_log_minutes)
  ]
  if errors or dead_incomplete or stale:
    return {
      **state,
      "status": "FLAT_METHOD_TRAINING_ALERT",
      "alert": {
        "error_jobs": [row["pid"] for row in errors],
        "dead_incomplete_jobs": [row["pid"] for row in dead_incomplete],
        "stale_jobs": [row["pid"] for row in stale],
        "automatic_restart_forbidden": True,
      },
    }
  if not all(row["final_checkpoint_present"] for row in health):
    return {**state, "status": "FLAT_METHOD_TRAINING_ACTIVE"}
  if any(row["process_alive"] for row in health):
    return {**state, "status": "FLAT_METHOD_TRAINING_FINALIZING"}

  gpu_processes = _gpu_processes()
  state["gpu_compute_processes"] = gpu_processes
  if gpu_processes:
    return {**state, "status": "FLAT_METHOD_WAITING_GPU_IDLE"}
  manifest_cfg = FlatMethodManifestCfg(
    launch_manifest=launch_path,
    output_dir=cfg.manifest_dir,
    protocol=cfg.protocol,
    logs_root=cfg.logs_root,
  )
  try:
    index = write_manifests(manifest_cfg)
  except Exception as error:
    return {
      **state,
      "status": "FLAT_METHOD_MANIFEST_ALERT",
      "error": f"{type(error).__name__}: {error}",
    }
  state["manifests"] = {
    "status": index["status"],
    "index": str((cfg.manifest_dir / "index.json").resolve()),
    "index_id": index["index_id"],
    "manifest_count": len(index["manifests"]),
    "checkpoint_entry_count": index["checkpoint_entry_count"],
  }
  _, manifest_rows = _validate_index(cfg.manifest_dir / "index.json")
  analysis_cfg = _analysis_cfg(cfg)
  marker_path = cfg.evaluation_root / "active_evaluation.json"
  if marker_path.is_file():
    marker = _load(marker_path)
    if (
      marker.get("plan_id") != _PLAN_ID
      or marker.get("protocol_sha256") != _PROTOCOL_SHA256
      or marker.get("attempt") != 1
    ):
      return {**state, "status": "FLAT_METHOD_EVAL_ALERT", "error": "active marker drifted"}
    if _pid_alive(int(marker["pid"])):
      return {**state, "status": "FLAT_METHOD_EVALUATION_ACTIVE", "active_evaluation": marker}
    key = (int(marker["policy_seed"]), int(marker["checkpoint_step"]))
    try:
      _audit_matrix(analysis_cfg, manifest_rows[key], key[0], key[1])
    except Exception as error:
      return {
        **state,
        "status": "FLAT_METHOD_EVAL_ALERT",
        "error": f"dead evaluator left incomplete or invalid evidence: {type(error).__name__}: {error}",
        "active_evaluation": marker,
        "automatic_restart_forbidden": True,
      }
    marker_path.unlink()

  completed = []
  for gate in _GATES:
    for seed in _SEEDS:
      matrix_dir = cfg.evaluation_root / f"gate_{gate}" / f"seed_{seed}"
      if (matrix_dir / "_COMPLETE.json").is_file():
        try:
          _audit_matrix(analysis_cfg, manifest_rows[(seed, gate)], seed, gate)
        except Exception as error:
          return {
            **state,
            "status": "FLAT_METHOD_EVAL_ALERT",
            "error": f"invalid completed matrix: {type(error).__name__}: {error}",
          }
        completed.append({"checkpoint_step": gate, "policy_seed": seed})
        continue
      if matrix_dir.exists() and any(matrix_dir.iterdir()):
        return {
          **state,
          "status": "FLAT_METHOD_EVAL_ALERT",
          "error": f"partial matrix exists without an active evaluator: {matrix_dir}",
          "automatic_restart_forbidden": True,
        }
      state["evaluation"] = {
        "completed_matrix_count": len(completed),
        "required_matrix_count": 12,
        "next_checkpoint_step": gate,
        "next_policy_seed": seed,
      }
      if not cfg.launch_evaluations_when_ready:
        return {**state, "status": "FLAT_METHOD_READY_FOR_FROZEN_EVALUATION"}
      if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
        return {**state, "status": "CODE_SYNC_ALERT", "error": "tracked worktree is dirty"}
      gpu_processes = _gpu_processes()
      if gpu_processes:
        return {**state, "status": "FLAT_METHOD_WAITING_GPU_IDLE", "gpu_compute_processes": gpu_processes}
      try:
        preflight = _disk_preflight(cfg.evaluation_root.parent)
      except RuntimeError as error:
        return {**state, "status": "DISK_SPACE_ALERT", "error": str(error)}
      marker = _launch_matrix(
        cfg,
        Path(manifest_rows[(seed, gate)]["path"]),
        seed,
        gate,
        matrix_dir,
        preflight,
      )
      return {**state, "status": "FLAT_METHOD_EVALUATION_LAUNCHED", "active_evaluation": marker}

  try:
    result = write_analysis(analysis_cfg)
  except Exception as error:
    return {
      **state,
      "status": "FLAT_METHOD_ANALYSIS_ALERT",
      "error": f"{type(error).__name__}: {error}",
    }
  return {
    **state,
    "status": result["status"],
    "evaluation": {"completed_matrix_count": 12, "required_matrix_count": 12},
    "analysis": str(cfg.analysis_json.resolve()),
    "promotion": result["promotion"],
  }


def main(cfg: FlatMethodAdvanceCfg) -> None:
  try:
    result = advance(cfg)
  except Exception as error:
    repo_root = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)
    result = {
      **_base_state(repo_root, now),
      "status": "FLAT_METHOD_AUTOMATION_ALERT",
      "error": f"{type(error).__name__}: {error}",
    }
  _atomic_json(cfg.state, result)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(FlatMethodAdvanceCfg))
