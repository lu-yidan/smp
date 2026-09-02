"""Fail-closed launcher for the preregistered A13 continuous-reset fine-tune."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import tyro

_PROTOCOL_SHA256 = "d584340770777df238eaa59cf3e649469b9c8afff97cae9a039c41ee75f38bf3"
_MINIMUM_COMMIT = "f249b3e"
_TASK = "Smp-Getup-Scratch-A13-F2S2-Continuous-Reset-G1"
_EXPECTED_RUNNER = "SmpCurriculumWarmStartRunner"
_SOURCE_SHA256 = "c477624f4c79a6a628ad1fdfa53443a1edb65c47cc75be06610b16f497e37c15"
_GIB = 1024**3


@dataclass(frozen=True)
class ContinuousResetFinetuneCfg:
  protocol: Path = Path("docs/ral_continuous_reset_finetune_v1.json")
  control_dir: Path = Path(
    "run_control/continuous_reset_finetune_v1/training_offline_retry_after_auth"
  )
  launch: bool = False


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _git(repo_root: Path, *args: str) -> str:
  return subprocess.run(
    ("git", *args), cwd=repo_root, check=True, capture_output=True, text=True
  ).stdout.strip()


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


def _disk_preflight(repo_root: Path) -> dict[str, float]:
  usage = shutil.disk_usage(repo_root)
  stats = os.statvfs(repo_root)
  free_gib = usage.free / _GIB
  inode_fraction = stats.f_favail / stats.f_files if stats.f_files else 0.0
  if free_gib < 100.0 or inode_fraction < 0.10:
    raise RuntimeError(
      f"DISK_SPACE_ALERT: free_gib={free_gib:.1f}, "
      f"inode_free_fraction={inode_fraction:.3f}"
    )
  return {"free_gib": free_gib, "inode_free_fraction": inode_fraction}


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _collect_tensors(value: Any) -> list[torch.Tensor]:
  if torch.is_tensor(value):
    return [value]
  if isinstance(value, dict):
    return [tensor for item in value.values() for tensor in _collect_tensors(item)]
  if isinstance(value, (tuple, list)):
    return [tensor for item in value for tensor in _collect_tensors(item)]
  return []


def _validate_checkpoint(path: Path, *, expected_iteration: int) -> dict[str, Any]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  tensors = _collect_tensors(payload)
  actor_shape = tuple(payload["actor_state_dict"]["mlp.0.weight"].shape)
  critic_shape = tuple(payload["critic_state_dict"]["mlp.0.weight"].shape)
  if (
    payload.get("iter") != expected_iteration
    or actor_shape != (512, 93)
    or critic_shape != (512, 960)
    or not tensors
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError(f"A13_CHECKPOINT_ALERT: invalid checkpoint {path}")
  return {
    "iteration": payload["iter"],
    "actor_input_dim": actor_shape[1],
    "critic_input_dim": critic_shape[1],
    "tensor_count": len(tensors),
    "tensor_elements": sum(tensor.numel() for tensor in tensors),
    "all_tensors_finite": True,
  }


def _validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
  digest = _sha256(path) if path.is_file() else ""
  if digest != _PROTOCOL_SHA256:
    raise ValueError("A13 continuous-reset protocol SHA-256 mismatch")
  protocol = _load_json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != "smp-continuous-reset-finetune-v1"
    or protocol.get("study_role")
    != "ENGINEERING_RESET_COVERAGE_FINETUNE_NOT_PROMOTION_OR_RAL_EVIDENCE"
  ):
    raise ValueError("A13 protocol is not launch eligible")
  source = protocol.get("source_policy", {})
  if (
    source.get("checkpoint_name") != "model_3500.pt"
    or source.get("checkpoint_sha256") != _SOURCE_SHA256
    or source.get("embedded_iteration") != 3500
    or source.get("actor_input_dim") != 93
    or source.get("critic_input_dim") != 960
  ):
    raise ValueError("A13 source policy drifted")
  reset = protocol.get("treatment", {}).get("reset_distribution", {})
  expected_reset = {
    "procedural_probability": 1.0,
    "mode_weights": [3.0, 1.0, 1.0, 1.0],
    "orientation_roll_pitch_noise_rad": 0.35,
    "joint_noise_levels_rad": [0.12, 0.20, 0.30],
    "joint_noise_weights": [0.70, 0.20, 0.10],
    "joint_limit_margin_rad": 0.02,
    "root_linear_velocity": 0.0,
    "root_angular_velocity": 0.0,
  }
  for key, value in expected_reset.items():
    if reset.get(key) != value:
      raise ValueError(f"A13 reset protocol {key} drifted")
  expected_training = {
    "policy_seed": 20261501,
    "environment_seed": 20261501,
    "warm_start": "actor_critic_and_normalizers_only",
    "reset_optimizer_iteration_and_environment_steps": True,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 5000,
    "save_interval": 500,
    "evaluation_gates": [0, 500, 1000, 2000, 3500, 4999],
    "learning_rate": 1e-05,
    "wandb_mode": "offline",
    "device": 0,
    "reserved_idle_devices": [1, 2, 3, 4, 5, 6, 7],
    "no_automatic_restart": True,
  }
  training = protocol.get("training_protocol", {})
  for key, value in expected_training.items():
    if training.get(key) != value:
      raise ValueError(f"A13 training protocol {key} drifted")
  audit = protocol.get("implementation_audit", {})
  if (
    audit.get("status") != "PASSED_REAL_MUJOCO_WARM_START_SMOKE"
    or audit.get("code_commit") != "f249b3eba3cd3d242615e0665ea399cf1ec4d5c2"
    or audit.get("source_checkpoint_sha256") != _SOURCE_SHA256
  ):
    raise ValueError("A13 implementation smoke drifted")
  return protocol, digest


def _validate_smoke(protocol: dict[str, Any], repo_root: Path) -> dict[str, Any]:
  audit = protocol["implementation_audit"]
  for name, row in audit["runtime_files"].items():
    path = repo_root / row["path"]
    if not path.is_file() or _sha256(path) != row["sha256"]:
      raise RuntimeError(f"A13_SMOKE_ALERT: {name} missing or drifted")
  log = (repo_root / audit["runtime_files"]["log"]["path"]).read_text(errors="replace")
  required = (
    "Learning iteration 0/1",
    "Total steps: 384",
    "Linear(in_features=93, out_features=512",
    "Linear(in_features=960, out_features=512",
    "procedural_joint_noise_level",
    "procedural_orientation_offset",
  )
  fatal = (
    "Traceback",
    "CUDA out of memory",
    "OutOfMemoryError",
    "PHYSICAL_RESET_ALERT",
  )
  if any(fragment not in log for fragment in required) or any(
    fragment in log for fragment in fatal
  ):
    raise RuntimeError("A13_SMOKE_ALERT: runtime log contract failed")
  checkpoint = repo_root / audit["runtime_files"]["checkpoint"]["path"]
  integrity = _validate_checkpoint(checkpoint, expected_iteration=0)
  verified = audit["verified"]
  if (
    integrity["tensor_count"] != verified["checkpoint_tensor_count"]
    or integrity["tensor_elements"] != verified["checkpoint_tensor_elements"]
  ):
    raise RuntimeError("A13_SMOKE_ALERT: checkpoint tensor inventory drifted")
  return {"status": audit["status"], "code_commit": audit["code_commit"]}


def build_plan(cfg: ContinuousResetFinetuneCfg) -> dict[str, Any]:
  repo_root = Path(__file__).resolve().parents[1]
  protocol_path = (
    cfg.protocol if cfg.protocol.is_absolute() else repo_root / cfg.protocol
  )
  protocol, protocol_sha = _validate_protocol(protocol_path)
  commit = _git(repo_root, "rev-parse", "HEAD")
  _git(repo_root, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  from mjlab.tasks.registry import load_runner_cls

  import smp.rl.tasks  # noqa: F401

  runner = load_runner_cls(_TASK)
  if runner is None or runner.__name__ != _EXPECTED_RUNNER:
    raise RuntimeError("A13 task lacks the fresh-optimizer warm-start runner")
  source = protocol["source_policy"]
  source_checkpoint = (repo_root / source["checkpoint_path"]).resolve()
  if not source_checkpoint.is_file() or _sha256(source_checkpoint) != _SOURCE_SHA256:
    raise RuntimeError("A13_SOURCE_ALERT: checkpoint missing or drifted")
  source_integrity = _validate_checkpoint(source_checkpoint, expected_iteration=3500)
  training = protocol["training_protocol"]
  source_name = "a13_source_a12_seed20261401_gate3500"
  experiment_root = repo_root / "logs/rsl_rl/smp_scratch_a13_f2s2_continuous_reset_g1"
  source_link = experiment_root / source_name / source["checkpoint_name"]
  run_name = "a13_continuous_reset_5k_seed20261501_offline_retry_after_auth"
  command = [
    str(repo_root / ".venv/bin/python"),
    "scripts/train.py",
    _TASK,
    "--env.scene.num-envs",
    str(training["num_envs"]),
    "--agent.seed",
    str(training["policy_seed"]),
    "--env.seed",
    str(training["environment_seed"]),
    "--agent.resume",
    "True",
    "--agent.load-run",
    f"^{source_name}$",
    "--agent.load-checkpoint",
    f"^{source['checkpoint_name']}$",
    "--agent.max-iterations",
    str(training["max_iterations"]),
    "--agent.save-interval",
    str(training["save_interval"]),
    "--agent.algorithm.learning-rate",
    str(training["learning_rate"]),
    "--agent.run-name",
    run_name,
  ]
  material = {
    "protocol_sha256": protocol_sha,
    "code_commit": commit,
    "task": _TASK,
    "source_checkpoint_sha256": _SOURCE_SHA256,
    "policy_seed": training["policy_seed"],
    "environment_seed": training["environment_seed"],
    "command": command,
    "wandb_mode": training["wandb_mode"],
  }
  plan_id = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  control_dir = (
    cfg.control_dir if cfg.control_dir.is_absolute() else repo_root / cfg.control_dir
  )
  return {
    "schema_version": 1,
    "status": "PLANNED",
    "study_id": protocol["study_id"],
    "study_role": protocol["study_role"],
    "plan_id": plan_id,
    "protocol": str(protocol_path.resolve()),
    "protocol_sha256": protocol_sha,
    "code_commit": commit,
    "task": _TASK,
    "runner": _EXPECTED_RUNNER,
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_sha256": _SOURCE_SHA256,
    "source_checkpoint_integrity": source_integrity,
    "source_link": str(source_link),
    "experiment_root": str(experiment_root),
    "policy_seed": training["policy_seed"],
    "environment_seed": training["environment_seed"],
    "num_envs": training["num_envs"],
    "max_iterations": training["max_iterations"],
    "save_interval": training["save_interval"],
    "evaluation_gates": training["evaluation_gates"],
    "learning_rate": training["learning_rate"],
    "wandb_mode": training["wandb_mode"],
    "gpu": training["device"],
    "reserved_idle_devices": training["reserved_idle_devices"],
    "run_name": run_name,
    "log": str(control_dir / "gpu0_a13_seed20261501.log"),
    "pid_file": str(control_dir / "gpu0_a13_seed20261501.pid"),
    "command": command,
    "pid": None,
    "claim_boundary": protocol["claim_boundary"],
  }


def launch_finetune(cfg: ContinuousResetFinetuneCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = Path(planned["log"]).parent / "launch_manifest.json"
  if state_path.exists():
    existing = _load_json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing A13 run has a different immutable plan")
    if existing.get("status") == "LAUNCHED":
      pid = existing.get("pid")
      finals = list(
        Path(existing["experiment_root"]).glob(
          f"*_{existing['run_name']}/model_4999.pt"
        )
      )
      if pid is not None and not _pid_alive(int(pid)) and len(finals) != 1:
        raise RuntimeError("A13_TRAINING_ALERT: worker died; no automatic restart")
      return existing
    raise RuntimeError(
      f"A13_TRAINING_ALERT: inadmissible state {existing.get('status')}"
    )
  if not cfg.launch:
    return planned
  repo_root = Path(__file__).resolve().parents[1]
  if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing A13 launch from tracked-dirty worktree")
  processes = _gpu_processes()
  if processes:
    raise RuntimeError(f"refusing A13 launch while GPU compute is active: {processes}")
  protocol = _load_json(Path(planned["protocol"]))
  planned["implementation_smoke"] = _validate_smoke(protocol, repo_root)
  planned["resource_preflight"] = _disk_preflight(repo_root)
  source = Path(planned["source_checkpoint"])
  link = Path(planned["source_link"])
  link.parent.mkdir(parents=True, exist_ok=True)
  if link.exists() or link.is_symlink():
    if not link.is_symlink() or link.resolve() != source.resolve():
      raise ValueError(f"conflicting A13 warm-start link: {link}")
  else:
    link.symlink_to(source)
  state_path.parent.mkdir(parents=True, exist_ok=True)
  planned["status"] = "LAUNCHING"
  planned["started_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  environment = os.environ.copy()
  environment["CUDA_VISIBLE_DEVICES"] = str(planned["gpu"])
  environment["WANDB_MODE"] = str(planned["wandb_mode"])
  with Path(planned["log"]).open("a") as stream:
    process = subprocess.Popen(
      planned["command"],
      cwd=repo_root,
      env=environment,
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  planned["pid"] = process.pid
  Path(planned["pid_file"]).write_text(f"{process.pid}\n")
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: ContinuousResetFinetuneCfg) -> None:
  result = launch_finetune(cfg)
  print(f"{result['status']}: plan {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(ContinuousResetFinetuneCfg))
