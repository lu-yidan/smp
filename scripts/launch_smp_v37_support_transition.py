"""Fail-closed launcher for the paired V37 support-transition study."""

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

_PROTOCOL_SHA256 = "834331f230ec9a6232a59b2698d26baaf4ec7ca8d03bd4354270d12d69d55005"
_MINIMUM_COMMIT = "4adef9194d1cd3710c14680bb92d8db83b3caff7"
_EXPECTED_RUNNER = "SmpCurriculumWarmStartRunner"
_GIB = 1024**3

_ARMS = {
  "A": (
    "Smp-Getup-V37-93D-Actuator-G1",
    "smp_getup_v37_93d_actuator_g1",
    False,
  ),
  "B": (
    "Smp-Getup-V37-93D-Support-G1",
    "smp_getup_v37_93d_support_g1",
    True,
  ),
}


@dataclass(frozen=True)
class V37SupportTransitionCfg:
  protocol: Path = Path("docs/v37_93d_support_transition_study_v1.json")
  control_dir: Path = Path("run_control/v37_93d_support_transition_v1/training")
  launch: bool = False


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


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ("git", *args), cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


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


def _tensors(value: Any) -> list[torch.Tensor]:
  if torch.is_tensor(value):
    return [value]
  if isinstance(value, dict):
    return [tensor for item in value.values() for tensor in _tensors(item)]
  if isinstance(value, (tuple, list)):
    return [tensor for item in value for tensor in _tensors(item)]
  return []


def _validate_checkpoint(path: Path, expected_sha: str) -> dict[str, Any]:
  if not path.is_file() or _sha256(path) != expected_sha:
    raise RuntimeError(f"V37_SOURCE_ALERT: missing or drifted checkpoint {path}")
  payload = torch.load(path, map_location="cpu", weights_only=False)
  tensors = _tensors(payload)
  actor_shape = tuple(payload["actor_state_dict"]["mlp.0.weight"].shape)
  critic_shape = tuple(payload["critic_state_dict"]["mlp.0.weight"].shape)
  if (
    payload.get("iter") != 5999
    or actor_shape != (512, 93)
    or critic_shape != (512, 960)
    or not tensors
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError(f"V37_SOURCE_ALERT: invalid checkpoint {path}")
  return {
    "embedded_iteration": 5999,
    "actor_input_dim": 93,
    "critic_input_dim": 960,
    "tensor_count": len(tensors),
    "tensor_elements": sum(tensor.numel() for tensor in tensors),
    "all_tensors_finite": True,
  }


def _validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
  digest = _sha256(path) if path.is_file() else ""
  if digest != _PROTOCOL_SHA256:
    raise ValueError("V37 support-transition protocol SHA-256 mismatch")
  protocol = _json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != "smp-v37-93d-support-transition-study-v1"
    or protocol.get("study_role")
    != "ENGINEERING_SIM2REAL_TRANSITION_STUDY_NOT_RAL_OR_HARDWARE_EVIDENCE"
  ):
    raise ValueError("V37 protocol is not launch eligible")
  expected_arms = [
    {
      "arm": "A",
      "task": _ARMS["A"][0],
      "treatment": "actuator feasibility control",
    },
    {
      "arm": "B",
      "task": _ARMS["B"][0],
      "treatment": (
        "A plus bilateral support transition and seated-trap reset cohort"
      ),
    },
  ]
  if protocol.get("arms") != expected_arms:
    raise ValueError("V37 arm definitions drifted")
  source = protocol["source_policy"]
  if (
    source.get("wandb_run_id") != "2yehj243"
    or source.get("checkpoint_sha256")
    != "5a2c09641ec3389b699f344b50e846a248309ba4777994611838b0b149ba9311"
    or source.get("actor_input_dim") != 93
    or source.get("critic_input_dim") != 960
  ):
    raise ValueError("V37 source policy drifted")
  training = protocol["training_protocol"]
  if (
    training.get("training_seeds") != [20262001, 20262002, 20262003]
    or training.get("num_envs") != 4096
    or training.get("rollout_steps_per_update") != 24
    or training.get("max_iterations") != 6000
    or training.get("save_interval") != 500
    or training.get("learning_rate") != 1e-5
    or training.get("wandb_mode") != "online"
    or not training.get("no_automatic_restart")
  ):
    raise ValueError("V37 training protocol drifted")
  expected_gpu = {
    f"{arm}/{seed}": arm_index * 3 + seed_index
    for arm_index, arm in enumerate(_ARMS)
    for seed_index, seed in enumerate((20262001, 20262002, 20262003))
  }
  if training.get("gpu_assignment") != expected_gpu:
    raise ValueError("V37 GPU assignment drifted")
  return protocol, digest


def _validate_tasks() -> None:
  from mjlab.tasks.registry import load_env_cfg, load_runner_cls

  import smp.rl.tasks  # noqa: F401

  for arm, (task, _, treatment) in _ARMS.items():
    runner = load_runner_cls(task)
    if runner is None or runner.__name__ != _EXPECTED_RUNNER:
      raise RuntimeError(f"V37_CONFIG_ALERT: {task} lacks warm-start runner")
    cfg = load_env_cfg(task)
    if tuple(cfg.observations["actor"].terms) != (
      "base_ang_vel",
      "projected_gravity",
      "joint_pos",
      "joint_vel",
      "actions",
    ):
      raise RuntimeError(f"V37_CONFIG_ALERT: {task} actor is not frozen 93D")
    action = cfg.actions["joint_pos"]
    if (
      action.__class__.__name__ != "RateLimitedJointPositionActionCfg"
      or action.max_target_velocity != 2.5
      or action.max_target_acceleration != 15.0
    ):
      raise RuntimeError(f"V37_CONFIG_ALERT: {task} action envelope drifted")
    if cfg.events["reset_escape_obstacle"].params["obstacle_probability"] != 0.0:
      raise RuntimeError(f"V37_CONFIG_ALERT: {task} plate is active")
    forbidden = {
      "failure_state_replay_reset",
      "record_failure_states",
      "stratified_post_stand_wrench",
      "post_stand_body_wrench",
    }
    if forbidden.intersection(cfg.events):
      raise RuntimeError(f"V37_CONFIG_ALERT: {task} has a forbidden event")
    has_treatment = "photo_informed_seated_trap_reset" in cfg.events
    if has_treatment != treatment:
      raise RuntimeError(f"V37_CONFIG_ALERT: {arm} treatment factor drifted")
    if treatment:
      if cfg.events["update_recovery_stage"].func.__name__ != (
        "update_recovery_stage_with_bilateral_support"
      ):
        raise RuntimeError("V37_CONFIG_ALERT: bilateral stage gate is absent")
      if cfg.events["photo_informed_seated_trap_reset"].params["probability"] != 0.15:
        raise RuntimeError("V37_CONFIG_ALERT: trap probability drifted")


def _resource_preflight(repo: Path) -> dict[str, Any]:
  usage = shutil.disk_usage(repo)
  stats = os.statvfs(repo)
  free_gib = usage.free / _GIB
  inode_free = stats.f_favail / stats.f_files if stats.f_files else 0.0
  if free_gib < 100.0 or inode_free < 0.10:
    raise RuntimeError(
      f"DISK_SPACE_ALERT: free_gib={free_gib:.1f}, inode_free_fraction={inode_free:.3f}"
    )
  inventory_rows = subprocess.run(
    ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"),
    check=True,
    capture_output=True,
    text=True,
  ).stdout.splitlines()
  inventory = {
    int(index.strip()): uuid.strip()
    for index, uuid in (row.split(",", maxsplit=1) for row in inventory_rows)
  }
  if set(inventory) != set(range(8)):
    raise RuntimeError(f"V37_RESOURCE_ALERT: expected GPUs 0--7, got {inventory}")
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
  active = [row.strip() for row in active if row.strip()]
  if active:
    raise RuntimeError(f"V37_RESOURCE_ALERT: GPU compute is active: {active}")
  return {
    "free_gib": free_gib,
    "inode_free_fraction": inode_free,
    "gpu_uuid_by_physical_index": inventory,
    "gpu_compute_processes": [],
  }


def build_plan(cfg: V37SupportTransitionCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo / cfg.protocol
  protocol, protocol_sha = _validate_protocol(protocol_path)
  commit = _git(repo, "rev-parse", "HEAD")
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  _validate_tasks()
  source_row = protocol["source_policy"]
  source = (repo / source_row["checkpoint_path"]).resolve()
  source_audit = {
    "path": str(source),
    "sha256": source_row["checkpoint_sha256"],
    "wandb_run_id": source_row["wandb_run_id"],
    "integrity": _validate_checkpoint(source, source_row["checkpoint_sha256"]),
  }
  training = protocol["training_protocol"]
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  jobs = []
  for arm, (task, experiment, _) in _ARMS.items():
    source_name = "v37_source_v36_sd_seed20261902_gate5999"
    source_link = repo / "logs/rsl_rl" / experiment / source_name / "model_5999.pt"
    for seed in training["training_seeds"]:
      gpu = training["gpu_assignment"][f"{arm}/{seed}"]
      run_name = f"v37_{arm.lower()}_6k_seed{seed}_online"
      command = [
        str(repo / ".venv/bin/python"),
        "scripts/train.py",
        task,
        "--env.scene.num-envs",
        str(training["num_envs"]),
        "--agent.seed",
        str(seed),
        "--env.seed",
        str(seed),
        "--agent.resume",
        "True",
        "--agent.load-run",
        f"^{source_name}$",
        "--agent.load-checkpoint",
        "^model_5999.pt$",
        "--agent.max-iterations",
        str(training["max_iterations"]),
        "--agent.save-interval",
        str(training["save_interval"]),
        "--agent.algorithm.learning-rate",
        str(training["learning_rate"]),
        "--agent.run-name",
        run_name,
      ]
      jobs.append(
        {
          "arm": arm,
          "task": task,
          "experiment": experiment,
          "seed": seed,
          "gpu": gpu,
          "run_name": run_name,
          "source_checkpoint": str(source),
          "source_sha256": source_row["checkpoint_sha256"],
          "source_link": str(source_link),
          "log": str(control / f"gpu{gpu}_{arm.lower()}_seed{seed}.log"),
          "pid_file": str(control / f"gpu{gpu}_{arm.lower()}_seed{seed}.pid"),
          "command": command,
          "pid": None,
        }
      )
  material = {
    "protocol_sha256": protocol_sha,
    "code_commit": commit,
    "source": source_audit,
    "jobs": jobs,
  }
  plan_id = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  return {
    "schema_version": 1,
    "status": "PLANNED",
    "study_id": protocol["study_id"],
    "study_role": protocol["study_role"],
    "plan_id": plan_id,
    "protocol": str(protocol_path.resolve()),
    "protocol_sha256": protocol_sha,
    "code_commit": commit,
    "source_audit": source_audit,
    "max_iterations": training["max_iterations"],
    "save_interval": training["save_interval"],
    "evaluation_gates": training["evaluation_gates"],
    "wandb_mode": training["wandb_mode"],
    "jobs": jobs,
    "claim_boundary": protocol["claim_boundary"],
  }


def launch(cfg: V37SupportTransitionCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  control = Path(planned["jobs"][0]["log"]).parent
  state_path = control / "launch_manifest.json"
  if state_path.exists():
    existing = _json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing V37 run has a different immutable plan")
    if existing.get("status") != "LAUNCHED":
      raise RuntimeError(f"V37_TRAINING_ALERT: state {existing.get('status')}")
    repo = Path(__file__).resolve().parents[1]
    for job in existing["jobs"]:
      finals = list(
        (repo / "logs/rsl_rl" / job["experiment"]).glob(
          f"*_{job['run_name']}/model_5999.pt"
        )
      )
      if not _pid_alive(int(job["pid"])) and len(finals) != 1:
        raise RuntimeError(
          f"V37_TRAINING_ALERT: {job['arm']}/{job['seed']} died; no restart"
        )
    return existing
  if not cfg.launch:
    return planned

  repo = Path(__file__).resolve().parents[1]
  if _git(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing V37 launch from tracked-dirty worktree")
  planned["resource_preflight"] = _resource_preflight(repo)
  for job in planned["jobs"]:
    source = Path(job["source_checkpoint"])
    link = Path(job["source_link"])
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
      if not link.is_symlink() or link.resolve() != source:
        raise RuntimeError(f"V37_SOURCE_ALERT: conflicting link {link}")
    else:
      link.symlink_to(source)

  control.mkdir(parents=True, exist_ok=True)
  planned["status"] = "LAUNCHING"
  planned["started_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  try:
    for job in planned["jobs"]:
      environment = os.environ.copy()
      environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
      environment["WANDB_MODE"] = planned["wandb_mode"]
      with Path(job["log"]).open("a") as stream:
        process = subprocess.Popen(
          job["command"],
          cwd=repo,
          env=environment,
          stdout=stream,
          stderr=subprocess.STDOUT,
          start_new_session=True,
        )
      job["pid"] = process.pid
      Path(job["pid_file"]).write_text(f"{process.pid}\n")
      _atomic_json(state_path, planned)
  except Exception:
    planned["status"] = "V37_TRAINING_ALERT_PARTIAL_LAUNCH"
    planned["alert_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, planned)
    raise
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: V37SupportTransitionCfg) -> None:
  result = launch(cfg)
  print(
    f"{result['status']}: plan {result['plan_id']} "
    f"pids={[job['pid'] for job in result['jobs']]}"
  )


if __name__ == "__main__":
  tyro.cli(main)
