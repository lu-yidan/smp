"""Fail-closed launcher for the V36 safe mixed-versus-stairs continuation."""

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

_PROTOCOL_SHA256 = "b829e243e6f04d0c6d28c2996e28ebc443c9b692f5064bacfcc280133c382ba2"
_MINIMUM_COMMIT = "81b031c7907dd009e4ed686eaec9963a54ced447"
_EXPECTED_RUNNER = "SmpCurriculumWarmStartRunner"
_GIB = 1024**3

_ARMS = {
  "M": (
    "Smp-Getup-V36-93D-Safe-Mixed-G1",
    "smp_getup_v36_93d_safe_mixed_g1",
    ("flat", "slope", "stairs", "rough"),
    False,
  ),
  "MD": (
    "Smp-Getup-V36-93D-Safe-Mixed-Wrench-G1",
    "smp_getup_v36_93d_safe_mixed_wrench_g1",
    ("flat", "slope", "stairs", "rough"),
    True,
  ),
  "S": (
    "Smp-Getup-V36-93D-Safe-Stairs-G1",
    "smp_getup_v36_93d_safe_stairs_g1",
    ("stairs",),
    False,
  ),
  "SD": (
    "Smp-Getup-V36-93D-Safe-Stairs-Wrench-G1",
    "smp_getup_v36_93d_safe_stairs_wrench_g1",
    ("stairs",),
    True,
  ),
}


@dataclass(frozen=True)
class V36SafeTerrainCfg:
  protocol: Path = Path("docs/v36_93d_safe_terrain_stairs_study_v1.json")
  control_dir: Path = Path("run_control/v36_93d_safe_terrain_study_v1/training")
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
    raise RuntimeError(f"V36_SOURCE_ALERT: missing or drifted checkpoint {path}")
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
    raise RuntimeError(f"V36_SOURCE_ALERT: invalid checkpoint {path}")
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
    raise ValueError("V36 safe-terrain protocol SHA-256 mismatch")
  protocol = _json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != "smp-v36-93d-safe-terrain-stairs-study-v1"
    or protocol.get("study_role")
    != "ENGINEERING_TERRAIN_CONTINUATION_NOT_PROMOTION_OR_RAL_EVIDENCE"
  ):
    raise ValueError("V36 safe-terrain protocol is not launch eligible")
  sources = protocol.get("source_policies")
  if not isinstance(sources, list) or len(sources) != 2:
    raise ValueError("V36 safe-terrain protocol requires two paired sources")
  by_seed = {int(row["training_seed"]): row for row in sources}
  if set(by_seed) != {20261901, 20261902}:
    raise ValueError("V36 safe-terrain source seed mapping drifted")
  expected_arms = [
    {
      "arm": arm,
      "task": task,
      "terrain_family": "mixed" if len(names) == 4 else "stairs_only",
      "continued_post_stand_wrench": wrench,
    }
    for arm, (task, _, names, wrench) in _ARMS.items()
  ]
  if protocol.get("arms") != expected_arms:
    raise ValueError("V36 safe-terrain arm definitions drifted")
  training = protocol["training_protocol"]
  if (
    training.get("training_seeds") != [20261901, 20261902]
    or training.get("num_envs") != 4096
    or training.get("rollout_steps_per_update") != 24
    or training.get("max_iterations") != 6000
    or training.get("save_interval") != 500
    or training.get("learning_rate") != 1e-5
    or training.get("wandb_mode") != "online"
    or not training.get("no_automatic_restart")
  ):
    raise ValueError("V36 safe-terrain training protocol drifted")
  expected_gpu = {
    f"{arm}/{seed}": arm_index * 2 + seed_index
    for arm_index, arm in enumerate(_ARMS)
    for seed_index, seed in enumerate((20261901, 20261902))
  }
  if training.get("gpu_assignment") != expected_gpu:
    raise ValueError("V36 safe-terrain GPU assignment drifted")
  return protocol, digest


def _validate_tasks() -> None:
  from mjlab.tasks.registry import load_env_cfg, load_runner_cls

  import smp.rl.tasks  # noqa: F401

  for arm, (task, _, expected_names, wrench) in _ARMS.items():
    runner = load_runner_cls(task)
    if runner is None or runner.__name__ != _EXPECTED_RUNNER:
      raise RuntimeError(f"V36_CONFIG_ALERT: {task} lacks warm-start runner")
    cfg = load_env_cfg(task)
    if tuple(cfg.observations["actor"].terms) != (
      "base_ang_vel",
      "projected_gravity",
      "joint_pos",
      "joint_vel",
      "actions",
    ):
      raise RuntimeError(f"V36_CONFIG_ALERT: {task} actor is not frozen 93D")
    generator = cfg.scene.terrain.terrain_generator
    if generator is None or tuple(generator.sub_terrains) != expected_names:
      raise RuntimeError(f"V36_CONFIG_ALERT: {task} terrain family drifted")
    if cfg.events["reset_escape_obstacle"].params["obstacle_probability"] != 0.0:
      raise RuntimeError(f"V36_CONFIG_ALERT: {task} plate is active")
    if "failure_state_replay_reset" in cfg.events:
      raise RuntimeError(f"V36_CONFIG_ALERT: {task} failure replay is active")
    required = {
      "ground_fall_on_training_terrain",
      "validate_training_terrain_contact",
    }
    if not required.issubset(cfg.events):
      raise RuntimeError(f"V36_CONFIG_ALERT: {task} reset audit is missing")
    has_wrench = "stratified_post_stand_wrench" in cfg.events
    if has_wrench != wrench:
      raise RuntimeError(f"V36_CONFIG_ALERT: {arm} wrench factor drifted")


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
    raise RuntimeError(f"V36_RESOURCE_ALERT: expected GPUs 0--7, got {inventory}")
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
    raise RuntimeError(f"V36_RESOURCE_ALERT: GPU compute is active: {active}")
  return {
    "free_gib": free_gib,
    "inode_free_fraction": inode_free,
    "gpu_uuid_by_physical_index": inventory,
    "gpu_compute_processes": [],
  }


def build_plan(cfg: V36SafeTerrainCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo / cfg.protocol
  protocol, protocol_sha = _validate_protocol(protocol_path)
  commit = _git(repo, "rev-parse", "HEAD")
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  _validate_tasks()

  source_by_seed = {
    int(row["training_seed"]): row for row in protocol["source_policies"]
  }
  source_audits = {}
  for seed, row in source_by_seed.items():
    source = (repo / row["checkpoint_path"]).resolve()
    source_audits[str(seed)] = {
      "path": str(source),
      "sha256": row["checkpoint_sha256"],
      "integrity": _validate_checkpoint(source, row["checkpoint_sha256"]),
    }

  training = protocol["training_protocol"]
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  jobs = []
  for arm, (task, experiment, _, _) in _ARMS.items():
    for seed in training["training_seeds"]:
      source_row = source_by_seed[seed]
      source_name = f"v36_source_v35_rd_seed{source_row['source_seed']}_gate5999"
      source_link = repo / "logs/rsl_rl" / experiment / source_name / "model_5999.pt"
      gpu = training["gpu_assignment"][f"{arm}/{seed}"]
      run_name = f"v36_{arm.lower()}_6k_seed{seed}_online"
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
          "source_checkpoint": source_audits[str(seed)]["path"],
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
    "sources": source_audits,
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
    "source_audits": source_audits,
    "max_iterations": training["max_iterations"],
    "save_interval": training["save_interval"],
    "evaluation_gates": training["evaluation_gates"],
    "wandb_mode": training["wandb_mode"],
    "jobs": jobs,
    "claim_boundary": protocol["claim_boundary"],
  }


def launch(cfg: V36SafeTerrainCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  control = Path(planned["jobs"][0]["log"]).parent
  state_path = control / "launch_manifest.json"
  if state_path.exists():
    existing = _json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing V36 run has a different immutable plan")
    if existing.get("status") != "LAUNCHED":
      raise RuntimeError(
        f"V36_TRAINING_ALERT: inadmissible state {existing.get('status')}"
      )
    repo = Path(__file__).resolve().parents[1]
    for job in existing["jobs"]:
      finals = list(
        (repo / "logs/rsl_rl" / job["experiment"]).glob(
          f"*_{job['run_name']}/model_5999.pt"
        )
      )
      if not _pid_alive(int(job["pid"])) and len(finals) != 1:
        raise RuntimeError(
          f"V36_TRAINING_ALERT: {job['arm']}/{job['seed']} died; "
          "no automatic restart"
        )
    return existing
  if not cfg.launch:
    return planned

  repo = Path(__file__).resolve().parents[1]
  if _git(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing V36 launch from tracked-dirty worktree")
  planned["resource_preflight"] = _resource_preflight(repo)
  for job in planned["jobs"]:
    source = Path(job["source_checkpoint"])
    link = Path(job["source_link"])
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
      if not link.is_symlink() or link.resolve() != source:
        raise RuntimeError(f"V36_SOURCE_ALERT: conflicting warm-start link {link}")
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
    planned["status"] = "V36_TRAINING_ALERT_PARTIAL_LAUNCH"
    planned["alert_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, planned)
    raise
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: V36SafeTerrainCfg) -> None:
  result = launch(cfg)
  pids = [job["pid"] for job in result["jobs"]]
  print(f"{result['status']}: plan {result['plan_id']} pids={pids}")


if __name__ == "__main__":
  tyro.cli(main)
