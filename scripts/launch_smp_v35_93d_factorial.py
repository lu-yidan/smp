"""Fail-closed launcher for the V35 93D reset/terrain/wrench study."""

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

_PROTOCOL_SHA256 = "b8536e03df6d87b61a9332c45fbe690b84e3f02fdd03ce7d2be9b1cef5bc1ca0"
_MINIMUM_COMMIT = "ccd22575e683f327c939ad0b3ce795c3a5b91925"
_SOURCE_SHA256 = "471fb41444a5d8f2bce1f93b0e2d65a613e86c1d417ef4cba36ba48b9779474b"
_EXPECTED_RUNNER = "SmpCurriculumWarmStartRunner"
_GIB = 1024**3

_ARMS = {
  "R": (
    "Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-G1",
    "smp_getup_escape_plate_v35_93d_reset_stability_g1",
  ),
  "RD": (
    "Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-Wrench-G1",
    "smp_getup_escape_plate_v35_93d_reset_stability_wrench_g1",
  ),
  "RT": (
    "Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-Terrain-G1",
    "smp_getup_escape_plate_v35_93d_reset_stability_terrain_g1",
  ),
  "RTD": (
    "Smp-Getup-Escape-Plate-V35-93D-Reset-Stability-Terrain-Wrench-G1",
    "smp_getup_escape_plate_v35_93d_reset_stability_terrain_wrench_g1",
  ),
}


@dataclass(frozen=True)
class V35FactorialCfg:
  protocol: Path = Path("docs/ral_v34_93d_reset_stability_finetune_v1.json")
  control_dir: Path = Path("run_control/v35_93d_factorial_v1/training")
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


def _validate_checkpoint(path: Path, iteration: int) -> dict[str, Any]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  tensors = _tensors(payload)
  actor_shape = tuple(payload["actor_state_dict"]["mlp.0.weight"].shape)
  critic_shape = tuple(payload["critic_state_dict"]["mlp.0.weight"].shape)
  if (
    payload.get("iter") != iteration
    or actor_shape != (512, 93)
    or critic_shape != (512, 960)
    or not tensors
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError(f"V35_CHECKPOINT_ALERT: invalid checkpoint {path}")
  return {
    "embedded_iteration": iteration,
    "actor_input_dim": 93,
    "critic_input_dim": 960,
    "tensor_count": len(tensors),
    "tensor_elements": sum(tensor.numel() for tensor in tensors),
    "all_tensors_finite": True,
  }


def _validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
  digest = _sha256(path) if path.is_file() else ""
  if digest != _PROTOCOL_SHA256:
    raise ValueError("V35 factorial protocol SHA-256 mismatch")
  protocol = _json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != "smp-v35-93d-reset-terrain-wrench-factorial-v1"
    or protocol.get("study_role")
    != "ENGINEERING_2X2_FINETUNE_NOT_PROMOTION_OR_RAL_EVIDENCE"
  ):
    raise ValueError("V35 protocol is not launch eligible")
  source = protocol["source_policy"]
  if (
    source.get("checkpoint_sha256") != _SOURCE_SHA256
    or source.get("embedded_iteration") != 6000
    or source.get("actor_input_dim") != 93
    or source.get("critic_input_dim") != 960
  ):
    raise ValueError("V35 source policy drifted")
  expected_arms = [
    {"arm": "R", "task": _ARMS["R"][0], "terrain": False, "physical_wrench": False},
    {"arm": "RD", "task": _ARMS["RD"][0], "terrain": False, "physical_wrench": True},
    {"arm": "RT", "task": _ARMS["RT"][0], "terrain": True, "physical_wrench": False},
    {"arm": "RTD", "task": _ARMS["RTD"][0], "terrain": True, "physical_wrench": True},
  ]
  if protocol.get("arms") != expected_arms:
    raise ValueError("V35 arm definitions drifted")
  expected_training = {
    "fine_tune_seeds": [20261801, 20261802],
    "agent_and_environment_seed_identical": True,
    "gpu_assignment": {
      "R/20261801": 0,
      "R/20261802": 1,
      "RD/20261801": 2,
      "RD/20261802": 3,
      "RT/20261801": 4,
      "RT/20261802": 5,
      "RTD/20261801": 6,
      "RTD/20261802": 7,
    },
    "warm_start": "same_v34_93d_actor_critic_and_normalizers_only",
    "reset_optimizer_iteration_and_environment_steps": True,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 6000,
    "save_interval": 500,
    "evaluation_gates": [0, 500, 1000, 2000, 3500, 5000, 5999],
    "learning_rate": 1e-5,
    "wandb_mode": "online",
    "no_automatic_restart": True,
  }
  if protocol.get("training_protocol") != expected_training:
    raise ValueError("V35 training protocol drifted")
  audit = protocol.get("implementation_audit", {})
  if (
    audit.get("status") != "PASSED_FOUR_REAL_MUJOCO_WARM_START_SMOKES"
    or audit.get("code_commit") != _MINIMUM_COMMIT
    or audit.get("source_checkpoint_sha256") != _SOURCE_SHA256
  ):
    raise ValueError("V35 implementation audit drifted")
  return protocol, digest


def _validate_smokes(protocol: dict[str, Any], repo: Path) -> dict[str, Any]:
  audit = protocol["implementation_audit"]
  fatal = re.compile(
    r"traceback|cuda out of memory|outofmemoryerror|physical_reset_alert|"
    r"(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)",
    re.IGNORECASE | re.MULTILINE,
  )
  for arm, files in audit["arms"].items():
    for row in files.values():
      path = repo / row["path"]
      if not path.is_file() or _sha256(path) != row["sha256"]:
        raise RuntimeError(f"V35_SMOKE_ALERT: {arm} artifact missing or drifted")
    log = (repo / files["log"]["path"]).read_text(errors="replace")
    required = (
      "Learning iteration 0/1",
      "Total steps: 384",
      "Linear(in_features=93, out_features=512",
      "Linear(in_features=960, out_features=512",
      "curriculum_validated_fall_reset",
    )
    arm_events = {
      "R": (),
      "RD": ("stratified_post_stand_wrench",),
      "RT": ("ground_fall_on_training_terrain",),
      "RTD": ("ground_fall_on_training_terrain", "stratified_post_stand_wrench"),
    }
    if fatal.search(log) or any(token not in log for token in (*required, *arm_events[arm])):
      raise RuntimeError(f"V35_SMOKE_ALERT: {arm} log contract failed")
    integrity = _validate_checkpoint(repo / files["checkpoint"]["path"], 0)
    if integrity["tensor_count"] != 73 or integrity["tensor_elements"] != 2620640:
      raise RuntimeError(f"V35_SMOKE_ALERT: {arm} tensor inventory drifted")
  return {"status": audit["status"], "code_commit": audit["code_commit"]}


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
    raise RuntimeError(f"V35_RESOURCE_ALERT: expected physical GPUs 0--7, got {inventory}")
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
    raise RuntimeError(f"V35_RESOURCE_ALERT: GPU compute is active: {active}")
  return {
    "free_gib": free_gib,
    "inode_free_fraction": inode_free,
    "gpu_uuid_by_physical_index": inventory,
    "gpu_compute_processes": [],
  }


def build_plan(cfg: V35FactorialCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo / cfg.protocol
  protocol, protocol_sha = _validate_protocol(protocol_path)
  commit = _git(repo, "rev-parse", "HEAD")
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  source = (repo / protocol["source_policy"]["checkpoint_path"]).resolve()
  if not source.is_file() or _sha256(source) != _SOURCE_SHA256:
    raise RuntimeError("V35_SOURCE_ALERT: checkpoint missing or drifted")
  source_integrity = _validate_checkpoint(source, 6000)
  from mjlab.tasks.registry import load_runner_cls

  import smp.rl.tasks  # noqa: F401

  training = protocol["training_protocol"]
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  jobs = []
  for arm, (task, experiment) in _ARMS.items():
    runner = load_runner_cls(task)
    if runner is None or runner.__name__ != _EXPECTED_RUNNER:
      raise RuntimeError(f"V35 task {task} lacks the warm-start runner")
    source_name = "v35_source_v34_93d_seed20261701_gate6000"
    source_link = repo / "logs/rsl_rl" / experiment / source_name / "model_6000.pt"
    for seed in training["fine_tune_seeds"]:
      gpu = training["gpu_assignment"][f"{arm}/{seed}"]
      run_name = f"v35_{arm.lower()}_6k_seed{seed}_online"
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
        "^model_6000.pt$",
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
    "source_checkpoint_sha256": _SOURCE_SHA256,
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
    "source_checkpoint": str(source),
    "source_checkpoint_sha256": _SOURCE_SHA256,
    "source_checkpoint_integrity": source_integrity,
    "max_iterations": training["max_iterations"],
    "save_interval": training["save_interval"],
    "evaluation_gates": training["evaluation_gates"],
    "wandb_mode": training["wandb_mode"],
    "jobs": jobs,
    "claim_boundary": protocol["claim_boundary"],
  }


def launch(cfg: V35FactorialCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  control = Path(planned["jobs"][0]["log"]).parent
  state_path = control / "launch_manifest.json"
  if state_path.exists():
    existing = _json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing V35 run has a different immutable plan")
    if existing.get("status") != "LAUNCHED":
      raise RuntimeError(
        f"V35_TRAINING_ALERT: inadmissible state {existing.get('status')}"
      )
    for job in existing["jobs"]:
      final = list(
        (Path(__file__).resolve().parents[1] / "logs/rsl_rl" / job["experiment"]).glob(
          f"*_{job['run_name']}/model_5999.pt"
        )
      )
      if not _pid_alive(int(job["pid"])) and len(final) != 1:
        raise RuntimeError(
          f"V35_TRAINING_ALERT: {job['arm']}/{job['seed']} died; no automatic restart"
        )
    return existing
  if not cfg.launch:
    return planned
  repo = Path(__file__).resolve().parents[1]
  if _git(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing V35 launch from tracked-dirty worktree")
  protocol = _json(Path(planned["protocol"]))
  planned["implementation_smoke"] = _validate_smokes(protocol, repo)
  planned["resource_preflight"] = _resource_preflight(repo)
  source = Path(planned["source_checkpoint"])
  for job in planned["jobs"]:
    link = Path(job["source_link"])
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
      if not link.is_symlink() or link.resolve() != source:
        raise RuntimeError(f"V35_SOURCE_ALERT: conflicting warm-start link {link}")
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
    planned["status"] = "V35_TRAINING_ALERT_PARTIAL_LAUNCH"
    planned["alert_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, planned)
    raise
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: V35FactorialCfg) -> None:
  result = launch(cfg)
  pids = [job["pid"] for job in result["jobs"]]
  print(f"{result['status']}: plan {result['plan_id']} pids={pids}")


if __name__ == "__main__":
  tyro.cli(main)
