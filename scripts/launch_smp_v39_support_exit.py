"""Fail-closed launcher for the paired V39 support-exit study."""

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

_PROTOCOL_SHA256 = "08b2535c39dcaf0448a01353132c0f295ee3452b7cac6b17ebfe8fa76a036e6d"
_MINIMUM_COMMIT = "a046a6bc35af3602f9349d80fd0550fef4ec9503"
_EXPECTED_RUNNER = "SmpCurriculumWarmStartRunner"
_SEED_MAP = {20262201: 20262101, 20262202: 20262102, 20262203: 20262103}
_ARMS = {
  "A": ("Smp-Getup-V39-93D-Control-G1", "smp_getup_v39_93d_control_g1"),
  "B": (
    "Smp-Getup-V39-93D-Support-Exit-G1",
    "smp_getup_v39_93d_support_exit_g1",
  ),
}
_GIB = 1024**3


@dataclass(frozen=True)
class V39SupportExitCfg:
  protocol: Path = Path("docs/v39_93d_support_exit_study_v1.json")
  control_dir: Path = Path("run_control/v39_93d_support_exit_v1/training")
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
    raise RuntimeError(f"V39_SOURCE_ALERT: missing or drifted checkpoint {path}")
  payload = torch.load(path, map_location="cpu", weights_only=False)
  tensors = _tensors(payload)
  actor_shape = tuple(payload["actor_state_dict"]["mlp.0.weight"].shape)
  critic_shape = tuple(payload["critic_state_dict"]["mlp.0.weight"].shape)
  if (
    payload.get("iter") != 7999
    or actor_shape != (512, 93)
    or critic_shape != (512, 960)
    or not tensors
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError(f"V39_SOURCE_ALERT: invalid checkpoint {path}")
  return {
    "embedded_iteration": 7999,
    "actor_input_dim": 93,
    "critic_input_dim": 960,
    "tensor_count": len(tensors),
    "tensor_elements": sum(tensor.numel() for tensor in tensors),
    "all_tensors_finite": True,
  }


def _validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
  digest = _sha256(path) if path.is_file() else ""
  if digest != _PROTOCOL_SHA256:
    raise ValueError("V39 support-exit protocol SHA-256 mismatch")
  protocol = _json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != "smp-v39-93d-support-exit-v1"
    or protocol.get("study_role")
    != "ENGINEERING_SIM2REAL_SUPPORT_EXIT_STUDY_NOT_RAL_OR_HARDWARE_EVIDENCE"
  ):
    raise ValueError("V39 protocol is not launch eligible")
  expected_arms = [
    {
      "arm": "A",
      "task": _ARMS["A"][0],
      "treatment": "exact V38-B environment continuation control",
    },
    {
      "arm": "B",
      "task": _ARMS["B"][0],
      "treatment": (
        "A plus time-limited post-waypoint hand/knee support, pure best-height "
        "progress, and balanced two-foot transfer with support release"
      ),
    },
  ]
  if protocol.get("arms") != expected_arms:
    raise ValueError("V39 arm definitions drifted")
  sources = protocol.get("source_policies")
  if not isinstance(sources, list) or len(sources) != 3:
    raise ValueError("V39 requires exactly three V38-B source policies")
  if {row["source_policy_seed"] for row in sources} != set(_SEED_MAP.values()):
    raise ValueError("V39 source seed set drifted")
  training = protocol["training_protocol"]
  expected_gpu = {
    f"{arm}/{seed}": arm_index * 3 + seed_index
    for arm_index, arm in enumerate(_ARMS)
    for seed_index, seed in enumerate(_SEED_MAP)
  }
  if (
    training.get("training_seeds") != list(_SEED_MAP)
    or training.get("gpu_assignment") != expected_gpu
    or training.get("num_envs") != 4096
    or training.get("rollout_steps_per_update") != 24
    or training.get("max_iterations") != 12000
    or training.get("save_interval") != 500
    or training.get("learning_rate") != 1e-5
    or training.get("wandb_mode") != "online"
    or not training.get("no_automatic_restart")
  ):
    raise ValueError("V39 training protocol drifted")
  outcomes = protocol["evaluation_protocol"]["required_outcomes"]
  if not any("maximum head height" in outcome for outcome in outcomes):
    raise ValueError("V39 failure telemetry maxima are not preregistered")
  return protocol, digest


def _validate_tasks() -> None:
  from mjlab.tasks.registry import load_env_cfg, load_runner_cls

  import smp.rl.tasks  # noqa: F401

  for arm, (task, _) in _ARMS.items():
    runner = load_runner_cls(task)
    if runner is None or runner.__name__ != _EXPECTED_RUNNER:
      raise RuntimeError(f"V39_CONFIG_ALERT: {task} lacks warm-start runner")
    cfg = load_env_cfg(task)
    if tuple(cfg.observations["actor"].terms) != (
      "base_ang_vel",
      "projected_gravity",
      "joint_pos",
      "joint_vel",
      "actions",
    ):
      raise RuntimeError(f"V39_CONFIG_ALERT: {task} actor is not frozen 93D")
    if cfg.episode_length_s != 20.0:
      raise RuntimeError(f"V39_CONFIG_ALERT: {task} horizon drifted")
    action = cfg.actions["joint_pos"]
    if (
      action.__class__.__name__ != "RateLimitedJointPositionActionCfg"
      or action.max_target_velocity != 2.5
      or action.max_target_acceleration != 15.0
    ):
      raise RuntimeError(f"V39_CONFIG_ALERT: {task} action envelope drifted")
    has_treatment = "update_v39_support_exit" in cfg.events
    if has_treatment != (arm == "B"):
      raise RuntimeError(f"V39_CONFIG_ALERT: {arm} treatment factor drifted")
    if arm == "B":
      params = cfg.events["update_v39_support_exit"].params
      if params.get("grace_steps") != 60 or params.get("full_penalty_steps") != 180:
        raise RuntimeError("V39_CONFIG_ALERT: support-exit timing drifted")
      expected = {
        "v39_support_dwell": -0.24,
        "v39_head_best_progress": 0.55,
        "v39_foot_transfer": 0.35,
      }
      if any(cfg.rewards[name].weight != weight for name, weight in expected.items()):
        raise RuntimeError("V39_CONFIG_ALERT: support-exit reward drifted")


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
    raise RuntimeError(f"V39_RESOURCE_ALERT: expected GPUs 0--7, got {inventory}")
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
    raise RuntimeError(f"V39_RESOURCE_ALERT: GPU compute is active: {active}")
  return {
    "free_gib": free_gib,
    "inode_free_fraction": inode_free,
    "gpu_uuid_by_physical_index": inventory,
    "gpu_compute_processes": [],
  }


def build_plan(cfg: V39SupportExitCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo / cfg.protocol
  protocol, protocol_sha = _validate_protocol(protocol_path)
  commit = _git(repo, "rev-parse", "HEAD")
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  _validate_tasks()
  source_rows = {row["source_policy_seed"]: row for row in protocol["source_policies"]}
  source_audits = {}
  for source_seed, row in source_rows.items():
    source = (repo / row["checkpoint_path"]).resolve()
    source_audits[str(source_seed)] = {
      "path": str(source),
      "sha256": row["checkpoint_sha256"],
      "wandb_run_id": row["wandb_run_id"],
      "integrity": _validate_checkpoint(source, row["checkpoint_sha256"]),
    }

  training = protocol["training_protocol"]
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  jobs = []
  for arm, (task, experiment) in _ARMS.items():
    for seed in training["training_seeds"]:
      source_seed = _SEED_MAP[seed]
      source = Path(source_audits[str(source_seed)]["path"])
      source_name = f"v39_source_v38b_seed{source_seed}_gate7999"
      source_link = repo / "logs/rsl_rl" / experiment / source_name / "model_7999.pt"
      gpu = training["gpu_assignment"][f"{arm}/{seed}"]
      run_name = f"v39_{arm.lower()}_12k_seed{seed}_online"
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
        "^model_7999.pt$",
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
          "source_policy_seed": source_seed,
          "gpu": gpu,
          "run_name": run_name,
          "source_checkpoint": str(source),
          "source_sha256": source_audits[str(source_seed)]["sha256"],
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


def launch(cfg: V39SupportExitCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  control = Path(planned["jobs"][0]["log"]).parent
  state_path = control / "launch_manifest.json"
  if state_path.exists():
    existing = _json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing V39 run has a different immutable plan")
    if existing.get("status") != "LAUNCHED":
      raise RuntimeError(f"V39_TRAINING_ALERT: state {existing.get('status')}")
    repo = Path(__file__).resolve().parents[1]
    for job in existing["jobs"]:
      finals = list(
        (repo / "logs/rsl_rl" / job["experiment"]).glob(
          f"*_{job['run_name']}/model_11999.pt"
        )
      )
      if not _pid_alive(int(job["pid"])) and len(finals) != 1:
        raise RuntimeError(
          f"V39_TRAINING_ALERT: {job['arm']}/{job['seed']} died; no restart"
        )
    return existing
  if not cfg.launch:
    return planned

  repo = Path(__file__).resolve().parents[1]
  if _git(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing V39 launch from tracked-dirty worktree")
  planned["resource_preflight"] = _resource_preflight(repo)
  for job in planned["jobs"]:
    source = Path(job["source_checkpoint"])
    link = Path(job["source_link"])
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
      if not link.is_symlink() or link.resolve() != source:
        raise RuntimeError(f"V39_SOURCE_ALERT: conflicting link {link}")
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
    planned["status"] = "V39_TRAINING_ALERT_PARTIAL_LAUNCH"
    planned["alert_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, planned)
    raise
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: V39SupportExitCfg) -> None:
  result = launch(cfg)
  print(
    f"{result['status']}: plan {result['plan_id']} "
    f"pids={[job['pid'] for job in result['jobs']]}"
  )


if __name__ == "__main__":
  tyro.cli(main)
