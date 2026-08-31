"""Fail-closed launcher for the preregistered flat objective-alignment study."""

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

from launch_smp_flat_method_study import (
  _atomic_json,
  _disk_preflight,
  _git,
  _gpu_processes,
  _load_json,
  _pid_alive,
  _sha256,
)

_PROTOCOL_SHA256 = "00c6f21db33bee1ff76d659ee85ec7e99817905e49fb92b7d7ed4c8a4a60dbbb"
_MINIMUM_COMMIT = "1274c3b"
_IMPLEMENTATION_COMMIT = "94fb0f415bd2ead78c21a79ec1b436b1323a2b19"
_ARM_ORDER = ("a6_replication_control", "a9_objective_aligned")
_POLICY_SEEDS = (20261101, 20261102, 20261103)
_DEVICES = (0, 1, 2, 3, 4, 5)
_RESERVED_DEVICES = (6, 7)


@dataclass(frozen=True)
class FlatObjectiveAlignmentCfg:
  protocol: Path = Path("docs/ral_flat_objective_alignment_v1.json")
  control_dir: Path = Path("run_control/flat_objective_alignment_v1_training")
  launch: bool = False


def _validate_protocol(path: Path, repo_root: Path) -> tuple[dict[str, Any], str]:
  protocol_sha = _sha256(path) if path.is_file() else ""
  if protocol_sha != _PROTOCOL_SHA256:
    raise ValueError("flat objective-alignment protocol SHA-256 mismatch")
  protocol = _load_json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != "smp-flat-objective-alignment-v1"
  ):
    raise ValueError("flat objective-alignment protocol is not launch-eligible")

  training = protocol.get("training_protocol", {})
  expected_training = {
    "policy_seeds": list(_POLICY_SEEDS),
    "environment_seed_equals_policy_seed": True,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 30000,
    "save_interval": 1000,
    "actor_observation_dim": 93,
    "actor_history_steps": 1,
    "critic_observation_dim": 960,
    "devices": list(_DEVICES),
    "reserved_idle_devices": list(_RESERVED_DEVICES),
  }
  for key, expected in expected_training.items():
    if training.get(key) != expected:
      raise ValueError(f"training protocol {key} drifted")

  evaluation = protocol.get("evaluation_protocol", {})
  expected_evaluation = {
    "evaluation_seed": 20261110,
    "num_envs": 512,
    "steps": 500,
    "evaluation_schema_version": 2,
    "failure_diagnosis_schema_version": 1,
    "reset_modes": ["native_gsi", "prone", "supine", "left_side", "right_side"],
  }
  for key, expected in expected_evaluation.items():
    if evaluation.get(key) != expected:
      raise ValueError(f"evaluation protocol {key} drifted")

  arms = protocol.get("arms")
  if not isinstance(arms, dict) or tuple(arms) != _ARM_ORDER:
    raise ValueError("objective-alignment arm ordering drifted")
  expected_tasks = {
    "a6_replication_control": "Smp-Getup-Scratch-A6-F2S2-Mix-Bridge-G1",
    "a9_objective_aligned": "Smp-Getup-Scratch-A9-F2S2-Objective-Aligned-G1",
  }
  for arm, task in expected_tasks.items():
    row = arms[arm]
    promotion_eligible = arm == "a9_objective_aligned"
    if (
      row.get("task") != task
      or row.get("prior") != "F2S2"
      or row.get("procedural_probability") != 0.2
      or row.get("procedural_pose_weights") != [1.0, 1.0, 1.0, 1.0]
      or row.get("gsi_only_low_smp_termination") is not True
      or row.get("actor_observation_dim") != 93
      or row.get("actor_history_steps") != 1
      or row.get("promotion_eligible") is not promotion_eligible
    ):
      raise ValueError(f"objective-alignment arm {arm} drifted")

  retained = protocol.get("retained_results", {})
  required_statuses = {
    "original_causal_screen": "NO_PROMOTION",
    "late_emergence_followup": "NO_PROMOTION",
    "procedural_coverage_study": "FLAT_METHOD_COMPLETE_NO_PROMOTION",
  }
  for name, status in required_statuses.items():
    row = retained.get(name, {})
    if row.get("status") != status or row.get("must_remain_unchanged") is not True:
      raise ValueError(f"retained null-result boundary drifted: {name}")

  audit = protocol.get("implementation_audit", {})
  if (
    audit.get("status") != "PASSED_REAL_MUJOCO_SMOKE"
    or audit.get("code_commit") != _IMPLEMENTATION_COMMIT
    or audit.get("task") != expected_tasks["a9_objective_aligned"]
    or audit.get("evidence_role") != "NON_PERFORMANCE_IMPLEMENTATION_EVIDENCE"
  ):
    raise ValueError("objective-alignment implementation audit drifted")
  return protocol, protocol_sha


def _validate_smoke(protocol: dict[str, Any], repo_root: Path) -> dict[str, Any]:
  audit = protocol["implementation_audit"]
  runtime = audit.get("runtime_files", {})
  required = ("log", "checkpoint", "agent_config", "environment_config", "git_provenance")
  resolved: dict[str, Path] = {}
  for name in required:
    row = runtime.get(name, {})
    path = repo_root / row.get("path", "")
    if not path.is_file() or _sha256(path) != row.get("sha256"):
      raise RuntimeError(f"FLAT_OBJECTIVE_SMOKE_ALERT: {name} missing or drifted")
    resolved[name] = path

  log_text = resolved["log"].read_text(errors="replace")
  forbidden = re.compile(
    r"traceback|cuda out of memory|outofmemoryerror|fatal|segmentation fault|"
    r"(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)",
    re.IGNORECASE | re.MULTILINE,
  )
  required_log = (
    "Learning iteration 0/1",
    "Total steps: 384",
    "Linear(in_features=93, out_features=512",
    "Linear(in_features=960, out_features=512",
  )
  if forbidden.search(log_text) or any(fragment not in log_text for fragment in required_log):
    raise RuntimeError("FLAT_OBJECTIVE_SMOKE_ALERT: MuJoCo smoke log is invalid")

  agent_text = resolved["agent_config"].read_text(errors="replace")
  env_text = resolved["environment_config"].read_text(errors="replace")
  for pattern in (
    r"^seed: 20261100$",
    r"^num_steps_per_env: 24$",
    r"^max_iterations: 1$",
    r"^save_interval: 1$",
  ):
    if re.search(pattern, agent_text, re.MULTILINE) is None:
      raise RuntimeError("FLAT_OBJECTIVE_SMOKE_ALERT: agent config drifted")
  for pattern in (
    r"^  num_envs: 16$",
    r"^      procedural_probability: 0\.2$",
    r"^seed: 20261100$",
    r"^      smp_floor: 0\.35$",
    r"^      head_height: 1\.1$",
    r"^      min_upright: 0\.85$",
    r"^      max_angular_speed: 1\.0$",
  ):
    if re.search(pattern, env_text, re.MULTILINE) is None:
      raise RuntimeError("FLAT_OBJECTIVE_SMOKE_ALERT: environment config drifted")
  for term in (
    "recovery_initiation_progress",
    "track_head_height",
    "upright_posture",
    "feet_stationary_when_upright",
    "base_stationary_when_upright",
    "stable_stand_metric",
    "head_vertical_overspeed",
    "action_rate_l2",
  ):
    if term not in env_text:
      raise RuntimeError(f"FLAT_OBJECTIVE_SMOKE_ALERT: missing objective term {term}")

  import torch

  try:
    checkpoint = torch.load(resolved["checkpoint"], map_location="cpu", weights_only=False)
  except Exception as error:
    raise RuntimeError("FLAT_OBJECTIVE_SMOKE_ALERT: checkpoint is not loadable") from error

  def collect(value: Any) -> list[Any]:
    if torch.is_tensor(value):
      return [value]
    if isinstance(value, dict):
      return [tensor for item in value.values() for tensor in collect(item)]
    if isinstance(value, (list, tuple)):
      return [tensor for item in value for tensor in collect(item)]
    return []

  actor = checkpoint.get("actor_state_dict", {})
  critic = checkpoint.get("critic_state_dict", {})
  tensors = collect(checkpoint)
  verified = audit.get("verified", {})
  if (
    checkpoint.get("iter") != 0
    or tuple(actor.get("mlp.0.weight", torch.empty(0)).shape) != (512, 93)
    or tuple(critic.get("mlp.0.weight", torch.empty(0)).shape) != (512, 960)
    or len(tensors) != verified.get("checkpoint_tensor_count")
    or sum(tensor.numel() for tensor in tensors) != verified.get("checkpoint_tensor_elements")
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError("FLAT_OBJECTIVE_SMOKE_ALERT: checkpoint integrity failed")
  return {
    "status": audit["status"],
    "code_commit": audit["code_commit"],
    "runtime_sha256": {name: runtime[name]["sha256"] for name in required},
  }


def build_plan(cfg: FlatObjectiveAlignmentCfg) -> dict[str, Any]:
  repo_root = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo_root / cfg.protocol
  protocol, protocol_sha = _validate_protocol(protocol_path, repo_root)
  commit = _git(repo_root, "rev-parse", "HEAD")
  try:
    _git(repo_root, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  except subprocess.CalledProcessError as error:
    raise ValueError(f"code commit is older than {_MINIMUM_COMMIT}") from error

  jobs = []
  pairs = ((arm, seed) for arm in _ARM_ORDER for seed in _POLICY_SEEDS)
  for gpu, (arm, seed) in zip(_DEVICES, pairs, strict=True):
    task = protocol["arms"][arm]["task"]
    run_name = f"flat_objective_v1_{arm}_30k_seed{seed}"
    log = cfg.control_dir / f"gpu{gpu}_{arm}_seed{seed}.log"
    pid_file = cfg.control_dir / f"gpu{gpu}_{arm}_seed{seed}.pid"
    command = [
      "uv", "run", "scripts/train.py", task,
      "--env.scene.num-envs", "4096",
      "--agent.seed", str(seed),
      "--env.seed", str(seed),
      "--agent.max-iterations", "30000",
      "--agent.save-interval", "1000",
      "--agent.run-name", run_name,
    ]
    jobs.append({
      "arm": arm,
      "task": task,
      "promotion_eligible": protocol["arms"][arm]["promotion_eligible"],
      "policy_seed": seed,
      "environment_seed": seed,
      "gpu": gpu,
      "run_name": run_name,
      "log": str(log.resolve()),
      "pid_file": str(pid_file.resolve()),
      "command": command,
      "pid": None,
    })

  material = {
    "protocol_sha256": protocol_sha,
    "code_commit": commit,
    "jobs": [{key: value for key, value in job.items() if key != "pid"} for job in jobs],
    "reserved_idle_devices": list(_RESERVED_DEVICES),
  }
  plan_id = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  return {
    "schema_version": 1,
    "status": "PLANNED",
    "study_id": protocol["study_id"],
    "plan_id": plan_id,
    "protocol": str(protocol_path.resolve()),
    "protocol_sha256": protocol_sha,
    "protocol_status": protocol["status"],
    "code_commit": commit,
    "policy_seeds": list(_POLICY_SEEDS),
    "devices": list(_DEVICES),
    "reserved_idle_devices": list(_RESERVED_DEVICES),
    "random_actor_critic_and_normalizers": True,
    "actor_observation_dim": 93,
    "actor_history_steps": 1,
    "critic_observation_dim": 960,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 30000,
    "save_interval": 1000,
    "jobs": jobs,
  }


def launch_study(cfg: FlatObjectiveAlignmentCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = cfg.control_dir / "launch_manifest.json"
  if state_path.exists():
    existing = _load_json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing flat objective study has a different frozen plan")
    if existing.get("status") == "LAUNCHED":
      return existing
    if existing.get("status") != "LAUNCHING":
      raise ValueError(f"inadmissible launch state: {existing.get('status')}")
    planned = existing
  if not cfg.launch:
    return planned

  repo_root = Path(__file__).resolve().parents[1]
  protocol = _load_json(Path(planned["protocol"]))
  planned["implementation_smoke"] = _validate_smoke(protocol, repo_root)
  if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing launch from a tracked-dirty worktree")
  gpu_processes = _gpu_processes()
  if gpu_processes:
    raise RuntimeError(f"refusing launch while GPU compute is active: {gpu_processes}")
  planned["resource_preflight"] = _disk_preflight(repo_root)
  cfg.control_dir.mkdir(parents=True, exist_ok=True)
  planned["status"] = "LAUNCHING"
  planned.setdefault("started_at_utc", datetime.now(timezone.utc).isoformat())
  _atomic_json(state_path, planned)

  for job in planned["jobs"]:
    if job.get("pid") is not None:
      if not _pid_alive(int(job["pid"])):
        raise RuntimeError(f"partially launched job exited: {job['run_name']}")
      continue
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    with Path(job["log"]).open("a") as stream:
      process = subprocess.Popen(
        job["command"], cwd=repo_root, env=environment,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
      )
    job["pid"] = process.pid
    Path(job["pid_file"]).write_text(f"{process.pid}\n")
    _atomic_json(state_path, planned)
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: FlatObjectiveAlignmentCfg) -> None:
  result = launch_study(cfg)
  print(f"{result['status']}: {len(result['jobs'])} jobs, plan {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(FlatObjectiveAlignmentCfg))
