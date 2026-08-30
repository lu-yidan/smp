"""Fail-closed launcher for the preregistered fresh-seed flat method study."""

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

import tyro

_PROTOCOL_SHA256 = "6ca241aa3bfb303084de8eac4f1cd6e02a4728ef5969a632dc7ba2b54750e0e0"
_MINIMUM_COMMIT = "a881567"
_ARM_ORDER = ("a6_replication_control", "a8_balanced_bridge")
_POLICY_SEEDS = (20261001, 20261002, 20261003)
_DEVICES = (0, 1, 2, 3, 4, 5)
_RESERVED_DEVICES = (6, 7)
_GIB = 1024**3


@dataclass(frozen=True)
class FlatMethodStudyCfg:
  protocol: Path = Path("docs/ral_flat_method_study_v1.json")
  control_dir: Path = Path("run_control/flat_method_study_v1_training")
  launch: bool = False


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise ValueError(f"required JSON artifact is missing: {path}")
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"{path} must contain a JSON object")
  return payload


def _git(repo_root: Path, *args: str) -> str:
  result = subprocess.run(
    ("git", *args),
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


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


def _validate_protocol(path: Path, repo_root: Path) -> tuple[dict[str, Any], str]:
  protocol_sha = _sha256(path) if path.is_file() else ""
  if protocol_sha != _PROTOCOL_SHA256:
    raise ValueError("flat method study protocol SHA-256 mismatch")
  protocol = _load_json(path)
  if protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING":
    raise ValueError("flat method study protocol is not launch-eligible")
  training = protocol.get("training_protocol", {})
  evaluation = protocol.get("evaluation_protocol", {})
  expected_training = {
    "policy_seeds": list(_POLICY_SEEDS),
    "environment_seed_equals_policy_seed": True,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 30000,
    "save_interval": 1000,
    "actor_observation_dim": 93,
    "actor_history_steps": 1,
    "devices": list(_DEVICES),
    "reserved_idle_devices": list(_RESERVED_DEVICES),
  }
  for key, expected in expected_training.items():
    if training.get(key) != expected:
      raise ValueError(f"training protocol {key} drifted")
  expected_evaluation = {
    "evaluation_seed": 20261010,
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
    raise ValueError("flat method study arms or ordering drifted")
  expected_arms = {
    "a6_replication_control": (
      "Smp-Getup-Scratch-A6-F2S2-Mix-Bridge-G1",
      0.20,
      False,
    ),
    "a8_balanced_bridge": (
      "Smp-Getup-Scratch-A8-F2S2-Balanced-Bridge-G1",
      0.50,
      True,
    ),
  }
  for arm, (task, probability, promotion_eligible) in expected_arms.items():
    row = arms[arm]
    if (
      row.get("task") != task
      or row.get("prior") != "F2S2"
      or row.get("procedural_probability") != probability
      or row.get("procedural_pose_weights") != [1.0, 1.0, 1.0, 1.0]
      or row.get("gsi_only_low_smp_termination") is not True
      or row.get("procedural_smp_floor") != 0.10
      or row.get("promotion_eligible") is not promotion_eligible
    ):
      raise ValueError(f"flat method arm {arm} drifted")
  prior = protocol.get("prior_null_result", {})
  if (
    prior.get("status") != "COMPLETE_NO_PROMOTION"
    or prior.get("must_remain_unchanged") is not True
    or prior.get("failure_diagnosis_action") != "STOP_NO_SAFE_AUTOMATIC_RETRAINING"
  ):
    raise ValueError("prior null-result boundary drifted")
  for source in protocol.get("sources", []):
    source_path = repo_root / source.get("path", "")
    if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
      raise ValueError(f"preregistration source changed: {source_path}")
  return protocol, protocol_sha


def _validate_smoke(protocol: dict[str, Any], repo_root: Path) -> dict[str, Any]:
  audit = protocol.get("implementation_audit", {})
  if (
    audit.get("status") != "PASSED_REAL_MUJOCO_SMOKE"
    or audit.get("code_commit") != "e9f8f051e472bb911c9aeaec252edd2a140bd96d"
    or audit.get("task") != "Smp-Getup-Scratch-A8-F2S2-Balanced-Bridge-G1"
    or audit.get("evidence_role") != "NON_PERFORMANCE_IMPLEMENTATION_EVIDENCE"
  ):
    raise RuntimeError("FLAT_METHOD_SMOKE_ALERT: implementation audit drifted")
  runtime = audit.get("runtime_files", {})
  required = ("log", "checkpoint", "agent_config", "environment_config", "git_provenance")
  resolved: dict[str, Path] = {}
  for name in required:
    row = runtime.get(name, {})
    path = repo_root / row.get("path", "")
    if not path.is_file() or _sha256(path) != row.get("sha256"):
      raise RuntimeError(f"FLAT_METHOD_SMOKE_ALERT: {name} is missing or drifted")
    resolved[name] = path

  log_text = resolved["log"].read_text(errors="replace")
  error_pattern = re.compile(
    r"traceback|cuda out of memory|outofmemoryerror|"
    r"(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)|fatal|segmentation fault",
    re.IGNORECASE | re.MULTILINE,
  )
  required_log_fragments = (
    "Learning iteration 0/1",
    "Total steps: 384",
    "Linear(in_features=93, out_features=512",
    "Linear(in_features=960, out_features=512",
  )
  if error_pattern.search(log_text) or any(
    fragment not in log_text for fragment in required_log_fragments
  ):
    raise RuntimeError("FLAT_METHOD_SMOKE_ALERT: MuJoCo smoke log is invalid")

  agent_text = resolved["agent_config"].read_text()
  env_text = resolved["environment_config"].read_text()
  for pattern in (
    r"^seed: 20261000$",
    r"^num_steps_per_env: 24$",
    r"^max_iterations: 1$",
    r"^save_interval: 1$",
  ):
    if re.search(pattern, agent_text, re.MULTILINE) is None:
      raise RuntimeError("FLAT_METHOD_SMOKE_ALERT: agent config contract drifted")
  for pattern in (
    r"^  num_envs: 16$",
    r"^      procedural_probability: 0\.5$",
    r"^seed: 20261000$",
  ):
    if re.search(pattern, env_text, re.MULTILINE) is None:
      raise RuntimeError("FLAT_METHOD_SMOKE_ALERT: environment config contract drifted")

  import torch

  try:
    checkpoint = torch.load(
      resolved["checkpoint"], map_location="cpu", weights_only=False
    )
  except Exception as error:
    raise RuntimeError(
      "FLAT_METHOD_SMOKE_ALERT: checkpoint is not loadable"
    ) from error
  actor = checkpoint.get("actor_state_dict", {})
  critic = checkpoint.get("critic_state_dict", {})

  def collect_tensors(value: Any) -> list[Any]:
    if torch.is_tensor(value):
      return [value]
    if isinstance(value, dict):
      return [tensor for item in value.values() for tensor in collect_tensors(item)]
    if isinstance(value, (list, tuple)):
      return [tensor for item in value for tensor in collect_tensors(item)]
    return []

  tensors = collect_tensors(checkpoint)
  verified = audit.get("verified", {})
  if (
    checkpoint.get("iter") != 0
    or tuple(actor.get("mlp.0.weight", torch.empty(0)).shape) != (512, 93)
    or tuple(critic.get("mlp.0.weight", torch.empty(0)).shape) != (512, 960)
    or len(tensors) != verified.get("checkpoint_tensor_count")
    or sum(tensor.numel() for tensor in tensors)
    != verified.get("checkpoint_tensor_elements")
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError("FLAT_METHOD_SMOKE_ALERT: checkpoint integrity failed")
  return {
    "status": audit["status"],
    "code_commit": audit["code_commit"],
    "runtime_sha256": {name: runtime[name]["sha256"] for name in required},
  }


def build_plan(cfg: FlatMethodStudyCfg) -> dict[str, Any]:
  repo_root = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo_root / cfg.protocol
  protocol, protocol_sha = _validate_protocol(protocol_path, repo_root)
  commit = _git(repo_root, "rev-parse", "HEAD")
  try:
    _git(repo_root, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  except subprocess.CalledProcessError as error:
    raise ValueError(f"code commit is older than {_MINIMUM_COMMIT}") from error

  jobs = []
  for gpu, (arm, seed) in zip(
    _DEVICES,
    ((arm, seed) for arm in _ARM_ORDER for seed in _POLICY_SEEDS),
    strict=True,
  ):
    task = protocol["arms"][arm]["task"]
    run_name = f"flat_method_v1_{arm}_30k_seed{seed}"
    log = cfg.control_dir / f"gpu{gpu}_{arm}_seed{seed}.log"
    pid_file = cfg.control_dir / f"gpu{gpu}_{arm}_seed{seed}.pid"
    command = [
      "uv",
      "run",
      "scripts/train.py",
      task,
      "--env.scene.num-envs",
      "4096",
      "--agent.seed",
      str(seed),
      "--env.seed",
      str(seed),
      "--agent.max-iterations",
      "30000",
      "--agent.save-interval",
      "1000",
      "--agent.run-name",
      run_name,
    ]
    jobs.append(
      {
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
      }
    )

  plan_material = {
    "protocol_sha256": protocol_sha,
    "code_commit": commit,
    "jobs": [{key: value for key, value in job.items() if key != "pid"} for job in jobs],
    "reserved_idle_devices": list(_RESERVED_DEVICES),
  }
  plan_id = hashlib.sha256(
    json.dumps(plan_material, sort_keys=True).encode()
  ).hexdigest()
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
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 30000,
    "save_interval": 1000,
    "jobs": jobs,
  }


def _disk_preflight(repo_root: Path) -> dict[str, float]:
  usage = shutil.disk_usage(repo_root)
  stats = os.statvfs(repo_root)
  inode_fraction = stats.f_favail / stats.f_files if stats.f_files else 0.0
  free_gib = usage.free / _GIB
  if free_gib < 100.0 or inode_fraction < 0.10:
    raise RuntimeError(
      f"DISK_SPACE_ALERT: free_gib={free_gib:.1f}, inode_free_fraction={inode_fraction:.3f}"
    )
  return {"free_gib": free_gib, "inode_free_fraction": inode_fraction}


def launch_study(cfg: FlatMethodStudyCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = cfg.control_dir / "launch_manifest.json"
  if state_path.exists():
    existing = _load_json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing flat method study has a different frozen plan")
    if existing.get("status") == "LAUNCHED":
      return existing
    if existing.get("status") != "LAUNCHING":
      raise ValueError(f"inadmissible launch state: {existing.get('status')}")
    planned = existing
  if not cfg.launch:
    return planned
  if planned.get("protocol_status") != "PREREGISTERED_READY_FOR_TRAINING":
    raise RuntimeError(
      "protocol is still awaiting implementation audit; training launch is forbidden"
    )

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
        job["command"],
        cwd=repo_root,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
    job["pid"] = process.pid
    Path(job["pid_file"]).write_text(f"{process.pid}\n")
    _atomic_json(state_path, planned)
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: FlatMethodStudyCfg) -> None:
  result = launch_study(cfg)
  print(f"{result['status']}: {len(result['jobs'])} jobs, plan {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(FlatMethodStudyCfg))
