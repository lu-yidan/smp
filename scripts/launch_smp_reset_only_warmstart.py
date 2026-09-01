"""Fail-closed launcher for the reset-only A6 warm-start engineering canary."""

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

_PROTOCOL_SHA256 = "7416457635c9fbb9c30880b07aebfc865f2a0d19e2e0581737069ffec2e84dbf"
_MINIMUM_COMMIT = "8037fb2"
_TASK = "Smp-Getup-Scratch-A10-F2S2-Physical-Reset-G1"
_EXPECTED_RUNNER = "SmpCurriculumWarmStartRunner"
_GIB = 1024**3


@dataclass(frozen=True)
class ResetOnlyWarmStartCfg:
  protocol: Path = Path("docs/ral_reset_only_warmstart_v1.json")
  control_dir: Path = Path("run_control/reset_only_warmstart_v1/training")
  launch: bool = False


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise ValueError(f"required JSON artifact missing: {path}")
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _git(repo_root: Path, *args: str) -> str:
  result = subprocess.run(
    ("git", *args), cwd=repo_root, check=True, capture_output=True, text=True
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


def _disk_preflight(repo_root: Path) -> dict[str, float]:
  usage = shutil.disk_usage(repo_root)
  stats = os.statvfs(repo_root)
  inode_fraction = stats.f_favail / stats.f_files if stats.f_files else 0.0
  free_gib = usage.free / _GIB
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


def _validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
  protocol_sha = _sha256(path) if path.is_file() else ""
  if protocol_sha != _PROTOCOL_SHA256:
    raise ValueError("reset-only warm-start protocol SHA-256 mismatch")
  protocol = _load_json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_CANARY"
    or protocol.get("study_id") != "smp-reset-only-warmstart-canary-v1"
    or protocol.get("study_role")
    != "ENGINEERING_CANARY_NOT_PROMOTION_OR_PERFORMANCE_EVIDENCE"
  ):
    raise ValueError("reset-only protocol is not canary-launch eligible")
  source = protocol.get("source_policy", {})
  if (
    source.get("policy_seed") != 20261102
    or source.get("checkpoint_name") != "model_15000.pt"
    or source.get("checkpoint_sha256")
    != "533fdb6c2072fc9f3436c3e33593c36b29a9c33df7a38667c593845922f016fc"
  ):
    raise ValueError("warm-start source drifted")
  treatment = protocol.get("treatment", {})
  if (
    treatment.get("only_changed_component") != "reset_sampling_and_physical_validation"
    or treatment.get("task") != _TASK
  ):
    raise ValueError("reset-only treatment drifted")
  training = protocol.get("training_protocol", {})
  expected = {
    "policy_seed": 20261201,
    "environment_seed": 20261201,
    "source_policy_seed": 20261102,
    "warm_start": "actor_critic_and_normalizers_only",
    "reset_optimizer_iteration_and_environment_steps": True,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 5000,
    "save_interval": 500,
    "learning_rate": 0.0001,
    "device": 0,
    "reserved_idle_devices": [1, 2, 3, 4, 5, 6, 7],
  }
  for key, value in expected.items():
    if training.get(key) != value:
      raise ValueError(f"reset-only training protocol {key} drifted")
  audit = protocol.get("implementation_audit", {})
  if (
    audit.get("status") != "PASSED_REAL_MUJOCO_SMOKE"
    or audit.get("code_commit") != "8037fb259f23bec3f5224499a2a12b89c4f65138"
    or audit.get("task") != _TASK
    or audit.get("evidence_role") != "NON_PERFORMANCE_IMPLEMENTATION_EVIDENCE"
  ):
    raise ValueError("reset-only implementation audit drifted")
  return protocol, protocol_sha


def _validate_smoke(protocol: dict[str, Any], repo_root: Path) -> dict[str, Any]:
  audit = protocol["implementation_audit"]
  runtime = audit.get("runtime_files", {})
  required = (
    "log",
    "checkpoint",
    "agent_config",
    "environment_config",
    "git_provenance",
  )
  resolved: dict[str, Path] = {}
  for name in required:
    row = runtime.get(name, {})
    path = repo_root / row.get("path", "")
    if not path.is_file() or _sha256(path) != row.get("sha256"):
      raise RuntimeError(f"RESET_ONLY_SMOKE_ALERT: {name} missing or drifted")
    resolved[name] = path
  log_text = resolved["log"].read_text(errors="replace")
  forbidden = re.compile(
    r"traceback|cuda out of memory|outofmemoryerror|fatal|segmentation fault|"
    r"physical_reset_alert|(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)",
    re.IGNORECASE | re.MULTILINE,
  )
  required_log = (
    "Learning iteration 0/1",
    "Total steps: 384",
    "Linear(in_features=93, out_features=512",
    "Linear(in_features=960, out_features=512",
    "curriculum_validated_fall_reset",
  )
  if forbidden.search(log_text) or any(text not in log_text for text in required_log):
    raise RuntimeError("RESET_ONLY_SMOKE_ALERT: smoke log contract failed")
  checkpoint = torch.load(
    resolved["checkpoint"], map_location="cpu", weights_only=False
  )

  def collect(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
      return [value]
    if isinstance(value, dict):
      return [tensor for item in value.values() for tensor in collect(item)]
    if isinstance(value, (tuple, list)):
      return [tensor for item in value for tensor in collect(item)]
    return []

  tensors = collect(checkpoint)
  verified = audit["verified"]
  if (
    checkpoint.get("iter") != 0
    or tuple(checkpoint["actor_state_dict"]["mlp.0.weight"].shape) != (512, 93)
    or tuple(checkpoint["critic_state_dict"]["mlp.0.weight"].shape) != (512, 960)
    or len(tensors) != verified.get("checkpoint_tensor_count")
    or sum(tensor.numel() for tensor in tensors)
    != verified.get("checkpoint_tensor_elements")
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError("RESET_ONLY_SMOKE_ALERT: checkpoint integrity failed")
  return {
    "status": audit["status"],
    "code_commit": audit["code_commit"],
    "runtime_sha256": {name: runtime[name]["sha256"] for name in required},
  }


def _validate_warm_start_smoke(
  protocol: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
  audit = protocol.get("warm_start_path_audit", {})
  if (
    audit.get("status") != "PASSED_REAL_MUJOCO_WARM_START_SMOKE"
    or audit.get("code_commit") != "23215977a104c9a8b78188753ed22b9c34ad6ce5"
    or audit.get("runner") != _EXPECTED_RUNNER
    or audit.get("source_checkpoint_sha256")
    != protocol["source_policy"]["checkpoint_sha256"]
    or audit.get("evidence_role") != "NON_PERFORMANCE_IMPLEMENTATION_EVIDENCE"
  ):
    raise RuntimeError("RESET_ONLY_WARM_START_SMOKE_ALERT: audit drifted")
  runtime = audit.get("runtime_files", {})
  required = (
    "log",
    "checkpoint",
    "agent_config",
    "environment_config",
    "git_provenance",
  )
  resolved: dict[str, Path] = {}
  for name in required:
    row = runtime.get(name, {})
    path = repo_root / row.get("path", "")
    if not path.is_file() or _sha256(path) != row.get("sha256"):
      raise RuntimeError(
        f"RESET_ONLY_WARM_START_SMOKE_ALERT: {name} missing or drifted"
      )
    resolved[name] = path
  log_text = resolved["log"].read_text(errors="replace")
  required_log = (
    "Loading model checkpoint from:",
    "reset_only_source_a6_seed20261102_gate15000/model_15000.pt",
    "Learning iteration 0/1",
    "Total steps: 384",
  )
  if any(fragment not in log_text for fragment in required_log):
    raise RuntimeError("RESET_ONLY_WARM_START_SMOKE_ALERT: load path/clock drifted")
  agent_text = resolved["agent_config"].read_text(errors="replace")
  for pattern in (
    r"^seed: 20261200$",
    r"^max_iterations: 1$",
    r"^save_interval: 1$",
    r"^resume: true$",
    r"^load_run: \^reset_only_source_a6_seed20261102_gate15000\$$",
    r"^load_checkpoint: \^model_15000\.pt\$$",
    r"^  learning_rate: 0\.0001$",
  ):
    if re.search(pattern, agent_text, re.MULTILINE) is None:
      raise RuntimeError("RESET_ONLY_WARM_START_SMOKE_ALERT: agent config drifted")
  verified = audit.get("verified", {})
  if (
    verified.get("learning_iteration_started_at") != 0
    or verified.get("environment_steps_started_at") != 0
    or verified.get("optimizer_restored") is not False
    or verified.get("all_checkpoint_tensors_finite") is not True
  ):
    raise RuntimeError("RESET_ONLY_WARM_START_SMOKE_ALERT: runner audit drifted")
  return {
    "status": audit["status"],
    "code_commit": audit["code_commit"],
    "runtime_sha256": {name: runtime[name]["sha256"] for name in required},
  }


def build_plan(cfg: ResetOnlyWarmStartCfg) -> dict[str, Any]:
  repo_root = Path(__file__).resolve().parents[1]
  protocol_path = (
    cfg.protocol if cfg.protocol.is_absolute() else repo_root / cfg.protocol
  )
  protocol, protocol_sha = _validate_protocol(protocol_path)
  commit = _git(repo_root, "rev-parse", "HEAD")
  try:
    _git(repo_root, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  except subprocess.CalledProcessError as error:
    raise ValueError(f"code commit is older than {_MINIMUM_COMMIT}") from error
  from mjlab.tasks.registry import load_runner_cls

  import smp.rl.tasks  # noqa: F401

  runner = load_runner_cls(_TASK)
  if runner is None or runner.__name__ != _EXPECTED_RUNNER:
    raise RuntimeError("reset-only task lacks the fresh-optimizer warm-start runner")
  source = protocol["source_policy"]
  source_checkpoint = (repo_root / source["checkpoint_path"]).resolve()
  if (
    not source_checkpoint.is_file()
    or _sha256(source_checkpoint) != source["checkpoint_sha256"]
  ):
    raise RuntimeError("RESET_ONLY_SOURCE_ALERT: source checkpoint missing or drifted")
  training = protocol["training_protocol"]
  source_name = "reset_only_source_a6_seed20261102_gate15000"
  source_link = (
    repo_root
    / "logs/rsl_rl/smp_scratch_a10_f2s2_physical_reset_g1"
    / source_name
    / source["checkpoint_name"]
  )
  run_name = "reset_only_v1_a10_5k_seed20261201"
  log = cfg.control_dir / "gpu0_a10_seed20261201.log"
  command = [
    "uv",
    "run",
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
    "source_checkpoint_sha256": source["checkpoint_sha256"],
    "policy_seed": training["policy_seed"],
    "environment_seed": training["environment_seed"],
    "command": command,
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
    "task": _TASK,
    "runner": _EXPECTED_RUNNER,
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_sha256": source["checkpoint_sha256"],
    "source_link": str(source_link),
    "experiment_root": str(source_link.parents[1]),
    "policy_seed": training["policy_seed"],
    "environment_seed": training["environment_seed"],
    "num_envs": training["num_envs"],
    "max_iterations": training["max_iterations"],
    "save_interval": training["save_interval"],
    "learning_rate": training["learning_rate"],
    "gpu": training["device"],
    "reserved_idle_devices": training["reserved_idle_devices"],
    "run_name": run_name,
    "log": str(log.resolve()),
    "pid_file": str((cfg.control_dir / "gpu0_a10_seed20261201.pid").resolve()),
    "command": command,
    "pid": None,
    "claim_boundary": protocol["claim_boundary"],
  }


def launch_canary(cfg: ResetOnlyWarmStartCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = cfg.control_dir / "launch_manifest.json"
  if state_path.exists():
    existing = _load_json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing reset-only canary has a different immutable plan")
    if existing.get("status") == "LAUNCHED":
      pid = existing.get("pid")
      final_candidates = list(
        Path(existing["experiment_root"]).glob(
          f"*_{existing['run_name']}/model_4999.pt"
        )
      )
      if pid is not None and not _pid_alive(int(pid)) and len(final_candidates) != 1:
        raise RuntimeError(
          "RESET_ONLY_TRAINING_ALERT: worker died; no automatic restart"
        )
      return existing
    if existing.get("status") == "LAUNCHING":
      raise RuntimeError("RESET_ONLY_TRAINING_ALERT: partial launch marker exists")
    raise ValueError(f"inadmissible reset-only launch state: {existing.get('status')}")
  if not cfg.launch:
    return planned
  repo_root = Path(__file__).resolve().parents[1]
  if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing reset-only launch from tracked-dirty worktree")
  gpu_processes = _gpu_processes()
  if gpu_processes:
    raise RuntimeError(
      f"refusing reset-only launch while GPU compute is active: {gpu_processes}"
    )
  protocol = _load_json(Path(planned["protocol"]))
  planned["implementation_smoke"] = _validate_smoke(protocol, repo_root)
  planned["warm_start_smoke"] = _validate_warm_start_smoke(protocol, repo_root)
  planned["resource_preflight"] = _disk_preflight(repo_root)
  source = Path(planned["source_checkpoint"])
  link = Path(planned["source_link"])
  link.parent.mkdir(parents=True, exist_ok=True)
  if link.exists() or link.is_symlink():
    if not link.is_symlink() or link.resolve() != source.resolve():
      raise ValueError(f"conflicting warm-start source link: {link}")
  else:
    link.symlink_to(source)
  cfg.control_dir.mkdir(parents=True, exist_ok=True)
  planned["status"] = "LAUNCHING"
  planned["started_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  environment = os.environ.copy()
  environment["CUDA_VISIBLE_DEVICES"] = str(planned["gpu"])
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


def main(cfg: ResetOnlyWarmStartCfg) -> None:
  result = launch_canary(cfg)
  print(f"{result['status']}: plan {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(ResetOnlyWarmStartCfg))
