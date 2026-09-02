"""Fail-closed launcher for the preregistered A14 velocity-envelope fine-tune."""

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

_PROTOCOL_SHA256 = "5b3c9c3b1b252133899448aa63494956c391f38b799f8996db4f851f6379528a"
_MINIMUM_COMMIT = "25c90be72eb36f6e04f68ede9e4a87c054e1407f"
_TASK = "Smp-Getup-Scratch-A14-F2S2-Velocity-Envelope-G1"
_EXPECTED_RUNNER = "SmpCurriculumWarmStartRunner"
_SOURCE_SHA256 = "c4d05589645aa6b993e220c0cf3a9533fa83d50f5b9ed6764bb724b643cf1f94"
_GIB = 1024**3


@dataclass(frozen=True)
class VelocityEnvelopeFinetuneCfg:
  protocol: Path = Path("docs/ral_velocity_envelope_finetune_v1.json")
  control_dir: Path = Path("run_control/velocity_envelope_finetune_v1/training")
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
  return [row.strip() for row in result.stdout.splitlines() if row.strip()]


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
  actor = tuple(payload["actor_state_dict"]["mlp.0.weight"].shape)
  critic = tuple(payload["critic_state_dict"]["mlp.0.weight"].shape)
  if (
    payload.get("iter") != iteration
    or actor != (512, 93)
    or critic != (512, 960)
    or not tensors
    or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
  ):
    raise RuntimeError(f"A14_CHECKPOINT_ALERT: invalid checkpoint {path}")
  return {
    "iteration": iteration,
    "actor_input_dim": actor[1],
    "critic_input_dim": critic[1],
    "tensor_count": len(tensors),
    "tensor_elements": sum(tensor.numel() for tensor in tensors),
    "all_tensors_finite": True,
  }


def _validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
  digest = _sha256(path) if path.is_file() else ""
  if digest != _PROTOCOL_SHA256:
    raise ValueError("A14 velocity-envelope protocol SHA-256 mismatch")
  protocol = _json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != "smp-velocity-envelope-finetune-v1"
    or protocol.get("study_role")
    != "ENGINEERING_SIM_TO_REAL_VELOCITY_TAIL_FINETUNE_NOT_PROMOTION_OR_RAL_EVIDENCE"
  ):
    raise ValueError("A14 protocol is not launch eligible")
  source = protocol.get("source_policy", {})
  if (
    source.get("checkpoint_name") != "model_4999.pt"
    or source.get("checkpoint_sha256") != _SOURCE_SHA256
    or source.get("embedded_iteration") != 4999
    or source.get("actor_input_dim") != 93
    or source.get("critic_input_dim") != 960
  ):
    raise ValueError("A14 source policy drifted")
  envelope = protocol.get("treatment", {}).get("action_envelope", {})
  expected_envelope = {
    "application_rate_hz": 50,
    "max_joint_position_target_velocity_rad_s": 4.0,
    "max_joint_position_target_acceleration_rad_s2": 30.0,
    "applied_before_pd_actuation": True,
    "uses_only_deployment_available_state": True,
    "same_transform_required_in_deployment": True,
  }
  for key, value in expected_envelope.items():
    if envelope.get(key) != value:
      raise ValueError(f"A14 action envelope {key} drifted")
  expected_training = {
    "policy_seed": 20261601,
    "environment_seed": 20261601,
    "warm_start": "actor_critic_and_normalizers_only",
    "reset_optimizer_iteration_and_environment_steps": True,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 8000,
    "save_interval": 500,
    "evaluation_gates": [0, 500, 1000, 2000, 3500, 5000, 7999],
    "learning_rate": 1e-5,
    "wandb_mode": "offline",
    "device": 0,
    "reserved_idle_devices": [1, 2, 3, 4, 5, 6, 7],
    "no_automatic_restart": True,
  }
  training = protocol.get("training_protocol", {})
  for key, value in expected_training.items():
    if training.get(key) != value:
      raise ValueError(f"A14 training protocol {key} drifted")
  audit = protocol.get("implementation_audit", {})
  if (
    audit.get("status") != "PASSED_REAL_MUJOCO_WARM_START_SMOKE"
    or audit.get("code_commit") != _MINIMUM_COMMIT
    or audit.get("source_checkpoint_sha256") != _SOURCE_SHA256
  ):
    raise ValueError("A14 implementation smoke drifted")
  return protocol, digest


def _validate_smoke(protocol: dict[str, Any], repo: Path) -> dict[str, Any]:
  audit = protocol["implementation_audit"]
  for name, row in audit["runtime_files"].items():
    path = repo / row["path"]
    if not path.is_file() or _sha256(path) != row["sha256"]:
      raise RuntimeError(f"A14_SMOKE_ALERT: {name} missing or drifted")
  log = (repo / audit["runtime_files"]["log"]["path"]).read_text(errors="replace")
  required = (
    "Learning iteration 0/1",
    "Total steps: 384",
    "Linear(in_features=93, out_features=512",
    "Linear(in_features=960, out_features=512",
    "target_velocity_max",
    "target_acceleration_max",
    "target_limited_fraction",
  )
  fatal = ("Traceback", "CUDA out of memory", "OutOfMemoryError", "PHYSICAL_RESET_ALERT")
  if any(token not in log for token in required) or any(token in log for token in fatal):
    raise RuntimeError("A14_SMOKE_ALERT: runtime log contract failed")
  checkpoint = repo / audit["runtime_files"]["checkpoint"]["path"]
  integrity = _validate_checkpoint(checkpoint, 0)
  verified = audit["verified"]
  if (
    integrity["tensor_count"] != verified["checkpoint_tensor_count"]
    or integrity["tensor_elements"] != verified["checkpoint_tensor_elements"]
  ):
    raise RuntimeError("A14_SMOKE_ALERT: checkpoint tensor inventory drifted")
  return {"status": audit["status"], "code_commit": audit["code_commit"]}


def build_plan(cfg: VelocityEnvelopeFinetuneCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo / cfg.protocol
  protocol, protocol_sha = _validate_protocol(protocol_path)
  commit = _git(repo, "rev-parse", "HEAD")
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  from mjlab.tasks.registry import load_runner_cls

  import smp.rl.tasks  # noqa: F401

  runner = load_runner_cls(_TASK)
  if runner is None or runner.__name__ != _EXPECTED_RUNNER:
    raise RuntimeError("A14 task lacks the fresh-optimizer warm-start runner")
  source = protocol["source_policy"]
  source_checkpoint = (repo / source["checkpoint_path"]).resolve()
  if not source_checkpoint.is_file() or _sha256(source_checkpoint) != _SOURCE_SHA256:
    raise RuntimeError("A14_SOURCE_ALERT: checkpoint missing or drifted")
  source_integrity = _validate_checkpoint(source_checkpoint, 4999)
  training = protocol["training_protocol"]
  source_name = "a14_source_a13_seed20261501_gate4999"
  experiment_root = repo / "logs/rsl_rl/smp_scratch_a14_f2s2_velocity_envelope_g1"
  source_link = experiment_root / source_name / source["checkpoint_name"]
  run_name = "a14_velocity_envelope_8k_seed20261601_offline"
  command = [
    str(repo / ".venv/bin/python"), "scripts/train.py", _TASK,
    "--env.scene.num-envs", str(training["num_envs"]),
    "--agent.seed", str(training["policy_seed"]),
    "--env.seed", str(training["environment_seed"]),
    "--agent.resume", "True", "--agent.load-run", f"^{source_name}$",
    "--agent.load-checkpoint", f"^{source['checkpoint_name']}$",
    "--agent.max-iterations", str(training["max_iterations"]),
    "--agent.save-interval", str(training["save_interval"]),
    "--agent.algorithm.learning-rate", str(training["learning_rate"]),
    "--agent.run-name", run_name,
  ]
  material = {
    "protocol_sha256": protocol_sha, "code_commit": commit, "task": _TASK,
    "source_checkpoint_sha256": _SOURCE_SHA256,
    "policy_seed": training["policy_seed"], "environment_seed": training["environment_seed"],
    "command": command, "wandb_mode": training["wandb_mode"],
  }
  plan_id = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
  return {
    "schema_version": 1, "status": "PLANNED", "study_id": protocol["study_id"],
    "study_role": protocol["study_role"], "plan_id": plan_id,
    "protocol": str(protocol_path.resolve()), "protocol_sha256": protocol_sha,
    "code_commit": commit, "task": _TASK, "runner": _EXPECTED_RUNNER,
    "source_checkpoint": str(source_checkpoint), "source_checkpoint_sha256": _SOURCE_SHA256,
    "source_checkpoint_integrity": source_integrity, "source_link": str(source_link),
    "experiment_root": str(experiment_root), "policy_seed": training["policy_seed"],
    "environment_seed": training["environment_seed"], "num_envs": training["num_envs"],
    "max_iterations": training["max_iterations"], "save_interval": training["save_interval"],
    "evaluation_gates": training["evaluation_gates"], "learning_rate": training["learning_rate"],
    "wandb_mode": training["wandb_mode"], "gpu": training["device"],
    "reserved_idle_devices": training["reserved_idle_devices"], "run_name": run_name,
    "log": str(control / "gpu0_a14_seed20261601.log"),
    "pid_file": str(control / "gpu0_a14_seed20261601.pid"),
    "command": command, "pid": None, "claim_boundary": protocol["claim_boundary"],
  }


def launch_finetune(cfg: VelocityEnvelopeFinetuneCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = Path(planned["log"]).parent / "launch_manifest.json"
  if state_path.exists():
    existing = _json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing A14 run has a different immutable plan")
    if existing.get("status") == "LAUNCHED":
      pid = existing.get("pid")
      finals = list(Path(existing["experiment_root"]).glob(f"*_{existing['run_name']}/model_7999.pt"))
      if pid is not None and not _pid_alive(int(pid)) and len(finals) != 1:
        raise RuntimeError("A14_TRAINING_ALERT: worker died; no automatic restart")
      return existing
    raise RuntimeError(f"A14_TRAINING_ALERT: inadmissible state {existing.get('status')}")
  if not cfg.launch:
    return planned
  repo = Path(__file__).resolve().parents[1]
  if _git(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing A14 launch from tracked-dirty worktree")
  processes = _gpu_processes()
  if processes:
    raise RuntimeError(f"refusing A14 launch while GPU compute is active: {processes}")
  protocol = _json(Path(planned["protocol"]))
  planned["implementation_smoke"] = _validate_smoke(protocol, repo)
  planned["resource_preflight"] = _disk_preflight(repo)
  source, link = Path(planned["source_checkpoint"]), Path(planned["source_link"])
  link.parent.mkdir(parents=True, exist_ok=True)
  if link.exists() or link.is_symlink():
    if not link.is_symlink() or link.resolve() != source.resolve():
      raise ValueError(f"conflicting A14 warm-start link: {link}")
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
      planned["command"], cwd=repo, env=environment, stdout=stream,
      stderr=subprocess.STDOUT, start_new_session=True,
    )
  planned["pid"] = process.pid
  Path(planned["pid_file"]).write_text(f"{process.pid}\n")
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: VelocityEnvelopeFinetuneCfg) -> None:
  result = launch_finetune(cfg)
  print(f"{result['status']}: plan {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(VelocityEnvelopeFinetuneCfg))
