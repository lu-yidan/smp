"""Fail-closed launcher for the matched V34 96D-to-93D ablation."""

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

_PROTOCOL_SHA256 = "11a0828e38ce83c27693b7b50a59778a5a67c690b4329bdf0fc2357a036d18ca"
_MINIMUM_COMMIT = "6e8447efacf3fea45e412b0d24aed1e1e04a119d"
_SOURCE_SHA256 = "fa54ac58f09a1a0ed0b46f96fb920f18de20422190c9ee92207f3080a3cbe393"
_PROJECTED_SHA256 = "b24d3aa9003fe2a685b8be42a5144ee95b330e43440831bdf29364ebd5953f7a"
_PROJECTION_MANIFEST_SHA256 = (
  "ac94496c1ce54b2cd6dfc2150bb50c09a66557913e859bcd642258cc866227bb"
)
_TASK = "Smp-Getup-Escape-Plate-V34-93D-G1"
_EXPECTED_RUNNER = "SmpCurriculumWarmStartRunner"
_GIB = 1024**3


@dataclass(frozen=True)
class V34NinetyThreeDimAblationCfg:
  protocol: Path = Path("docs/ral_v34_93d_observation_ablation_v1.json")
  control_dir: Path = Path("run_control/v34_93d_control/training")
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


def _gpu_preflight(device: int) -> dict[str, Any]:
  inventory = subprocess.run(
    ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"),
    check=True,
    capture_output=True,
    text=True,
  ).stdout.splitlines()
  mapping: dict[int, str] = {}
  for row in inventory:
    index, uuid = (value.strip() for value in row.split(",", maxsplit=1))
    mapping[int(index)] = uuid
  if device not in mapping:
    raise RuntimeError(f"V34_93D_RESOURCE_ALERT: GPU {device} does not exist")
  processes = subprocess.run(
    (
      "nvidia-smi",
      "--query-compute-apps=gpu_uuid,pid,process_name",
      "--format=csv,noheader",
    ),
    check=True,
    capture_output=True,
    text=True,
  ).stdout.splitlines()
  active = [row.strip() for row in processes if row.strip()]
  conflicts = [row for row in active if row.startswith(mapping[device] + ",")]
  if conflicts:
    raise RuntimeError(
      f"V34_93D_RESOURCE_ALERT: physical GPU {device} is occupied: {conflicts}"
    )
  return {
    "physical_device": device,
    "gpu_uuid": mapping[device],
    "other_gpu_compute_processes_at_launch": active,
  }


def _tensor_inventory(value: Any) -> tuple[int, int, bool]:
  tensors: list[torch.Tensor] = []

  def collect(item: Any) -> None:
    if torch.is_tensor(item):
      tensors.append(item)
    elif isinstance(item, dict):
      for child in item.values():
        collect(child)
    elif isinstance(item, (tuple, list)):
      for child in item:
        collect(child)

  collect(value)
  return (
    len(tensors),
    sum(tensor.numel() for tensor in tensors),
    bool(tensors) and all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
  )


def _validate_projected_checkpoint(path: Path) -> dict[str, Any]:
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  actor = checkpoint["actor_state_dict"]
  critic = checkpoint["critic_state_dict"]
  inventory = _tensor_inventory({"actor": actor, "critic": critic})
  if (
    checkpoint.get("iter") != 98000
    or tuple(actor["mlp.0.weight"].shape) != (512, 93)
    or tuple(actor["obs_normalizer._mean"].shape) != (1, 93)
    or tuple(critic["mlp.0.weight"].shape) != (512, 960)
    or checkpoint.get("optimizer_state_dict") != {}
    or not inventory[2]
  ):
    raise RuntimeError("V34_93D_PROJECTION_ALERT: projected checkpoint is invalid")
  return {
    "embedded_iteration": 98000,
    "actor_input_dim": 93,
    "critic_input_dim": 960,
    "tensor_count": inventory[0],
    "tensor_elements": inventory[1],
    "all_tensors_finite": True,
    "optimizer_discarded": True,
  }


def _validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
  digest = _sha256(path) if path.is_file() else ""
  if digest != _PROTOCOL_SHA256:
    raise ValueError("V34 93D ablation protocol SHA-256 mismatch")
  protocol = _json(path)
  if (
    protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING"
    or protocol.get("study_id") != "smp-v34-93d-observation-ablation-v1"
  ):
    raise ValueError("V34 93D ablation protocol is not launch eligible")
  projection = protocol["projection"]
  expected_projection = {
    "source_checkpoint_sha256": _SOURCE_SHA256,
    "projected_checkpoint_sha256": _PROJECTED_SHA256,
    "manifest_sha256": _PROJECTION_MANIFEST_SHA256,
    "removed_actor_indices": [0, 1, 2],
    "removed_actor_term": "base_lin_vel",
    "bias_reference_raw_base_lin_vel": [0.0, 0.0, 0.0],
    "critic_unchanged": True,
    "optimizer_discarded": True,
    "projected_actor_input_dim": 93,
    "critic_input_dim": 960,
  }
  for key, value in expected_projection.items():
    if projection.get(key) != value:
      raise ValueError(f"V34 93D projection field {key} drifted")
  training = protocol["training_protocol"]
  expected_training = {
    "policy_seed": 20261701,
    "environment_seed": 20261701,
    "warm_start": "projected_actor_plus_unchanged_critic_and_normalizers",
    "reset_optimizer_iteration_and_environment_steps": True,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 12000,
    "save_interval": 1000,
    "evaluation_gates": [0, 1000, 3000, 6000, 9000, 11999],
    "learning_rate": 0.0001,
    "wandb_mode": "offline",
    "physical_device": 1,
    "no_automatic_restart": True,
  }
  for key, value in expected_training.items():
    if training.get(key) != value:
      raise ValueError(f"V34 93D training field {key} drifted")
  return protocol, digest


def _validate_runtime_artifacts(protocol: dict[str, Any], repo: Path) -> dict[str, Any]:
  projection = protocol["projection"]
  expected = {
    projection["source_checkpoint_path"]: _SOURCE_SHA256,
    projection["projected_checkpoint_path"]: _PROJECTED_SHA256,
    projection["manifest_path"]: _PROJECTION_MANIFEST_SHA256,
  }
  for relative, digest in expected.items():
    path = repo / relative
    if not path.is_file() or _sha256(path) != digest:
      raise RuntimeError(f"V34_93D_SOURCE_ALERT: {relative} missing or drifted")
  manifest = _json(repo / projection["manifest_path"])
  if (
    manifest.get("status") != "PROJECTED_SOURCE_READY_NOT_PERFORMANCE_EVIDENCE"
    or manifest.get("projected_checkpoint_sha256") != _PROJECTED_SHA256
    or manifest.get("source_checkpoint_sha256") != _SOURCE_SHA256
  ):
    raise RuntimeError("V34_93D_PROJECTION_ALERT: projection manifest drifted")
  smoke = protocol["implementation_audit"]
  if smoke.get("status") != "PASSED_REAL_MUJOCO_PROJECTED_WARM_START_SMOKE":
    raise RuntimeError("V34_93D_SMOKE_ALERT: smoke status is not passed")
  for row in smoke["runtime_files"].values():
    path = repo / row["path"]
    if not path.is_file() or _sha256(path) != row["sha256"]:
      raise RuntimeError(f"V34_93D_SMOKE_ALERT: {path} missing or drifted")
  log = (repo / smoke["runtime_files"]["log"]["path"]).read_text(errors="replace")
  required = (
    "Learning iteration 0/2",
    "Learning iteration 1/2",
    "Linear(in_features=93, out_features=512",
    "Linear(in_features=960, out_features=512",
  )
  fatal = ("Traceback", "CUDA out of memory", "OutOfMemoryError")
  if any(token not in log for token in required) or any(token in log for token in fatal):
    raise RuntimeError("V34_93D_SMOKE_ALERT: log contract failed")
  checkpoint = repo / smoke["runtime_files"]["checkpoint"]["path"]
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  inventory = _tensor_inventory(payload)
  if (
    payload.get("iter") != 0
    or tuple(payload["actor_state_dict"]["mlp.0.weight"].shape) != (512, 93)
    or tuple(payload["critic_state_dict"]["mlp.0.weight"].shape) != (512, 960)
    or inventory[:2]
    != (smoke["checkpoint_tensor_count"], smoke["checkpoint_tensor_elements"])
    or not inventory[2]
  ):
    raise RuntimeError("V34_93D_SMOKE_ALERT: checkpoint contract failed")
  return _validate_projected_checkpoint(repo / projection["projected_checkpoint_path"])


def build_plan(cfg: V34NinetyThreeDimAblationCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  protocol_path = cfg.protocol if cfg.protocol.is_absolute() else repo / cfg.protocol
  protocol, protocol_sha = _validate_protocol(protocol_path)
  commit = _git(repo, "rev-parse", "HEAD")
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, commit)
  from mjlab.tasks.registry import load_runner_cls

  import smp.rl.tasks  # noqa: F401

  runner = load_runner_cls(_TASK)
  if runner is None or runner.__name__ != _EXPECTED_RUNNER:
    raise RuntimeError("V34 93D task lacks the fresh-optimizer warm-start runner")
  projection = protocol["projection"]
  projected = (repo / projection["projected_checkpoint_path"]).resolve()
  experiment_root = repo / "logs/rsl_rl/smp_getup_escape_plate_v34_93d_g1"
  source_name = "v34_model98000_zero_velocity_projection"
  source_link = experiment_root / source_name / "model_98000.pt"
  training = protocol["training_protocol"]
  run_name = "v34_93d_ablation_12k_seed20261701_offline"
  command = [
    str(repo / ".venv/bin/python"),
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
    "^model_98000.pt$",
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
    "source_checkpoint_sha256": _PROJECTED_SHA256,
    "policy_seed": training["policy_seed"],
    "environment_seed": training["environment_seed"],
    "command": command,
    "physical_device": training["physical_device"],
  }
  plan_id = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  control = cfg.control_dir if cfg.control_dir.is_absolute() else repo / cfg.control_dir
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
    "source_checkpoint": str(projected),
    "source_checkpoint_sha256": _PROJECTED_SHA256,
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
    "physical_device": training["physical_device"],
    "run_name": run_name,
    "log": str(control / "gpu1_v34_93d_seed20261701.log"),
    "pid_file": str(control / "gpu1_v34_93d_seed20261701.pid"),
    "command": command,
    "pid": None,
    "claim_boundary": protocol["claim_boundary"],
  }


def launch_ablation(cfg: V34NinetyThreeDimAblationCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = Path(planned["log"]).parent / "launch_manifest.json"
  if state_path.exists():
    existing = _json(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing V34 93D run has a different immutable plan")
    if existing.get("status") == "LAUNCHED":
      pid = existing.get("pid")
      finals = list(
        Path(existing["experiment_root"]).glob(
          f"*_{existing['run_name']}/model_11999.pt"
        )
      )
      if pid is not None and not _pid_alive(int(pid)) and len(finals) != 1:
        raise RuntimeError("V34_93D_TRAINING_ALERT: worker died; no automatic restart")
      return existing
    raise RuntimeError(
      f"V34_93D_TRAINING_ALERT: inadmissible state {existing.get('status')}"
    )
  if not cfg.launch:
    return planned
  repo = Path(__file__).resolve().parents[1]
  if _git(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("refusing V34 93D launch from tracked-dirty worktree")
  protocol = _json(Path(planned["protocol"]))
  planned["projected_source_audit"] = _validate_runtime_artifacts(protocol, repo)
  planned["disk_preflight"] = _disk_preflight(repo)
  planned["gpu_preflight"] = _gpu_preflight(int(planned["physical_device"]))
  source = Path(planned["source_checkpoint"])
  link = Path(planned["source_link"])
  link.parent.mkdir(parents=True, exist_ok=True)
  if link.exists() or link.is_symlink():
    if not link.is_symlink() or link.resolve() != source.resolve():
      raise ValueError(f"conflicting V34 93D warm-start link: {link}")
  else:
    link.symlink_to(source)
  state_path.parent.mkdir(parents=True, exist_ok=True)
  planned["status"] = "LAUNCHING"
  planned["started_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  environment = os.environ.copy()
  environment["CUDA_VISIBLE_DEVICES"] = str(planned["physical_device"])
  environment["WANDB_MODE"] = str(planned["wandb_mode"])
  environment["PYTHONUNBUFFERED"] = "1"
  with Path(planned["log"]).open("a") as stream:
    process = subprocess.Popen(
      planned["command"],
      cwd=repo,
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


def main(cfg: V34NinetyThreeDimAblationCfg) -> None:
  result = launch_ablation(cfg)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(V34NinetyThreeDimAblationCfg))
