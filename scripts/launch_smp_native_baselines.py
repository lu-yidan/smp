"""Safely launch the three native matched-bank Tier-A PPO baselines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from audit_smp_baseline_registry import audit as audit_registry
from build_smp_causal_manifest import _ARMS
from select_smp_confirmed_flat_arm import FlatPromotionCfg, select

_NATIVE_METHOD_TASK_NAMES = {
  "task_only_ppo": "TaskOnly",
  "original_product_smp": "OriginalSMP",
  "proposed_smp_recovery": "ProposedSMP",
}
_ADAPTER_METHODS = {"firm_r_deployable", "recovery_tracking"}
_ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_FATAL_PATTERNS = ("Traceback", "CUDA out of memory", "Segmentation fault")


@dataclass(frozen=True)
class NativeBaselineLaunchCfg:
  promotion: Path
  runtime_registry: Path
  control_dir: Path = Path("run_control/ral_baselines/native_training")
  devices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
  launch: bool = False


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _git_commit(repo_root: Path) -> str:
  result = subprocess.run(
    ("git", "rev-parse", "HEAD"),
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _gpu_processes() -> list[str]:
  result = subprocess.run(
    ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"),
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


def _validate_promotion(path: Path) -> dict[str, Any]:
  promotion = _load(path)
  if promotion.get("status") != "PROMOTE_TP_SPECIALISTS":
    raise ValueError("flat promotion does not authorize Tier-A baselines")
  recomputed = select(
    FlatPromotionCfg(
      aggregate=Path(promotion["aggregate"]),
      confirmation_manifest_index=Path(promotion["confirmation_manifest_index"]),
      protocol=Path(promotion["protocol"]),
      output=path,
    )
  )
  if recomputed["promotion_id"] != promotion.get("promotion_id"):
    raise ValueError("flat promotion sources or decision changed")
  return promotion


def _validate_registry(
  path: Path, promotion_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
  registry = _load(path)
  report = audit_registry(registry, path)
  reports = {method["id"]: method for method in report["methods"]}
  if not report["reset_bank_ready"]:
    raise ValueError("matched reset bank is not ready")
  for method in _NATIVE_METHOD_TASK_NAMES:
    if (
      reports[method]["status"] != "ready_for_training" or reports[method]["blocked_on"]
    ):
      raise ValueError(f"native baseline is not ready for training: {method}")
  for method in _ADAPTER_METHODS:
    if reports[method]["status"] != "blocked":
      raise ValueError(f"adapter baseline must remain separately blocked: {method}")

  bank = registry["shared_reset_bank"]
  bank_path = Path(bank["result_path"])
  bank_path = bank_path if bank_path.is_absolute() else path.parent / bank_path
  manifest_path = Path(bank["manifest_path"])
  manifest_path = (
    manifest_path if manifest_path.is_absolute() else path.parent / manifest_path
  )
  manifest = _load(manifest_path)
  if manifest.get("promotion_id") != promotion_id:
    raise ValueError("matched reset bank belongs to a different flat promotion")
  return registry, report


def _worker_command(jobs: list[dict[str, Any]]) -> str:
  commands = ["set -euo pipefail"]
  for job in jobs:
    commands.append(f"{shlex.join(job['command'])} >> {shlex.quote(job['log'])} 2>&1")
  return "\n".join(commands)


def _job_log_completed(path: Path, max_updates: int) -> bool:
  if not path.is_file():
    return False
  text = path.read_text(errors="replace")
  if any(pattern in text for pattern in _FATAL_PATTERNS):
    return False
  matches = list(_ITERATION.finditer(text))
  if not matches:
    return False
  iteration, maximum = (int(value) for value in matches[-1].groups())
  return maximum == max_updates and iteration >= maximum - 1


def _validate_launched_workers(plan: dict[str, Any]) -> None:
  jobs = {job["job_id"]: job for job in plan["jobs"]}
  failures = []
  for worker in plan["workers"]:
    pid = worker.get("pid")
    if pid is not None and _pid_alive(int(pid)):
      continue
    incomplete = [
      job_id
      for job_id in worker["job_ids"]
      if not _job_log_completed(Path(jobs[job_id]["log"]), int(plan["max_updates"]))
    ]
    if incomplete:
      failures.append(f"gpu{worker['gpu']}:{','.join(incomplete)}")
  if failures:
    raise RuntimeError(
      "Tier-A worker exited before its immutable queue completed: "
      + "; ".join(failures)
    )


def build_plan(cfg: NativeBaselineLaunchCfg) -> dict[str, Any]:
  if not cfg.devices or len(set(cfg.devices)) != len(cfg.devices):
    raise ValueError("baseline launch requires one or more unique GPUs")
  if any(device < 0 for device in cfg.devices):
    raise ValueError("GPU indexes must be non-negative")

  promotion = _validate_promotion(cfg.promotion)
  registry, readiness = _validate_registry(
    cfg.runtime_registry, promotion["promotion_id"]
  )
  budget = registry["training_budget"]
  seeds = tuple(int(seed) for seed in budget["policy_seeds"])
  if len(seeds) != 3 or len(set(seeds)) != 3:
    raise ValueError("Tier-A training requires exactly three unique policy seeds")
  num_envs = int(budget["num_envs"])
  max_updates = int(budget["max_updates"])
  rollout_steps = int(budget["transitions_per_env_per_update"])
  save_interval = int(budget["save_interval"])
  if save_interval != 1000:
    raise ValueError("Tier-A checkpoint interval must remain 1000 updates")

  selected_arm = str(promotion["selected_arm"])
  arm_index = int(promotion["selected_arm_index"])
  catalog = {index: arm["name"] for index, arm in enumerate(_ARMS)}
  if catalog.get(arm_index) != selected_arm:
    raise ValueError("selected arm index disagrees with the frozen arm catalog")

  bank = registry["shared_reset_bank"]
  bank_path = Path(bank["result_path"]).resolve()
  bank_sha256 = str(bank["sha256"])
  bank_manifest = Path(bank["manifest_path"]).resolve()
  repo_root = Path(__file__).resolve().parents[1]
  code_commit = _git_commit(repo_root)

  jobs: list[dict[str, Any]] = []
  combinations = [
    (method, seed) for seed in seeds for method in _NATIVE_METHOD_TASK_NAMES
  ]
  for index, (method, seed) in enumerate(combinations):
    gpu = cfg.devices[index % len(cfg.devices)]
    method_task_name = _NATIVE_METHOD_TASK_NAMES[method]
    task = f"Smp-Getup-RAL-B-{method_task_name}-A{arm_index}-G1"
    run_name = f"ral_b_{method}_a{arm_index}_30k_seed{seed}"
    log = cfg.control_dir / f"gpu{gpu}_{method}_seed{seed}.log"
    command = [
      "uv",
      "run",
      "scripts/train.py",
      task,
      "--env.scene.num-envs",
      str(num_envs),
      "--agent.seed",
      str(seed),
      "--env.seed",
      str(seed),
      "--agent.resume",
      "False",
      "--agent.max-iterations",
      str(max_updates),
      "--agent.save-interval",
      str(save_interval),
      "--agent.run-name",
      run_name,
      "--env.events.init-matched-reset-bank.params.bank-path",
      str(bank_path),
      "--env.events.init-matched-reset-bank.params.bank-sha256",
      bank_sha256,
      "--env.events.init-matched-reset-bank.params.expected-num-states",
      str(bank["num_states"]),
    ]
    jobs.append(
      {
        "job_id": f"{method}_seed{seed}",
        "method": method,
        "task": task,
        "selected_arm": selected_arm,
        "arm_index": arm_index,
        "policy_seed": seed,
        "environment_seed": seed,
        "gpu": gpu,
        "run_name": run_name,
        "log": str(log.resolve()),
        "command": command,
      }
    )

  workers = []
  for gpu in cfg.devices:
    assigned = [job for job in jobs if job["gpu"] == gpu]
    if not assigned:
      continue
    workers.append(
      {
        "gpu": gpu,
        "job_ids": [job["job_id"] for job in assigned],
        "worker_log": str((cfg.control_dir / f"worker_gpu{gpu}.log").resolve()),
        "pid_file": str((cfg.control_dir / f"worker_gpu{gpu}.pid").resolve()),
        "command": _worker_command(assigned),
        "pid": None,
      }
    )

  material = {
    "promotion_id": promotion["promotion_id"],
    "promotion_sha256": _sha256(cfg.promotion),
    "runtime_registry_sha256": _sha256(cfg.runtime_registry),
    "bank_sha256": bank_sha256,
    "bank_manifest_sha256": _sha256(bank_manifest),
    "code_commit": code_commit,
    "jobs": jobs,
    "workers": [{k: v for k, v in worker.items() if k != "pid"} for worker in workers],
  }
  plan_id = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  return {
    "schema_version": 1,
    "status": "PLANNED",
    "plan_id": plan_id,
    "promotion": str(cfg.promotion.resolve()),
    "promotion_sha256": _sha256(cfg.promotion),
    "promotion_id": promotion["promotion_id"],
    "runtime_registry": str(cfg.runtime_registry.resolve()),
    "runtime_registry_sha256": _sha256(cfg.runtime_registry),
    "readiness_status": readiness["status"],
    "bank": str(bank_path),
    "bank_sha256": bank_sha256,
    "bank_manifest": str(bank_manifest),
    "bank_manifest_sha256": _sha256(bank_manifest),
    "code_commit": code_commit,
    "selected_arm": selected_arm,
    "arm_index": arm_index,
    "policy_seeds": list(seeds),
    "num_envs": num_envs,
    "max_updates": max_updates,
    "save_interval": save_interval,
    "transitions_per_run": max_updates * num_envs * rollout_steps,
    "scheduler": "round_robin_one_sequential_worker_per_physical_gpu",
    "jobs": jobs,
    "workers": workers,
    "claim_boundary": "A launch manifest is protocol provenance, not performance evidence.",
  }


def launch_baselines(cfg: NativeBaselineLaunchCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = cfg.control_dir / "launch_manifest.json"
  if state_path.exists():
    existing = _load(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing Tier-A launch has a different frozen plan")
    if existing.get("status") == "LAUNCHED":
      _validate_launched_workers(existing)
      return existing
    if existing.get("status") != "LAUNCHING":
      raise ValueError(f"inadmissible Tier-A launch state: {existing.get('status')}")
    planned = existing
  if not cfg.launch:
    return planned
  if _gpu_processes():
    raise RuntimeError("refusing Tier-A launch while a GPU process is active")

  cfg.control_dir.mkdir(parents=True, exist_ok=True)
  planned["status"] = "LAUNCHING"
  planned.setdefault("started_at_utc", datetime.now(timezone.utc).isoformat())
  _atomic_json(state_path, planned)
  repo_root = Path(__file__).resolve().parents[1]
  for worker in planned["workers"]:
    if worker.get("pid") is not None:
      if not _pid_alive(int(worker["pid"])):
        raise RuntimeError(f"partially launched worker exited: gpu{worker['gpu']}")
      continue
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
    with Path(worker["worker_log"]).open("a") as stream:
      process = subprocess.Popen(
        ("bash", "-lc", worker["command"]),
        cwd=repo_root,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
    worker["pid"] = process.pid
    Path(worker["pid_file"]).write_text(f"{process.pid}\n")
    _atomic_json(state_path, planned)
  planned["status"] = "LAUNCHED"
  planned["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
  _atomic_json(state_path, planned)
  return planned


def main(cfg: NativeBaselineLaunchCfg) -> None:
  result = launch_baselines(cfg)
  print(
    f"{result['status']}: {len(result['jobs'])} jobs on "
    f"{len(result['workers'])} GPU workers, plan {result['plan_id']}"
  )


if __name__ == "__main__":
  main(tyro.cli(NativeBaselineLaunchCfg))
