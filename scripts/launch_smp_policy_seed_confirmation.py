"""Safely launch preregistered policy-seed confirmation training jobs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from build_smp_causal_manifest import _ARMS


@dataclass(frozen=True)
class ConfirmationCfg:
  selection: Path
  control_dir: Path
  devices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
  seeds: tuple[int, ...] = (20260901, 20260902, 20260903)
  num_envs: int = 4096
  max_iterations: int = 30000
  save_interval: int = 1000
  launch: bool = False


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


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
    (
      "nvidia-smi",
      "--query-compute-apps=pid",
      "--format=csv,noheader,nounits",
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


def _validate_selection(path: Path) -> tuple[dict[str, Any], str]:
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = json.loads(path.read_text())
  if payload.get("status") != "PROMOTE_FOR_POLICY_SEEDS":
    raise ValueError("stable selection does not authorize policy-seed training")
  candidates = payload.get("promoted_candidates")
  if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
    raise ValueError("stable selection must promote one or two candidates")
  for source in payload.get("sources", []):
    source_path = Path(source["path"])
    if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
      raise ValueError(f"stable selection source changed: {source_path}")
  return payload, _sha256(path)


def build_plan(cfg: ConfirmationCfg) -> dict[str, Any]:
  selection, selection_sha = _validate_selection(cfg.selection)
  if len(cfg.seeds) != 3 or len(set(cfg.seeds)) != 3:
    raise ValueError("confirmation requires exactly three unique seeds")
  catalog = {arm["name"]: arm for arm in _ARMS}
  candidates = selection["promoted_candidates"]
  unknown = [arm for arm in candidates if arm not in catalog]
  if unknown:
    raise ValueError(f"selection contains unknown arms: {unknown}")
  job_count = len(candidates) * len(cfg.seeds)
  if len(cfg.devices) < job_count or len(set(cfg.devices)) != len(cfg.devices):
    raise ValueError("confirmation requires one unique GPU per concurrent job")

  jobs = []
  for index, (arm_name, seed) in enumerate(
    (arm, seed) for arm in candidates for seed in cfg.seeds
  ):
    arm = catalog[arm_name]
    gpu = cfg.devices[index]
    run_name = f"confirm_{arm_name}_30k_seed{seed}"
    log = cfg.control_dir / f"gpu{gpu}_{arm_name}_seed{seed}.log"
    pid_file = cfg.control_dir / f"gpu{gpu}_{arm_name}_seed{seed}.pid"
    command = [
      "uv",
      "run",
      "scripts/train.py",
      arm["task"],
      "--env.scene.num-envs",
      str(cfg.num_envs),
      "--agent.seed",
      str(seed),
      "--env.seed",
      str(seed),
      "--agent.max-iterations",
      str(cfg.max_iterations),
      "--agent.save-interval",
      str(cfg.save_interval),
      "--agent.run-name",
      run_name,
    ]
    jobs.append(
      {
        "arm": arm_name,
        "task": arm["task"],
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
    "selection_sha256": selection_sha,
    "candidates": candidates,
    "seeds": list(cfg.seeds),
    "devices": list(cfg.devices[:job_count]),
    "num_envs": cfg.num_envs,
    "max_iterations": cfg.max_iterations,
    "save_interval": cfg.save_interval,
    "jobs": [{k: v for k, v in job.items() if k != "pid"} for job in jobs],
  }
  plan_id = hashlib.sha256(
    json.dumps(plan_material, sort_keys=True).encode()
  ).hexdigest()
  repo_root = Path(__file__).resolve().parents[1]
  return {
    "schema_version": 1,
    "status": "PLANNED",
    "plan_id": plan_id,
    "selection": str(cfg.selection.resolve()),
    "selection_sha256": selection_sha,
    "screen_policy_seed": selection["policy_seed"],
    "confirmation_seeds": list(cfg.seeds),
    "code_commit": _git_commit(repo_root),
    "num_envs": cfg.num_envs,
    "max_iterations": cfg.max_iterations,
    "transitions_per_full_run": cfg.max_iterations * cfg.num_envs * 24,
    "jobs": jobs,
  }


def launch_confirmation(cfg: ConfirmationCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = cfg.control_dir / "launch_manifest.json"
  if state_path.exists():
    existing = json.loads(state_path.read_text())
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing confirmation launch has a different frozen plan")
    if existing.get("status") == "LAUNCHED":
      return existing
    if existing.get("status") != "LAUNCHING":
      raise ValueError(f"inadmissible confirmation state: {existing.get('status')}")
    planned = existing
  if not cfg.launch:
    return planned
  if _gpu_processes():
    raise RuntimeError("refusing to launch confirmation while a GPU process is active")

  cfg.control_dir.mkdir(parents=True, exist_ok=True)
  planned["status"] = "LAUNCHING"
  planned.setdefault("started_at_utc", datetime.now(timezone.utc).isoformat())
  _atomic_json(state_path, planned)
  repo_root = Path(__file__).resolve().parents[1]
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


def main(cfg: ConfirmationCfg) -> None:
  result = launch_confirmation(cfg)
  print(f"{result['status']}: {len(result['jobs'])} jobs, plan {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(ConfirmationCfg))
