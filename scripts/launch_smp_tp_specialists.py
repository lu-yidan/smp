"""Safely launch preregistered terrain and plate specialists for matched seeds."""

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

from select_smp_confirmed_flat_arm import FlatPromotionCfg, select


@dataclass(frozen=True)
class SpecialistLaunchCfg:
  promotion: Path
  control_dir: Path
  smoke_test: Path
  protocol: Path = Path("docs/ral_terrain_plate_protocol.json")
  logs_root: Path = Path("logs/rsl_rl")
  devices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
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
    raise ValueError("flat promotion does not authorize T/P specialists")
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


def _validate_smoke(
  path: Path, promotion_id: str, code_commit: str, tasks: set[str]
) -> str:
  smoke = _load(path)
  if (
    smoke.get("status") != "PASS"
    or smoke.get("promotion_id") != promotion_id
    or smoke.get("code_commit") != code_commit
    or set(smoke.get("tasks", [])) != tasks
  ):
    raise ValueError("physical smoke test does not match promotion/code/tasks")
  return _sha256(path)


def build_plan(cfg: SpecialistLaunchCfg) -> dict[str, Any]:
  promotion = _validate_promotion(cfg.promotion)
  protocol = _load(cfg.protocol)
  if _sha256(cfg.protocol) != promotion.get("protocol_sha256"):
    raise ValueError("launch protocol differs from flat promotion protocol")
  shared = protocol["shared_training"]
  seeds = tuple(int(seed) for seed in promotion["policy_seeds"])
  if tuple(sorted(seeds)) != tuple(sorted(shared["policy_seeds"])):
    raise ValueError("specialist seeds differ from frozen protocol")
  if len(cfg.devices) < 2 * len(seeds) or len(set(cfg.devices)) != len(cfg.devices):
    raise ValueError("T/P specialists require one unique GPU per concurrent job")

  arm = promotion["selected_arm"]
  arm_index = int(promotion["selected_arm_index"])
  repo_root = Path(__file__).resolve().parents[1]
  code_commit = _git_commit(repo_root)
  tasks = {
    "T": f"Smp-Getup-RAL-T-A{arm_index}-G1",
    "P": f"Smp-Getup-RAL-P-A{arm_index}-G1",
  }
  jobs = []
  combinations = [(phase, seed) for phase in ("T", "P") for seed in seeds]
  for index, (phase, seed) in enumerate(combinations):
    phase_lower = phase.lower()
    experiment = f"smp_ral_{phase_lower}_a{arm_index}_g1"
    source_name = f"flat_source_{arm}_seed{seed}"
    checkpoint_source = promotion["matched_flat_checkpoints"][str(seed)]
    checkpoint = Path(checkpoint_source["checkpoint"])
    if (
      not checkpoint.is_file()
      or _sha256(checkpoint) != checkpoint_source["checkpoint_sha256"]
    ):
      raise ValueError(f"matched flat checkpoint changed: {checkpoint}")
    source_link = cfg.logs_root / experiment / source_name / checkpoint.name
    run_name = f"ral_{phase_lower}_{arm}_20k_seed{seed}"
    gpu = cfg.devices[index]
    log = cfg.control_dir / f"gpu{gpu}_{phase_lower}_{arm}_seed{seed}.log"
    pid_file = cfg.control_dir / f"gpu{gpu}_{phase_lower}_{arm}_seed{seed}.pid"
    command = [
      "uv",
      "run",
      "scripts/train.py",
      tasks[phase],
      "--env.scene.num-envs",
      str(shared["num_envs"]),
      "--agent.seed",
      str(seed),
      "--env.seed",
      str(seed),
      "--agent.resume",
      "True",
      "--agent.load-run",
      f"^{source_name}$",
      "--agent.load-checkpoint",
      f"^{checkpoint.name}$",
      "--agent.max-iterations",
      str(shared["max_updates"]),
      "--agent.save-interval",
      str(shared["save_interval"]),
      "--agent.algorithm.learning-rate",
      str(shared["learning_rate"]),
      "--agent.run-name",
      run_name,
    ]
    jobs.append(
      {
        "phase": phase,
        "arm": arm,
        "arm_index": arm_index,
        "task": tasks[phase],
        "policy_seed": seed,
        "environment_seed": seed,
        "gpu": gpu,
        "experiment": experiment,
        "run_name": run_name,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_source["checkpoint_sha256"],
        "source_link": str(source_link.resolve()),
        "log": str(log.resolve()),
        "pid_file": str(pid_file.resolve()),
        "command": command,
        "pid": None,
      }
    )

  material = {
    "promotion_id": promotion["promotion_id"],
    "code_commit": code_commit,
    "protocol_sha256": _sha256(cfg.protocol),
    "jobs": [{k: v for k, v in job.items() if k != "pid"} for job in jobs],
  }
  plan_id = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
  return {
    "schema_version": 1,
    "status": "PLANNED",
    "plan_id": plan_id,
    "promotion": str(cfg.promotion.resolve()),
    "promotion_sha256": _sha256(cfg.promotion),
    "promotion_id": promotion["promotion_id"],
    "protocol": str(cfg.protocol.resolve()),
    "protocol_sha256": _sha256(cfg.protocol),
    "code_commit": code_commit,
    "selected_arm": arm,
    "policy_seeds": list(seeds),
    "num_envs": shared["num_envs"],
    "max_updates": shared["max_updates"],
    "transitions_per_run": shared["transitions_per_run"],
    "jobs": jobs,
  }


def launch_specialists(cfg: SpecialistLaunchCfg) -> dict[str, Any]:
  planned = build_plan(cfg)
  state_path = cfg.control_dir / "launch_manifest.json"
  if state_path.exists():
    existing = _load(state_path)
    if existing.get("plan_id") != planned["plan_id"]:
      raise ValueError("existing T/P launch has a different frozen plan")
    if existing.get("status") == "LAUNCHED":
      return existing
    if existing.get("status") != "LAUNCHING":
      raise ValueError(f"inadmissible T/P launch state: {existing.get('status')}")
    planned = existing
  if not cfg.launch:
    return planned
  if _gpu_processes():
    raise RuntimeError("refusing to launch T/P while a GPU process is active")
  smoke_sha = _validate_smoke(
    cfg.smoke_test,
    planned["promotion_id"],
    planned["code_commit"],
    {job["task"] for job in planned["jobs"]},
  )

  cfg.control_dir.mkdir(parents=True, exist_ok=True)
  planned["status"] = "LAUNCHING"
  planned["smoke_test"] = str(cfg.smoke_test.resolve())
  planned["smoke_test_sha256"] = smoke_sha
  planned.setdefault("started_at_utc", datetime.now(timezone.utc).isoformat())
  _atomic_json(state_path, planned)
  repo_root = Path(__file__).resolve().parents[1]
  for job in planned["jobs"]:
    if job.get("pid") is not None:
      if not _pid_alive(int(job["pid"])):
        raise RuntimeError(f"partially launched job exited: {job['run_name']}")
      continue
    source = Path(job["source_checkpoint"])
    link = Path(job["source_link"])
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
      if not link.is_symlink() or link.resolve() != source.resolve():
        raise ValueError(f"conflicting warm-start link: {link}")
    else:
      link.symlink_to(source.resolve())
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


def main(cfg: SpecialistLaunchCfg) -> None:
  result = launch_specialists(cfg)
  print(f"{result['status']}: {len(result['jobs'])} jobs, plan {result['plan_id']}")


if __name__ == "__main__":
  main(tyro.cli(SpecialistLaunchCfg))
