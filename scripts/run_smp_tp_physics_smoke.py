"""Run one real MuJoCo update for the promoted T and P task configurations."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from build_smp_causal_manifest import _recorded_seed, _sha256
from launch_smp_tp_specialists import SpecialistLaunchCfg, build_plan


@dataclass(frozen=True)
class TpSmokeCfg:
  promotion: Path
  output: Path
  work_dir: Path
  protocol: Path = Path("docs/ral_terrain_plate_protocol.json")
  device: int = 0
  num_envs: int = 256
  timeout_s: int = 1800
  run: bool = False


def _gpu_processes() -> list[str]:
  result = subprocess.run(
    ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"),
    check=True,
    capture_output=True,
    text=True,
  )
  return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _set_arg(command: list[str], flag: str, value: str) -> None:
  index = command.index(flag)
  command[index + 1] = value


def _metric_values(text: str, name: str) -> list[float]:
  pattern = rf"Episode_Metrics/{re.escape(name)}:\s*([-+0-9.eE]+)"
  return [float(value) for value in re.findall(pattern, text)]


def _source_link(job: dict[str, Any]) -> None:
  source = Path(job["source_checkpoint"])
  link = Path(job["source_link"])
  if not source.is_file() or _sha256(source) != job["source_checkpoint_sha256"]:
    raise ValueError(f"smoke source checkpoint changed: {source}")
  link.parent.mkdir(parents=True, exist_ok=True)
  if link.exists() or link.is_symlink():
    if not link.is_symlink() or link.resolve() != source.resolve():
      raise ValueError(f"conflicting smoke warm-start link: {link}")
  else:
    link.symlink_to(source.resolve())


def _completed_result_is_current(
  path: Path, promotion_id: str, code_commit: str, tasks: set[str]
) -> dict[str, Any] | None:
  if not path.is_file():
    return None
  payload = json.loads(path.read_text())
  if (
    payload.get("status") == "PASS"
    and payload.get("promotion_id") == promotion_id
    and payload.get("code_commit") == code_commit
    and set(payload.get("tasks", [])) == tasks
  ):
    return payload
  raise ValueError("existing smoke result conflicts with current promotion/code")


def run_smoke(cfg: TpSmokeCfg) -> dict[str, Any]:
  launch_cfg = SpecialistLaunchCfg(
    promotion=cfg.promotion,
    control_dir=cfg.work_dir / "unused_launch_control",
    smoke_test=cfg.output,
    protocol=cfg.protocol,
    logs_root=cfg.work_dir / "logs",
    devices=(cfg.device,) * 6,
  )
  # build_plan requires unique production devices. Use a temporary valid plan;
  # the smoke itself intentionally runs T and P sequentially on one GPU.
  plan = build_plan(replace(launch_cfg, devices=(0, 1, 2, 3, 4, 5)))
  tasks = {job["task"] for job in plan["jobs"]}
  existing = _completed_result_is_current(
    cfg.output, plan["promotion_id"], plan["code_commit"], tasks
  )
  if existing is not None:
    return existing
  if not cfg.run:
    return {
      "status": "PLANNED",
      "promotion_id": plan["promotion_id"],
      "code_commit": plan["code_commit"],
      "tasks": sorted(tasks),
      "num_envs": cfg.num_envs,
    }
  if _gpu_processes():
    raise RuntimeError("refusing physics smoke while a GPU process is active")

  cfg.work_dir.mkdir(parents=True, exist_ok=True)
  selected = {
    phase: next(job for job in plan["jobs"] if job["phase"] == phase)
    for phase in ("T", "P")
  }
  results = []
  repo_root = Path(__file__).resolve().parents[1]
  for phase, job in selected.items():
    _source_link(job)
    command = list(job["command"])
    _set_arg(command, "--env.scene.num-envs", str(cfg.num_envs))
    _set_arg(command, "--agent.max-iterations", "1")
    _set_arg(command, "--agent.save-interval", "1")
    smoke_run_name = f"smoke_{phase.lower()}_{job['arm']}_seed{job['policy_seed']}"
    _set_arg(command, "--agent.run-name", smoke_run_name)
    command.extend(("--log-root", str((cfg.work_dir / "logs").resolve())))
    log = cfg.work_dir / f"{phase.lower()}_physics_smoke.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    environment["WANDB_MODE"] = "offline"
    with log.open("w") as stream:
      completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        timeout=cfg.timeout_s,
      )
    text = log.read_text(errors="replace")
    if completed.returncode != 0:
      raise RuntimeError(f"{phase} physics smoke failed; inspect {log}")
    for fatal in ("Traceback (most recent call last)", "CUDA out of memory"):
      if fatal in text:
        raise RuntimeError(f"{phase} physics smoke logged {fatal!r}")
    if "Loading model checkpoint from:" not in text:
      raise RuntimeError(f"{phase} smoke did not warm-start the flat checkpoint")

    experiment_root = cfg.work_dir / "logs" / job["experiment"]
    runs = sorted(
      (path for path in experiment_root.glob(f"*_{smoke_run_name}") if path.is_dir()),
      key=lambda path: path.stat().st_mtime,
    )
    if not runs:
      raise RuntimeError(f"{phase} smoke did not create a run directory")
    run_dir = runs[-1]
    checkpoint = run_dir / "model_0.pt"
    if not checkpoint.is_file():
      raise RuntimeError(f"{phase} warm start did not reset iteration to model_0.pt")
    seed = int(job["policy_seed"])
    if (
      _recorded_seed(run_dir, "agent.yaml") != seed
      or _recorded_seed(run_dir, "env.yaml") != seed
    ):
      raise RuntimeError(f"{phase} smoke saved incorrect seed provenance")
    phase_metrics = {}
    if phase == "P":
      obstacle = _metric_values(text, "escape_obstacle_episode")
      if not obstacle or max(obstacle) <= 0.0 or min(obstacle) >= 1.0:
        raise RuntimeError("P smoke did not exercise both plate and unpinned episodes")
      phase_metrics["escape_obstacle_episode"] = obstacle[-1]
    if phase == "T":
      edge = _metric_values(text, "terrain_edge_reset")
      if not edge:
        raise RuntimeError("T smoke did not report terrain edge reset metrics")
      phase_metrics["terrain_edge_reset"] = edge[-1]
    results.append(
      {
        "phase": phase,
        "task": job["task"],
        "policy_seed": seed,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "log": str(log.resolve()),
        "log_sha256": _sha256(log),
        "metrics": phase_metrics,
      }
    )

  result = {
    "schema_version": 1,
    "status": "PASS",
    "promotion": str(cfg.promotion.resolve()),
    "promotion_id": plan["promotion_id"],
    "code_commit": plan["code_commit"],
    "tasks": sorted(tasks),
    "num_envs": cfg.num_envs,
    "results": results,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(cfg.output, result)
  return result


def main(cfg: TpSmokeCfg) -> None:
  result = run_smoke(cfg)
  print(f"{result['status']}: tasks={result['tasks']}")


if __name__ == "__main__":
  main(tyro.cli(TpSmokeCfg))
