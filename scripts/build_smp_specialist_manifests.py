"""Build immutable checkpoint manifests for matched-seed T/P specialists."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from build_smp_causal_manifest import _recorded_seed, _sha256


@dataclass(frozen=True)
class SpecialistManifestCfg:
  launch_manifest: Path
  output_dir: Path
  logs_root: Path = Path("logs/rsl_rl")
  checkpoint_steps: tuple[int, ...] = (2000, 5000, 10000, 19999)
  expected_seeds: tuple[int, ...] = (20260901, 20260902, 20260903)
  phases: tuple[str, ...] = ("T", "P")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _discover_run(logs_root: Path, experiment: str, run_name: str) -> Path:
  task_dir = logs_root / experiment
  if not task_dir.is_dir():
    raise FileNotFoundError(task_dir)
  candidates = sorted(
    (
      path
      for path in task_dir.iterdir()
      if path.is_dir() and path.name.endswith(run_name)
    ),
    key=lambda path: path.stat().st_mtime,
  )
  if len(candidates) != 1:
    raise ValueError(
      f"expected exactly one run ending {run_name!r}, got {len(candidates)}"
    )
  return candidates[0].resolve()


def _wandb_run_id(log: Path) -> str | None:
  if not log.is_file():
    return None
  matches = re.findall(r"https://wandb\.ai/[^\s]+/runs/([A-Za-z0-9]+)", log.read_text())
  return matches[-1] if matches else None


def _validate_launch(cfg: SpecialistManifestCfg) -> tuple[dict[str, Any], list[dict]]:
  launch = _load(cfg.launch_manifest)
  if launch.get("status") != "LAUNCHED":
    raise ValueError("T/P launch is not frozen as LAUNCHED")
  jobs = launch.get("jobs")
  if not isinstance(jobs, list) or not jobs:
    raise ValueError("T/P launch has no jobs")
  phases = {str(job.get("phase")) for job in jobs}
  seeds = {int(job["policy_seed"]) for job in jobs}
  if phases != set(cfg.phases):
    raise ValueError(f"unexpected specialist phases: {sorted(phases)}")
  if seeds != set(cfg.expected_seeds):
    raise ValueError(f"unexpected specialist seeds: {sorted(seeds)}")
  expected_pairs = {
    (phase, seed) for phase in cfg.phases for seed in cfg.expected_seeds
  }
  observed_pairs = [(str(job["phase"]), int(job["policy_seed"])) for job in jobs]
  if (
    len(observed_pairs) != len(set(observed_pairs))
    or set(observed_pairs) != expected_pairs
  ):
    raise ValueError("T/P launch must contain exactly one job per phase and seed")
  if any(int(job["environment_seed"]) != int(job["policy_seed"]) for job in jobs):
    raise ValueError("policy and environment seed provenance differs")
  return launch, jobs


def build(
  cfg: SpecialistManifestCfg,
) -> tuple[dict[str, Any], dict[tuple[str, int, int], dict]]:
  launch, jobs = _validate_launch(cfg)
  launch_hash = _sha256(cfg.launch_manifest)
  payloads: dict[tuple[str, int, int], dict] = {}
  for job in jobs:
    phase = str(job["phase"])
    seed = int(job["policy_seed"])
    run_dir = _discover_run(cfg.logs_root, job["experiment"], job["run_name"])
    policy_seed = _recorded_seed(run_dir, "agent.yaml")
    environment_seed = _recorded_seed(run_dir, "env.yaml")
    if policy_seed != seed or environment_seed != seed:
      raise ValueError(
        f"saved seed mismatch for {phase}/{seed}: "
        f"{policy_seed}/{environment_seed} != {seed}"
      )
    source = Path(job["source_checkpoint"])
    if not source.is_file() or _sha256(source) != job["source_checkpoint_sha256"]:
      raise ValueError(f"matched flat source changed: {source}")
    wandb_id = _wandb_run_id(Path(job["log"]))
    for checkpoint_step in cfg.checkpoint_steps:
      checkpoint = run_dir / f"model_{checkpoint_step}.pt"
      if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
      run = {
        "name": f"{phase.lower()}_{job['arm']}_seed{seed}",
        "phase": phase,
        "task": job["task"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": checkpoint_step,
        "policy_seed": seed,
        "environment_seed": seed,
        "run_dir": str(run_dir),
        "run_name": job["run_name"],
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": job["source_checkpoint_sha256"],
        "seed_provenance": {
          "agent_config": str(run_dir / "params" / "agent.yaml"),
          "environment_config": str(run_dir / "params" / "env.yaml"),
        },
      }
      if wandb_id is not None:
        run["wandb_run_id"] = wandb_id
        run["wandb_url"] = f"https://wandb.ai/tabletennis/smp/runs/{wandb_id}"
      payloads[(phase, seed, checkpoint_step)] = {
        "schema_version": 1,
        "experiment": "smp-ral-specialist-frozen-evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "checkpoint_step": checkpoint_step,
        "policy_seed": seed,
        "environment_seed": seed,
        "training_code_commit": launch["code_commit"],
        "launch_plan_id": launch["plan_id"],
        "launch_manifest": str(cfg.launch_manifest.resolve()),
        "launch_manifest_sha256": launch_hash,
        "promotion_id": launch["promotion_id"],
        "protocol": launch["protocol"],
        "protocol_sha256": launch["protocol_sha256"],
        "evaluation_protocol": {
          "evaluation_seed": 20260910,
          "num_envs_per_stratum": 256,
          "steps": 750,
          "selection_rule": "complete frozen matrices only",
        },
        "runs": [run],
      }

  index = {
    "schema_version": 1,
    "status": "READY",
    "launch_plan_id": launch["plan_id"],
    "launch_manifest": str(cfg.launch_manifest.resolve()),
    "launch_manifest_sha256": launch_hash,
    "promotion_id": launch["promotion_id"],
    "protocol": launch["protocol"],
    "protocol_sha256": launch["protocol_sha256"],
    "phases": list(cfg.phases),
    "policy_seeds": list(cfg.expected_seeds),
    "checkpoint_steps": list(cfg.checkpoint_steps),
  }
  return index, payloads


def _manifest_identity(payload: dict[str, Any]) -> tuple[Any, ...]:
  run = payload["runs"][0]
  return (
    payload.get("phase"),
    payload.get("checkpoint_step"),
    payload.get("policy_seed"),
    payload.get("launch_plan_id"),
    payload.get("launch_manifest_sha256"),
    payload.get("protocol_sha256"),
    run.get("checkpoint"),
    run.get("checkpoint_sha256"),
    run.get("source_checkpoint_sha256"),
  )


def write_manifests(cfg: SpecialistManifestCfg) -> dict[str, Any]:
  index, payloads = build(cfg)
  rows = []
  for (phase, seed, checkpoint_step), payload in sorted(payloads.items()):
    path = cfg.output_dir / phase.lower() / f"seed_{seed}_gate_{checkpoint_step}.json"
    if path.exists():
      existing = _load(path)
      if _manifest_identity(existing) != _manifest_identity(payload):
        raise ValueError(f"existing specialist manifest conflicts: {path}")
    else:
      _atomic_json(path, payload)
    rows.append(
      {
        "phase": phase,
        "policy_seed": seed,
        "checkpoint_step": checkpoint_step,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
      }
    )
  index["manifests"] = rows
  _atomic_json(cfg.output_dir / "index.json", index)
  return index


def main(cfg: SpecialistManifestCfg) -> None:
  index = write_manifests(cfg)
  print(
    f"{index['status']}: {len(index['manifests'])} manifests for "
    f"{index['phases']} x {index['policy_seeds']}"
  )


if __name__ == "__main__":
  main(tyro.cli(SpecialistManifestCfg))
