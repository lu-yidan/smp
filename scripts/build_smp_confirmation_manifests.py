"""Build immutable final-checkpoint manifests for confirmation policy seeds."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from build_smp_causal_manifest import _ARMS, _recorded_seed, _sha256


@dataclass(frozen=True)
class ConfirmationManifestCfg:
  launch_manifest: Path
  output_dir: Path
  logs_root: Path = Path("logs/rsl_rl")
  checkpoint_step: int = 29999
  expected_seeds: tuple[int, ...] = (20260901, 20260902, 20260903)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _discover_run(logs_root: Path, log_dir: str, run_name: str) -> Path:
  task_dir = logs_root / log_dir
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


def build(cfg: ConfirmationManifestCfg) -> tuple[dict[str, Any], dict[int, dict]]:
  launch = json.loads(cfg.launch_manifest.read_text())
  if launch.get("status") != "LAUNCHED":
    raise ValueError("confirmation launch is not frozen as LAUNCHED")
  jobs = launch.get("jobs")
  if not isinstance(jobs, list) or not jobs:
    raise ValueError("confirmation launch has no jobs")
  seeds = sorted({int(job["policy_seed"]) for job in jobs})
  if tuple(seeds) != tuple(sorted(cfg.expected_seeds)):
    raise ValueError(f"unexpected confirmation seeds: {seeds}")
  if any(int(job["environment_seed"]) != int(job["policy_seed"]) for job in jobs):
    raise ValueError("policy and environment seed provenance differs")

  catalog = {arm["name"]: arm for arm in _ARMS}
  discovered: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
  seen = set()
  for job in jobs:
    arm_name = job["arm"]
    seed = int(job["policy_seed"])
    key = (arm_name, seed)
    if key in seen:
      raise ValueError(f"duplicate confirmation job: {key}")
    seen.add(key)
    if arm_name not in catalog:
      raise ValueError(f"unknown confirmation arm: {arm_name}")
    arm = catalog[arm_name]
    run_dir = _discover_run(cfg.logs_root, arm["log_dir"], job["run_name"])
    policy_seed = _recorded_seed(run_dir, "agent.yaml")
    environment_seed = _recorded_seed(run_dir, "env.yaml")
    if policy_seed != seed or environment_seed != seed:
      raise ValueError(
        f"saved seed mismatch for {arm_name}: {policy_seed}/{environment_seed} != {seed}"
      )
    checkpoint = run_dir / f"model_{cfg.checkpoint_step}.pt"
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
    run = {
      "name": arm_name,
      "task": arm["task"],
      "checkpoint": str(checkpoint.resolve()),
      "checkpoint_sha256": _sha256(checkpoint),
      "policy_seed": seed,
      "environment_seed": seed,
      "run_dir": str(run_dir),
      "run_name": job["run_name"],
      "seed_provenance": {
        "agent_config": str(run_dir / "params" / "agent.yaml"),
        "environment_config": str(run_dir / "params" / "env.yaml"),
      },
    }
    wandb_id = _wandb_run_id(Path(job["log"]))
    if wandb_id is not None:
      run["wandb_run_id"] = wandb_id
      run["wandb_url"] = f"https://wandb.ai/tabletennis/smp/runs/{wandb_id}"
    discovered[seed].append(run)

  arm_sets = {
    tuple(sorted(run["name"] for run in runs)) for runs in discovered.values()
  }
  if len(arm_sets) != 1:
    raise ValueError("confirmation seeds do not contain the same arm set")
  manifests = {}
  for seed, runs in discovered.items():
    manifests[seed] = {
      "schema_version": 1,
      "experiment": f"scratch-causal-confirmation-seed-{seed}",
      "generated_at_utc": datetime.now(timezone.utc).isoformat(),
      "training_code_commit": launch["code_commit"],
      "launch_plan_id": launch["plan_id"],
      "launch_manifest": str(cfg.launch_manifest.resolve()),
      "launch_manifest_sha256": _sha256(cfg.launch_manifest),
      "stable_selection": launch["selection"],
      "stable_selection_sha256": launch["selection_sha256"],
      "checkpoint_step": cfg.checkpoint_step,
      "policy_seed": seed,
      "environment_seed": seed,
      "evaluation_protocol": {
        "reset_modes": [
          "native_gsi",
          "prone",
          "supine",
          "left_side",
          "right_side",
        ],
        "num_envs": 512,
        "steps": 500,
        "evaluation_seed": 20260829,
      },
      "runs": sorted(runs, key=lambda run: run["name"]),
    }
  index = {
    "schema_version": 1,
    "status": "READY",
    "checkpoint_step": cfg.checkpoint_step,
    "policy_seeds": seeds,
    "arms": list(next(iter(arm_sets))),
    "launch_plan_id": launch["plan_id"],
  }
  return index, manifests


def write_manifests(cfg: ConfirmationManifestCfg) -> dict[str, Any]:
  index, payloads = build(cfg)
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for seed, payload in sorted(payloads.items()):
    path = cfg.output_dir / f"seed_{seed}.json"
    if path.exists():
      existing = json.loads(path.read_text())
      existing_runs = [
        (
          run.get("name"),
          run.get("checkpoint"),
          run.get("checkpoint_sha256"),
          run.get("policy_seed"),
          run.get("environment_seed"),
        )
        for run in existing.get("runs", [])
      ]
      expected_runs = [
        (
          run["name"],
          run["checkpoint"],
          run["checkpoint_sha256"],
          run["policy_seed"],
          run["environment_seed"],
        )
        for run in payload["runs"]
      ]
      if (
        existing.get("launch_plan_id") != payload["launch_plan_id"]
        or existing.get("policy_seed") != seed
        or existing.get("stable_selection_sha256") != payload["stable_selection_sha256"]
        or existing_runs != expected_runs
      ):
        raise ValueError(f"existing confirmation manifest conflicts: {path}")
    else:
      _atomic_json(path, payload)
    rows.append(
      {"policy_seed": seed, "path": str(path.resolve()), "sha256": _sha256(path)}
    )
  index["manifests"] = rows
  _atomic_json(cfg.output_dir / "index.json", index)
  return index


def main(cfg: ConfirmationManifestCfg) -> None:
  index = write_manifests(cfg)
  print(f"{index['status']}: seeds={index['policy_seeds']}, arms={index['arms']}")


if __name__ == "__main__":
  main(tyro.cli(ConfirmationManifestCfg))
