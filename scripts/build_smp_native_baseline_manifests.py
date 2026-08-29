"""Build immutable checkpoint manifests for native Tier-A baseline training."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from build_smp_causal_manifest import _recorded_seed, _sha256

_METHOD_TASK_NAMES = {
  "task_only_ppo": "TaskOnly",
  "original_product_smp": "OriginalSMP",
  "proposed_smp_recovery": "ProposedSMP",
}
_ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)


@dataclass(frozen=True)
class NativeBaselineManifestCfg:
  launch_manifest: Path
  output_dir: Path
  logs_root: Path = Path("logs/rsl_rl")
  checkpoint_steps: tuple[int, ...] = (8000, 15000, 25000, 29999)
  expected_seeds: tuple[int, ...] = (20260901, 20260902, 20260903)


def _load(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _verify_source(path_value: Any, digest: Any, label: str) -> Path:
  if not isinstance(path_value, str) or not isinstance(digest, str):
    raise ValueError(f"launch manifest lacks {label} provenance")
  path = Path(path_value)
  if not path.is_file() or _sha256(path) != digest:
    raise ValueError(f"frozen {label} changed: {path}")
  return path.resolve()


def _discover_run(logs_root: Path, experiment: str, run_name: str) -> Path:
  experiment_dir = logs_root / experiment
  if not experiment_dir.is_dir():
    raise FileNotFoundError(experiment_dir)
  matches = sorted(
    path
    for path in experiment_dir.iterdir()
    if path.is_dir() and path.name.endswith(run_name)
  )
  if len(matches) != 1:
    raise ValueError(f"expected one run ending {run_name!r}, got {len(matches)}")
  return matches[0].resolve()


def _top_level_scalar(path: Path, name: str) -> str:
  match = re.search(
    rf"^{re.escape(name)}:\s*([^\n]+?)\s*$",
    path.read_text(),
    flags=re.MULTILINE,
  )
  if match is None:
    raise ValueError(f"top-level {name} missing from {path}")
  return match.group(1)


def _recorded_actor_terms(env_path: Path) -> tuple[str, ...]:
  match = re.search(
    r"^observations:\n  actor:\n    terms:\n(?P<body>.*?)(?=^    concatenate_terms:)",
    env_path.read_text(),
    flags=re.MULTILINE | re.DOTALL,
  )
  if match is None:
    raise ValueError(f"actor observation block missing from {env_path}")
  return tuple(
    re.findall(r"^      ([A-Za-z0-9_]+):\s*$", match.group("body"), re.MULTILINE)
  )


def _verify_saved_config(
  run_dir: Path, job: dict[str, Any], launch: dict[str, Any], experiment: str
) -> None:
  agent = run_dir / "params" / "agent.yaml"
  env = run_dir / "params" / "env.yaml"
  seed = int(job["policy_seed"])
  if _recorded_seed(run_dir, "agent.yaml") != seed:
    raise ValueError(f"saved policy seed mismatch: {run_dir}")
  if _recorded_seed(run_dir, "env.yaml") != seed:
    raise ValueError(f"saved environment seed mismatch: {run_dir}")
  expected_agent = {
    "num_steps_per_env": 24,
    "max_iterations": int(launch["max_updates"]),
    "save_interval": int(launch["save_interval"]),
    "experiment_name": experiment,
    "run_name": job["run_name"],
    "resume": "false",
  }
  for name, expected in expected_agent.items():
    actual = _top_level_scalar(agent, name)
    if str(actual).lower() != str(expected).lower():
      raise ValueError(f"saved agent {name} drifted: {actual} != {expected}")
  env_text = env.read_text()
  num_envs = re.search(r"^  num_envs:\s*(\d+)\s*$", env_text, re.MULTILINE)
  if num_envs is None or int(num_envs.group(1)) != int(launch["num_envs"]):
    raise ValueError(f"saved environment count drifted: {run_dir}")
  if _recorded_actor_terms(env) != _ACTOR_TERMS:
    raise ValueError(f"saved actor observation contract drifted: {run_dir}")
  actor = re.search(
    r"^observations:\n  actor:\n(?P<body>.*?)(?=^  critic:)",
    env_text,
    flags=re.MULTILINE | re.DOTALL,
  )
  if actor is None or not re.search(
    r"^    history_length:\s*null\s*$", actor.group("body"), re.MULTILINE
  ):
    raise ValueError(f"saved actor is not one frame: {run_dir}")
  if str(launch["bank_sha256"]) not in env_text:
    raise ValueError(f"saved environment lacks matched bank SHA: {run_dir}")
  if str(Path(launch["bank"]).resolve()) not in env_text:
    raise ValueError(f"saved environment lacks matched bank path: {run_dir}")


def _wandb_run_id(log: Path) -> str | None:
  if not log.is_file():
    return None
  matches = re.findall(r"https://wandb\.ai/[^\s]+/runs/([A-Za-z0-9]+)", log.read_text())
  return matches[-1] if matches else None


def _validate_launch(cfg: NativeBaselineManifestCfg) -> dict[str, Any]:
  launch = _load(cfg.launch_manifest)
  if launch.get("schema_version") != 1 or launch.get("status") != "LAUNCHED":
    raise ValueError("native baseline launch is not frozen as LAUNCHED")
  if tuple(launch.get("policy_seeds", ())) != tuple(cfg.expected_seeds):
    raise ValueError("native baseline policy seeds differ from the frozen protocol")
  if not isinstance(launch.get("jobs"), list) or len(launch["jobs"]) != 9:
    raise ValueError("native baseline launch must contain nine jobs")
  _verify_source(launch.get("promotion"), launch.get("promotion_sha256"), "promotion")
  _verify_source(
    launch.get("runtime_registry"),
    launch.get("runtime_registry_sha256"),
    "runtime registry",
  )
  _verify_source(launch.get("bank"), launch.get("bank_sha256"), "reset bank")
  _verify_source(
    launch.get("bank_manifest"),
    launch.get("bank_manifest_sha256"),
    "reset-bank manifest",
  )
  job_ids = [job.get("job_id") for job in launch["jobs"]]
  queued = [
    job_id for worker in launch.get("workers", ()) for job_id in worker["job_ids"]
  ]
  if sorted(job_ids) != sorted(queued) or len(queued) != len(set(queued)):
    raise ValueError("worker queues do not cover every launch job exactly once")
  return launch


def build(
  cfg: NativeBaselineManifestCfg,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
  launch = _validate_launch(cfg)
  if tuple(cfg.checkpoint_steps) != (8000, 15000, 25000, 29999):
    raise ValueError("native baseline checkpoint gates differ from the frozen protocol")
  expected_pairs = {
    (method, seed) for method in _METHOD_TASK_NAMES for seed in cfg.expected_seeds
  }
  observed_pairs = {
    (str(job["method"]), int(job["policy_seed"])) for job in launch["jobs"]
  }
  if observed_pairs != expected_pairs:
    raise ValueError("native launch does not contain the full method/seed factorial")

  runs_by_seed: dict[int, list[tuple[dict[str, Any], Path]]] = {
    seed: [] for seed in cfg.expected_seeds
  }
  arm_index = int(launch["arm_index"])
  for job in launch["jobs"]:
    method = str(job["method"])
    seed = int(job["policy_seed"])
    if int(job["environment_seed"]) != seed:
      raise ValueError(f"policy/environment seed mismatch: {job['job_id']}")
    expected_task = f"Smp-Getup-RAL-B-{_METHOD_TASK_NAMES[method]}-A{arm_index}-G1"
    if job["task"] != expected_task:
      raise ValueError(f"native baseline task drifted: {job['job_id']}")
    experiment = f"smp_ral_b_{method}_a{arm_index}_g1"
    run_dir = _discover_run(cfg.logs_root, experiment, job["run_name"])
    _verify_saved_config(run_dir, job, launch, experiment)
    runs_by_seed[seed].append((job, run_dir))

  missing = []
  for seed, jobs in runs_by_seed.items():
    for job, run_dir in jobs:
      for gate in cfg.checkpoint_steps:
        checkpoint = run_dir / f"model_{gate}.pt"
        if not checkpoint.is_file():
          missing.append(f"{job['method']} seed {seed} gate {gate}: {checkpoint}")
  if missing:
    raise FileNotFoundError(
      "refusing partial native baseline manifests; missing:\n"
      + "\n".join(f"- {item}" for item in missing)
    )

  payloads: dict[tuple[int, int], dict[str, Any]] = {}
  launch_hash = _sha256(cfg.launch_manifest)
  for seed, jobs in runs_by_seed.items():
    for gate in cfg.checkpoint_steps:
      runs = []
      for job, run_dir in jobs:
        checkpoint = run_dir / f"model_{gate}.pt"
        run = {
          "name": job["method"],
          "method": job["method"],
          "task": job["task"],
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
        runs.append(run)
      stable = {
        "schema_version": 1,
        "experiment": "ral-native-tier-a-baselines",
        "training_code_commit": launch["code_commit"],
        "launch_plan_id": launch["plan_id"],
        "launch_manifest": str(cfg.launch_manifest.resolve()),
        "launch_manifest_sha256": launch_hash,
        "promotion_id": launch["promotion_id"],
        "runtime_registry_sha256": launch["runtime_registry_sha256"],
        "training_reset_bank_sha256": launch["bank_sha256"],
        "selected_arm": launch["selected_arm"],
        "checkpoint_step": gate,
        "policy_seed": seed,
        "environment_seed": seed,
        "actor_observation": "g1-deployable-proprio-f1-v1",
        "evaluation_status": "BLOCKED_ON_MATCHED_HELD_OUT_RESET_BANK",
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
          "requires_disjoint_matched_reset_bank": True,
        },
        "runs": sorted(runs, key=lambda run: run["name"]),
      }
      stable["manifest_id"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode()
      ).hexdigest()
      payloads[(seed, gate)] = {
        **stable,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
          "Checkpoint provenance is not performance evidence; evaluation remains "
          "blocked until a disjoint matched held-out reset bank is bound."
        ),
      }

  index_material = {
    "schema_version": 1,
    "status": "CHECKPOINTS_READY_EVALUATION_BANK_BLOCKED",
    "launch_plan_id": launch["plan_id"],
    "selected_arm": launch["selected_arm"],
    "methods": sorted(_METHOD_TASK_NAMES),
    "policy_seeds": list(cfg.expected_seeds),
    "checkpoint_steps": list(cfg.checkpoint_steps),
    "manifest_ids": sorted(payload["manifest_id"] for payload in payloads.values()),
  }
  index_material["index_id"] = hashlib.sha256(
    json.dumps(index_material, sort_keys=True).encode()
  ).hexdigest()
  return index_material, payloads


def write_manifests(cfg: NativeBaselineManifestCfg) -> dict[str, Any]:
  index, payloads = build(cfg)
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  index_path = cfg.output_dir / "index.json"
  existing_index = _load(index_path) if index_path.exists() else None
  if existing_index is not None and existing_index.get("index_id") != index["index_id"]:
    raise ValueError(f"existing native baseline index conflicts: {index_path}")
  existing_rows = {
    (int(row["policy_seed"]), int(row["checkpoint_step"])): row
    for row in (existing_index or {}).get("manifests", ())
  }
  manifest_paths = set(cfg.output_dir.glob("gate_*_seed_*.json"))
  if existing_index is None and manifest_paths:
    raise ValueError("partial native baseline manifests exist without an index")
  rows = []
  for (seed, gate), payload in sorted(payloads.items()):
    path = cfg.output_dir / f"gate_{gate}_seed_{seed}.json"
    if path.exists():
      existing = _load(path)
      if existing.get("manifest_id") != payload["manifest_id"]:
        raise ValueError(f"existing native baseline manifest conflicts: {path}")
      frozen_row = existing_rows.get((seed, gate))
      if (
        frozen_row is None
        or frozen_row.get("path") != str(path.resolve())
        or frozen_row.get("sha256") != _sha256(path)
      ):
        raise ValueError(f"native baseline manifest changed after indexing: {path}")
    else:
      if existing_index is not None:
        raise ValueError(f"indexed native baseline manifest is missing: {path}")
      _atomic_json(path, payload)
    rows.append(
      {
        "policy_seed": seed,
        "checkpoint_step": gate,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
      }
    )
  index["manifests"] = rows
  if existing_index is not None:
    if existing_index.get("manifests") != rows:
      raise ValueError(f"existing native baseline index rows conflict: {index_path}")
  else:
    _atomic_json(index_path, index)
  return index


def main(cfg: NativeBaselineManifestCfg) -> None:
  result = write_manifests(cfg)
  print(
    f"{result['status']}: {len(result['manifests'])} manifests, "
    f"index {result['index_id']}"
  )


if __name__ == "__main__":
  main(tyro.cli(NativeBaselineManifestCfg))
