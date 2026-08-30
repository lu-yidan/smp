"""Build immutable four-gate manifests for the fresh-seed flat method study."""

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

_PROTOCOL_SHA256 = "6ca241aa3bfb303084de8eac4f1cd6e02a4728ef5969a632dc7ba2b54750e0e0"
_PLAN_ID = "3e8a7aee720f4ac408b3d91eb912b9ace14c20dfd0ca4e82c8d8d2c9185ceb28"
_LAUNCH_COMMIT = "d56dafd831de6697e76bda151de3d0d50b27029f"
_SEEDS = (20261001, 20261002, 20261003)
_GATES = (8000, 15000, 25000, 29999)
_ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)
_ARMS = {
  "a6_replication_control": {
    "task": "Smp-Getup-Scratch-A6-F2S2-Mix-Bridge-G1",
    "experiment": "smp_scratch_a6_f2s2_mix_bridge_g1",
    "procedural_probability": 0.20,
    "promotion_eligible": False,
  },
  "a8_balanced_bridge": {
    "task": "Smp-Getup-Scratch-A8-F2S2-Balanced-Bridge-G1",
    "experiment": "smp_scratch_a8_f2s2_balanced_bridge_g1",
    "procedural_probability": 0.50,
    "promotion_eligible": True,
  },
}


@dataclass(frozen=True)
class FlatMethodManifestCfg:
  launch_manifest: Path = Path(
    "run_control/flat_method_study_v1_training/launch_manifest.json"
  )
  output_dir: Path = Path("run_control/flat_method_study_v1_eval/manifests")
  protocol: Path = Path("docs/ral_flat_method_study_v1.json")
  logs_root: Path = Path("logs/rsl_rl")
  checkpoint_steps: tuple[int, ...] = _GATES
  expected_seeds: tuple[int, ...] = _SEEDS


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


def _named_block(text: str, section: str, name: str) -> str:
  lines = text.splitlines()
  section_line = f"{section}:"
  target_line = f"  {name}:"
  in_section = False
  start: int | None = None
  for index, line in enumerate(lines):
    if line == section_line:
      in_section = True
      continue
    if in_section and line and not line.startswith(" "):
      break
    if in_section and line == target_line:
      start = index + 1
      break
  if start is None:
    raise ValueError(f"saved config lacks {section}.{name}")
  end = len(lines)
  for index in range(start, len(lines)):
    line = lines[index]
    if (line and not line.startswith(" ")) or re.fullmatch(
      r"  [A-Za-z0-9_]+:\s*", line
    ):
      end = index
      break
  return "\n".join(lines[start:end])


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


def _wandb_run_id(log: Path) -> str | None:
  if not log.is_file():
    return None
  matches = re.findall(r"https://wandb\.ai/[^\s]+/runs/([A-Za-z0-9]+)", log.read_text())
  return matches[-1] if matches else None


def _expected_command(arm: str, seed: int, run_name: str) -> list[str]:
  return [
    "uv",
    "run",
    "scripts/train.py",
    _ARMS[arm]["task"],
    "--env.scene.num-envs",
    "4096",
    "--agent.seed",
    str(seed),
    "--env.seed",
    str(seed),
    "--agent.max-iterations",
    "30000",
    "--agent.save-interval",
    "1000",
    "--agent.run-name",
    run_name,
  ]


def _validate_launch(cfg: FlatMethodManifestCfg) -> tuple[dict[str, Any], dict[str, Any]]:
  launch = _load(cfg.launch_manifest)
  protocol = _load(cfg.protocol)
  if _sha256(cfg.protocol) != _PROTOCOL_SHA256:
    raise ValueError("flat method protocol SHA-256 drifted")
  if protocol.get("status") != "PREREGISTERED_READY_FOR_TRAINING":
    raise ValueError("flat method protocol status drifted")
  if (
    launch.get("schema_version") != 1
    or launch.get("status") != "LAUNCHED"
    or launch.get("plan_id") != _PLAN_ID
    or launch.get("protocol_sha256") != _PROTOCOL_SHA256
    or launch.get("code_commit") != _LAUNCH_COMMIT
  ):
    raise ValueError("flat method launch lineage drifted")
  if Path(launch.get("protocol", "")).resolve() != cfg.protocol.resolve():
    raise ValueError("flat method launch protocol path drifted")
  expected_top = {
    "policy_seeds": list(_SEEDS),
    "devices": [0, 1, 2, 3, 4, 5],
    "reserved_idle_devices": [6, 7],
    "random_actor_critic_and_normalizers": True,
    "actor_observation_dim": 93,
    "actor_history_steps": 1,
    "num_envs": 4096,
    "rollout_steps_per_update": 24,
    "max_iterations": 30000,
    "save_interval": 1000,
  }
  for name, expected in expected_top.items():
    if launch.get(name) != expected:
      raise ValueError(f"flat method launch {name} drifted")
  smoke = launch.get("implementation_smoke", {})
  if smoke.get("status") != "PASSED_REAL_MUJOCO_SMOKE":
    raise ValueError("flat method launch lacks the frozen smoke audit")
  preflight = launch.get("resource_preflight", {})
  if preflight.get("free_gib", 0) < 100 or preflight.get("inode_free_fraction", 0) < 0.10:
    raise ValueError("flat method launch resource preflight is invalid")
  jobs = launch.get("jobs")
  if not isinstance(jobs, list) or len(jobs) != 6:
    raise ValueError("flat method launch must contain six jobs")
  expected_pairs = {(arm, seed) for arm in _ARMS for seed in _SEEDS}
  observed_pairs = {(str(job.get("arm")), int(job.get("policy_seed"))) for job in jobs}
  if observed_pairs != expected_pairs:
    raise ValueError("flat method launch does not contain the full arm/seed factorial")
  expected_order = [(arm, seed) for arm in _ARMS for seed in _SEEDS]
  if [(job["arm"], int(job["policy_seed"])) for job in jobs] != expected_order:
    raise ValueError("flat method launch job order drifted")
  for gpu, job in enumerate(jobs):
    arm = str(job["arm"])
    seed = int(job["policy_seed"])
    if (
      int(job["environment_seed"]) != seed
      or int(job["gpu"]) != gpu
      or job["task"] != _ARMS[arm]["task"]
      or job["promotion_eligible"] is not _ARMS[arm]["promotion_eligible"]
      or job["command"] != _expected_command(arm, seed, job["run_name"])
    ):
      raise ValueError(f"flat method launch job drifted: {arm}/{seed}")
  return launch, protocol


def _verify_saved_config(run_dir: Path, job: dict[str, Any]) -> None:
  arm = str(job["arm"])
  seed = int(job["policy_seed"])
  expected = _ARMS[arm]
  agent = run_dir / "params" / "agent.yaml"
  env = run_dir / "params" / "env.yaml"
  if _recorded_seed(run_dir, "agent.yaml") != seed:
    raise ValueError(f"saved policy seed mismatch: {run_dir}")
  if _recorded_seed(run_dir, "env.yaml") != seed:
    raise ValueError(f"saved environment seed mismatch: {run_dir}")
  expected_agent = {
    "num_steps_per_env": 24,
    "max_iterations": 30000,
    "save_interval": 1000,
    "experiment_name": expected["experiment"],
    "run_name": job["run_name"],
    "resume": "false",
  }
  for name, value in expected_agent.items():
    actual = _top_level_scalar(agent, name)
    if str(actual).lower() != str(value).lower():
      raise ValueError(f"saved agent {name} drifted: {actual} != {value}")
  text = env.read_text()
  num_envs = re.search(r"^  num_envs:\s*(\d+)\s*$", text, re.MULTILINE)
  if num_envs is None or int(num_envs.group(1)) != 4096:
    raise ValueError(f"saved environment count drifted: {run_dir}")
  if _recorded_actor_terms(env) != _ACTOR_TERMS:
    raise ValueError(f"saved actor observation contract drifted: {run_dir}")
  actor = re.search(
    r"^observations:\n  actor:\n(?P<body>.*?)(?=^  critic:)",
    text,
    flags=re.MULTILINE | re.DOTALL,
  )
  if actor is None or re.search(
    r"^    history_length:\s*null\s*$", actor.group("body"), re.MULTILINE
  ) is None:
    raise ValueError(f"saved actor is not one frame: {run_dir}")
  reset = _named_block(text, "events", "mixed_fall_reset")
  probability = re.search(r"^      procedural_probability:\s*([0-9.]+)$", reset, re.MULTILINE)
  if probability is None or float(probability.group(1)) != expected["procedural_probability"]:
    raise ValueError(f"saved procedural probability drifted: {run_dir}")
  reward = _named_block(text, "rewards", "task_smp_product")
  if "procedural_smp_floor: 0.1" not in reward:
    raise ValueError(f"saved reward bridge drifted: {run_dir}")
  termination = _named_block(text, "terminations", "smp_too_low")
  if "smp_too_low_gsi_only" not in termination:
    raise ValueError(f"saved reset-aware termination drifted: {run_dir}")


def _checkpoint_integrity(path: Path, gate: int) -> dict[str, Any]:
  import torch

  try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
  except Exception as error:
    raise ValueError(f"checkpoint is not loadable: {path}") from error
  actor = payload.get("actor_state_dict", {})
  critic = payload.get("critic_state_dict", {})

  def collect(value: Any) -> list[Any]:
    if torch.is_tensor(value):
      return [value]
    if isinstance(value, dict):
      return [tensor for item in value.values() for tensor in collect(item)]
    if isinstance(value, (list, tuple)):
      return [tensor for item in value for tensor in collect(item)]
    return []

  actor_tensors = collect(actor)
  critic_tensors = collect(critic)
  all_tensors = collect(payload)
  if (
    payload.get("iter") != gate
    or tuple(actor.get("obs_normalizer._mean", torch.empty(0)).shape) != (1, 93)
    or tuple(critic.get("obs_normalizer._mean", torch.empty(0)).shape) != (1, 960)
    or not actor_tensors
    or not critic_tensors
    or not all_tensors
    or not all(bool(torch.isfinite(tensor).all()) for tensor in all_tensors)
  ):
    raise ValueError(f"checkpoint integrity failed: {path}")
  return {
    "embedded_iteration": gate,
    "actor_tensor_count": len(actor_tensors),
    "critic_tensor_count": len(critic_tensors),
    "total_tensor_count": len(all_tensors),
    "total_tensor_elements": sum(tensor.numel() for tensor in all_tensors),
    "all_tensors_finite": True,
  }


def build(
  cfg: FlatMethodManifestCfg,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
  launch, protocol = _validate_launch(cfg)
  if tuple(cfg.expected_seeds) != _SEEDS or tuple(cfg.checkpoint_steps) != _GATES:
    raise ValueError("flat method seed or checkpoint gates differ from the frozen protocol")
  runs: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
  for job in launch["jobs"]:
    arm = str(job["arm"])
    seed = int(job["policy_seed"])
    run_dir = _discover_run(cfg.logs_root, _ARMS[arm]["experiment"], job["run_name"])
    _verify_saved_config(run_dir, job)
    runs[(arm, seed)] = (job, run_dir)
  missing = []
  for (arm, seed), (_, run_dir) in runs.items():
    for gate in _GATES:
      checkpoint = run_dir / f"model_{gate}.pt"
      if not checkpoint.is_file():
        missing.append(f"{arm} seed {seed} gate {gate}: {checkpoint}")
  if missing:
    raise FileNotFoundError(
      "refusing partial flat method manifests; missing:\n"
      + "\n".join(f"- {item}" for item in missing)
    )

  launch_sha = _sha256(cfg.launch_manifest)
  payloads: dict[tuple[int, int], dict[str, Any]] = {}
  for seed in _SEEDS:
    for gate in _GATES:
      manifest_runs = []
      for arm in _ARMS:
        job, run_dir = runs[(arm, seed)]
        checkpoint = run_dir / f"model_{gate}.pt"
        row = {
          "name": arm,
          "task": _ARMS[arm]["task"],
          "promotion_eligible": _ARMS[arm]["promotion_eligible"],
          "procedural_probability": _ARMS[arm]["procedural_probability"],
          "checkpoint": str(checkpoint.resolve()),
          "checkpoint_sha256": _sha256(checkpoint),
          "checkpoint_integrity": _checkpoint_integrity(checkpoint, gate),
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
          row["wandb_run_id"] = wandb_id
          row["wandb_url"] = f"https://wandb.ai/tabletennis/smp/runs/{wandb_id}"
        manifest_runs.append(row)
      stable = {
        "schema_version": 1,
        "status": "READY_FOR_FROZEN_EVALUATION",
        "study_id": protocol["study_id"],
        "training_code_commit": launch["code_commit"],
        "launch_plan_id": launch["plan_id"],
        "launch_manifest": str(cfg.launch_manifest.resolve()),
        "launch_manifest_sha256": launch_sha,
        "protocol": str(cfg.protocol.resolve()),
        "protocol_sha256": _PROTOCOL_SHA256,
        "checkpoint_step": gate,
        "policy_seed": seed,
        "environment_seed": seed,
        "actor_observation": "g1-deployable-proprio-f1-v1",
        "evaluation_protocol": protocol["evaluation_protocol"],
        "failure_reason_codebook": protocol["failure_reason_codebook"],
        "runs": manifest_runs,
      }
      stable["manifest_id"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode()
      ).hexdigest()
      payloads[(seed, gate)] = {
        **stable,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
          "Checkpoint provenance is not performance evidence. Only complete frozen "
          "five-pose evaluation over all three policy seeds may support promotion."
        ),
      }
  index = {
    "schema_version": 1,
    "status": "READY_FOR_FROZEN_EVALUATION",
    "study_id": protocol["study_id"],
    "launch_plan_id": launch["plan_id"],
    "protocol_sha256": _PROTOCOL_SHA256,
    "arms": list(_ARMS),
    "policy_seeds": list(_SEEDS),
    "checkpoint_steps": list(_GATES),
    "checkpoint_entry_count": len(_ARMS) * len(_SEEDS) * len(_GATES),
    "manifest_ids": sorted(payload["manifest_id"] for payload in payloads.values()),
  }
  index["index_id"] = hashlib.sha256(
    json.dumps(index, sort_keys=True).encode()
  ).hexdigest()
  return index, payloads


def write_manifests(cfg: FlatMethodManifestCfg) -> dict[str, Any]:
  index, payloads = build(cfg)
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  index_path = cfg.output_dir / "index.json"
  existing_index = _load(index_path) if index_path.exists() else None
  if existing_index is not None and existing_index.get("index_id") != index["index_id"]:
    raise ValueError(f"existing flat method index conflicts: {index_path}")
  manifest_paths = set(cfg.output_dir.glob("gate_*_seed_*.json"))
  if existing_index is None and manifest_paths:
    raise ValueError("partial flat method manifests exist without an index")
  existing_rows = {
    (int(row["policy_seed"]), int(row["checkpoint_step"])): row
    for row in (existing_index or {}).get("manifests", ())
  }
  rows = []
  for (seed, gate), payload in sorted(payloads.items()):
    path = cfg.output_dir / f"gate_{gate}_seed_{seed}.json"
    if path.exists():
      existing = _load(path)
      frozen = existing_rows.get((seed, gate))
      if existing.get("manifest_id") != payload["manifest_id"]:
        raise ValueError(f"existing flat method manifest conflicts: {path}")
      if (
        frozen is None
        or frozen.get("path") != str(path.resolve())
        or frozen.get("sha256") != _sha256(path)
      ):
        raise ValueError(f"flat method manifest changed after indexing: {path}")
    else:
      if existing_index is not None:
        raise ValueError(f"indexed flat method manifest is missing: {path}")
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
      raise ValueError(f"existing flat method index rows conflict: {index_path}")
  else:
    _atomic_json(index_path, index)
  return index


def main(cfg: FlatMethodManifestCfg) -> None:
  result = write_manifests(cfg)
  print(
    f"{result['status']}: {len(result['manifests'])} manifests, "
    f"{result['checkpoint_entry_count']} checkpoints, index {result['index_id']}"
  )


if __name__ == "__main__":
  main(tyro.cli(FlatMethodManifestCfg))
