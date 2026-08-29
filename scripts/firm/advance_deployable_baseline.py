"""Advance the frozen deployable FIRM-R external-reference pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PROTOCOL = "docs/firm_deployable_baseline_protocol.json"
DEFAULT_STATE = "run_control/firm_r_deployable_93d_v1/state.json"


def sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
  with path.open() as stream:
    value = json.load(stream)
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object in {path}")
  return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _now() -> str:
  return datetime.now(UTC).isoformat()


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except (OSError, ValueError):
    return False
  return True


def _gpu_processes() -> list[int]:
  result = subprocess.run(
    [
      "nvidia-smi",
      "--query-compute-apps=pid",
      "--format=csv,noheader,nounits",
    ],
    check=False,
    capture_output=True,
    text=True,
  )
  if result.returncode != 0:
    raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
  return [int(line) for line in result.stdout.splitlines() if line.strip()]


def _resolve(repo_root: Path, value: str) -> Path:
  path = Path(value).expanduser()
  return path if path.is_absolute() else repo_root / path


def _replicate_paths(protocol: dict[str, Any], seed: int) -> tuple[Path, Path]:
  outputs = protocol["outputs"]
  rollout = Path(outputs["rollout_root"]) / f"seed_{seed}"
  action = Path(outputs["action_root"]) / f"seed_{seed}"
  return rollout, action


def _adapter_path(protocol: dict[str, Any], seed: int, name: str) -> Path:
  return Path(protocol["outputs"]["adapter_root"]) / name / f"seed_{seed}"


def _validate_artifact(path: Path, expected_sha256: str, label: str) -> None:
  if not path.is_file():
    raise FileNotFoundError(f"{label} missing: {path}")
  actual = sha256_file(path)
  if actual != expected_sha256:
    raise RuntimeError(
      f"{label} checksum mismatch: expected {expected_sha256}, got {actual}"
    )


def validate_rollout_manifest(
  manifest_path: Path,
  protocol: dict[str, Any],
  seed: int,
  *,
  verify_shards: bool = True,
) -> None:
  manifest = _read_json(manifest_path)
  expected = protocol["collection"]
  config = manifest.get("config", {})
  if manifest.get("task_id") != protocol["expert"]["task_id"]:
    raise RuntimeError(f"wrong rollout task in {manifest_path}")
  required_config = {
    "seed": seed,
    "num_start_frames": expected["num_start_frames"],
    "episodes_per_frame": expected["episodes_per_frame"],
    "max_steps": expected["max_steps"],
    "standing_hold_steps": expected["standing_hold_steps"],
    "observation_corruption": expected["observation_corruption"],
    "physical_disturbances": expected["physical_disturbances"],
  }
  mismatches = {
    key: (config.get(key), value)
    for key, value in required_config.items()
    if config.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"rollout config mismatch in {manifest_path}: {mismatches}")
  shape = manifest.get("layout", {}).get("observation", {}).get("shape")
  if shape != [protocol["expert"]["deployable_state_dim"]]:
    raise RuntimeError(f"rollout is not frozen 93D input: shape={shape}")
  artifacts = manifest.get("artifacts", {})
  for key in ("checkpoint_sha256", "motion_sha256"):
    if artifacts.get(key) != protocol["expert"][key]:
      raise RuntimeError(f"rollout {key} mismatch in {manifest_path}")
  episodes = expected["num_start_frames"] * expected["episodes_per_frame"]
  if manifest.get("episodes") != episodes:
    raise RuntimeError(f"rollout episode count mismatch in {manifest_path}")
  successes = int(manifest.get("successful_episodes", 0))
  if successes / episodes < expected["minimum_success_fraction"]:
    raise RuntimeError(
      f"rollout success fraction {successes / episodes:.3f} is below frozen "
      f"minimum {expected['minimum_success_fraction']:.3f}"
    )
  if (
    expected["physical_disturbances"] and manifest.get("disturbed_transitions", 0) <= 0
  ):
    raise RuntimeError(f"rollout has no disturbed transitions: {manifest_path}")
  if int(manifest.get("total_samples", 0)) <= 0:
    raise RuntimeError(f"rollout has no samples: {manifest_path}")
  if verify_shards:
    for shard in manifest.get("shards", []):
      shard_path = manifest_path.parent / shard["file"]
      _validate_artifact(shard_path, shard["sha256"], "rollout shard")


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
  import torch

  payload = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(payload, dict):
    raise TypeError(f"expected checkpoint dictionary in {path}")
  return payload


def validate_action_checkpoint(
  checkpoint_path: Path,
  manifest_path: Path,
  protocol: dict[str, Any],
  seed: int,
) -> None:
  payload = _load_torch_checkpoint(checkpoint_path)
  config = payload.get("config", {})
  expected = protocol["action_training"]
  required = {
    "seed": seed,
    "observation_dim": protocol["expert"]["deployable_state_dim"],
    "horizon": expected["horizon"],
    "num_epochs": expected["num_epochs"],
  }
  mismatches = {
    key: (config.get(key), value)
    for key, value in required.items()
    if config.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"action checkpoint config mismatch: {mismatches}")
  if payload.get("manifest_sha256") != sha256_file(manifest_path):
    raise RuntimeError("action checkpoint rollout manifest checksum mismatch")


def validate_adapter_checkpoint(
  checkpoint_path: Path,
  action_path: Path,
  protocol: dict[str, Any],
  seed: int,
  history_steps: int,
) -> None:
  payload = _load_torch_checkpoint(checkpoint_path)
  config = payload.get("config", {})
  required = {
    "seed": seed,
    "observation_dim": protocol["expert"]["deployable_state_dim"],
    "history_steps": history_steps,
    "num_epochs": protocol["adapter_training"]["num_epochs"],
  }
  mismatches = {
    key: (config.get(key), value)
    for key, value in required.items()
    if config.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"adapter checkpoint config mismatch: {mismatches}")
  artifacts = payload.get("artifacts", {})
  if artifacts.get("action_checkpoint_sha256") != sha256_file(action_path):
    raise RuntimeError("adapter action checkpoint checksum mismatch")


def _launch_job(
  *,
  state: dict[str, Any],
  key: str,
  command: list[str],
  gpu_id: int,
  log_path: Path,
  repo_root: Path,
) -> None:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  environment = dict(os.environ)
  environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
  environment["PYTHONUNBUFFERED"] = "1"
  source_path = str(repo_root / "src")
  existing_pythonpath = environment.get("PYTHONPATH")
  environment["PYTHONPATH"] = (
    f"{source_path}:{existing_pythonpath}" if existing_pythonpath else source_path
  )
  with log_path.open("a") as stream:
    process = subprocess.Popen(
      command,
      cwd=repo_root,
      env=environment,
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  state.setdefault("jobs", {})[key] = {
    "command": command,
    "gpu_id": gpu_id,
    "log": str(log_path),
    "pid": process.pid,
    "started_at": _now(),
  }


def _active_or_failed_jobs(
  state: dict[str, Any], keys: list[str]
) -> tuple[list[str], list[str]]:
  active: list[str] = []
  failed: list[str] = []
  jobs = state.get("jobs", {})
  for key in keys:
    job = jobs.get(key)
    if job is None:
      continue
    if _pid_alive(int(job["pid"])):
      active.append(key)
    else:
      failed.append(key)
  return active, failed


def _set_state(
  state: dict[str, Any], status: str, action: str, state_path: Path
) -> dict[str, Any]:
  state["status"] = status
  state["action"] = action
  state["updated_at"] = _now()
  _write_json(state_path, state)
  return state


def _collection_command(
  repo_root: Path,
  protocol: dict[str, Any],
  seed: int,
  output_dir: Path,
) -> list[str]:
  expert = protocol["expert"]
  cfg = protocol["collection"]
  command = [
    sys.executable,
    "scripts/firm/collect_rollouts.py",
    "--task-id",
    expert["task_id"],
    "--checkpoint-file",
    str(_resolve(repo_root, expert["checkpoint_file"])),
    "--motion-file",
    str(_resolve(repo_root, expert["motion_file"])),
    "--num-start-frames",
    str(cfg["num_start_frames"]),
    "--episodes-per-frame",
    str(cfg["episodes_per_frame"]),
    "--max-steps",
    str(cfg["max_steps"]),
    "--standing-hold-steps",
    str(cfg["standing_hold_steps"]),
    "--seed",
    str(seed),
    "--device",
    "cuda:0",
    "--output-dir",
    str(output_dir),
    "--shard-size",
    str(cfg["shard_size"]),
  ]
  command.append(
    "--observation-corruption"
    if cfg["observation_corruption"]
    else "--no-observation-corruption"
  )
  command.append(
    "--physical-disturbances"
    if cfg["physical_disturbances"]
    else "--no-physical-disturbances"
  )
  return command


def _action_command(
  protocol: dict[str, Any], seed: int, manifest: Path, output_dir: Path
) -> list[str]:
  cfg = protocol["action_training"]
  command = [
    sys.executable,
    "scripts/firm/train_action_diffusion.py",
    "--manifest-file",
    str(manifest),
    "--horizon",
    str(cfg["horizon"]),
    "--train-fraction",
    str(cfg["train_fraction"]),
    "--batch-size",
    str(cfg["batch_size"]),
    "--num-epochs",
    str(cfg["num_epochs"]),
    "--learning-rate",
    str(cfg["learning_rate"]),
    "--weight-decay",
    str(cfg["weight_decay"]),
    "--num-timesteps",
    str(cfg["num_timesteps"]),
    "--save-interval",
    str(cfg["save_interval"]),
    "--seed",
    str(seed),
    "--device",
    "cuda:0",
    "--run-name",
    f"firm_r_93d_action_seed_{seed}",
    "--output-dir",
    str(output_dir),
  ]
  command.append(
    "--successful-only" if cfg["successful_only"] else "--no-successful-only"
  )
  return command


def _adapter_command(
  protocol: dict[str, Any],
  seed: int,
  history_steps: int,
  action_path: Path,
  manifest_path: Path,
  output_dir: Path,
  variant: str,
) -> list[str]:
  cfg = protocol["adapter_training"]
  command = [
    sys.executable,
    "scripts/firm/train_goal_adapter.py",
    "--action-checkpoint-file",
    str(action_path),
    "--manifest-file",
    str(manifest_path),
    "--history-steps",
    str(history_steps),
    "--train-fraction",
    str(cfg["train_fraction"]),
    "--batch-size",
    str(cfg["batch_size"]),
    "--num-epochs",
    str(cfg["num_epochs"]),
    "--learning-rate",
    str(cfg["learning_rate"]),
    "--weight-decay",
    str(cfg["weight_decay"]),
    "--seed",
    str(seed),
    "--device",
    "cuda:0",
    "--run-name",
    f"{variant}_seed_{seed}",
    "--output-dir",
    str(output_dir),
  ]
  command.append(
    "--balance-goal-sampling"
    if cfg["balance_goal_sampling"]
    else "--no-balance-goal-sampling"
  )
  return command


def advance(
  *,
  repo_root: Path,
  protocol_path: Path,
  state_path: Path,
  launch_when_ready: bool,
) -> dict[str, Any]:
  protocol = _read_json(protocol_path)
  protocol_sha = sha256_file(protocol_path)
  state = _read_json(state_path) if state_path.is_file() else {"jobs": {}}
  if state.get("protocol_sha256") not in {None, protocol_sha}:
    return _set_state(
      state,
      "PROTOCOL_DRIFT_ALERT",
      "Restore the frozen protocol; do not tune it after observing results.",
      state_path,
    )
  state["protocol_sha256"] = protocol_sha
  state["protocol_id"] = protocol["protocol_id"]

  try:
    expert = protocol["expert"]
    _validate_artifact(
      _resolve(repo_root, expert["checkpoint_file"]),
      expert["checkpoint_sha256"],
      "expert checkpoint",
    )
    _validate_artifact(
      _resolve(repo_root, expert["motion_file"]),
      expert["motion_sha256"],
      "expert motion",
    )
  except Exception as error:
    return _set_state(state, "ARTIFACT_ALERT", str(error), state_path)

  launch_cfg = protocol["launch"]
  upstream_path = Path(launch_cfg["upstream_state_file"])
  if not upstream_path.is_file():
    return _set_state(
      state,
      "WAITING_UPSTREAM",
      f"Upstream state is missing: {upstream_path}",
      state_path,
    )
  upstream = _read_json(upstream_path)
  state["upstream_status"] = upstream.get("status")
  if upstream.get("status") not in launch_cfg["upstream_terminal_statuses"]:
    return _set_state(
      state,
      "WAITING_UPSTREAM",
      "Preserve all GPUs for the matched SMP Tier-A pipeline.",
      state_path,
    )

  seeds = protocol["replicate_seeds"]
  rollout_keys = [f"rollout:{seed}" for seed in seeds]
  rollout_complete: list[int] = []
  try:
    for seed in seeds:
      rollout_dir, _ = _replicate_paths(protocol, seed)
      manifest = rollout_dir / "manifest.json"
      if manifest.is_file():
        validate_rollout_manifest(manifest, protocol, seed)
        rollout_complete.append(seed)
      elif rollout_dir.exists() and f"rollout:{seed}" not in state.get("jobs", {}):
        raise RuntimeError(f"unregistered partial rollout directory: {rollout_dir}")
  except Exception as error:
    return _set_state(state, "ROLLOUT_ALERT", str(error), state_path)

  if len(rollout_complete) != len(seeds):
    active, dead = _active_or_failed_jobs(state, rollout_keys)
    missing_dead = [
      key for key in dead if int(key.split(":")[1]) not in rollout_complete
    ]
    if missing_dead:
      return _set_state(
        state,
        "ROLLOUT_ALERT",
        f"Rollout workers exited without valid manifests: {missing_dead}",
        state_path,
      )
    if active:
      return _set_state(
        state,
        "ROLLOUT_COLLECTION_RUNNING",
        f"Active rollout workers: {active}",
        state_path,
      )
    if not launch_when_ready:
      return _set_state(
        state,
        "ROLLOUT_COLLECTION_READY",
        "Run again with --launch-when-ready after confirming the frozen protocol.",
        state_path,
      )
    if launch_cfg["require_zero_gpu_processes"] and _gpu_processes():
      return _set_state(
        state,
        "WAITING_FREE_GPU",
        "GPU processes are active; do not contend with them.",
        state_path,
      )
    for seed, gpu_id in zip(seeds, launch_cfg["collection_gpu_ids"], strict=True):
      rollout_dir, _ = _replicate_paths(protocol, seed)
      key = f"rollout:{seed}"
      _launch_job(
        state=state,
        key=key,
        command=_collection_command(repo_root, protocol, seed, rollout_dir),
        gpu_id=gpu_id,
        log_path=state_path.parent / "logs" / f"rollout_{seed}.log",
        repo_root=repo_root,
      )
    return _set_state(
      state,
      "ROLLOUT_COLLECTION_RUNNING",
      "Launched three independently seeded 93D disturbed rollout banks.",
      state_path,
    )

  action_keys = [f"action:{seed}" for seed in seeds]
  action_complete: list[int] = []
  try:
    for seed in seeds:
      rollout_dir, action_dir = _replicate_paths(protocol, seed)
      checkpoint = action_dir / "firm_action_diffusion.pt"
      if checkpoint.is_file():
        validate_action_checkpoint(
          checkpoint, rollout_dir / "manifest.json", protocol, seed
        )
        action_complete.append(seed)
      elif action_dir.exists() and f"action:{seed}" not in state.get("jobs", {}):
        raise RuntimeError(f"unregistered partial action directory: {action_dir}")
  except Exception as error:
    return _set_state(state, "ACTION_TRAINING_ALERT", str(error), state_path)

  if len(action_complete) != len(seeds):
    active, dead = _active_or_failed_jobs(state, action_keys)
    missing_dead = [
      key for key in dead if int(key.split(":")[1]) not in action_complete
    ]
    if missing_dead:
      return _set_state(
        state,
        "ACTION_TRAINING_ALERT",
        f"Action workers exited without valid checkpoints: {missing_dead}",
        state_path,
      )
    if active:
      return _set_state(
        state, "ACTION_TRAINING_RUNNING", f"Active action workers: {active}", state_path
      )
    if not launch_when_ready:
      return _set_state(
        state,
        "ACTION_TRAINING_READY",
        "Run again with --launch-when-ready to train three action seeds.",
        state_path,
      )
    if launch_cfg["require_zero_gpu_processes"] and _gpu_processes():
      return _set_state(
        state, "WAITING_FREE_GPU", "GPU processes are active.", state_path
      )
    for seed, gpu_id in zip(seeds, launch_cfg["action_gpu_ids"], strict=True):
      rollout_dir, action_dir = _replicate_paths(protocol, seed)
      _launch_job(
        state=state,
        key=f"action:{seed}",
        command=_action_command(
          protocol, seed, rollout_dir / "manifest.json", action_dir
        ),
        gpu_id=gpu_id,
        log_path=state_path.parent / "logs" / f"action_{seed}.log",
        repo_root=repo_root,
      )
    return _set_state(
      state,
      "ACTION_TRAINING_RUNNING",
      "Launched three independent 93D action-diffusion seeds.",
      state_path,
    )

  adapter_specs = [
    (seed, variant["name"], variant["history_steps"])
    for variant in protocol["adapter_variants"]
    for seed in seeds
  ]
  adapter_keys = [f"adapter:{name}:{seed}" for seed, name, _ in adapter_specs]
  adapter_complete: list[str] = []
  try:
    for seed, name, history_steps in adapter_specs:
      rollout_dir, action_dir = _replicate_paths(protocol, seed)
      adapter_dir = _adapter_path(protocol, seed, name)
      checkpoint = adapter_dir / "firm_goal_adapter.pt"
      if checkpoint.is_file():
        validate_adapter_checkpoint(
          checkpoint,
          action_dir / "firm_action_diffusion.pt",
          protocol,
          seed,
          history_steps,
        )
        adapter_complete.append(f"adapter:{name}:{seed}")
      elif adapter_dir.exists() and f"adapter:{name}:{seed}" not in state.get(
        "jobs", {}
      ):
        raise RuntimeError(f"unregistered partial adapter directory: {adapter_dir}")
  except Exception as error:
    return _set_state(state, "ADAPTER_TRAINING_ALERT", str(error), state_path)

  if len(adapter_complete) == len(adapter_specs):
    return _set_state(
      state,
      "READY_FOR_MATCHED_EVAL_ADAPTER",
      "93D FIRM-R artifacts are complete; matched-state evaluation integration remains required.",
      state_path,
    )
  active, dead = _active_or_failed_jobs(state, adapter_keys)
  missing_dead = [key for key in dead if key not in adapter_complete]
  if missing_dead:
    return _set_state(
      state,
      "ADAPTER_TRAINING_ALERT",
      f"Adapter workers exited without valid checkpoints: {missing_dead}",
      state_path,
    )
  if active:
    return _set_state(
      state,
      "ADAPTER_TRAINING_RUNNING",
      f"Active adapter workers: {active}",
      state_path,
    )
  if not launch_when_ready:
    return _set_state(
      state,
      "ADAPTER_TRAINING_READY",
      "Run again with --launch-when-ready to train 1-frame and 50-frame adapters.",
      state_path,
    )
  if launch_cfg["require_zero_gpu_processes"] and _gpu_processes():
    return _set_state(
      state, "WAITING_FREE_GPU", "GPU processes are active.", state_path
    )
  for (seed, name, history_steps), gpu_id in zip(
    adapter_specs, launch_cfg["adapter_gpu_ids"], strict=True
  ):
    rollout_dir, action_dir = _replicate_paths(protocol, seed)
    adapter_dir = _adapter_path(protocol, seed, name)
    _launch_job(
      state=state,
      key=f"adapter:{name}:{seed}",
      command=_adapter_command(
        protocol,
        seed,
        history_steps,
        action_dir / "firm_action_diffusion.pt",
        rollout_dir / "manifest.json",
        adapter_dir,
        name,
      ),
      gpu_id=gpu_id,
      log_path=state_path.parent / "logs" / f"adapter_{name}_{seed}.log",
      repo_root=repo_root,
    )
  return _set_state(
    state,
    "ADAPTER_TRAINING_RUNNING",
    "Launched paired 1-frame and causal 50-frame adapter variants.",
    state_path,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo-root", type=Path, default=Path.cwd())
  parser.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
  parser.add_argument("--state", type=Path, default=Path(DEFAULT_STATE))
  parser.add_argument("--launch-when-ready", action="store_true")
  args = parser.parse_args()
  repo_root = args.repo_root.resolve()
  protocol_path = _resolve(repo_root, str(args.protocol))
  state_path = _resolve(repo_root, str(args.state))
  result = advance(
    repo_root=repo_root,
    protocol_path=protocol_path,
    state_path=state_path,
    launch_when_ready=args.launch_when_ready,
  )
  print(f"{result['status']}: {result['action']}")


if __name__ == "__main__":
  main()
