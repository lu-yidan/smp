"""Collect sharded observation-goal-action rollouts from a FIRM expert."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import smp.rl.tasks  # noqa: F401
from smp.firm.expert_runtime import (
  actor_base_observation,
  create_expert_runtime,
  runtime_metadata,
  sha256_file,
)
from smp.rl.tasks.firm.keyframe_env_cfg import MOTION_FILE
from smp.rl.tasks.getup.mdp.events import random_body_wrench

DEFAULT_TASK_ID = "Firm-Keyframe-G1"


@dataclass(frozen=True)
class CollectRolloutsConfig:
  """Pilot rollout collection configuration."""

  task_id: str = DEFAULT_TASK_ID
  """Registered expert task; deployable observations use Firm-Keyframe-Deployable-G1."""
  checkpoint_file: str | None = None
  """Local model checkpoint. Mutually exclusive with wandb_run_path."""
  wandb_run_path: str | None = None
  """W&B run path, for example tabletennis/smp/j0q8fell."""
  wandb_checkpoint_name: str | None = None
  motion_file: str = MOTION_FILE
  num_start_frames: int = 25
  start_frame_range: tuple[int, int] | None = None
  """Optional inclusive dense-start range; defaults to the complete motion."""
  episodes_per_frame: int = 8
  """Eight replicas produce at most 100,000 transitions over 500 steps."""
  max_steps: int = 500
  standing_hold_steps: int = 25
  root_height_threshold: float = 0.65
  upright_threshold: float = 0.85
  root_linear_speed_threshold: float = 0.50
  root_angular_speed_threshold: float = 0.50
  observation_corruption: bool = True
  physical_disturbances: bool = False
  """Apply finite-duration torso wrenches while the expert acts."""
  disturbance_interval_steps: tuple[int, int] = (50, 150)
  disturbance_duration_steps: tuple[int, int] = (5, 10)
  disturbance_recovery_steps: int = 40
  disturbance_force_range: tuple[float, float] = (20.0, 80.0)
  disturbance_torque_range: tuple[float, float] = (2.0, 10.0)
  """Ranges ramp from the first to second value during a long training run.

  Corrective rollout capture uses the upper value after its first interval.
  """
  seed: int = 42
  device: str | None = None
  output_dir: str = "datasets/firm/rollouts/c003_stage0_pilot"
  shard_size: int = 50_000
  log_root: str = "logs/rsl_rl"


class ShardWriter:
  """Buffer transition batches and write independently checksummed NPZ shards."""

  def __init__(self, output_dir: Path, shard_size: int):
    if shard_size <= 0:
      raise ValueError(f"shard_size must be positive, got {shard_size}")
    self.output_dir = output_dir
    self.shard_size = shard_size
    self.buffers: dict[str, list[np.ndarray]] = {}
    self.buffered = 0
    self.total = 0
    self.shards: list[dict[str, int | str]] = []

  def add(self, batch: dict[str, np.ndarray]) -> None:
    sizes = {value.shape[0] for value in batch.values()}
    if len(sizes) != 1:
      raise ValueError(f"inconsistent transition batch sizes: {sizes}")
    size = sizes.pop()
    for name, value in batch.items():
      self.buffers.setdefault(name, []).append(value)
    self.buffered += size
    if self.buffered >= self.shard_size:
      self.flush()

  def flush(self) -> None:
    if self.buffered == 0:
      return
    arrays = {
      name: np.concatenate(parts, axis=0) for name, parts in self.buffers.items()
    }
    shard_index = len(self.shards)
    shard_path = self.output_dir / f"shard_{shard_index:04d}.npz"
    np.savez_compressed(shard_path, **arrays)
    checksum = sha256_file(shard_path)
    samples = next(iter(arrays.values())).shape[0]
    self.shards.append(
      {
        "file": shard_path.name,
        "samples": int(samples),
        "sha256": checksum,
      }
    )
    self.total += samples
    self.buffers.clear()
    self.buffered = 0
    print(
      f"[INFO] wrote {shard_path.name}: {samples} samples, sha256={checksum[:12]}..."
    )


def _cpu(array: torch.Tensor, dtype: torch.dtype | None = None) -> np.ndarray:
  if dtype is not None:
    array = array.to(dtype=dtype)
  return array.detach().cpu().numpy()


def _prepare_output(path: Path) -> None:
  path.mkdir(parents=True, exist_ok=True)
  conflicts = [path / "manifest.json", *path.glob("shard_*.npz")]
  conflicts = [item for item in conflicts if item.exists()]
  if conflicts:
    names = ", ".join(item.name for item in conflicts[:5])
    raise FileExistsError(
      f"{path} already contains a rollout dataset ({names}); choose a new output_dir"
    )


def _observation_layout(observation_dim: int) -> list[dict[str, object]]:
  if observation_dim == 90:
    return [
      {"name": "root_angular_velocity", "slice": [0, 3]},
      {"name": "joint_position", "slice": [3, 32]},
      {"name": "joint_velocity", "slice": [32, 61]},
      {"name": "previous_action", "slice": [61, 90]},
    ]
  if observation_dim == 93:
    return [
      {"name": "root_angular_velocity", "slice": [0, 3]},
      {"name": "projected_gravity", "slice": [3, 6]},
      {"name": "joint_position", "slice": [6, 35]},
      {"name": "joint_velocity", "slice": [35, 64]},
      {"name": "previous_action", "slice": [64, 93]},
    ]
  raise ValueError(f"unsupported deployable observation dimension: {observation_dim}")


def run_collection(cfg: CollectRolloutsConfig) -> dict:
  """Collect sequential expert transitions and write a reproducible manifest."""
  if cfg.max_steps <= 0 or cfg.standing_hold_steps <= 0:
    raise ValueError("max_steps and standing_hold_steps must be positive")
  if cfg.physical_disturbances:
    integer_ranges = {
      "disturbance_interval_steps": cfg.disturbance_interval_steps,
      "disturbance_duration_steps": cfg.disturbance_duration_steps,
    }
    for name, bounds in integer_ranges.items():
      if bounds[0] <= 0 or bounds[0] > bounds[1]:
        raise ValueError(f"{name} must be positive and ordered, got {bounds}")
    if cfg.disturbance_recovery_steps < 0:
      raise ValueError("disturbance_recovery_steps must be non-negative")
  output_dir = Path(cfg.output_dir)
  _prepare_output(output_dir)

  runtime = create_expert_runtime(
    task_id=cfg.task_id,
    motion_file=cfg.motion_file,
    checkpoint_file=cfg.checkpoint_file,
    wandb_run_path=cfg.wandb_run_path,
    wandb_checkpoint_name=cfg.wandb_checkpoint_name,
    log_root=cfg.log_root,
    num_start_frames=cfg.num_start_frames,
    episodes_per_frame=cfg.episodes_per_frame,
    seed=cfg.seed,
    device=cfg.device,
    observation_corruption=cfg.observation_corruption,
    start_frame_range=cfg.start_frame_range,
  )
  env = runtime.env
  raw_env = env.unwrapped
  robot = raw_env.scene["robot"]
  command = runtime.command
  n = env.num_envs
  device = env.device
  writer = ShardWriter(output_dir, cfg.shard_size)
  disturbed_transitions = torch.zeros((), dtype=torch.long, device=device)

  done = torch.zeros(n, dtype=torch.bool, device=device)
  success = torch.zeros_like(done)
  unsafe = torch.zeros_like(done)
  timed_out = torch.zeros_like(done)
  active_steps = torch.zeros(n, dtype=torch.long, device=device)
  stable_hold = torch.zeros(n, dtype=torch.long, device=device)
  episode_ids = torch.arange(n, dtype=torch.long, device=device)
  obs = env.get_observations()
  observation_dim = int(actor_base_observation(obs).shape[-1])

  try:
    for step in range(cfg.max_steps):
      active = ~done
      active_ids = torch.where(active)[0]
      if active_ids.numel() == 0:
        break

      state_observation = actor_base_observation(obs)
      motion_frames = command.time_steps.clone()
      goal_frames = command.goal_steps.clone()
      goals = command.joint_pos.clone()
      with torch.no_grad():
        actions = runtime.policy(obs)
      if env.clip_actions is not None:
        actions = actions.clamp(-env.clip_actions, env.clip_actions)

      upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
      root_linear_speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)
      root_angular_speed = torch.linalg.norm(robot.data.root_link_ang_vel_w, dim=-1)
      stable = (
        (robot.data.root_link_pos_w[:, 2] >= cfg.root_height_threshold)
        & (upright >= cfg.upright_threshold)
        & (root_linear_speed <= cfg.root_linear_speed_threshold)
        & (root_angular_speed <= cfg.root_angular_speed_threshold)
      )
      stable_hold = torch.where(active & stable, stable_hold + 1, 0)

      transition_disturbed = torch.zeros(n, dtype=torch.bool, device=device)
      if cfg.physical_disturbances:
        random_body_wrench(
          raw_env,
          env_ids=active_ids,
          interval_steps=cfg.disturbance_interval_steps,
          duration_steps=cfg.disturbance_duration_steps,
          recovery_steps=cfg.disturbance_recovery_steps,
          force_range=cfg.disturbance_force_range,
          torque_range=cfg.disturbance_torque_range,
          curriculum_steps=1,
        )
        applied_forces = raw_env._robust_forces  # type: ignore[attr-defined]
        transition_disturbed[active_ids] = (
          applied_forces[active_ids].abs().amax(dim=(1, 2)) > 0.0
        )
      disturbed_transitions += transition_disturbed.sum()

      next_obs, _, dones, _ = env.step(actions)
      terminated = raw_env.termination_manager.terminated.bool()
      timeouts = raw_env.termination_manager.time_outs.bool()
      transition_done = dones.bool() & active
      transition_unsafe = terminated & active
      transition_timeout = timeouts & active

      ids = active_ids
      count = ids.numel()
      writer.add(
        {
          "observation": _cpu(state_observation[ids], torch.float32),
          "goal": _cpu(goals[ids], torch.float32),
          "action": _cpu(actions[ids], torch.float32),
          "episode_id": _cpu(episode_ids[ids], torch.int64),
          "episode_step": np.full(count, step, dtype=np.int32),
          "start_frame": _cpu(runtime.env_start_frames[ids], torch.int32),
          "motion_frame": _cpu(motion_frames[ids], torch.int32),
          "goal_frame": _cpu(goal_frames[ids], torch.int32),
          "transition_disturbed": _cpu(transition_disturbed[ids], torch.bool),
          "done": _cpu(transition_done[ids], torch.bool),
          "unsafe": _cpu(transition_unsafe[ids], torch.bool),
          "timeout": _cpu(transition_timeout[ids], torch.bool),
        }
      )
      active_steps += active.long()

      newly_done = transition_done
      if newly_done.any():
        unsafe[newly_done] = terminated[newly_done]
        timed_out[newly_done] = timeouts[newly_done]
        success[newly_done] = (
          timeouts[newly_done]
          & ~terminated[newly_done]
          & (stable_hold[newly_done] >= cfg.standing_hold_steps)
        )
        done[newly_done] = True
      obs = next_obs

      if step % 50 == 0 or newly_done.any():
        print(
          f"[INFO] step={step:03d} samples={writer.total + writer.buffered} "
          f"active={int((~done).sum())} success={int(success.sum())} "
          f"unsafe={int(unsafe.sum())}"
        )
  finally:
    env.close()
  writer.flush()

  episode_records = [
    {
      "episode_id": index,
      "start_frame": int(runtime.env_start_frames[index].item()),
      "samples": int(active_steps[index].item()),
      "completed": bool(done[index].item()),
      "success": bool(success[index].item()),
      "unsafe": bool(unsafe[index].item()),
      "timeout": bool(timed_out[index].item()),
    }
    for index in range(n)
  ]
  manifest = {
    "format_version": 1,
    "task_id": cfg.task_id,
    "config": asdict(cfg),
    "artifacts": runtime_metadata(runtime),
    "layout": {
      "observation": {
        "shape": [observation_dim],
        "components": _observation_layout(observation_dim),
      },
      "goal": {"shape": [29], "description": "target keyframe joint position"},
      "action": {"shape": [29], "description": "clipped expert policy action"},
      "transition_disturbed": {
        "shape": [],
        "description": "wrench applied during this transition",
      },
      "ordering": "step-major; use episode_id and episode_step for sequences",
    },
    "total_samples": writer.total,
    "disturbed_transitions": int(disturbed_transitions.item()),
    "episodes": n,
    "successful_episodes": int(success.sum().item()),
    "unsafe_episodes": int(unsafe.sum().item()),
    "shards": writer.shards,
    "episode_records": episode_records,
  }
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(
    f"[INFO] Rollout manifest written to {manifest_path}: "
    f"{writer.total} samples, {int(success.sum())}/{n} successful episodes"
  )
  return manifest


def main() -> None:
  run_collection(tyro.cli(CollectRolloutsConfig))


if __name__ == "__main__":
  main()
