"""Shared runtime helpers for FIRM expert evaluation and rollout collection."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from smp.rl.tasks.firm.mdp.commands import (
  SparseKeyframeCommand,
  SparseKeyframeCommandCfg,
)

Policy = Callable[[TensorDict], torch.Tensor]


@dataclass(frozen=True)
class ExpertRuntime:
  """Constructed environment, inference policy, and deterministic start mapping."""

  env: RslRlVecEnvWrapper
  policy: Policy
  command: SparseKeyframeCommand
  start_frames: np.ndarray
  env_start_frames: torch.Tensor
  checkpoint_path: Path
  motion_path: Path


def sha256_file(path: str | Path) -> str:
  """Return a streaming SHA256 checksum for a local artifact."""
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def resolve_checkpoint(
  *,
  experiment_name: str,
  checkpoint_file: str | None,
  wandb_run_path: str | None,
  wandb_checkpoint_name: str | None,
  log_root: str,
) -> Path:
  """Resolve exactly one local or W&B checkpoint source."""
  if (checkpoint_file is None) == (wandb_run_path is None):
    msg = "provide exactly one of checkpoint_file or wandb_run_path"
    raise ValueError(msg)

  if checkpoint_file is not None:
    checkpoint_path = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint_path.is_file():
      raise FileNotFoundError(checkpoint_path)
    return checkpoint_path

  assert wandb_run_path is not None
  checkpoint_path, was_cached = get_wandb_checkpoint_path(
    (Path(log_root) / experiment_name).resolve(),
    Path(wandb_run_path),
    wandb_checkpoint_name,
  )
  state = "cached" if was_cached else "downloaded"
  print(f"[INFO] W&B checkpoint {state}: {checkpoint_path}")
  return checkpoint_path


def dense_start_frames(
  total_frames: int,
  count: int,
  frame_range: tuple[int, int] | None = None,
) -> np.ndarray:
  """Choose unique, evenly spaced reference frames over an inclusive range."""
  if count < 2:
    msg = f"num_start_frames must be at least 2, got {count}"
    raise ValueError(msg)
  lower, upper = (0, total_frames - 1) if frame_range is None else frame_range
  if lower < 0 or upper < lower or upper >= total_frames:
    msg = (
      f"start_frame_range must satisfy 0 <= lower <= upper < {total_frames}, "
      f"got {(lower, upper)}"
    )
    raise ValueError(msg)
  available = upper - lower + 1
  if count > available:
    msg = (
      f"num_start_frames={count} exceeds selected frame count={available} "
      f"for range {(lower, upper)}"
    )
    raise ValueError(msg)
  frames = np.rint(np.linspace(lower, upper, count)).astype(np.int64)
  if len(np.unique(frames)) != count:
    msg = "dense start-frame schedule contains duplicates"
    raise RuntimeError(msg)
  return frames


def _reset_to_start_schedule(
  env: RslRlVecEnvWrapper,
  command: SparseKeyframeCommand,
  start_frames: np.ndarray,
  episodes_per_frame: int,
) -> torch.Tensor:
  device = env.unwrapped.device
  mapping = torch.empty(env.num_envs, dtype=torch.long, device=device)
  for group, frame in enumerate(start_frames.tolist()):
    begin = group * episodes_per_frame
    end = begin + episodes_per_frame
    env_ids = torch.arange(begin, end, device=device)
    command.reset_to_frame(env_ids, int(frame))
    mapping[env_ids] = int(frame)

  env.unwrapped.sim.forward()
  command.update_relative_body_poses()
  env.unwrapped.episode_length_buf.zero_()
  return mapping


def create_expert_runtime(
  *,
  task_id: str,
  motion_file: str,
  checkpoint_file: str | None,
  wandb_run_path: str | None,
  wandb_checkpoint_name: str | None,
  log_root: str,
  num_start_frames: int,
  episodes_per_frame: int,
  seed: int,
  device: str | None,
  observation_corruption: bool,
  start_frame_range: tuple[int, int] | None = None,
) -> ExpertRuntime:
  """Build a trained FIRM expert over a deterministic dense-frame schedule."""
  if episodes_per_frame <= 0:
    msg = f"episodes_per_frame must be positive, got {episodes_per_frame}"
    raise ValueError(msg)

  configure_torch_backends()
  resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  motion_path = Path(motion_file).expanduser().resolve()
  if not motion_path.is_file():
    raise FileNotFoundError(motion_path)

  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = num_start_frames * episodes_per_frame
  env_cfg.observations["actor"].enable_corruption = observation_corruption

  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, SparseKeyframeCommandCfg):
    msg = f"{task_id} does not expose a SparseKeyframeCommandCfg"
    raise TypeError(msg)
  motion_cfg.motion_file = str(motion_path)
  motion_cfg.sampling_mode = "start"
  motion_cfg.debug_vis = False

  checkpoint_path = resolve_checkpoint(
    experiment_name=agent_cfg.experiment_name,
    checkpoint_file=checkpoint_file,
    wandb_run_path=wandb_run_path,
    wandb_checkpoint_name=wandb_checkpoint_name,
    log_root=log_root,
  )

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=resolved_device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=resolved_device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=resolved_device,
  )
  policy = cast(Policy, runner.get_inference_policy(device=resolved_device))
  command = cast(
    SparseKeyframeCommand, env.unwrapped.command_manager.get_term("motion")
  )

  start_frames = dense_start_frames(
    command.motion.time_step_total,
    num_start_frames,
    start_frame_range,
  )
  env_start_frames = _reset_to_start_schedule(
    env, command, start_frames, episodes_per_frame
  )
  print(
    f"[INFO] FIRM runtime: {env.num_envs} envs, "
    f"{len(start_frames)} dense starts x {episodes_per_frame}, "
    f"frames {start_frames[0]}..{start_frames[-1]}"
  )
  return ExpertRuntime(
    env=env,
    policy=policy,
    command=command,
    start_frames=start_frames,
    env_start_frames=env_start_frames,
    checkpoint_path=checkpoint_path,
    motion_path=motion_path,
  )


def actor_base_observation(obs: TensorDict) -> torch.Tensor:
  """Return FIRM's 90-D state observation without goal error or phase."""
  actor = obs["actor"]
  if actor.shape[-1] != 120:
    msg = f"expected 120-D FIRM actor observation, got {tuple(actor.shape)}"
    raise RuntimeError(msg)
  return actor[:, :90]


def runtime_metadata(runtime: ExpertRuntime) -> dict[str, Any]:
  """Return immutable artifact identifiers shared by evaluation datasets."""
  return {
    "checkpoint_file": str(runtime.checkpoint_path),
    "checkpoint_sha256": sha256_file(runtime.checkpoint_path),
    "motion_file": str(runtime.motion_path),
    "motion_sha256": sha256_file(runtime.motion_path),
    "motion_frames": int(runtime.command.motion.time_step_total),
    "start_frames": runtime.start_frames.tolist(),
  }
