"""Interactively play a trained FIRM action-diffusion policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from tensordict import TensorDict

import smp.rl.tasks  # noqa: F401
from smp.firm.action_diffusion import (
  FirmActionDiffusion,
  denormalize_actions,
  load_action_diffusion_checkpoint,
  normalize_action_condition,
  sample_action_horizon,
)
from smp.firm.expert_runtime import actor_base_observation, create_expert_runtime
from smp.pretrain.scheduler import DDPMScheduler
from smp.rl.tasks.firm.keyframe_env_cfg import MOTION_FILE

TASK_ID = "Firm-Keyframe-G1"


@dataclass(frozen=True)
class PlayDiffusionPolicyConfig:
  """Interactive FIRM diffusion-policy viewer configuration."""

  action_checkpoint_file: str | None = None
  action_wandb_run_path: str | None = None
  action_wandb_checkpoint_name: str | None = "firm_action_diffusion.pt"
  """Action-diffusion checkpoint name within the W&B run."""
  action_log_root: str = "logs/firm_action_diffusion"
  expert_checkpoint_file: str | None = None
  expert_wandb_run_path: str | None = None
  expert_wandb_checkpoint_name: str | None = None
  motion_file: str = MOTION_FILE
  start_frame: int = 0
  num_envs: int = 1
  num_action_samples: int = 4
  """Independent DDPM horizons averaged at every control step."""
  use_ema: bool = True
  observation_corruption: bool = False
  disable_terminations: bool = True
  viewer: Literal["auto", "native", "viser"] = "auto"
  frame_rate: float = 60.0
  enable_perturbations: bool = True
  """Allow mouse-applied forces in the native MuJoCo viewer."""
  seed: int = 42
  device: str | None = None
  log_root: str = "logs/rsl_rl"


class _DiffusionPolicy:
  def __init__(
    self,
    *,
    env,
    command,
    model: FirmActionDiffusion,
    scheduler: DDPMScheduler,
    statistics: dict[str, torch.Tensor],
    num_action_samples: int,
  ) -> None:
    self.env = env
    self.command = command
    self.model = model
    self.scheduler = scheduler
    self.statistics = statistics
    self.num_action_samples = num_action_samples

  @torch.inference_mode()
  def __call__(self, obs: TensorDict) -> torch.Tensor:
    observation = actor_base_observation(obs)
    normalized_observation, current_joint, normalized_goal = normalize_action_condition(
      observation,
      self.command.joint_pos,
      self.statistics,
    )
    batch = observation.shape[0]
    if self.num_action_samples > 1:
      normalized_observation = normalized_observation.repeat_interleave(
        self.num_action_samples, dim=0
      )
      current_joint = current_joint.repeat_interleave(self.num_action_samples, dim=0)
      normalized_goal = normalized_goal.repeat_interleave(
        self.num_action_samples, dim=0
      )
    horizon = sample_action_horizon(
      self.model,
      self.scheduler,
      normalized_observation,
      current_joint,
      normalized_goal,
    )
    if self.num_action_samples > 1:
      horizon = horizon.view(
        batch,
        self.num_action_samples,
        self.model.horizon,
        self.model.action_dim,
      ).mean(dim=1)
    actions = denormalize_actions(horizon[:, 0], self.statistics)
    if self.env.clip_actions is not None:
      actions = actions.clamp(-self.env.clip_actions, self.env.clip_actions)
    return actions


def _resolve_action_checkpoint(cfg: PlayDiffusionPolicyConfig) -> Path:
  if (cfg.action_checkpoint_file is None) == (cfg.action_wandb_run_path is None):
    raise ValueError(
      "provide exactly one of action_checkpoint_file or action_wandb_run_path"
    )
  if cfg.action_checkpoint_file is not None:
    path = Path(cfg.action_checkpoint_file).expanduser().resolve()
    if not path.is_file():
      raise FileNotFoundError(path)
    return path

  if cfg.action_wandb_checkpoint_name is None:
    raise ValueError("action_wandb_checkpoint_name is required for a W&B run")
  assert cfg.action_wandb_run_path is not None
  cache_dir = (
    (Path(cfg.action_log_root) / Path(cfg.action_wandb_run_path).name)
    .expanduser()
    .resolve()
  )
  target = cache_dir / cfg.action_wandb_checkpoint_name
  if target.is_file():
    print(f"[INFO] Action-diffusion W&B checkpoint cached: {target}")
    return target

  import wandb

  run = wandb.Api().run(cfg.action_wandb_run_path)
  remote_file = run.file(cfg.action_wandb_checkpoint_name)
  if remote_file is None:
    raise FileNotFoundError(
      f"{cfg.action_wandb_checkpoint_name!r} is absent from {cfg.action_wandb_run_path}"
    )
  cache_dir.mkdir(parents=True, exist_ok=True)
  downloaded = Path(remote_file.download(root=str(cache_dir), replace=True))
  if downloaded.is_dir():
    downloaded = downloaded / cfg.action_wandb_checkpoint_name
  if not downloaded.is_file():
    raise FileNotFoundError(downloaded)
  print(f"[INFO] Action-diffusion W&B checkpoint downloaded: {downloaded}")
  return downloaded


def run_play(cfg: PlayDiffusionPolicyConfig) -> None:
  """Build the matched FIRM environment and launch an interactive viewer."""
  if cfg.start_frame < 0:
    raise ValueError("start_frame must be non-negative")
  if cfg.num_envs <= 0 or cfg.num_action_samples <= 0:
    raise ValueError("num_envs and num_action_samples must be positive")
  if cfg.frame_rate <= 0:
    raise ValueError("frame_rate must be positive")

  runtime = create_expert_runtime(
    task_id=TASK_ID,
    motion_file=cfg.motion_file,
    checkpoint_file=cfg.expert_checkpoint_file,
    wandb_run_path=cfg.expert_wandb_run_path,
    wandb_checkpoint_name=cfg.expert_wandb_checkpoint_name,
    log_root=cfg.log_root,
    num_start_frames=1,
    episodes_per_frame=cfg.num_envs,
    seed=cfg.seed,
    device=cfg.device,
    observation_corruption=cfg.observation_corruption,
    start_frame_range=(cfg.start_frame, cfg.start_frame),
    play=True,
    debug_vis=True,
    disable_terminations=cfg.disable_terminations,
  )
  env = runtime.env
  device = torch.device(env.device)
  action_checkpoint = _resolve_action_checkpoint(cfg)
  model, scheduler, statistics, _ = load_action_diffusion_checkpoint(
    action_checkpoint,
    device,
    use_ema=cfg.use_ema,
  )
  policy = _DiffusionPolicy(
    env=env,
    command=runtime.command,
    model=model,
    scheduler=scheduler,
    statistics=statistics,
    num_action_samples=cfg.num_action_samples,
  )

  resolved_viewer = cfg.viewer
  if resolved_viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  print(
    f"[INFO] FIRM diffusion play: viewer={resolved_viewer}, "
    f"start_frame={cfg.start_frame}, samples={cfg.num_action_samples}, "
    f"terminations={'off' if cfg.disable_terminations else 'on'}"
  )
  try:
    if resolved_viewer == "native":
      from mjlab.viewer import NativeMujocoViewer

      NativeMujocoViewer(
        env,
        policy,
        frame_rate=cfg.frame_rate,
        enable_perturbations=cfg.enable_perturbations,
      ).run()
    elif resolved_viewer == "viser":
      from mjlab.viewer import ViserPlayViewer

      ViserPlayViewer(env, policy, frame_rate=cfg.frame_rate).run()
    else:
      raise RuntimeError(f"unsupported viewer: {resolved_viewer}")
  finally:
    env.close()


def main() -> None:
  run_play(tyro.cli(PlayDiffusionPolicyConfig))


if __name__ == "__main__":
  main()
