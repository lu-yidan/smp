"""Collect long expert rescues from states visited by action diffusion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from collect_onpolicy_corrections import _diffusion_action
from collect_rollouts import ShardWriter, _cpu, _prepare_output

import smp.rl.tasks  # noqa: F401
from smp.firm.action_diffusion import load_action_diffusion_checkpoint
from smp.firm.expert_runtime import (
  actor_base_observation,
  create_expert_runtime,
  runtime_metadata,
  sha256_file,
)
from smp.rl.tasks.firm.keyframe_env_cfg import MOTION_FILE

TASK_ID = "Firm-Keyframe-G1"


@dataclass(frozen=True)
class CollectOnPolicyRescuesConfig:
  """Diffusion rollout followed by a coherent expert takeover."""

  action_checkpoint_file: str
  expert_checkpoint_file: str | None = None
  expert_wandb_run_path: str | None = None
  expert_wandb_checkpoint_name: str | None = None
  motion_file: str = MOTION_FILE
  start_frame: int = 324
  episodes_per_frame: int = 64
  minimum_diffusion_steps: int = 40
  disagreement_threshold: float = 0.12
  max_steps: int = 500
  standing_hold_steps: int = 25
  root_height_threshold: float = 0.65
  upright_threshold: float = 0.85
  root_linear_speed_threshold: float = 0.50
  root_angular_speed_threshold: float = 0.50
  observation_corruption: bool = True
  use_ema: bool = True
  num_action_samples: int = 4
  seed: int = 45
  device: str | None = None
  output_dir: str = "datasets/firm/rollouts/c003_onpolicy_rescues"
  shard_size: int = 50_000
  log_root: str = "logs/rsl_rl"


def _validate(cfg: CollectOnPolicyRescuesConfig) -> None:
  if cfg.start_frame < 0:
    raise ValueError("start_frame must be non-negative")
  if cfg.episodes_per_frame <= 0 or cfg.max_steps <= 0:
    raise ValueError("episodes_per_frame and max_steps must be positive")
  if not 0 <= cfg.minimum_diffusion_steps < cfg.max_steps:
    raise ValueError("minimum_diffusion_steps must be in [0, max_steps)")
  if cfg.disagreement_threshold < 0.0:
    raise ValueError("disagreement_threshold must be non-negative")
  if cfg.standing_hold_steps <= 0 or cfg.num_action_samples <= 0:
    raise ValueError("standing_hold_steps and num_action_samples must be positive")


def run_collection(cfg: CollectOnPolicyRescuesConfig) -> dict:
  """Save all expert transitions after an on-policy rescue trigger."""
  _validate(cfg)
  output_dir = Path(cfg.output_dir).expanduser()
  _prepare_output(output_dir)
  runtime = create_expert_runtime(
    task_id=TASK_ID,
    motion_file=cfg.motion_file,
    checkpoint_file=cfg.expert_checkpoint_file,
    wandb_run_path=cfg.expert_wandb_run_path,
    wandb_checkpoint_name=cfg.expert_wandb_checkpoint_name,
    log_root=cfg.log_root,
    num_start_frames=1,
    episodes_per_frame=cfg.episodes_per_frame,
    seed=cfg.seed,
    device=cfg.device,
    observation_corruption=cfg.observation_corruption,
    start_frame_range=(cfg.start_frame, cfg.start_frame),
  )
  env = runtime.env
  raw_env = env.unwrapped
  robot = raw_env.scene["robot"]
  command = runtime.command
  device = torch.device(env.device)
  model, scheduler, statistics, action_checkpoint = load_action_diffusion_checkpoint(
    cfg.action_checkpoint_file,
    device,
    use_ema=cfg.use_ema,
  )

  n = env.num_envs
  writer = ShardWriter(output_dir, cfg.shard_size)
  done = torch.zeros(n, dtype=torch.bool, device=device)
  success = torch.zeros_like(done)
  unsafe = torch.zeros_like(done)
  timed_out = torch.zeros_like(done)
  rescuing = torch.zeros_like(done)
  stable_hold = torch.zeros(n, dtype=torch.long, device=device)
  rescue_steps = torch.zeros(n, dtype=torch.long, device=device)
  trigger_steps = torch.full((n,), -1, dtype=torch.long, device=device)
  trigger_rmse = torch.full((n,), float("nan"), device=device)
  simulation_steps = torch.zeros(n, dtype=torch.long, device=device)
  episode_ids = torch.arange(n, dtype=torch.long, device=device)
  obs = env.get_observations()

  try:
    for step in range(cfg.max_steps):
      active = ~done
      if not active.any():
        break
      state_observation = actor_base_observation(obs)
      goals = command.joint_pos.clone()
      motion_frames = command.time_steps.clone()
      goal_frames = command.goal_steps.clone()
      diffusion_action = _diffusion_action(
        state_observation,
        goals,
        model=model,
        scheduler=scheduler,
        statistics=statistics,
        num_samples=cfg.num_action_samples,
      )
      with torch.no_grad():
        expert_action = runtime.policy(obs)
      if env.clip_actions is not None:
        diffusion_action = diffusion_action.clamp(
          -env.clip_actions, env.clip_actions
        )
        expert_action = expert_action.clamp(-env.clip_actions, env.clip_actions)
      disagreement = torch.sqrt(
        torch.mean(torch.square(diffusion_action - expert_action), dim=-1)
      )

      new_triggers = (
        active
        & ~rescuing
        & (step >= cfg.minimum_diffusion_steps)
        & (disagreement >= cfg.disagreement_threshold)
      )
      if new_triggers.any():
        rescuing[new_triggers] = True
        trigger_steps[new_triggers] = step
        trigger_rmse[new_triggers] = disagreement[new_triggers]

      rescue_ids = torch.where(active & rescuing)[0]
      actions = diffusion_action.clone()
      actions[rescue_ids] = expert_action[rescue_ids]

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

      next_obs, _, dones, _ = env.step(actions)
      terminated = raw_env.termination_manager.terminated.bool()
      timeouts = raw_env.termination_manager.time_outs.bool()
      transition_done = dones.bool() & active
      transition_unsafe = terminated & active
      transition_timeout = timeouts & active

      if rescue_ids.numel() > 0:
        count = rescue_ids.numel()
        writer.add(
          {
            "observation": _cpu(state_observation[rescue_ids], torch.float32),
            "goal": _cpu(goals[rescue_ids], torch.float32),
            "action": _cpu(expert_action[rescue_ids], torch.float32),
            "episode_id": _cpu(episode_ids[rescue_ids], torch.int64),
            "episode_step": _cpu(rescue_steps[rescue_ids], torch.int32),
            "start_frame": np.full(count, cfg.start_frame, dtype=np.int32),
            "source_env_id": _cpu(rescue_ids, torch.int32),
            "motion_frame": _cpu(motion_frames[rescue_ids], torch.int32),
            "goal_frame": _cpu(goal_frames[rescue_ids], torch.int32),
            "trigger_action_rmse": _cpu(
              trigger_rmse[rescue_ids], torch.float32
            ),
            "expert_intervention": np.ones(count, dtype=np.bool_),
            "transition_disturbed": np.zeros(count, dtype=np.bool_),
            "done": _cpu(transition_done[rescue_ids], torch.bool),
            "unsafe": _cpu(transition_unsafe[rescue_ids], torch.bool),
            "timeout": _cpu(transition_timeout[rescue_ids], torch.bool),
          }
        )
        rescue_steps[rescue_ids] += 1
      simulation_steps += active.long()

      if transition_done.any():
        unsafe[transition_done] = terminated[transition_done]
        timed_out[transition_done] = timeouts[transition_done]
        success[transition_done] = (
          timeouts[transition_done]
          & ~terminated[transition_done]
          & (stable_hold[transition_done] >= cfg.standing_hold_steps)
        )
        done[transition_done] = True
      obs = next_obs
      if step % 50 == 0 or new_triggers.any() or transition_done.any():
        print(
          f"[INFO] step={step:03d} active={int((~done).sum())} "
          f"rescuing={int((rescuing & ~done).sum())} "
          f"samples={writer.total + writer.buffered} "
          f"success={int(success.sum())} unsafe={int(unsafe.sum())}"
        )
  finally:
    env.close()
  writer.flush()

  episode_records = [
    {
      "episode_id": index,
      "source_env_id": index,
      "start_frame": cfg.start_frame,
      "trigger_step": int(trigger_steps[index].item()),
      "trigger_action_rmse": (
        float(trigger_rmse[index].item())
        if torch.isfinite(trigger_rmse[index])
        else None
      ),
      "samples": int(rescue_steps[index].item()),
      "simulation_steps": int(simulation_steps[index].item()),
      "completed": bool(done[index].item()),
      "success": bool(success[index].item()),
      "unsafe": bool(unsafe[index].item()),
      "timeout": bool(timed_out[index].item()),
    }
    for index in range(n)
  ]
  action_path = Path(cfg.action_checkpoint_file).expanduser().resolve()
  manifest = {
    "format_version": 1,
    "task_id": TASK_ID,
    "collector": "onpolicy_long_expert_rescue",
    "config": asdict(cfg),
    "artifacts": {
      **runtime_metadata(runtime),
      "action_checkpoint_file": str(action_path),
      "action_checkpoint_sha256": sha256_file(action_path),
      "action_checkpoint_epoch": int(action_checkpoint["epoch"]),
      "action_weights": "ema" if cfg.use_ema else "online",
    },
    "layout": {
      "observation": {
        "shape": [90],
        "description": "actual state during coherent expert rescue",
      },
      "goal": {
        "shape": [29],
        "description": "active sparse-keyframe goal during rescue",
      },
      "action": {
        "shape": [29],
        "description": "clipped expert action at the recorded state",
      },
      "ordering": "step-major; episode_step begins at expert takeover",
    },
    "total_samples": writer.total,
    "episodes": n,
    "triggered_episodes": int((trigger_steps >= 0).sum().item()),
    "successful_episodes": int(success.sum().item()),
    "unsafe_episodes": int(unsafe.sum().item()),
    "shards": writer.shards,
    "episode_records": episode_records,
  }
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(
    f"[INFO] Rescue manifest written to {manifest_path}: "
    f"{writer.total} samples, {int(success.sum())}/{n} successful episodes"
  )
  return manifest


def main() -> None:
  run_collection(tyro.cli(CollectOnPolicyRescuesConfig))


if __name__ == "__main__":
  main()
