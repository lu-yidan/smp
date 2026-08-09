"""Evaluate the FIRM action diffusion model in closed-loop simulation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from evaluate_expert import _aggregate, _masked_max
from mjlab.tasks.tracking.mdp.metrics import (
  compute_mpkpe,
  compute_root_relative_mpkpe,
)

import smp.rl.tasks  # noqa: F401
from smp.firm.action_diffusion import (
  denormalize_actions,
  load_action_diffusion_checkpoint,
  normalize_action_condition,
  sample_action_horizon,
)
from smp.firm.expert_runtime import (
  actor_base_observation,
  create_expert_runtime,
  runtime_metadata,
  sha256_file,
)
from smp.rl.tasks.firm.keyframe_env_cfg import MOTION_FILE

TASK_ID = "Firm-Keyframe-G1"


@dataclass(frozen=True)
class EvaluateDiffusionPolicyConfig:
  """Fixed-start closed-loop diffusion-policy evaluation configuration."""

  action_checkpoint_file: str
  expert_checkpoint_file: str | None = None
  """Stage-0 expert checkpoint used only to construct the matched runtime."""
  expert_wandb_run_path: str | None = None
  expert_wandb_checkpoint_name: str | None = None
  motion_file: str = MOTION_FILE
  num_start_frames: int = 25
  episodes_per_frame: int = 32
  max_steps: int = 500
  standing_hold_steps: int = 25
  root_height_threshold: float = 0.65
  upright_threshold: float = 0.85
  root_linear_speed_threshold: float = 0.50
  root_angular_speed_threshold: float = 0.50
  observation_corruption: bool = True
  use_ema: bool = True
  seed: int = 42
  device: str | None = None
  output_file: str | None = None
  log_root: str = "logs/rsl_rl"


def run_evaluation(cfg: EvaluateDiffusionPolicyConfig) -> dict:
  """Run receding-horizon diffusion inference, executing its first action."""
  if cfg.max_steps <= 0 or cfg.standing_hold_steps <= 0:
    raise ValueError("max_steps and standing_hold_steps must be positive")

  runtime = create_expert_runtime(
    task_id=TASK_ID,
    motion_file=cfg.motion_file,
    checkpoint_file=cfg.expert_checkpoint_file,
    wandb_run_path=cfg.expert_wandb_run_path,
    wandb_checkpoint_name=cfg.expert_wandb_checkpoint_name,
    log_root=cfg.log_root,
    num_start_frames=cfg.num_start_frames,
    episodes_per_frame=cfg.episodes_per_frame,
    seed=cfg.seed,
    device=cfg.device,
    observation_corruption=cfg.observation_corruption,
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

  active_steps = torch.zeros(n, dtype=torch.long, device=device)
  done = torch.zeros(n, dtype=torch.bool, device=device)
  success = torch.zeros_like(done)
  unsafe = torch.zeros_like(done)
  timed_out = torch.zeros_like(done)
  stable_hold = torch.zeros(n, dtype=torch.long, device=device)
  mpkpe_sum = torch.zeros(n, device=device)
  root_relative_mpkpe_sum = torch.zeros(n, device=device)
  joint_position_rmse_sum = torch.zeros(n, device=device)
  action_rate_sq_sum = torch.zeros(n, device=device)
  max_joint_speed = torch.zeros(n, device=device)
  max_joint_acceleration = torch.zeros(n, device=device)
  max_actuator_force = torch.zeros(n, device=device)
  max_root_vertical_speed = torch.zeros(n, device=device)
  previous_action = torch.zeros(n, env.num_actions, device=device)
  sampling_seconds = 0.0
  sampled_windows = 0

  obs = env.get_observations()
  try:
    for step in range(cfg.max_steps):
      active = ~done
      if not active.any():
        break

      state_observation = actor_base_observation(obs)
      normalized_observation, current_joint, normalized_goal = (
        normalize_action_condition(state_observation, command.joint_pos, statistics)
      )
      sample_start = time.perf_counter()
      normalized_horizon = sample_action_horizon(
        model,
        scheduler,
        normalized_observation,
        current_joint,
        normalized_goal,
      )
      sampling_seconds += time.perf_counter() - sample_start
      sampled_windows += n
      actions = denormalize_actions(normalized_horizon[:, 0], statistics)
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

      mpkpe_sum += torch.where(active, compute_mpkpe(command), 0.0)
      root_relative_mpkpe_sum += torch.where(
        active, compute_root_relative_mpkpe(command), 0.0
      )
      joint_rmse = torch.sqrt(
        torch.mean(torch.square(command.robot_joint_pos - command.joint_pos), dim=-1)
      )
      joint_position_rmse_sum += torch.where(active, joint_rmse, 0.0)
      action_rate_sq_sum += torch.where(
        active, torch.mean(torch.square(actions - previous_action), dim=-1), 0.0
      )
      previous_action = torch.where(active[:, None], actions, previous_action)
      max_joint_speed = _masked_max(
        max_joint_speed, robot.data.joint_vel.abs().amax(dim=-1), active
      )
      max_joint_acceleration = _masked_max(
        max_joint_acceleration, robot.data.joint_acc.abs().amax(dim=-1), active
      )
      max_actuator_force = _masked_max(
        max_actuator_force, robot.data.actuator_force.abs().amax(dim=-1), active
      )
      max_root_vertical_speed = _masked_max(
        max_root_vertical_speed,
        robot.data.root_link_lin_vel_w[:, 2].abs(),
        active,
      )
      active_steps += active.long()

      obs, _, dones, _ = env.step(actions)
      terminated = raw_env.termination_manager.terminated.bool()
      timeouts = raw_env.termination_manager.time_outs.bool()
      newly_done = dones.bool() & active
      if newly_done.any():
        unsafe[newly_done] = terminated[newly_done]
        timed_out[newly_done] = timeouts[newly_done]
        success[newly_done] = (
          timeouts[newly_done]
          & ~terminated[newly_done]
          & (stable_hold[newly_done] >= cfg.standing_hold_steps)
        )
        done[newly_done] = True

      if step % 50 == 0 or newly_done.any():
        print(
          f"[INFO] step={step:03d} active={int((~done).sum())} "
          f"done={int(done.sum())} success={int(success.sum())} "
          f"unsafe={int(unsafe.sum())}"
        )
  finally:
    env.close()

  aggregates = _aggregate(
    runtime,
    active_steps=active_steps,
    done=done,
    success=success,
    unsafe=unsafe,
    timed_out=timed_out,
    mpkpe_sum=mpkpe_sum,
    root_relative_mpkpe_sum=root_relative_mpkpe_sum,
    joint_position_rmse_sum=joint_position_rmse_sum,
    action_rate_sq_sum=action_rate_sq_sum,
    max_joint_speed=max_joint_speed,
    max_joint_acceleration=max_joint_acceleration,
    max_actuator_force=max_actuator_force,
    max_root_vertical_speed=max_root_vertical_speed,
  )
  action_path = Path(cfg.action_checkpoint_file).expanduser().resolve()
  result = {
    "format_version": 1,
    "task_id": TASK_ID,
    "policy": "firm_action_diffusion_receding_horizon_first_action",
    "config": asdict(cfg),
    "artifacts": {
      **runtime_metadata(runtime),
      "action_checkpoint_file": str(action_path),
      "action_checkpoint_sha256": sha256_file(action_path),
      "action_checkpoint_epoch": int(action_checkpoint["epoch"]),
      "action_weights": "ema" if cfg.use_ema else "online",
    },
    "inference": {
      "sampled_windows": sampled_windows,
      "sampling_seconds": sampling_seconds,
      "windows_per_second": sampled_windows / max(sampling_seconds, 1.0e-9),
      "ddpm_steps_per_window": scheduler.num_timesteps,
      "executed_actions_per_window": 1,
    },
    **aggregates,
  }
  print(
    json.dumps(
      {"overall": result["overall"], "inference": result["inference"]}, indent=2
    )
  )
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[INFO] Evaluation written to {output_path}")
  return result


def main() -> None:
  run_evaluation(tyro.cli(EvaluateDiffusionPolicyConfig))


if __name__ == "__main__":
  main()
