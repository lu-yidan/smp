"""Quantitatively evaluate the Stage 0 FIRM sparse-keyframe expert."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.tasks.tracking.mdp.metrics import (
  compute_mpkpe,
  compute_root_relative_mpkpe,
)

import smp.rl.tasks  # noqa: F401
from smp.firm.expert_runtime import (
  ExpertRuntime,
  create_expert_runtime,
  runtime_metadata,
)
from smp.rl.tasks.firm.keyframe_env_cfg import MOTION_FILE

TASK_ID = "Firm-Keyframe-G1"


@dataclass(frozen=True)
class EvaluateExpertConfig:
  """Fixed-start evaluation configuration."""

  checkpoint_file: str | None = None
  """Local model checkpoint. Mutually exclusive with wandb_run_path."""
  wandb_run_path: str | None = None
  """W&B run path, for example tabletennis/smp/j0q8fell."""
  wandb_checkpoint_name: str | None = None
  """Checkpoint filename within the W&B run."""
  motion_file: str = MOTION_FILE
  """Validated local candidate-003 motion NPZ."""
  num_start_frames: int = 25
  """Number of evenly spaced dense motion frames used as initial states."""
  episodes_per_frame: int = 32
  """Parallel replicas for each initial frame."""
  max_steps: int = 500
  """Maximum control steps per episode."""
  standing_hold_steps: int = 25
  """Consecutive stable steps required at episode completion."""
  root_height_threshold: float = 0.65
  upright_threshold: float = 0.85
  root_linear_speed_threshold: float = 0.50
  root_angular_speed_threshold: float = 0.50
  observation_corruption: bool = True
  """Keep actor observation noise enabled during evaluation."""
  seed: int = 42
  device: str | None = None
  output_file: str | None = None
  """Optional JSON result path."""
  log_root: str = "logs/rsl_rl"


def _masked_max(
  current: torch.Tensor, value: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
  return torch.maximum(current, torch.where(active, value, torch.zeros_like(value)))


def _quantile(value: torch.Tensor, q: float) -> float:
  return float(torch.quantile(value.float(), q).item())


def _aggregate(
  runtime: ExpertRuntime,
  *,
  active_steps: torch.Tensor,
  done: torch.Tensor,
  success: torch.Tensor,
  unsafe: torch.Tensor,
  timed_out: torch.Tensor,
  mpkpe_sum: torch.Tensor,
  root_relative_mpkpe_sum: torch.Tensor,
  joint_position_rmse_sum: torch.Tensor,
  action_rate_sq_sum: torch.Tensor,
  max_joint_speed: torch.Tensor,
  max_joint_acceleration: torch.Tensor,
  max_actuator_force: torch.Tensor,
  max_root_vertical_speed: torch.Tensor,
) -> dict:
  divisor = active_steps.float().clamp_min(1.0)
  mpkpe = mpkpe_sum / divisor
  root_relative_mpkpe = root_relative_mpkpe_sum / divisor
  joint_position_rmse = joint_position_rmse_sum / divisor
  action_rate_rms = torch.sqrt(action_rate_sq_sum / divisor)

  def summary(ids: torch.Tensor) -> dict[str, float | int]:
    if ids.numel() == 0:
      return {"episodes": 0}
    return {
      "episodes": int(ids.numel()),
      "completion_rate": float(done[ids].float().mean().item()),
      "success_rate": float(success[ids].float().mean().item()),
      "unsafe_termination_rate": float(unsafe[ids].float().mean().item()),
      "timeout_rate": float(timed_out[ids].float().mean().item()),
      "episode_steps_mean": float(active_steps[ids].float().mean().item()),
      "mpkpe_mean_m": float(mpkpe[ids].mean().item()),
      "mpkpe_p95_m": _quantile(mpkpe[ids], 0.95),
      "root_relative_mpkpe_mean_m": float(root_relative_mpkpe[ids].mean().item()),
      "joint_position_rmse_mean_rad": float(joint_position_rmse[ids].mean().item()),
      "action_rate_rms_mean": float(action_rate_rms[ids].mean().item()),
      "max_joint_speed_p95_rad_s": _quantile(max_joint_speed[ids], 0.95),
      "max_joint_speed_max_rad_s": float(max_joint_speed[ids].max().item()),
      "max_joint_acceleration_p95_rad_s2": _quantile(max_joint_acceleration[ids], 0.95),
      "max_actuator_force_p95": _quantile(max_actuator_force[ids], 0.95),
      "max_root_vertical_speed_p95_m_s": _quantile(max_root_vertical_speed[ids], 0.95),
    }

  all_ids = torch.arange(runtime.env.num_envs, device=runtime.env.device)
  by_start_frame = {}
  for frame in runtime.start_frames.tolist():
    frame_ids = torch.where(runtime.env_start_frames == int(frame))[0]
    by_start_frame[str(frame)] = summary(frame_ids)
  return {"overall": summary(all_ids), "by_start_frame": by_start_frame}


def run_evaluation(cfg: EvaluateExpertConfig) -> dict:
  """Evaluate all replicas until timeout or numerical-safety termination."""
  if cfg.max_steps <= 0 or cfg.standing_hold_steps <= 0:
    raise ValueError("max_steps and standing_hold_steps must be positive")

  runtime = create_expert_runtime(
    task_id=TASK_ID,
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
  )
  env = runtime.env
  raw_env = env.unwrapped
  robot = raw_env.scene["robot"]
  command = runtime.command
  n = env.num_envs
  device = env.device

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

  obs = env.get_observations()
  try:
    for step in range(cfg.max_steps):
      active = ~done
      if not active.any():
        break

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

      mpkpe_sum += torch.where(active, compute_mpkpe(command), 0.0)
      root_relative_mpkpe_sum += torch.where(
        active, compute_root_relative_mpkpe(command), 0.0
      )
      joint_rmse = torch.sqrt(
        torch.mean(torch.square(command.robot_joint_pos - command.joint_pos), dim=-1)
      )
      joint_position_rmse_sum += torch.where(active, joint_rmse, 0.0)
      action_rate_sq = torch.mean(torch.square(actions - previous_action), dim=-1)
      action_rate_sq_sum += torch.where(active, action_rate_sq, 0.0)
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
  result = {
    "format_version": 1,
    "task_id": TASK_ID,
    "config": asdict(cfg),
    "artifacts": runtime_metadata(runtime),
    **aggregates,
  }
  print(json.dumps(result["overall"], indent=2))

  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[INFO] Evaluation written to {output_path}")
  return result


def main() -> None:
  run_evaluation(tyro.cli(EvaluateExpertConfig))


if __name__ == "__main__":
  main()
