"""Frozen evaluation for original-SMP observation-factorial policies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401
from smp.rl.tasks.getup import mdp

_RESET_WEIGHTS = {
  "prone": (1.0, 0.0, 0.0, 0.0),
  "supine": (0.0, 1.0, 0.0, 0.0),
  "left_side": (0.0, 0.0, 1.0, 0.0),
  "right_side": (0.0, 0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class EvalCfg:
  checkpoint: Path
  task: str = "Smp-Getup-G1"
  reset_mode: str = "native_gsi"
  num_envs: int = 512
  steps: int = 500
  seed: int = 20260829
  device: str = "cuda:0"
  native_pushes: bool = True


def _quantile(values: torch.Tensor, q: float) -> float:
  return float(torch.quantile(values, q)) if values.numel() else 0.0


def main(cfg: EvalCfg) -> None:
  valid_modes = ("native_gsi", *_RESET_WEIGHTS)
  if cfg.reset_mode not in valid_modes:
    choices = ", ".join(valid_modes)
    raise ValueError(f"reset_mode must be one of: {choices}")

  configure_torch_backends()
  env_cfg = load_env_cfg(cfg.task)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.terminations = {}
  env_cfg.episode_length_s = 1.0e9
  env_cfg.events.pop("gsi_refresh", None)
  env_cfg.events["init_smp_state"].params.update(
    {
      "compile_model": False,
      "gsi_buffer_size": max(1024, cfg.num_envs),
    }
  )

  if cfg.reset_mode == "native_gsi":
    if not cfg.native_pushes:
      env_cfg.events.pop("push_robot", None)
  else:
    env_cfg.events.pop("push_robot", None)
    env_cfg.events["forced_fall_reset"] = EventTermCfg(
      func=mdp.mixed_fall_reset,
      mode="reset",
      params={
        "procedural_probability": 1.0,
        "mode_weights": _RESET_WEIGHTS[cfg.reset_mode],
        "root_height_range": (0.48, 0.62),
        "joint_noise": 0.12,
        "orientation_noise": 0.0,
        "root_xy_range": 0.1,
        "root_linear_velocity": 0.1,
        "root_angular_velocity": 0.2,
      },
    )

  raw_env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(cfg.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=cfg.device,
  )
  policy = runner.get_inference_policy(device=cfg.device)
  obs = env.get_observations()

  robot = raw_env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  strict_first = torch.full(
    (raw_env.num_envs,), -1, dtype=torch.long, device=raw_env.device
  )
  baseline_first = torch.full_like(strict_first, -1)
  strict_hold = torch.zeros_like(strict_first)
  baseline_hold = torch.zeros_like(strict_first)
  max_joint_speed = torch.zeros(raw_env.num_envs, device=raw_env.device)
  max_root_linear_speed = torch.zeros_like(max_joint_speed)
  max_root_angular_speed = torch.zeros_like(max_joint_speed)
  max_torque = torch.zeros_like(max_joint_speed)
  max_power = torch.zeros_like(max_joint_speed)
  finite = torch.ones(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)

  initial_head_z = robot.data.site_pos_w[:, head_idx, 2].clone()
  initial_upright = torch.clamp(
    -robot.data.projected_gravity_b[:, 2], 0.0, 1.0
  ).clone()

  for step in range(cfg.steps):
    with torch.inference_mode():
      actions = policy(obs)
      finite &= torch.isfinite(actions).all(dim=-1)
      obs, _, _, _ = env.step(actions)

    head_z = robot.data.site_pos_w[:, head_idx, 2]
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
    linear_speed = torch.linalg.vector_norm(
      robot.data.root_link_lin_vel_w, dim=-1
    )
    angular_speed = torch.linalg.vector_norm(
      robot.data.root_link_ang_vel_w, dim=-1
    )
    strict_standing = (
      (head_z >= 1.10)
      & (upright >= 0.85)
      & (linear_speed < 0.50)
      & (angular_speed < 1.0)
    )
    baseline_standing = (head_z >= 1.20) & (linear_speed < 0.50)
    strict_hold = torch.where(
      strict_standing, strict_hold + 1, torch.zeros_like(strict_hold)
    )
    baseline_hold = torch.where(
      baseline_standing, baseline_hold + 1, torch.zeros_like(baseline_hold)
    )
    strict_first[(strict_first < 0) & (strict_hold >= 25)] = step + 1
    baseline_first[(baseline_first < 0) & (baseline_hold >= 25)] = step + 1

    max_joint_speed = torch.maximum(
      max_joint_speed, torch.abs(robot.data.joint_vel).amax(dim=-1)
    )
    max_root_linear_speed = torch.maximum(max_root_linear_speed, linear_speed)
    max_root_angular_speed = torch.maximum(max_root_angular_speed, angular_speed)
    max_torque = torch.maximum(max_torque, mdp.max_joint_torque_metric(raw_env))
    max_power = torch.maximum(max_power, mdp.max_joint_power_metric(raw_env))

  strict_success = strict_first >= 0
  baseline_success = baseline_first >= 0
  recovery_steps = strict_first[strict_success].float()
  result = {
    "checkpoint": cfg.checkpoint.name,
    "task": cfg.task,
    "reset_mode": cfg.reset_mode,
    "native_pushes": cfg.native_pushes if cfg.reset_mode == "native_gsi" else False,
    "seed": cfg.seed,
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "strict_success_rate": float(strict_success.float().mean()),
    "baseline_success_rate": float(baseline_success.float().mean()),
    "strict_recovery_time_median_s": (
      float(recovery_steps.median() * raw_env.step_dt)
      if recovery_steps.numel()
      else -1.0
    ),
    "strict_recovery_time_p90_s": (
      _quantile(recovery_steps * raw_env.step_dt, 0.90)
      if recovery_steps.numel()
      else -1.0
    ),
    "finite_action_rate": float(finite.float().mean()),
    "initial_head_z_mean_m": float(initial_head_z.mean()),
    "initial_upright_mean": float(initial_upright.mean()),
    "max_joint_speed_mean_rad_s": float(max_joint_speed.mean()),
    "max_joint_speed_p95_rad_s": _quantile(max_joint_speed, 0.95),
    "max_root_linear_speed_mean_m_s": float(max_root_linear_speed.mean()),
    "max_root_angular_speed_mean_rad_s": float(max_root_angular_speed.mean()),
    "max_torque_mean_nm": float(max_torque.mean()),
    "max_power_mean_w": float(max_power.mean()),
  }
  print("SMP_BASELINE_EVAL_JSON=" + json.dumps(result, sort_keys=True))
  raw_env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalCfg))
