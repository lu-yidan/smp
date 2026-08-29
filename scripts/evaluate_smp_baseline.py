"""Frozen evaluation for original-SMP observation-factorial policies."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
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
  output: Path | None = None
  policy_seed: int | None = None
  include_per_env: bool = False


def _quantile(values: torch.Tensor, q: float) -> float:
  return float(torch.quantile(values, q)) if values.numel() else 0.0


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
  """Return a 95% Wilson interval for rollout-level Bernoulli outcomes."""
  if total <= 0:
    return (0.0, 0.0)
  z = 1.959963984540054
  rate = successes / total
  denominator = 1.0 + z**2 / total
  center = (rate + z**2 / (2.0 * total)) / denominator
  radius = (
    z
    * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2))
    / denominator
  )
  # Preserve the observed rate exactly at the boundaries despite floating-point
  # roundoff (for example, the upper bound for 10/10 can be 0.9999999999999999).
  return (
    min(rate, max(0.0, center - radius)),
    max(rate, min(1.0, center + radius)),
  )


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

  foot_ground = ContactSensorCfg(
    name="baseline_foot_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"(left|right)_foot[1-7]_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
    history_length=2,
  )
  sensors = tuple(env_cfg.scene.sensors or ())
  if not any(sensor.name == foot_ground.name for sensor in sensors):
    env_cfg.scene.sensors = sensors + (foot_ground,)

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
  foot_ids = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]
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
  foot_slip_sum = torch.zeros_like(max_joint_speed)
  foot_contact_steps = torch.zeros_like(max_joint_speed)
  root_xy_start = robot.data.root_link_pos_w[:, :2].clone()
  max_root_planar_excursion = torch.zeros_like(max_joint_speed)
  root_xy_at_success = torch.zeros_like(root_xy_start)
  post_success_root_drift = torch.zeros_like(max_joint_speed)
  foot_separation_at_success = torch.full_like(max_joint_speed, torch.nan)
  secondary_fall_hold = torch.zeros_like(strict_first)
  secondary_fall = torch.zeros(
    raw_env.num_envs, dtype=torch.bool, device=raw_env.device
  )
  action_delta_sum = torch.zeros_like(max_joint_speed)
  action_second_difference_sum = torch.zeros_like(max_joint_speed)
  previous_actions: torch.Tensor | None = None
  previous_action_delta: torch.Tensor | None = None
  finite = torch.ones(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)

  initial_head_z = robot.data.site_pos_w[:, head_idx, 2].clone()
  initial_upright = torch.clamp(
    -robot.data.projected_gravity_b[:, 2], 0.0, 1.0
  ).clone()

  for step in range(cfg.steps):
    with torch.inference_mode():
      actions = policy(obs)
      finite &= torch.isfinite(actions).all(dim=-1)
      if previous_actions is None:
        action_delta = actions
      else:
        action_delta = actions - previous_actions
      action_delta_sum += torch.sqrt(torch.mean(action_delta**2, dim=-1))
      if previous_action_delta is not None:
        action_second_difference_sum += torch.sqrt(
          torch.mean((action_delta - previous_action_delta) ** 2, dim=-1)
        )
      previous_actions = actions.clone()
      previous_action_delta = action_delta.clone()
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
    newly_strict = (strict_first < 0) & (strict_hold >= 25)
    strict_first[newly_strict] = step + 1
    root_xy_at_success[newly_strict] = robot.data.root_link_pos_w[newly_strict, :2]
    foot_xy = robot.data.site_pos_w[:, foot_ids, :2]
    foot_separation = torch.linalg.vector_norm(foot_xy[:, 0] - foot_xy[:, 1], dim=-1)
    foot_separation_at_success[newly_strict] = foot_separation[newly_strict]
    baseline_first[(baseline_first < 0) & (baseline_hold >= 25)] = step + 1

    root_excursion = torch.linalg.vector_norm(
      robot.data.root_link_pos_w[:, :2] - root_xy_start, dim=-1
    )
    max_root_planar_excursion = torch.maximum(
      max_root_planar_excursion, root_excursion
    )
    after_success = strict_first >= 0
    root_drift = torch.linalg.vector_norm(
      robot.data.root_link_pos_w[:, :2] - root_xy_at_success, dim=-1
    )
    post_success_root_drift = torch.where(
      after_success,
      torch.maximum(post_success_root_drift, root_drift),
      post_success_root_drift,
    )
    fallen_after_success = after_success & ((head_z < 0.75) | (upright < 0.40))
    secondary_fall_hold = torch.where(
      fallen_after_success,
      secondary_fall_hold + 1,
      torch.zeros_like(secondary_fall_hold),
    )
    secondary_fall |= secondary_fall_hold >= 10

    found = raw_env.scene["baseline_foot_ground_contact"].data.found
    if found is None:
      raise RuntimeError("baseline foot contact sensor must expose found")
    in_contact = found.reshape(raw_env.num_envs, -1).any(dim=-1)
    foot_speed_xy = torch.linalg.vector_norm(
      robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1
    ).amax(dim=-1)
    foot_slip_sum += torch.where(in_contact, foot_speed_xy, 0.0)
    foot_contact_steps += in_contact.float()

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
  strict_successes = int(strict_success.sum())
  baseline_successes = int(baseline_success.sum())
  strict_ci = _wilson_interval(strict_successes, raw_env.num_envs)
  baseline_ci = _wilson_interval(baseline_successes, raw_env.num_envs)
  foot_slip = foot_slip_sum / torch.clamp(foot_contact_steps, min=1.0)
  action_delta_rms = action_delta_sum / cfg.steps
  action_second_difference_rms = action_second_difference_sum / max(
    cfg.steps - 1, 1
  )
  successful_secondary_fall = secondary_fall & strict_success
  successful_foot_separation = foot_separation_at_success[strict_success]
  result = {
    "checkpoint": cfg.checkpoint.name,
    "checkpoint_path": str(cfg.checkpoint.resolve()),
    "task": cfg.task,
    "reset_mode": cfg.reset_mode,
    "native_pushes": cfg.native_pushes if cfg.reset_mode == "native_gsi" else False,
    "policy_seed": cfg.policy_seed,
    "seed": cfg.seed,
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "strict_successes": strict_successes,
    "strict_success_rate": float(strict_success.float().mean()),
    "strict_success_rate_ci95_low": strict_ci[0],
    "strict_success_rate_ci95_high": strict_ci[1],
    "baseline_successes": baseline_successes,
    "baseline_success_rate": float(baseline_success.float().mean()),
    "baseline_success_rate_ci95_low": baseline_ci[0],
    "baseline_success_rate_ci95_high": baseline_ci[1],
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
    "contact_foot_slip_mean_m_s": float(foot_slip.mean()),
    "contact_foot_slip_p95_m_s": _quantile(foot_slip, 0.95),
    "root_planar_excursion_median_m": float(max_root_planar_excursion.median()),
    "root_planar_excursion_p95_m": _quantile(max_root_planar_excursion, 0.95),
    "post_success_root_drift_median_m": (
      float(post_success_root_drift[strict_success].median())
      if strict_success.any()
      else -1.0
    ),
    "post_success_root_drift_p95_m": (
      _quantile(post_success_root_drift[strict_success], 0.95)
      if strict_success.any()
      else -1.0
    ),
    "secondary_fall_rate_after_success": (
      float(successful_secondary_fall.sum() / strict_success.sum())
      if strict_success.any()
      else -1.0
    ),
    "foot_separation_at_success_median_m": (
      float(successful_foot_separation.median())
      if successful_foot_separation.numel()
      else -1.0
    ),
    "foot_separation_at_success_p95_m": (
      _quantile(successful_foot_separation, 0.95)
      if successful_foot_separation.numel()
      else -1.0
    ),
    "action_delta_rms_mean": float(action_delta_rms.mean()),
    "action_delta_rms_p95": _quantile(action_delta_rms, 0.95),
    "action_second_difference_rms_mean": float(
      action_second_difference_rms.mean()
    ),
    "action_second_difference_rms_p95": _quantile(
      action_second_difference_rms, 0.95
    ),
  }
  if cfg.include_per_env:
    result["per_env"] = {
      "strict_first_step": strict_first.cpu().tolist(),
      "baseline_first_step": baseline_first.cpu().tolist(),
      "finite_action": finite.cpu().tolist(),
      "initial_head_z_m": initial_head_z.cpu().tolist(),
      "initial_upright": initial_upright.cpu().tolist(),
      "max_joint_speed_rad_s": max_joint_speed.cpu().tolist(),
      "max_root_linear_speed_m_s": max_root_linear_speed.cpu().tolist(),
      "max_root_angular_speed_rad_s": max_root_angular_speed.cpu().tolist(),
      "max_torque_nm": max_torque.cpu().tolist(),
      "max_power_w": max_power.cpu().tolist(),
      "contact_foot_slip_m_s": foot_slip.cpu().tolist(),
      "root_planar_excursion_m": max_root_planar_excursion.cpu().tolist(),
      "post_success_root_drift_m": post_success_root_drift.cpu().tolist(),
      "secondary_fall_after_success": secondary_fall.cpu().tolist(),
      "foot_separation_at_success_m": foot_separation_at_success.cpu().tolist(),
      "action_delta_rms": action_delta_rms.cpu().tolist(),
      "action_second_difference_rms": action_second_difference_rms.cpu().tolist(),
    }
  if cfg.output is not None:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(cfg.output)
  logged_result = dict(result)
  logged_result.pop("per_env", None)
  print("SMP_BASELINE_EVAL_JSON=" + json.dumps(logged_result, sort_keys=True))
  raw_env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalCfg))
