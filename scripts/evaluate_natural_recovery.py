"""Deterministic reset-stratified evaluation for natural get-up policies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401  # task registration
from smp.rl.tasks.getup import mdp

_RESET_WEIGHTS = {
  "supine": (1.0, 0.0, 0.0, 0.0),
  "prone": (0.0, 1.0, 0.0, 0.0),
  "left_side": (0.0, 0.0, 1.0, 0.0),
  "right_side": (0.0, 0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class EvalCfg:
  checkpoint: Path
  task: str = "Smp-Getup-Robust-Smooth-V8-Natural-G1"
  reset_mode: str = "prone"
  num_envs: int = 512
  steps: int = 1000
  seed: int = 20260817
  device: str = "cuda:0"


def main(cfg: EvalCfg) -> None:
  if cfg.reset_mode not in _RESET_WEIGHTS:
    choices = ", ".join(_RESET_WEIGHTS)
    raise ValueError(f"reset_mode must be one of: {choices}")
  configure_torch_backends()
  env_cfg = load_env_cfg(cfg.task)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.terminations = {}
  env_cfg.events.pop("stratified_post_stand_wrench", None)
  env_cfg.events.pop("record_failure_states", None)
  env_cfg.events.pop("failure_state_replay_reset", None)
  env_cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 1.0,
      "mode_weights": _RESET_WEIGHTS[cfg.reset_mode],
    }
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
  foot_ids = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]
  initial_feet = robot.data.site_pos_w[:, foot_ids, :2].clone()
  max_foot_excursion = torch.zeros(raw_env.num_envs, device=raw_env.device)
  max_joint_speed = torch.zeros_like(max_foot_excursion)
  max_torque = torch.zeros_like(max_foot_excursion)
  max_power = torch.zeros_like(max_foot_excursion)
  max_splay = torch.zeros_like(max_foot_excursion)
  hand_support_sum = torch.zeros_like(max_foot_excursion)
  knee_support_sum = torch.zeros_like(max_foot_excursion)
  first_success = torch.full(
    (raw_env.num_envs,), -1, dtype=torch.long, device=raw_env.device
  )

  for step in range(cfg.steps):
    with torch.inference_mode():
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
    stage = raw_env._v4_recovery_stage  # type: ignore[attr-defined]
    first_success[(first_success < 0) & (stage == 3)] = step + 1
    foot_excursion = torch.linalg.vector_norm(
      robot.data.site_pos_w[:, foot_ids, :2] - initial_feet,
      dim=-1,
    ).amax(dim=-1)
    max_foot_excursion = torch.maximum(max_foot_excursion, foot_excursion)
    max_joint_speed = torch.maximum(
      max_joint_speed, torch.abs(robot.data.joint_vel).amax(dim=-1)
    )
    max_torque = torch.maximum(max_torque, mdp.max_joint_torque_metric(raw_env))
    max_power = torch.maximum(max_power, mdp.max_joint_power_metric(raw_env))
    max_splay = torch.maximum(max_splay, mdp.prone_leg_splay_excess_l2(raw_env))
    hand_support_sum += mdp.ground_support_contact_metric(
      raw_env, "natural_hand_ground_contact"
    )
    knee_support_sum += mdp.ground_support_contact_metric(
      raw_env, "natural_knee_ground_contact"
    )

  success = first_success >= 0
  recovery_steps = first_success[success].float()

  def quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q)) if values.numel() else 0.0

  result = {
    "checkpoint": cfg.checkpoint.name,
    "task": cfg.task,
    "reset_mode": cfg.reset_mode,
    "seed": cfg.seed,
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "success": int(success.sum()),
    "success_rate": float(success.float().mean()),
    "recovery_time_median_s": (
      float(recovery_steps.median() * raw_env.step_dt)
      if recovery_steps.numel()
      else -1.0
    ),
    "recovery_time_p90_s": (
      quantile(recovery_steps * raw_env.step_dt, 0.90)
      if recovery_steps.numel()
      else -1.0
    ),
    "hand_support_fraction_mean": float((hand_support_sum / cfg.steps).mean()),
    "knee_support_fraction_mean": float((knee_support_sum / cfg.steps).mean()),
    "leg_splay_excess_mean": float(max_splay.mean()),
    "leg_splay_excess_p95": quantile(max_splay, 0.95),
    "foot_excursion_median_m": float(max_foot_excursion.median()),
    "foot_excursion_p95_m": quantile(max_foot_excursion, 0.95),
    "max_joint_speed_mean_rad_s": float(max_joint_speed.mean()),
    "max_joint_speed_p95_rad_s": quantile(max_joint_speed, 0.95),
    "max_torque_mean_nm": float(max_torque.mean()),
    "max_power_mean_w": float(max_power.mean()),
  }
  print("NATURAL_RECOVERY_EVAL_JSON=" + json.dumps(result, sort_keys=True))
  raw_env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalCfg))
