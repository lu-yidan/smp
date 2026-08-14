"""Deterministic physical-validity evaluation for an escape checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401  # task registration
from smp.rl.tasks.getup import mdp


@dataclass(frozen=True)
class EvalCfg:
  checkpoint: Path
  task: str = "Smp-Getup-Escape-Plate-V3-G1"
  num_envs: int = 512
  steps: int = 1000
  seed: int = 20260814
  device: str = "cuda:0"


def main(cfg: EvalCfg) -> None:
  configure_torch_backends()
  env_cfg = load_env_cfg(cfg.task)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  # Keep terminal/invalid states intact for the complete audit horizon.
  env_cfg.terminations = {}

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

  active = (raw_env._escape_phase > 0).clone()  # type: ignore[attr-defined]
  first_contact = torch.full(
    (raw_env.num_envs,), -1, dtype=torch.long, device=raw_env.device
  )
  hand_support_sum = torch.zeros(raw_env.num_envs, device=raw_env.device)
  max_torque = torch.zeros_like(hand_support_sum)
  max_power = torch.zeros_like(hand_support_sum)

  for step in range(cfg.steps):
    with torch.inference_mode():
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
    obstacle_found = raw_env.scene["robot_obstacle_contact"].data.found
    assert obstacle_found is not None
    contact = torch.any(obstacle_found > 0, dim=-1)
    first_contact[(first_contact < 0) & contact & active] = step + 1
    hand_found = raw_env.scene["hand_ground_contact"].data.found
    assert hand_found is not None
    hand_support_sum += (hand_found > 0).float().mean(dim=-1)
    max_torque = torch.maximum(max_torque, mdp.max_joint_torque_metric(raw_env))
    max_power = torch.maximum(max_power, mdp.max_joint_power_metric(raw_env))

  phase = raw_env._escape_phase  # type: ignore[attr-defined]
  active_count = int(active.sum())
  contacted = (first_contact >= 0) & active
  invalid = raw_env._escape_invalid_contact & active  # type: ignore[attr-defined]
  escaped = (phase == 3) & active
  valid = active & (~invalid)
  first = first_contact[contacted].float()
  penetration = raw_env._escape_peak_penetration[active]  # type: ignore[attr-defined]
  force = raw_env._escape_peak_contact_force[active]  # type: ignore[attr-defined]

  def quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q)) if values.numel() else 0.0

  result = {
    "checkpoint": cfg.checkpoint.name,
    "seed": cfg.seed,
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "active": active_count,
    "contacted": int(contacted.sum()),
    "first_contact_step_median": float(first.median()) if first.numel() else -1.0,
    "escaped": int(escaped.sum()),
    "conditional_escape_rate": float(escaped.sum() / max(active_count, 1)),
    "valid_conditional_escape_rate": float(
      (escaped & valid).sum() / max(int(valid.sum()), 1)
    ),
    "invalid": int(invalid.sum()),
    "invalid_rate": float(invalid.sum() / max(active_count, 1)),
    "pending": int(((phase == 1) & active).sum()),
    "pinned": int(((phase == 2) & active).sum()),
    "penetration_median_m": float(penetration.median()),
    "penetration_p99_m": quantile(penetration, 0.99),
    "penetration_max_m": float(penetration.max()),
    "force_median_n": float(force.median()),
    "force_p99_n": quantile(force, 0.99),
    "force_max_n": float(force.max()),
    "hand_support_mean": float((hand_support_sum[active] / cfg.steps).mean()),
    "max_torque_mean_nm": float(max_torque[active].mean()),
    "max_power_mean_w": float(max_power[active].mean()),
  }
  print("ESCAPE_EVAL_JSON=" + json.dumps(result, sort_keys=True))
  raw_env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalCfg))
