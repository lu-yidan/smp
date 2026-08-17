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
  setup_invalid = raw_env._escape_invalid_setup & active  # type: ignore[attr-defined]
  any_invalid = invalid | setup_invalid
  escaped = (phase == 3) & active
  valid = active & (~any_invalid)
  first = first_contact[contacted].float()
  penetration = raw_env._escape_peak_penetration[active]  # type: ignore[attr-defined]
  force = raw_env._escape_peak_contact_force[active]  # type: ignore[attr-defined]
  separation = raw_env._escape_best_separation[active]  # type: ignore[attr-defined]
  clear_hold = raw_env._escape_clear_hold[active]  # type: ignore[attr-defined]
  initial_covered = getattr(raw_env, "_escape_initial_covered_geom_count", None)
  covered = getattr(raw_env, "_escape_covered_geom_count", None)
  best_covered = getattr(raw_env, "_escape_best_covered_geom_count", None)
  planar_clearance = getattr(raw_env, "_escape_planar_clearance", None)
  if initial_covered is not None:
    initial_covered = initial_covered[active]
    covered = covered[active]
    best_covered = best_covered[active]
    planar_clearance = planar_clearance[active]
  obstacle = raw_env.scene["escape_obstacle"]
  plate_body_ids, _ = obstacle.find_bodies(["escape_plate"], preserve_order=True)
  plate_mass = None
  if len(plate_body_ids) == 1:
    plate_local = torch.tensor(plate_body_ids, dtype=torch.long, device=raw_env.device)
    plate_body_id = obstacle.indexing.body_ids[plate_local][0].long()
    plate_mass = raw_env.sim.model.body_mass[:, plate_body_id][active]

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
    "setup_invalid": int(setup_invalid.sum()),
    "setup_invalid_rate": float(setup_invalid.sum() / max(active_count, 1)),
    "pending": int(((phase == 1) & active).sum()),
    "pinned": int(((phase == 2) & active).sum()),
    "penetration_median_m": float(penetration.median()),
    "penetration_p99_m": quantile(penetration, 0.99),
    "penetration_max_m": float(penetration.max()),
    "force_median_n": float(force.median()),
    "force_p99_n": quantile(force, 0.99),
    "force_max_n": float(force.max()),
    "separation_median_m": float(separation.median()),
    "separation_p90_m": quantile(separation, 0.90),
    "separation_p99_m": quantile(separation, 0.99),
    "separation_max_m": float(separation.max()),
    "separation_ready": int((separation >= 0.50).sum()),
    "clear_hold_median_steps": float(clear_hold.float().median()),
    "clear_hold_max_steps": int(clear_hold.max()),
    "hand_support_mean": float((hand_support_sum[active] / cfg.steps).mean()),
    "first_contact_head_height_median_m": float(
      raw_env._escape_first_contact_head_height[contacted].median()  # type: ignore[attr-defined]
    ),
    "first_contact_head_height_max_m": float(
      raw_env._escape_first_contact_head_height[contacted].max()  # type: ignore[attr-defined]
    ),
    "hand_support_steps_median": float(
      raw_env._escape_hand_support_steps[active].float().median()  # type: ignore[attr-defined]
    ),
    "hand_supported_progress_median_m": float(
      raw_env._escape_hand_supported_progress[active].median()  # type: ignore[attr-defined]
    ),
    "max_torque_mean_nm": float(max_torque[active].mean()),
    "max_power_mean_w": float(max_power[active].mean()),
  }
  if initial_covered is not None:
    result.update(
      {
        "initial_covered_geom_count_median": float(initial_covered.float().median()),
        "final_covered_geom_count_median": float(covered.float().median()),
        "best_covered_geom_count_median": float(best_covered.float().median()),
        "final_fully_clear_geometry": int(
          ((covered == 0) & (planar_clearance >= 0.025)).sum()
        ),
        "planar_clearance_median_m": float(planar_clearance.median()),
        "planar_clearance_p90_m": quantile(planar_clearance, 0.90),
      }
    )
  if plate_mass is not None:
    result.update(
      {
        "plate_mass_median_kg": float(plate_mass.median()),
        "plate_mass_max_kg": float(plate_mass.max()),
      }
    )
  print("ESCAPE_EVAL_JSON=" + json.dumps(result, sort_keys=True))
  raw_env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalCfg))
