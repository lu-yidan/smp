"""Reset-stratified zero-shot evaluation for the V3.5 terrain benchmark."""

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


@dataclass(frozen=True)
class EvalCfg:
  checkpoint: Path
  task: str = "Smp-Getup-Terrain-V35-G1"
  terrain_types: tuple[str, ...] = ("flat", "slope", "stairs", "rough")
  levels: tuple[int, ...] = (1,)
  reset_modes: tuple[str, ...] = ("prone", "supine", "left_side", "right_side")
  edge_cohorts: tuple[str, ...] = ()
  num_envs: int = 64
  steps: int = 750
  seed: int = 20260818
  device: str = "cuda:0"
  output: Path = Path("logs/evaluation/terrain_v35.jsonl")


def _quantile(values: torch.Tensor, q: float) -> float:
  return float(torch.quantile(values, q)) if values.numel() else 0.0


def _run_case(
  cfg: EvalCfg,
  terrain_type: str,
  level: int,
  reset_mode: str,
  edge_cohort: str | None = None,
) -> dict[str, object]:
  # Task registration happens only after CLI parsing.  The selected benchmark
  # case then replaces the prebuilt play terrain without changing observations.
  import smp.rl.tasks  # noqa: F401
  from smp.rl.tasks.getup import mdp
  from smp.rl.tasks.getup.terrain_v35_env_cfg import (
    RESET_POSE_WEIGHTS,
    TERRAIN_KINDS,
    terrain_generator_v35,
    terrain_surface_normals_v35,
  )
  from smp.rl.tasks.getup.terrain_v37_env_cfg import EDGE_RESET_COHORTS

  if terrain_type not in TERRAIN_KINDS or terrain_type == "mixed":
    raise ValueError("terrain_types must contain flat, slope, stairs, or rough")
  if level not in range(4):
    raise ValueError("levels must contain only 0, 1, 2, or 3")
  if reset_mode not in RESET_POSE_WEIGHTS or reset_mode == "mixed":
    raise ValueError("reset_modes must contain prone, supine, left_side, or right_side")
  if edge_cohort is not None:
    if terrain_type != "stairs" or edge_cohort not in EDGE_RESET_COHORTS:
      raise ValueError("edge_cohorts require stairs and a supported V3.7 cohort")

  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.scene.terrain.terrain_generator = terrain_generator_v35(
    terrain_type, level, cfg.seed
  )
  env_cfg.events["ground_procedural_fall_on_terrain"].params["surface_normals"] = (
    terrain_surface_normals_v35(terrain_type, level)
  )
  env_cfg.terminations = {}
  for event_name in (
    "stratified_post_stand_wrench",
    "record_failure_states",
    "failure_state_replay_reset",
  ):
    env_cfg.events.pop(event_name, None)
  env_cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 1.0,
      "mode_weights": RESET_POSE_WEIGHTS[reset_mode],
    }
  )
  if edge_cohort is not None:
    weights = tuple(float(name == edge_cohort) for name in EDGE_RESET_COHORTS)
    edge_event = env_cfg.events["sample_terrain_edge_reset"]
    edge_event.params["cohort_weights"] = weights

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
  origins = raw_env.scene.env_origins
  support_height = getattr(raw_env, "_terrain_reset_support_height", origins[:, 2])
  reset_anchor_xy = getattr(raw_env, "_terrain_reset_anchor_xy", origins[:, :2])
  initial_reset_offset = torch.linalg.vector_norm(
    reset_anchor_xy - origins[:, :2], dim=-1
  )
  initial_support_delta = support_height - origins[:, 2]
  root_xy_start = robot.data.root_link_pos_w[:, :2].clone()
  foot_ids = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  max_planar_displacement = torch.zeros(cfg.num_envs, device=raw_env.device)
  max_terrain_descent = torch.zeros_like(max_planar_displacement)
  max_joint_speed = torch.zeros_like(max_planar_displacement)
  max_torque = torch.zeros_like(max_planar_displacement)
  max_power = torch.zeros_like(max_planar_displacement)
  foot_slip_sum = torch.zeros_like(max_planar_displacement)
  foot_contact_steps = torch.zeros_like(max_planar_displacement)
  first_success = torch.full(
    (cfg.num_envs,), -1, dtype=torch.long, device=raw_env.device
  )
  stand_hold = torch.zeros_like(first_success)
  secondary_fall_hold = torch.zeros_like(first_success)
  secondary_fall = torch.zeros(cfg.num_envs, dtype=torch.bool, device=raw_env.device)
  terrain_exit = torch.zeros_like(secondary_fall)
  invalid_dynamics = torch.zeros_like(secondary_fall)
  terrain_generator = raw_env.scene.terrain.cfg.terrain_generator
  terrain_exit_radius = 0.5 * min(terrain_generator.size) - 0.5

  for step in range(cfg.steps):
    with torch.inference_mode():
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)

    raw_root_pos = robot.data.root_link_pos_w
    raw_displacement = torch.linalg.vector_norm(
      raw_root_pos[:, :2] - root_xy_start, dim=-1
    )
    finite = torch.isfinite(raw_root_pos).all(dim=-1) & torch.isfinite(
      robot.data.joint_vel
    ).all(dim=-1)
    invalid_dynamics |= ~finite
    terrain_exit |= (raw_displacement > terrain_exit_radius) | ~finite
    active = ~terrain_exit

    head_z = robot.data.site_pos_w[:, head_idx, 2]
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
    linear_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=-1)
    standing = (
      (head_z - support_height >= 1.10)
      & (upright >= 0.85)
      & (linear_speed < 0.50)
      & (angular_speed < 1.0)
    )
    stand_hold = torch.where(standing, stand_hold + 1, torch.zeros_like(stand_hold))
    newly_successful = active & (first_success < 0) & (stand_hold >= 25)
    first_success[newly_successful] = step + 1

    fallen_after_success = (first_success >= 0) & (
      ((head_z - support_height) < 0.75) | (upright < 0.40)
    )
    secondary_fall_hold = torch.where(
      fallen_after_success,
      secondary_fall_hold + 1,
      torch.zeros_like(secondary_fall_hold),
    )
    secondary_fall |= secondary_fall_hold >= 10
    secondary_fall |= terrain_exit & (first_success >= 0)

    displacement = torch.nan_to_num(
      raw_displacement,
      nan=terrain_exit_radius,
      posinf=terrain_exit_radius,
      neginf=terrain_exit_radius,
    )
    max_planar_displacement = torch.maximum(
      max_planar_displacement, torch.clamp(displacement, max=terrain_exit_radius)
    )
    descent = torch.nan_to_num(
      torch.clamp(support_height - raw_root_pos[:, 2], min=0.0),
      nan=2.0,
      posinf=2.0,
      neginf=0.0,
    )
    max_terrain_descent = torch.maximum(
      max_terrain_descent, torch.where(active, torch.clamp(descent, max=2.0), 0.0)
    )
    joint_speed = torch.nan_to_num(
      torch.abs(robot.data.joint_vel).amax(dim=-1),
      nan=0.0,
      posinf=0.0,
      neginf=0.0,
    )
    max_joint_speed = torch.maximum(
      max_joint_speed, torch.where(active, joint_speed, 0.0)
    )
    torque = torch.nan_to_num(mdp.max_joint_torque_metric(raw_env), nan=0.0)
    power = torch.nan_to_num(mdp.max_joint_power_metric(raw_env), nan=0.0)
    max_torque = torch.maximum(max_torque, torch.where(active, torque, 0.0))
    max_power = torch.maximum(max_power, torch.where(active, power, 0.0))

    found = raw_env.scene["terrain_foot_ground_contact"].data.found
    if found is None:
      raise RuntimeError("terrain foot contact sensor must expose found")
    in_contact = found.reshape(cfg.num_envs, -1).any(dim=-1)
    foot_speed_xy = torch.linalg.vector_norm(
      robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1
    ).amax(dim=-1)
    valid_contact = in_contact & active
    foot_slip_sum += torch.where(valid_contact, foot_speed_xy, 0.0)
    foot_contact_steps += valid_contact.float()

    # Once a rollout leaves its terrain patch, re-anchor it in a benign
    # state.  Its failure remains recorded, while one escaped body cannot
    # free-fall to numerical overflow and corrupt the rest of the batch.
    failed_ids = torch.nonzero(terrain_exit, as_tuple=False).flatten()
    if failed_ids.numel() > 0:
      safe_root = robot.data.default_root_state[failed_ids].clone()
      safe_root[:, :3] += origins[failed_ids]
      safe_root[:, 7:] = 0.0
      robot.write_root_state_to_sim(safe_root, env_ids=failed_ids)
      robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[failed_ids],
        torch.zeros_like(robot.data.default_joint_pos[failed_ids]),
        env_ids=failed_ids,
      )
      raw_env.sim.forward()

  success = first_success >= 0
  recovery_steps = first_success[success].float()
  successful_secondary_fall = secondary_fall & success
  foot_slip = foot_slip_sum / torch.clamp(foot_contact_steps, min=1.0)
  result: dict[str, object] = {
    "checkpoint": cfg.checkpoint.name,
    "task": cfg.task,
    "terrain_type": terrain_type,
    "terrain_level": level,
    "reset_mode": reset_mode,
    "edge_cohort": edge_cohort,
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
      _quantile(recovery_steps * raw_env.step_dt, 0.90)
      if recovery_steps.numel()
      else -1.0
    ),
    "secondary_fall_rate_after_success": (
      float(successful_secondary_fall.sum() / success.sum()) if success.any() else -1.0
    ),
    "terrain_exit_rate": float(terrain_exit.float().mean()),
    "initial_reset_offset_median_m": float(initial_reset_offset.median()),
    "initial_reset_offset_min_m": float(initial_reset_offset.min()),
    "initial_reset_offset_max_m": float(initial_reset_offset.max()),
    "initial_support_delta_median_m": float(initial_support_delta.median()),
    "terrain_exit_radius_m": terrain_exit_radius,
    "invalid_dynamics_rate": float(invalid_dynamics.float().mean()),
    "planar_displacement_median_m": float(max_planar_displacement.median()),
    "planar_displacement_p95_m": _quantile(max_planar_displacement, 0.95),
    "terrain_descent_median_m": float(max_terrain_descent.median()),
    "terrain_descent_p95_m": _quantile(max_terrain_descent, 0.95),
    "contact_foot_slip_mean_m_s": float(foot_slip.mean()),
    "contact_foot_slip_p95_m_s": _quantile(foot_slip, 0.95),
    "max_joint_speed_mean_rad_s": float(max_joint_speed.mean()),
    "max_joint_speed_p95_rad_s": _quantile(max_joint_speed, 0.95),
    "max_torque_mean_nm": float(max_torque.mean()),
    "max_power_mean_w": float(max_power.mean()),
  }
  raw_env.close()
  del policy, runner, env, raw_env
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  return result


def main(cfg: EvalCfg) -> None:
  configure_torch_backends()
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  results = []
  edge_cohorts: tuple[str | None, ...] = cfg.edge_cohorts or (None,)
  for terrain_type in cfg.terrain_types:
    for level in cfg.levels:
      for reset_mode in cfg.reset_modes:
        for edge_cohort in edge_cohorts:
          result = _run_case(
            cfg, terrain_type, level, reset_mode, edge_cohort=edge_cohort
          )
          results.append(result)
          print("TERRAIN_RECOVERY_EVAL_JSON=" + json.dumps(result, sort_keys=True))

  with cfg.output.open("w", encoding="utf-8") as stream:
    for result in results:
      stream.write(json.dumps(result, sort_keys=True) + "\n")
  aggregate = {
    "cases": len(results),
    "mean_success_rate": sum(float(r["success_rate"]) for r in results)
    / max(len(results), 1),
    "output": str(cfg.output),
  }
  print("TERRAIN_RECOVERY_SUMMARY_JSON=" + json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(EvalCfg))
