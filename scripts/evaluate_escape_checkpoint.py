"""Deterministic physical-validity evaluation for an escape checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import mujoco
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
  task: str = "Smp-Getup-Escape-Plate-V33-G1"
  num_envs: int = 512
  steps: int = 1000
  seed: int = 20260814
  device: str = "cuda:0"
  plate_mass_kg: float = 8.0
  plate_length_m: float = 0.90
  plate_width_m: float = 0.64
  plate_thickness_m: float = 0.07
  plate_friction: float = 1.20
  reset_pose: str = "prone"
  longitudinal_offset_m: float = -0.10
  lateral_offset_m: float = 0.0
  longitudinal_jitter_m: float = 0.0
  lateral_jitter_m: float = 0.0
  xy_jitter_m: float = 0.005
  stable_hold_steps: int = 25
  stand_head_height_m: float = 1.10
  stand_min_upright: float = 0.85
  stand_max_linear_speed_m_s: float = 0.50
  stand_max_angular_speed_rad_s: float = 1.0
  wide_stance_threshold_m: float = 0.45


def _get_evaluation_plate_spec(
  half_extents: tuple[float, float, float], friction: float
) -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Create the V3.3 plate with evaluation-specific geometry and friction."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="escape_plate")
  body.add_joint(
    name="escape_plate_slide",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=(0.0, 0.0, 1.0),
    limited=True,
    range=(-1.20, 0.0),
    damping=60.0,
  )
  geom = body.add_geom(
    name="escape_plate_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half_extents,
    mass=8.0,
    friction=(friction, 0.01, 0.001),
    rgba=(0.12, 0.72, 0.24, 0.82),
  )
  geom.priority = 1
  geom.solref = (0.01, 1.0)
  geom.solimp = (0.98, 0.995, 0.001, 0.5, 2.0)
  return spec


def main(cfg: EvalCfg) -> None:
  if cfg.plate_mass_kg <= 0.0:
    raise ValueError("plate_mass_kg must be positive")
  if min(cfg.plate_length_m, cfg.plate_width_m, cfg.plate_thickness_m) <= 0.0:
    raise ValueError("plate dimensions must be positive")
  if cfg.plate_friction <= 0.0:
    raise ValueError("plate_friction must be positive")
  pose_specs = {
    "supine": ((0.0, 1.0, 0.0, 0.0), (2,)),
    "prone": ((1.0, 0.0, 0.0, 0.0), (1,)),
    "mixed": ((1.0, 1.0, 0.0, 0.0), (1, 2)),
  }
  if cfg.reset_pose not in pose_specs:
    choices = ", ".join(pose_specs)
    raise ValueError(f"reset_pose must be one of {choices}, got {cfg.reset_pose!r}")
  pose_weights, eligible_reset_types = pose_specs[cfg.reset_pose]
  if cfg.stable_hold_steps <= 0:
    raise ValueError("stable_hold_steps must be positive")
  configure_torch_backends()
  env_cfg = load_env_cfg(cfg.task)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  plate_half_extents = (
    0.5 * cfg.plate_length_m,
    0.5 * cfg.plate_width_m,
    0.5 * cfg.plate_thickness_m,
  )
  env_cfg.scene.entities["escape_obstacle"].spec_fn = partial(
    _get_evaluation_plate_spec,
    half_extents=plate_half_extents,
    friction=cfg.plate_friction,
  )
  # Keep terminal/invalid states intact for the complete audit horizon.
  env_cfg.terminations = {}
  # Evaluate one explicit physical condition. Failure replay and post-stand
  # pushes otherwise change the requested reset distribution during an audit.
  env_cfg.events.pop("stratified_post_stand_wrench", None)
  env_cfg.events.pop("record_failure_states", None)
  env_cfg.events.pop("failure_state_replay_reset", None)
  env_cfg.events.pop("gsi_refresh", None)
  env_cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 1.0,
      "mode_weights": pose_weights,
    }
  )
  plate_reset = env_cfg.events["reset_escape_obstacle"].params
  plate_reset.update(
    {
      "obstacle_probability": 1.0,
      "eligible_reset_types": eligible_reset_types,
      "longitudinal_offset": cfg.longitudinal_offset_m,
      "lateral_offset": cfg.lateral_offset_m,
      "longitudinal_offset_curriculum": (
        cfg.longitudinal_jitter_m,
        cfg.longitudinal_jitter_m,
      ),
      "lateral_offset_curriculum": (
        cfg.lateral_jitter_m,
        cfg.lateral_jitter_m,
      ),
      "overlap_curriculum_steps": 1,
      "xy_offset_range": cfg.xy_jitter_m,
      "plate_mass_range": (cfg.plate_mass_kg, cfg.plate_mass_kg),
      "initial_max_mass": cfg.plate_mass_kg,
      "mass_curriculum_steps": 1,
      "plate_half_extents": plate_half_extents,
    }
  )
  env_cfg.events["update_escape_phase"].params["plate_half_extents"] = (
    plate_half_extents
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
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  active = (raw_env._escape_phase > 0).clone()  # type: ignore[attr-defined]
  reset_type = raw_env._robust_reset_type.clone()  # type: ignore[attr-defined]
  first_contact = torch.full(
    (raw_env.num_envs,), -1, dtype=torch.long, device=raw_env.device
  )
  first_escape = torch.full_like(first_contact, -1)
  first_stable_stand = torch.full_like(first_contact, -1)
  stand_hold = torch.zeros_like(first_contact)
  hand_support_sum = torch.zeros(raw_env.num_envs, device=raw_env.device)
  foot_separation_at_stand = torch.full_like(hand_support_sum, torch.nan)
  foot_speed_at_stand = torch.full_like(hand_support_sum, torch.nan)
  max_post_escape_foot_separation = torch.zeros_like(hand_support_sum)
  max_joint_speed = torch.zeros_like(hand_support_sum)
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
    phase = raw_env._escape_phase  # type: ignore[attr-defined]
    escaped_now = phase == 3
    first_escape[(first_escape < 0) & escaped_now & active] = step + 1
    foot_xy = robot.data.site_pos_w[:, foot_ids, :2]
    foot_separation = torch.linalg.vector_norm(foot_xy[:, 0] - foot_xy[:, 1], dim=-1)
    foot_speed = torch.linalg.vector_norm(
      robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1
    ).mean(dim=-1)
    max_post_escape_foot_separation = torch.where(
      escaped_now,
      torch.maximum(max_post_escape_foot_separation, foot_separation),
      max_post_escape_foot_separation,
    )
    head_z = robot.data.site_pos_w[:, head_idx, 2]
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
    linear_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=-1)
    standing = (
      (head_z >= cfg.stand_head_height_m)
      & (upright >= cfg.stand_min_upright)
      & (linear_speed < cfg.stand_max_linear_speed_m_s)
      & (angular_speed < cfg.stand_max_angular_speed_rad_s)
      & escaped_now
    )
    stand_hold = torch.where(standing, stand_hold + 1, torch.zeros_like(stand_hold))
    new_stable = (first_stable_stand < 0) & (stand_hold >= cfg.stable_hold_steps)
    first_stable_stand[new_stable] = step + 1
    foot_separation_at_stand[new_stable] = foot_separation[new_stable]
    foot_speed_at_stand[new_stable] = foot_speed[new_stable]
    max_joint_speed = torch.maximum(
      max_joint_speed, torch.abs(robot.data.joint_vel).amax(dim=-1)
    )
    max_torque = torch.maximum(max_torque, mdp.max_joint_torque_metric(raw_env))
    max_power = torch.maximum(max_power, mdp.max_joint_power_metric(raw_env))

  phase = raw_env._escape_phase  # type: ignore[attr-defined]
  active_count = int(active.sum())
  contacted = (first_contact >= 0) & active
  invalid = raw_env._escape_invalid_contact & active  # type: ignore[attr-defined]
  setup_invalid = raw_env._escape_invalid_setup & active  # type: ignore[attr-defined]
  any_invalid = invalid | setup_invalid
  escaped = (first_escape >= 0) & active
  valid = active & (~any_invalid)
  valid_escaped = escaped & valid
  stable_stand = (first_stable_stand >= 0) & valid
  first = first_contact[contacted].float()
  escape_steps = first_escape[valid_escaped].float()
  stable_steps = first_stable_stand[stable_stand].float()
  escape_to_stand_steps = (
    first_stable_stand[stable_stand] - first_escape[stable_stand]
  ).float()
  stable_foot_separation = foot_separation_at_stand[stable_stand]
  stable_foot_speed = foot_speed_at_stand[stable_stand]
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

  def median_or(values: torch.Tensor, default: float = -1.0) -> float:
    return float(values.median()) if values.numel() else default

  result = {
    "checkpoint": cfg.checkpoint.name,
    "seed": cfg.seed,
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "step_dt_s": raw_env.step_dt,
    "plate_mass_kg": cfg.plate_mass_kg,
    "plate_length_m": cfg.plate_length_m,
    "plate_width_m": cfg.plate_width_m,
    "plate_thickness_m": cfg.plate_thickness_m,
    "plate_friction": cfg.plate_friction,
    "reset_pose": cfg.reset_pose,
    "longitudinal_offset_m": cfg.longitudinal_offset_m,
    "lateral_offset_m": cfg.lateral_offset_m,
    "longitudinal_jitter_m": cfg.longitudinal_jitter_m,
    "lateral_jitter_m": cfg.lateral_jitter_m,
    "xy_jitter_m": cfg.xy_jitter_m,
    "stable_hold_steps": cfg.stable_hold_steps,
    "stand_head_height_m": cfg.stand_head_height_m,
    "stand_min_upright": cfg.stand_min_upright,
    "stand_max_linear_speed_m_s": cfg.stand_max_linear_speed_m_s,
    "stand_max_angular_speed_rad_s": cfg.stand_max_angular_speed_rad_s,
    "active": active_count,
    "contacted": int(contacted.sum()),
    "first_contact_step_median": float(first.median()) if first.numel() else -1.0,
    "escaped": int(escaped.sum()),
    "conditional_escape_rate": float(escaped.sum() / max(active_count, 1)),
    "valid_conditional_escape_rate": float(
      valid_escaped.sum() / max(int(valid.sum()), 1)
    ),
    "escape_time_median_s": (
      median_or(escape_steps) * raw_env.step_dt if escape_steps.numel() else -1.0
    ),
    "escape_time_p90_s": (
      quantile(escape_steps * raw_env.step_dt, 0.90) if escape_steps.numel() else -1.0
    ),
    "escaped_and_stably_stood": int(stable_stand.sum()),
    "escape_and_stand_rate": float(stable_stand.sum() / max(active_count, 1)),
    "valid_escape_and_stand_rate": float(stable_stand.sum() / max(int(valid.sum()), 1)),
    "stable_stand_given_escape_rate": float(
      stable_stand.sum() / max(int(valid_escaped.sum()), 1)
    ),
    "stable_stand_time_median_s": (
      median_or(stable_steps) * raw_env.step_dt if stable_steps.numel() else -1.0
    ),
    "escape_to_stand_time_median_s": (
      median_or(escape_to_stand_steps) * raw_env.step_dt
      if escape_to_stand_steps.numel()
      else -1.0
    ),
    "stable_foot_separation_median_m": median_or(stable_foot_separation),
    "stable_foot_separation_p90_m": (
      quantile(stable_foot_separation, 0.90) if stable_foot_separation.numel() else -1.0
    ),
    "stable_foot_separation_p95_m": (
      quantile(stable_foot_separation, 0.95) if stable_foot_separation.numel() else -1.0
    ),
    "wide_stance_threshold_m": cfg.wide_stance_threshold_m,
    "wide_stance_rate_at_stable": (
      float((stable_foot_separation > cfg.wide_stance_threshold_m).float().mean())
      if stable_foot_separation.numel()
      else -1.0
    ),
    "stable_foot_speed_median_m_s": median_or(stable_foot_speed),
    "max_post_escape_foot_separation_p95_m": (
      quantile(max_post_escape_foot_separation[valid_escaped], 0.95)
      if valid_escaped.any()
      else -1.0
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
    "max_joint_speed_mean_rad_s": float(max_joint_speed[active].mean()),
    "max_joint_speed_p95_rad_s": quantile(max_joint_speed[active], 0.95),
    "max_torque_mean_nm": float(max_torque[active].mean()),
    "max_power_mean_w": float(max_power[active].mean()),
  }
  for pose_name, pose_type in (("supine", 2), ("prone", 1)):
    pose_active = active & (reset_type == pose_type)
    pose_stable = stable_stand & pose_active
    pose_invalid = any_invalid & pose_active
    pose_count = int(pose_active.sum())
    pose_foot_separation = foot_separation_at_stand[pose_stable]
    result.update(
      {
        f"{pose_name}_active": pose_count,
        f"{pose_name}_escape_and_stand_rate": float(
          pose_stable.sum() / max(pose_count, 1)
        ),
        f"{pose_name}_invalid_rate": float(pose_invalid.sum() / max(pose_count, 1)),
        f"{pose_name}_stable_foot_separation_median_m": median_or(pose_foot_separation),
      }
    )
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
