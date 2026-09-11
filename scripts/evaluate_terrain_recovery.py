"""Reset-stratified zero-shot evaluation for the V3.5 terrain benchmark."""

from __future__ import annotations

import hashlib
import json
import os
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
  checkpoint_sha256: str = ""
  protocol: Path | None = None
  protocol_sha256: str = ""
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
  stand_head_height_m: float = 1.10
  stand_min_upright: float = 0.85
  stand_max_linear_speed_m_s: float = 0.50
  stand_max_angular_speed_rad_s: float = 1.0
  stand_max_abs_head_vertical_speed_m_s: float = 1.0e9
  stable_hold_steps: int = 25


def _quantile(values: torch.Tensor, q: float) -> float:
  return float(torch.quantile(values, q)) if values.numel() else 0.0


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _run_case(
  cfg: EvalCfg,
  terrain_type: str,
  level: int,
  reset_mode: str,
  edge_cohort: str | None = None,
) -> dict[str, object]:
  checkpoint_sha256 = _sha256(cfg.checkpoint)
  if cfg.checkpoint_sha256 and checkpoint_sha256 != cfg.checkpoint_sha256:
    raise RuntimeError("checkpoint SHA-256 does not match the frozen evaluation")
  protocol_sha256 = _sha256(cfg.protocol) if cfg.protocol is not None else None
  if cfg.protocol_sha256 and protocol_sha256 != cfg.protocol_sha256:
    raise RuntimeError("protocol SHA-256 does not match the frozen evaluation")
  # Task registration happens only after CLI parsing.  Select the play terrain
  # before constructing the task so V36 uses the same fixed-terrain and
  # contact-validation path as interactive playback.
  os.environ["SMP_PLAY_TERRAIN_TYPE"] = terrain_type
  os.environ["SMP_PLAY_TERRAIN_LEVEL"] = str(level)
  is_v37_trap = reset_mode == "synthetic_seated_trap"
  is_v38_post_roll = reset_mode == "post_roll_supine_crossed"
  is_support_transition = any(tag in cfg.task for tag in ("V37", "V38", "V39"))
  canonical_reset_mode = (
    "prone" if is_v37_trap else "supine" if is_v38_post_roll else reset_mode
  )
  # The parent V35/V36 selector only knows the four canonical lying poses.
  # Synthetic cohorts first construct a canonical grounded reset, then their
  # frozen event deterministically replaces the whole cohort.
  os.environ["SMP_PLAY_TERRAIN_RESET_POSE"] = canonical_reset_mode
  if is_v38_post_roll:
    os.environ["SMP_PLAY_V38_POST_ROLL_RESET"] = "1"
  else:
    os.environ.pop("SMP_PLAY_V38_POST_ROLL_RESET", None)
  os.environ.pop("SMP_PLAY_AUTO_DISTURBANCES", None)
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
  if (
    reset_mode not in RESET_POSE_WEIGHTS and not is_v37_trap and not is_v38_post_roll
  ) or reset_mode == "mixed":
    raise ValueError(
      "reset_modes must contain prone, supine, left_side, right_side, "
      "synthetic_seated_trap, or post_roll_supine_crossed"
    )
  if edge_cohort is not None:
    if terrain_type != "stairs" or edge_cohort not in EDGE_RESET_COHORTS:
      raise ValueError("edge_cohorts require stairs and a supported V3.7 cohort")

  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if "V38" in cfg.task or "V39" in cfg.task:
    # Evaluation-only sensors make A/B support telemetry symmetric.  They are
    # manager-side measurements and never enter the frozen 93D actor input.
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

    support_sensors = (
      ContactSensorCfg(
        name="v38_hand_ground_contact",
        primary=ContactMatch(
          mode="geom", pattern=r"(left|right)_hand_collision$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=4,
      ),
      ContactSensorCfg(
        name="v38_knee_ground_contact",
        primary=ContactMatch(
          mode="geom",
          pattern=r"(left|right)_(shin|linkage_brace)_collision$",
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=4,
      ),
    )
    existing_sensors = {sensor.name for sensor in env_cfg.scene.sensors or ()}
    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + tuple(
      sensor for sensor in support_sensors if sensor.name not in existing_sensors
    )
  # Older V35 tasks did not consume the play selectors during construction.
  # Keep their explicit replacement path while leaving V36's safe landing
  # island and audited reset ordering intact.
  if "V36" not in cfg.task and not is_support_transition:
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
  if "curriculum_validated_fall_reset" in env_cfg.events:
    reset_event = env_cfg.events["curriculum_validated_fall_reset"]
    reset_event.params["balanced_probability"] = 1.0
    reset_event.params["target_probability"] = 1.0
    reset_event.params["mode_weights"] = RESET_POSE_WEIGHTS[canonical_reset_mode]
  elif "mixed_fall_reset" in env_cfg.events:
    env_cfg.events["mixed_fall_reset"].params.update(
      {
        "procedural_probability": 1.0,
        "mode_weights": RESET_POSE_WEIGHTS[canonical_reset_mode],
      }
    )
  else:
    raise RuntimeError("terrain evaluation requires a supported fall reset event")
  if is_v37_trap:
    if not is_support_transition:
      raise ValueError("synthetic_seated_trap requires a V37/V38/V39 task")
    from mjlab.managers.event_manager import EventTermCfg

    trap = env_cfg.events.get("photo_informed_seated_trap_reset")
    if trap is None:
      trap = EventTermCfg(
        func=mdp.photo_informed_seated_trap_reset,
        mode="reset",
        params={
          "probability": 1.0,
          "joint_noise": 0.08,
          "joint_limit_margin": 0.03,
          "max_penetration": 0.012,
          "max_support_gap": 0.025,
        },
      )
      reordered = {}
      for name, term in env_cfg.events.items():
        reordered[name] = term
        if name == "curriculum_validated_fall_reset":
          reordered["photo_informed_seated_trap_reset"] = trap
      env_cfg.events = reordered
    else:
      trap.params["probability"] = 1.0
  if is_v38_post_roll:
    if "V38" not in cfg.task:
      raise ValueError("post_roll_supine_crossed requires a V38 task")
    from mjlab.managers.event_manager import EventTermCfg

    post_roll = env_cfg.events.get("post_roll_supine_failure_reset")
    if post_roll is None:
      post_roll = EventTermCfg(
        func=mdp.post_roll_supine_failure_reset,
        mode="reset",
        params={
          "probability": 1.0,
          "joint_noise": 0.08,
          "joint_limit_margin": 0.03,
          "max_penetration": 0.012,
          "max_support_gap": 0.025,
        },
      )
      after = (
        "photo_informed_seated_trap_reset"
        if "photo_informed_seated_trap_reset" in env_cfg.events
        else "curriculum_validated_fall_reset"
      )
      reordered = {}
      inserted = False
      for name, term in env_cfg.events.items():
        reordered[name] = term
        if name == after:
          reordered["post_roll_supine_failure_reset"] = post_roll
          inserted = True
      if not inserted:
        raise RuntimeError("V38_EVAL_ALERT: post-roll reset insertion point missing")
      env_cfg.events = reordered
    else:
      post_roll.params["probability"] = 1.0
  if edge_cohort is not None:
    weights = tuple(float(name == edge_cohort) for name in EDGE_RESET_COHORTS)
    edge_event = env_cfg.events["sample_terrain_edge_reset"]
    edge_event.params["cohort_weights"] = weights

  raw_env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  # Curriculum runners enforce training-only warm-start contracts.  Evaluation
  # is an actor-only strict load and must use the ordinary inference runner.
  if runner_cls.__name__ == "SmpCurriculumWarmStartRunner":
    runner_cls = MjlabOnPolicyRunner
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
  reset_contact_valid = getattr(raw_env, "_terrain_reset_contact_valid", None)
  reset_refinement_steps = getattr(raw_env, "_terrain_reset_refinement_steps", None)
  reset_min_distance = getattr(raw_env, "_terrain_reset_min_distance", None)
  if is_support_transition:
    from smp.rl.tasks.getup.mdp.events import _physical_reset_postcheck

    all_env_ids = torch.arange(cfg.num_envs, device=raw_env.device)
    reset_contact_valid = _physical_reset_postcheck(
      raw_env,
      all_env_ids,
      max_penetration=0.012,
      max_support_gap=0.025,
    )
    reset_refinement_steps = torch.zeros(
      cfg.num_envs, dtype=torch.long, device=raw_env.device
    )
    if not bool(reset_contact_valid.all()):
      raise RuntimeError("SUPPORT_TRANSITION_EVAL_ALERT: invalid grounded reset")
  elif reset_contact_valid is None or reset_refinement_steps is None:
    raise RuntimeError("terrain evaluation requires audited reset-contact telemetry")
  if is_v37_trap:
    trap_selected = getattr(raw_env, "_v37_seated_trap_reset", None)
    if trap_selected is None or not bool(trap_selected.all()):
      raise RuntimeError("V37_EVAL_ALERT: trap reset did not cover every environment")
  if is_v38_post_roll:
    selected = getattr(raw_env, "_v38_post_roll_supine_reset", None)
    if selected is None or not bool(selected.all()):
      raise RuntimeError(
        "V38_EVAL_ALERT: post-roll reset did not cover every environment"
      )
  root_xy_start = robot.data.root_link_pos_w[:, :2].clone()
  foot_ids = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  v37_joint_ids = (
    robot.find_joints(
      [
        "left_knee_joint",
        "right_knee_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
      ],
      preserve_order=True,
    )[0]
    if is_support_transition
    else []
  )
  max_planar_displacement = torch.zeros(cfg.num_envs, device=raw_env.device)
  max_terrain_descent = torch.zeros_like(max_planar_displacement)
  max_joint_speed = torch.zeros_like(max_planar_displacement)
  max_torque = torch.zeros_like(max_planar_displacement)
  max_power = torch.zeros_like(max_planar_displacement)
  max_stance_width = torch.zeros_like(max_planar_displacement)
  foot_slip_sum = torch.zeros_like(max_planar_displacement)
  foot_contact_steps = torch.zeros_like(max_planar_displacement)
  max_head_z = robot.data.site_pos_w[:, head_idx, 2].clone()
  max_upright = torch.zeros_like(max_planar_displacement)
  first_success = torch.full(
    (cfg.num_envs,), -1, dtype=torch.long, device=raw_env.device
  )
  stand_hold = torch.zeros_like(first_success)
  longest_stand_hold = torch.zeros_like(first_success)
  first_head_height = torch.full_like(first_success, -1)
  first_upright = torch.full_like(first_success, -1)
  first_linear_settled = torch.full_like(first_success, -1)
  first_angular_settled = torch.full_like(first_success, -1)
  first_head_vertical_settled = torch.full_like(first_success, -1)
  first_strict_candidate = torch.full_like(first_success, -1)
  secondary_fall_hold = torch.zeros_like(first_success)
  secondary_fall = torch.zeros(cfg.num_envs, dtype=torch.bool, device=raw_env.device)
  terrain_exit = torch.zeros_like(secondary_fall)
  invalid_dynamics = torch.zeros_like(secondary_fall)
  finite_action = torch.ones_like(secondary_fall)
  bilateral_support_steps = torch.zeros_like(max_planar_displacement)
  hand_support_steps = torch.zeros_like(max_planar_displacement)
  knee_support_steps = torch.zeros_like(max_planar_displacement)
  active_steps = torch.zeros_like(max_planar_displacement)
  minimum_foot_load_share_sum = torch.zeros_like(max_planar_displacement)
  trap_dwell_steps = torch.zeros_like(first_success)
  left_trap = torch.zeros_like(secondary_fall)
  leg_asymmetry_sum = torch.zeros_like(max_planar_displacement)
  stance_width_sum = torch.zeros_like(max_planar_displacement)
  action_delta_sum = torch.zeros_like(max_planar_displacement)
  action_delta2_sum = torch.zeros_like(max_planar_displacement)
  action_delta_steps = torch.zeros_like(max_planar_displacement)
  action_delta2_steps = torch.zeros_like(max_planar_displacement)
  action_delta_max = torch.zeros_like(max_planar_displacement)
  action_delta2_max = torch.zeros_like(max_planar_displacement)
  previous_action = None
  previous_delta = None
  terrain_generator = raw_env.scene.terrain.cfg.terrain_generator
  terrain_exit_radius = (
    4.0 if terrain_generator is None else 0.5 * min(terrain_generator.size) - 0.5
  )

  for step in range(cfg.steps):
    with torch.inference_mode():
      actions = policy(obs)
      action_is_finite = torch.isfinite(actions).all(dim=-1)
      finite_action &= action_is_finite
      actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
      obs, _, _, _ = env.step(actions)

    current_action = actions.detach()
    if previous_action is not None:
      current_delta = current_action - previous_action
      delta = torch.linalg.vector_norm(current_delta, dim=-1)
      action_delta_sum += delta
      action_delta_steps += 1.0
      action_delta_max = torch.maximum(action_delta_max, delta)
      if previous_delta is not None:
        delta2 = torch.linalg.vector_norm(current_delta - previous_delta, dim=-1)
        action_delta2_sum += delta2
        action_delta2_steps += 1.0
        action_delta2_max = torch.maximum(action_delta2_max, delta2)
      previous_delta = current_delta
    previous_action = current_action.clone()

    raw_root_pos = robot.data.root_link_pos_w
    raw_displacement = torch.linalg.vector_norm(
      raw_root_pos[:, :2] - root_xy_start, dim=-1
    )
    finite = torch.isfinite(raw_root_pos).all(dim=-1) & torch.isfinite(
      robot.data.joint_vel
    ).all(dim=-1)
    invalid_dynamics |= ~finite | ~finite_action
    terrain_exit |= (raw_displacement > terrain_exit_radius) | ~finite
    active = ~terrain_exit

    head_z = robot.data.site_pos_w[:, head_idx, 2]
    head_vertical_speed = torch.abs(robot.data.site_lin_vel_w[:, head_idx, 2])
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
    max_head_z = torch.maximum(max_head_z, torch.where(active, head_z, -torch.inf))
    max_upright = torch.maximum(max_upright, torch.where(active, upright, 0.0))
    linear_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=-1)
    standing = (
      (head_z - support_height >= cfg.stand_head_height_m)
      & (upright >= cfg.stand_min_upright)
      & (linear_speed < cfg.stand_max_linear_speed_m_s)
      & (angular_speed < cfg.stand_max_angular_speed_rad_s)
      & (head_vertical_speed <= cfg.stand_max_abs_head_vertical_speed_m_s)
    )
    first_head_height = torch.where(
      (first_head_height < 0) & ((head_z - support_height) >= cfg.stand_head_height_m),
      torch.full_like(first_head_height, step + 1),
      first_head_height,
    )
    first_upright = torch.where(
      (first_upright < 0) & (upright >= cfg.stand_min_upright),
      torch.full_like(first_upright, step + 1),
      first_upright,
    )
    first_linear_settled = torch.where(
      (first_linear_settled < 0) & (linear_speed < cfg.stand_max_linear_speed_m_s),
      torch.full_like(first_linear_settled, step + 1),
      first_linear_settled,
    )
    first_angular_settled = torch.where(
      (first_angular_settled < 0) & (angular_speed < cfg.stand_max_angular_speed_rad_s),
      torch.full_like(first_angular_settled, step + 1),
      first_angular_settled,
    )
    first_strict_candidate = torch.where(
      (first_strict_candidate < 0) & standing,
      torch.full_like(first_strict_candidate, step + 1),
      first_strict_candidate,
    )
    first_head_vertical_settled = torch.where(
      (first_head_vertical_settled < 0)
      & (head_vertical_speed <= cfg.stand_max_abs_head_vertical_speed_m_s),
      torch.full_like(first_head_vertical_settled, step + 1),
      first_head_vertical_settled,
    )
    stand_hold = torch.where(standing, stand_hold + 1, torch.zeros_like(stand_hold))
    longest_stand_hold = torch.maximum(longest_stand_hold, stand_hold)
    newly_successful = (
      active & (first_success < 0) & (stand_hold >= cfg.stable_hold_steps)
    )
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
    foot_xy = robot.data.site_pos_w[:, foot_ids, :2]
    stance_width = torch.linalg.vector_norm(foot_xy[:, 0] - foot_xy[:, 1], dim=-1)
    max_stance_width = torch.maximum(
      max_stance_width, torch.where(active, stance_width, 0.0)
    )

    active_steps += active.float()
    stance_width_sum += torch.where(active, stance_width, 0.0)
    if is_support_transition:
      v37_sensor = raw_env.scene["v37_foot_ground_contact"]
      v37_found = v37_sensor.data.found
      v37_force = v37_sensor.data.force
      if v37_found is None or v37_force is None:
        raise RuntimeError("V37 foot sensor must expose found and force")
      flat_found = v37_found.reshape(cfg.num_envs, -1) > 0
      found_split = max(flat_found.shape[1] // 2, 1)
      left_found = flat_found[:, :found_split].any(dim=-1)
      right_found = flat_found[:, found_split:].any(dim=-1)
      flat_force = v37_force.reshape(cfg.num_envs, -1, 3)
      force_split = max(flat_force.shape[1] // 2, 1)
      left_force = torch.linalg.vector_norm(flat_force[:, :force_split], dim=-1).amax(
        dim=-1
      )
      right_force = torch.linalg.vector_norm(flat_force[:, force_split:], dim=-1).amax(
        dim=-1
      )
      total_force = left_force + right_force
      minimum_share = torch.minimum(left_force, right_force) / torch.clamp(
        total_force, min=1.0
      )
      both_feet = left_found & right_found & active
      bilateral_support_steps += both_feet.float()
      minimum_foot_load_share_sum += torch.where(active, minimum_share, 0.0)
      joint = robot.data.joint_pos[:, v37_joint_ids]
      leg_asymmetry = torch.abs(joint[:, 0] - joint[:, 1]) + torch.abs(
        joint[:, 2] - joint[:, 3]
      )
      leg_asymmetry_sum += torch.where(active, leg_asymmetry, 0.0)
      if "V38" in cfg.task or "V39" in cfg.task:
        hand_found = raw_env.scene["v38_hand_ground_contact"].data.found
        knee_found = raw_env.scene["v38_knee_ground_contact"].data.found
        if hand_found is None or knee_found is None:
          raise RuntimeError("V38 support sensors must expose found")
        hand_support_steps += (
          hand_found.reshape(cfg.num_envs, -1).any(dim=-1) & active
        ).float()
        knee_support_steps += (
          knee_found.reshape(cfg.num_envs, -1).any(dim=-1) & active
        ).float()
    still_trapped = (head_z - support_height < 0.75) & ~left_trap & active
    trap_dwell_steps += still_trapped.long()
    left_trap |= (head_z - support_height >= 0.75) | ~active

    found = (
      v37_found
      if is_support_transition
      else raw_env.scene["terrain_foot_ground_contact"].data.found
    )
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
  action_delta_mean = action_delta_sum / torch.clamp(action_delta_steps, min=1.0)
  action_delta2_mean = action_delta2_sum / torch.clamp(action_delta2_steps, min=1.0)

  failure_reasons = []
  for index in range(cfg.num_envs):
    if bool(success[index]):
      reason = "success"
    elif not bool(reset_contact_valid[index]):
      reason = (
        "invalid_initialization" if "V38" in cfg.task else "invalid_reset_contact"
      )
    elif not bool(finite_action[index]) and "V38" in cfg.task:
      reason = "nonfinite_action"
    elif bool(invalid_dynamics[index]):
      reason = "invalid_dynamics"
    elif bool(terrain_exit[index]) and "V38" not in cfg.task:
      reason = "terrain_exit"
    elif int(first_head_height[index]) < 0:
      reason = "head_height_not_reached"
    elif int(first_upright[index]) < 0:
      reason = "upright_not_reached"
    elif int(first_linear_settled[index]) < 0 and "V38" not in cfg.task:
      reason = "linear_speed_not_settled"
    elif int(first_angular_settled[index]) < 0 and "V38" not in cfg.task:
      reason = "angular_speed_not_settled"
    elif int(first_head_vertical_settled[index]) < 0:
      reason = "head_vertical_speed_not_settled"
    else:
      reason = (
        "stable_hold_too_short" if "V38" in cfg.task else "strict_candidate_not_held"
      )
    failure_reasons.append(reason)
  reason_codebook = (
    (
      "success",
      "invalid_initialization",
      "invalid_dynamics",
      "nonfinite_action",
      "head_height_not_reached",
      "upright_not_reached",
      "head_vertical_speed_not_settled",
      "stable_hold_too_short",
    )
    if "V38" in cfg.task
    else (
      "success",
      "invalid_reset_contact",
      "invalid_dynamics",
      "terrain_exit",
      "head_height_not_reached",
      "upright_not_reached",
      "linear_speed_not_settled",
      "angular_speed_not_settled",
      "head_vertical_speed_not_settled",
      "strict_candidate_not_held",
    )
  )
  reason_counts = {reason: failure_reasons.count(reason) for reason in reason_codebook}
  if sum(reason_counts.values()) != cfg.num_envs or reason_counts["success"] != int(
    success.sum()
  ):
    raise RuntimeError("strict failure-reason accounting is inconsistent")

  def _list(value: torch.Tensor) -> list[object]:
    return value.detach().cpu().tolist()

  result: dict[str, object] = {
    "schema_version": 2,
    "checkpoint": str(cfg.checkpoint),
    "checkpoint_sha256": checkpoint_sha256,
    "protocol": str(cfg.protocol) if cfg.protocol is not None else None,
    "protocol_sha256": protocol_sha256,
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
    "finite_action_rate": float(finite_action.float().mean()),
    "strict_success_hold_steps": cfg.stable_hold_steps,
    "strict_success_thresholds": {
      "head_height_above_support_m": cfg.stand_head_height_m,
      "upright": cfg.stand_min_upright,
      "linear_speed_m_s": cfg.stand_max_linear_speed_m_s,
      "angular_speed_rad_s": cfg.stand_max_angular_speed_rad_s,
      "absolute_head_vertical_speed_m_s": (cfg.stand_max_abs_head_vertical_speed_m_s),
    },
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
    "terrain_reset_contact_valid_rate": float(reset_contact_valid.float().mean()),
    "terrain_reset_refinement_steps_mean": float(reset_refinement_steps.float().mean()),
    "terrain_reset_refinement_steps_max": int(reset_refinement_steps.max()),
    "terrain_reset_min_distance_min_m": (
      float(reset_min_distance.min()) if reset_min_distance is not None else None
    ),
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
    "max_power_p95_w": _quantile(max_power, 0.95),
    "max_torque_p95_nm": _quantile(max_torque, 0.95),
    "max_stance_width_mean_m": float(max_stance_width.mean()),
    "max_stance_width_p95_m": _quantile(max_stance_width, 0.95),
    "action_first_difference_mean_l2": float(action_delta_mean.mean()),
    "action_first_difference_p95_l2": _quantile(action_delta_mean, 0.95),
    "action_second_difference_mean_l2": float(action_delta2_mean.mean()),
    "action_second_difference_p95_l2": _quantile(action_delta2_mean, 0.95),
    "bilateral_foot_support_fraction_mean": float(
      (bilateral_support_steps / torch.clamp(active_steps, min=1.0)).mean()
    ),
    "minimum_foot_load_share_mean": float(
      (minimum_foot_load_share_sum / torch.clamp(active_steps, min=1.0)).mean()
    ),
    "stance_width_mean_m": float(
      (stance_width_sum / torch.clamp(active_steps, min=1.0)).mean()
    ),
    "leg_asymmetry_mean_rad": float(
      (leg_asymmetry_sum / torch.clamp(active_steps, min=1.0)).mean()
    ),
    "trap_dwell_time_mean_s": float(trap_dwell_steps.float().mean() * raw_env.step_dt),
    "strict_failure_diagnosis": {
      "schema_version": 1,
      "reason_codebook": list(reason_counts),
      "reason_counts": reason_counts,
    },
    "per_env": {
      "strict_success": _list(success),
      "strict_success_step": _list(first_success),
      "failure_reason": failure_reasons,
      "first_head_height_step": _list(first_head_height),
      "first_upright_step": _list(first_upright),
      "first_linear_speed_settled_step": _list(first_linear_settled),
      "first_angular_speed_settled_step": _list(first_angular_settled),
      "first_head_vertical_speed_settled_step": _list(first_head_vertical_settled),
      "first_strict_candidate_step": _list(first_strict_candidate),
      "longest_stable_stand_hold_steps": _list(longest_stand_hold),
      "secondary_fall": _list(secondary_fall),
      "terrain_exit": _list(terrain_exit),
      "invalid_dynamics": _list(invalid_dynamics),
      "terrain_reset_contact_valid": _list(reset_contact_valid),
      "terrain_reset_refinement_steps": _list(reset_refinement_steps),
      "foot_slip_mean_m_s": _list(foot_slip),
      "max_planar_displacement_m": _list(max_planar_displacement),
      "max_stance_width_m": _list(max_stance_width),
      "action_first_difference_mean_l2": _list(action_delta_mean),
      "action_second_difference_mean_l2": _list(action_delta2_mean),
      "max_joint_speed_rad_s": _list(max_joint_speed),
      "max_torque_nm": _list(max_torque),
      "max_power_w": _list(max_power),
      "finite_action": _list(finite_action),
      "bilateral_foot_support_fraction": _list(
        bilateral_support_steps / torch.clamp(active_steps, min=1.0)
      ),
      "minimum_foot_load_share_mean": _list(
        minimum_foot_load_share_sum / torch.clamp(active_steps, min=1.0)
      ),
      "stance_width_mean_m": _list(
        stance_width_sum / torch.clamp(active_steps, min=1.0)
      ),
      "leg_asymmetry_mean_rad": _list(
        leg_asymmetry_sum / torch.clamp(active_steps, min=1.0)
      ),
      "trap_dwell_steps": _list(trap_dwell_steps),
      # V38 frozen additive telemetry.  These aliases intentionally preserve
      # the older field names above so historical V37 results remain readable.
      "max_head_z": _list(max_head_z),
      "max_upright": _list(max_upright),
      "bilateral_support_fraction": _list(
        bilateral_support_steps / torch.clamp(active_steps, min=1.0)
      ),
      "hand_support_fraction": _list(
        hand_support_steps / torch.clamp(active_steps, min=1.0)
      ),
      "knee_support_fraction": _list(
        knee_support_steps / torch.clamp(active_steps, min=1.0)
      ),
      "foot_slip": _list(foot_slip),
      "root_drift": _list(max_planar_displacement),
      "action_first_difference": _list(action_delta_mean),
      "action_second_difference": _list(action_delta2_mean),
      "peak_joint_speed": _list(max_joint_speed),
      "peak_joint_torque": _list(max_torque),
      "peak_joint_power": _list(max_power),
      "first_head_vertical_speed_step": _list(first_head_vertical_settled),
      "longest_strict_hold_steps": _list(longest_stand_hold),
    },
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
