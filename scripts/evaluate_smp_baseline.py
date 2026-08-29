"""Frozen evaluation for original-SMP observation-factorial policies."""

from __future__ import annotations

import hashlib
import json
import math
import time
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
from smp.firm.deployable_policy import FirmDeployablePolicy
from smp.rl.tasks.getup import mdp

_RESET_WEIGHTS = {
  "prone": (1.0, 0.0, 0.0, 0.0),
  "supine": (0.0, 1.0, 0.0, 0.0),
  "left_side": (0.0, 0.0, 1.0, 0.0),
  "right_side": (0.0, 0.0, 0.0, 1.0),
}
_EVALUATION_SCHEMA_VERSION = 2
_SPECIALIST_PROFILES = ("flat", "terrain", "plate")
_TERRAIN_TYPES = ("flat", "slope", "stairs", "rough")
_STAIR_EDGE_COHORTS = ("center", "near_edge", "straddle", "lower_tread")


@dataclass(frozen=True)
class EvalCfg:
  checkpoint: Path
  policy_kind: str = "rsl_rl"
  firm_adapter_checkpoint: Path | None = None
  firm_goal_refresh_steps: int = 5
  firm_num_action_samples: int = 1
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
  evaluation_profile: str = "flat"
  terrain_type: str = ""
  terrain_level: int = -1
  stair_edge_cohort: str = ""
  plate_mode: str = ""
  plate_mass_kg: float = 0.0
  matched_eval_manifest: Path | None = None
  matched_eval_manifest_sha256: str = ""


def _validate_policy_configuration(cfg: EvalCfg) -> None:
  if cfg.policy_kind not in ("rsl_rl", "firm_r"):
    raise ValueError("policy_kind must be rsl_rl or firm_r")
  if cfg.policy_kind == "rsl_rl":
    if cfg.firm_adapter_checkpoint is not None:
      raise ValueError("rsl_rl policy cannot carry a FIRM adapter")
    return
  if cfg.firm_adapter_checkpoint is None:
    raise ValueError("firm_r policy requires a deployable adapter checkpoint")
  if cfg.firm_goal_refresh_steps <= 0 or cfg.firm_num_action_samples <= 0:
    raise ValueError("FIRM refresh steps and action samples must be positive")


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
    z * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) / denominator
  )
  # Preserve the observed rate exactly at the boundaries despite floating-point
  # roundoff (for example, the upper bound for 10/10 can be 0.9999999999999999).
  return (
    min(rate, max(0.0, center - radius)),
    max(rate, min(1.0, center + radius)),
  )


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _configure_matched_eval_bank(env_cfg, cfg: EvalCfg) -> dict | None:
  """Bind one held-out bank to a native Tier-A task, or reject ambiguity."""
  is_native_baseline = "init_matched_reset_bank" in env_cfg.events
  if cfg.matched_eval_manifest is None:
    if is_native_baseline:
      raise ValueError(
        "native Tier-A evaluation requires a matched held-out reset-bank manifest"
      )
    if cfg.matched_eval_manifest_sha256:
      raise ValueError("held-out manifest SHA was provided without a manifest")
    return None
  if not is_native_baseline:
    raise ValueError("matched held-out banks are only valid for native Tier-A tasks")
  if cfg.evaluation_profile != "flat":
    raise ValueError("matched held-out banks are only defined for flat evaluation")
  path = cfg.matched_eval_manifest
  if (
    not path.is_file()
    or len(cfg.matched_eval_manifest_sha256) != 64
    or _sha256(path) != cfg.matched_eval_manifest_sha256
  ):
    raise ValueError("matched held-out reset-bank manifest SHA-256 mismatch")
  manifest = json.loads(path.read_text())
  if (
    manifest.get("status") != "READY"
    or manifest.get("generation_seed") != cfg.seed
    or manifest.get("num_states_per_mode") != cfg.num_envs
    or tuple(manifest.get("modes", ())) != ("native_gsi", *_RESET_WEIGHTS)
    or manifest.get("exact_training_overlap_count") != 0
    or not isinstance(manifest.get("training_bank_sha256"), str)
    or len(manifest["training_bank_sha256"]) != 64
  ):
    raise ValueError("matched held-out reset-bank manifest violates the protocol")
  row = manifest.get("banks", {}).get(cfg.reset_mode)
  if not isinstance(row, dict):
    raise ValueError(f"held-out manifest lacks reset mode {cfg.reset_mode}")
  bank = Path(row.get("path", ""))
  bank_sha = row.get("sha256")
  expected_type = (
    0
    if cfg.reset_mode == "native_gsi"
    else list(_RESET_WEIGHTS).index(cfg.reset_mode) + 1
  )
  expected_counts = [0] * 5
  expected_counts[expected_type] = cfg.num_envs
  if (
    not bank.is_file()
    or not isinstance(bank_sha, str)
    or _sha256(bank) != bank_sha
    or row.get("num_states") != cfg.num_envs
    or row.get("reset_type_counts") != expected_counts
  ):
    raise ValueError(f"held-out {cfg.reset_mode} bank or provenance changed")
  loader = env_cfg.events["init_matched_reset_bank"]
  loader.params.update(
    {
      "bank_path": str(bank.resolve()),
      "bank_sha256": bank_sha,
      "expected_num_states": cfg.num_envs,
      "sampling_seed": cfg.seed,
    }
  )
  return {
    "manifest": str(path.resolve()),
    "manifest_sha256": cfg.matched_eval_manifest_sha256,
    "bank": str(bank.resolve()),
    "bank_sha256": bank_sha,
    "training_bank_sha256": manifest["training_bank_sha256"],
    "exact_training_overlap_count": 0,
  }


def _configure_specialist_stratum(env_cfg, cfg: EvalCfg) -> None:
  """Force one preregistered T/P stratum before environment construction."""
  if cfg.evaluation_profile not in _SPECIALIST_PROFILES:
    raise ValueError(f"evaluation_profile must be one of: {_SPECIALIST_PROFILES}")
  if cfg.evaluation_profile == "flat":
    if any(
      (
        cfg.terrain_type,
        cfg.stair_edge_cohort,
        cfg.plate_mode,
        cfg.terrain_level >= 0,
        cfg.plate_mass_kg > 0.0,
      )
    ):
      raise ValueError("flat evaluation cannot carry a specialist stratum")
    return
  if cfg.reset_mode not in _RESET_WEIGHTS:
    raise ValueError("specialist evaluation requires one fixed procedural pose")
  env_cfg.events.pop("push_robot", None)
  env_cfg.events["mixed_fall_reset"].params["mode_weights"] = _RESET_WEIGHTS[
    cfg.reset_mode
  ]

  if cfg.evaluation_profile == "terrain":
    if cfg.terrain_type not in _TERRAIN_TYPES:
      raise ValueError(f"terrain_type must be one of: {_TERRAIN_TYPES}")
    if cfg.terrain_level not in (0, 1, 2):
      raise ValueError("formal terrain_level must be 0, 1, or 2")
    if cfg.plate_mode or cfg.plate_mass_kg > 0.0:
      raise ValueError("terrain evaluation cannot carry a plate stratum")
    if cfg.terrain_type == "stairs":
      if cfg.stair_edge_cohort not in _STAIR_EDGE_COHORTS:
        raise ValueError(f"stair_edge_cohort must be one of: {_STAIR_EDGE_COHORTS}")
    elif cfg.stair_edge_cohort:
      raise ValueError("only stairs may specify a stair_edge_cohort")
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
      raise RuntimeError("formal terrain evaluation requires a generator")
    if tuple(generator.sub_terrains) != _TERRAIN_TYPES:
      raise RuntimeError("terrain generator order differs from frozen protocol")
    for name, sub_terrain in generator.sub_terrains.items():
      sub_terrain.proportion = float(name == cfg.terrain_type)
    level_count = len(
      env_cfg.events["sample_weighted_terrain_levels"].params["level_weights"]
    )
    level_weights = [0.0] * level_count
    level_weights[cfg.terrain_level] = 1.0
    env_cfg.events["sample_weighted_terrain_levels"].params["level_weights"] = tuple(
      level_weights
    )
    cohort_weights = [0.0] * len(_STAIR_EDGE_COHORTS)
    cohort_index = (
      _STAIR_EDGE_COHORTS.index(cfg.stair_edge_cohort)
      if cfg.terrain_type == "stairs"
      else 0
    )
    cohort_weights[cohort_index] = 1.0
    env_cfg.events["sample_terrain_edge_reset"].params["cohort_weights"] = tuple(
      cohort_weights
    )
    return

  if cfg.terrain_type or cfg.stair_edge_cohort or cfg.terrain_level >= 0:
    raise ValueError("plate evaluation cannot carry a terrain stratum")
  if cfg.plate_mode not in ("unpinned", "pinned"):
    raise ValueError("plate_mode must be unpinned or pinned")
  if cfg.plate_mode == "pinned":
    if cfg.reset_mode not in ("prone", "supine"):
      raise ValueError("the frozen physical fixture only pins prone or supine")
    if cfg.plate_mass_kg not in (4.0, 8.0, 12.0):
      raise ValueError("pinned plate_mass_kg must be 4, 8, or 12")
  elif cfg.plate_mass_kg != 0.0:
    raise ValueError("unpinned evaluation must use plate_mass_kg=0")
  reset_plate = env_cfg.events["reset_escape_obstacle"]
  reset_plate.params["obstacle_probability_by_reset_type"] = (
    (1.0, 1.0, 0.0, 0.0) if cfg.plate_mode == "pinned" else (0.0, 0.0, 0.0, 0.0)
  )
  if cfg.plate_mode == "pinned":
    reset_plate.params["plate_masses"] = (cfg.plate_mass_kg,)
    reset_plate.params["mass_weights"] = (1.0,)


def main(cfg: EvalCfg) -> None:
  _validate_policy_configuration(cfg)
  valid_modes = ("native_gsi", *_RESET_WEIGHTS)
  if cfg.reset_mode not in valid_modes:
    choices = ", ".join(valid_modes)
    raise ValueError(f"reset_mode must be one of: {choices}")

  configure_torch_backends()
  env_cfg = load_env_cfg(cfg.task)
  agent_cfg = load_rl_cfg(cfg.task)
  matched_eval = _configure_matched_eval_bank(env_cfg, cfg)
  _configure_specialist_stratum(env_cfg, cfg)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.terminations = {}
  env_cfg.episode_length_s = 1.0e9
  env_cfg.events.pop("gsi_refresh", None)
  if "init_smp_state" in env_cfg.events:
    prior_params = {
      "compile_model": False,
      "gsi_buffer_size": 1 if matched_eval is not None else max(1024, cfg.num_envs),
    }
    if matched_eval is not None:
      prior_params["gsi_batch_size"] = 1
    env_cfg.events["init_smp_state"].params.update(prior_params)

  if matched_eval is not None:
    if cfg.reset_mode != "native_gsi" or not cfg.native_pushes:
      env_cfg.events.pop("push_robot", None)
  elif cfg.evaluation_profile != "flat":
    pass
  elif cfg.reset_mode == "native_gsi":
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
  firm_policy: FirmDeployablePolicy | None = None
  if cfg.policy_kind == "rsl_rl":
    runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
    runner.load(
      str(cfg.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=cfg.device,
    )
    policy = runner.get_inference_policy(device=cfg.device)
  else:
    assert cfg.firm_adapter_checkpoint is not None
    firm_policy = FirmDeployablePolicy(
      cfg.checkpoint,
      cfg.firm_adapter_checkpoint,
      device=cfg.device,
      expected_seed=cfg.policy_seed,
      goal_refresh_steps=cfg.firm_goal_refresh_steps,
      num_action_samples=cfg.firm_num_action_samples,
    )
    policy = firm_policy
  rollout_rng_seed = cfg.seed + 1000003
  torch.manual_seed(rollout_rng_seed)
  obs = env.get_observations()

  terrain_type_ids = torch.full(
    (raw_env.num_envs,), -1, dtype=torch.long, device=raw_env.device
  )
  terrain_levels = torch.full_like(terrain_type_ids, -1)
  terrain_cohorts = torch.full_like(terrain_type_ids, -1)
  plate_present = torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)
  plate_mass = torch.zeros(raw_env.num_envs, device=raw_env.device)
  plate_friction = torch.zeros_like(plate_mass)
  plate_longitudinal_offset = torch.zeros_like(plate_mass)
  plate_lateral_offset = torch.zeros_like(plate_mass)
  if cfg.evaluation_profile == "terrain":
    terrain = raw_env.scene["terrain"]
    terrain_type_ids = terrain.terrain_types.clone()
    terrain_levels = terrain.terrain_levels.clone()
    terrain_cohorts = raw_env._terrain_reset_cohort.clone()
    generator = terrain.cfg.terrain_generator
    if generator is None:
      raise RuntimeError("formal terrain evaluation lost its generator")
    expected_type = tuple(generator.sub_terrains).index(cfg.terrain_type)
    expected_cohort = (
      _STAIR_EDGE_COHORTS.index(cfg.stair_edge_cohort)
      if cfg.terrain_type == "stairs"
      else 0
    )
    if not torch.all(terrain_type_ids == expected_type):
      raise RuntimeError("terrain family allocation did not match forced stratum")
    if not torch.all(terrain_levels == cfg.terrain_level):
      raise RuntimeError("terrain level allocation did not match forced stratum")
    if not torch.all(terrain_cohorts == expected_cohort):
      raise RuntimeError("stair cohort allocation did not match forced stratum")
  elif cfg.evaluation_profile == "plate":
    phase = getattr(raw_env, "_escape_phase", None)
    if phase is None:
      raise RuntimeError("formal plate evaluation has no escape phase state")
    plate_present = phase > 0
    expected_present = cfg.plate_mode == "pinned"
    if not torch.all(plate_present == expected_present):
      raise RuntimeError("plate presence did not match forced stratum")
    plate_mass = mdp.escape_plate_mass_metric(raw_env).clone()
    plate_friction = mdp.escape_plate_friction_metric(raw_env).clone()
    plate_longitudinal_offset = mdp.escape_plate_longitudinal_offset_metric(
      raw_env
    ).clone()
    plate_lateral_offset = mdp.escape_plate_lateral_offset_metric(raw_env).clone()
    if expected_present and not torch.allclose(
      plate_mass,
      torch.full_like(plate_mass, cfg.plate_mass_kg),
      atol=1.0e-5,
      rtol=0.0,
    ):
      raise RuntimeError("plate mass did not match forced stratum")

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
  invalid_dynamics = torch.zeros_like(finite)
  terrain_exit = torch.zeros_like(finite)
  escape_first = torch.full_like(strict_first, -1)
  invalid_escape_setup = torch.zeros_like(finite)
  invalid_escape_contact = torch.zeros_like(finite)
  hand_support = torch.zeros_like(finite)
  policy_inference_wall_s = 0.0
  policy_inference_steps = 0

  initial_head_z = robot.data.site_pos_w[:, head_idx, 2].clone()
  if cfg.evaluation_profile == "terrain":
    initial_head_z -= raw_env.scene.env_origins[:, 2]
  initial_upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0).clone()

  for step in range(cfg.steps):
    with torch.inference_mode():
      if raw_env.device.type == "cuda":
        torch.cuda.synchronize(raw_env.device)
      inference_start = time.perf_counter()
      actions = policy(obs)
      if raw_env.device.type == "cuda":
        torch.cuda.synchronize(raw_env.device)
      policy_inference_wall_s += time.perf_counter() - inference_start
      policy_inference_steps += 1
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

    head_z_world = robot.data.site_pos_w[:, head_idx, 2]
    head_z = (
      head_z_world - raw_env.scene.env_origins[:, 2]
      if cfg.evaluation_profile == "terrain"
      else head_z_world
    )
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
    linear_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=-1)
    escape_complete = torch.ones_like(plate_present)
    if cfg.evaluation_profile == "plate":
      phase = raw_env._escape_phase
      current_escape = phase == 3
      newly_escaped = (escape_first < 0) & current_escape & plate_present
      escape_first[newly_escaped] = step + 1
      escape_complete = (~plate_present) | current_escape
      invalid_escape_setup |= raw_env._escape_invalid_setup
      invalid_escape_contact |= raw_env._escape_invalid_contact
      hand_found = raw_env.scene["hand_ground_contact"].data.found
      if hand_found is None:
        raise RuntimeError("formal plate evaluation requires hand contact labels")
      hand_support |= hand_found.reshape(raw_env.num_envs, -1).any(dim=-1)
    strict_standing = (
      (head_z >= 1.10)
      & (upright >= 0.85)
      & (linear_speed < 0.50)
      & (angular_speed < 1.0)
      & escape_complete
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
    max_root_planar_excursion = torch.maximum(max_root_planar_excursion, root_excursion)
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

    state_finite = (
      torch.isfinite(robot.data.root_link_pose_w).all(dim=-1)
      & torch.isfinite(robot.data.root_link_lin_vel_w).all(dim=-1)
      & torch.isfinite(robot.data.root_link_ang_vel_w).all(dim=-1)
      & torch.isfinite(robot.data.joint_pos).all(dim=-1)
      & torch.isfinite(robot.data.joint_vel).all(dim=-1)
    )
    invalid_dynamics |= ~state_finite
    if cfg.evaluation_profile == "terrain":
      terrain_exit |= mdp.terrain_patch_exit(raw_env, margin=0.50)

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

  valid_episode = finite & (~invalid_dynamics) & (~terrain_exit)
  if cfg.evaluation_profile == "plate":
    valid_episode &= (~invalid_escape_setup) & (~invalid_escape_contact)
  strict_success = (strict_first >= 0) & valid_episode
  baseline_success = (baseline_first >= 0) & valid_episode
  recovery_steps = strict_first[strict_success].float()
  strict_successes = int(strict_success.sum())
  baseline_successes = int(baseline_success.sum())
  strict_ci = _wilson_interval(strict_successes, raw_env.num_envs)
  baseline_ci = _wilson_interval(baseline_successes, raw_env.num_envs)
  foot_slip = foot_slip_sum / torch.clamp(foot_contact_steps, min=1.0)
  action_delta_rms = action_delta_sum / cfg.steps
  action_second_difference_rms = action_second_difference_sum / max(cfg.steps - 1, 1)
  successful_secondary_fall = secondary_fall & strict_success
  successful_foot_separation = foot_separation_at_success[strict_success]
  pinned = plate_present
  escaped = escape_first >= 0
  escaped_steps = escape_first[pinned & escaped].float()
  pinned_count = int(pinned.sum())
  escape_successes = int((pinned & escaped).sum())
  escape_success_rate = escape_successes / pinned_count if pinned_count else None
  friction_tertiles = {"low": 0, "mid": 0, "high": 0}
  offset_cells = {
    f"long_{long_label}__lat_{lat_label}": 0
    for long_label in ("negative", "center", "positive")
    for lat_label in ("negative", "center", "positive")
  }
  if pinned_count:
    friction_bucket = torch.bucketize(
      plate_friction[pinned],
      torch.tensor((2.0 / 3.0, 14.0 / 15.0), device=raw_env.device),
    )
    for index, label in enumerate(friction_tertiles):
      friction_tertiles[label] = int((friction_bucket == index).sum())
    boundaries = torch.tensor((-0.04, 0.04), device=raw_env.device)
    longitudinal_bucket = torch.bucketize(plate_longitudinal_offset[pinned], boundaries)
    lateral_bucket = torch.bucketize(plate_lateral_offset[pinned], boundaries)
    labels = ("negative", "center", "positive")
    for longitudinal_index, longitudinal_label in enumerate(labels):
      for lateral_index, lateral_label in enumerate(labels):
        offset_cells[f"long_{longitudinal_label}__lat_{lateral_label}"] = int(
          (
            (longitudinal_bucket == longitudinal_index)
            & (lateral_bucket == lateral_index)
          ).sum()
        )
  stratum = {
    "evaluation_profile": cfg.evaluation_profile,
    "terrain_type": cfg.terrain_type or None,
    "terrain_level": cfg.terrain_level if cfg.terrain_level >= 0 else None,
    "stair_edge_cohort": cfg.stair_edge_cohort or None,
    "fall_pose": cfg.reset_mode,
    "plate_mode": cfg.plate_mode or None,
    "plate_present": bool(cfg.plate_mode == "pinned"),
    "plate_mass_kg": cfg.plate_mass_kg if cfg.plate_mass_kg > 0.0 else None,
  }
  result = {
    "evaluation_schema_version": _EVALUATION_SCHEMA_VERSION,
    "policy_kind": cfg.policy_kind,
    "checkpoint": cfg.checkpoint.name,
    "checkpoint_path": str(cfg.checkpoint.resolve()),
    "checkpoint_sha256": _sha256(cfg.checkpoint),
    "firm_adapter_checkpoint": (
      str(cfg.firm_adapter_checkpoint.resolve())
      if cfg.firm_adapter_checkpoint is not None
      else None
    ),
    "firm_adapter_checkpoint_sha256": (
      _sha256(cfg.firm_adapter_checkpoint)
      if cfg.firm_adapter_checkpoint is not None
      else None
    ),
    "policy_metadata": firm_policy.metadata() if firm_policy is not None else None,
    "task": cfg.task,
    "reset_mode": cfg.reset_mode,
    "evaluation_profile": cfg.evaluation_profile,
    "stratum": stratum,
    "native_pushes": cfg.native_pushes if cfg.reset_mode == "native_gsi" else False,
    "policy_seed": cfg.policy_seed,
    "seed": cfg.seed,
    "rollout_rng_seed": rollout_rng_seed,
    "matched_eval_manifest": matched_eval["manifest"] if matched_eval else None,
    "matched_eval_manifest_sha256": (
      matched_eval["manifest_sha256"] if matched_eval else None
    ),
    "matched_eval_bank": matched_eval["bank"] if matched_eval else None,
    "matched_eval_bank_sha256": matched_eval["bank_sha256"] if matched_eval else None,
    "matched_eval_training_bank_sha256": (
      matched_eval["training_bank_sha256"] if matched_eval else None
    ),
    "matched_eval_exact_training_overlap_count": (
      matched_eval["exact_training_overlap_count"] if matched_eval else None
    ),
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "physics_dt_s": float(raw_env.physics_dt),
    "control_dt_s": float(raw_env.step_dt),
    "actor_observation_dim": int(obs["actor"].shape[-1]),
    "critic_observation_dim": int(obs["critic"].shape[-1]),
    "strict_success_definition": {
      "head_height_min_m": 1.10,
      "head_height_relative_to_env_origin": cfg.evaluation_profile == "terrain",
      "upright_min": 0.85,
      "root_linear_speed_max_m_s": 0.50,
      "root_angular_speed_max_rad_s": 1.0,
      "hold_steps": 25,
      "requires_plate_escape": cfg.evaluation_profile == "plate"
      and cfg.plate_mode == "pinned",
      "requires_valid_dynamics_and_setup": True,
    },
    "strict_successes": strict_successes,
    "strict_success_rate": float(strict_success.float().mean()),
    "strict_success_rate_ci95_low": strict_ci[0],
    "strict_success_rate_ci95_high": strict_ci[1],
    "baseline_successes": baseline_successes,
    "baseline_success_rate": float(baseline_success.float().mean()),
    "baseline_success_rate_ci95_low": baseline_ci[0],
    "baseline_success_rate_ci95_high": baseline_ci[1],
    "terrain_exit_rate": float(terrain_exit.float().mean()),
    "invalid_dynamics_rate": float(invalid_dynamics.float().mean()),
    "invalid_escape_setup_rate": float(invalid_escape_setup.float().mean()),
    "invalid_escape_contact_rate": float(invalid_escape_contact.float().mean()),
    "escape_successes": escape_successes,
    "escape_trials": pinned_count,
    "escape_success_rate": escape_success_rate,
    "escape_time_median_s": (
      float(escaped_steps.median() * raw_env.step_dt) if escaped_steps.numel() else None
    ),
    "escape_time_p90_s": (
      _quantile(escaped_steps * raw_env.step_dt, 0.90)
      if escaped_steps.numel()
      else None
    ),
    "hand_support_rate": (
      float(hand_support[pinned].float().mean()) if pinned_count else None
    ),
    "plate_friction_min": (
      float(plate_friction[pinned].min()) if pinned_count else None
    ),
    "plate_friction_max": (
      float(plate_friction[pinned].max()) if pinned_count else None
    ),
    "plate_friction_tertile_counts": friction_tertiles,
    "plate_longitudinal_offset_abs_p95_m": (
      _quantile(plate_longitudinal_offset[pinned].abs(), 0.95) if pinned_count else None
    ),
    "plate_lateral_offset_abs_p95_m": (
      _quantile(plate_lateral_offset[pinned].abs(), 0.95) if pinned_count else None
    ),
    "plate_offset_cell_counts": offset_cells,
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
    "action_second_difference_rms_mean": float(action_second_difference_rms.mean()),
    "action_second_difference_rms_p95": _quantile(action_second_difference_rms, 0.95),
    "policy_inference_wall_s": policy_inference_wall_s,
    "policy_inference_batch_mean_ms": (
      1000.0 * policy_inference_wall_s / max(policy_inference_steps, 1)
    ),
    "policy_inference_env_actions_per_s": (
      raw_env.num_envs * policy_inference_steps / max(policy_inference_wall_s, 1.0e-12)
    ),
  }
  if cfg.include_per_env:
    result["per_env"] = {
      "terrain_type_id": terrain_type_ids.cpu().tolist(),
      "terrain_level": terrain_levels.cpu().tolist(),
      "stair_edge_cohort_id": terrain_cohorts.cpu().tolist(),
      "fall_pose_id": getattr(
        raw_env,
        "_robust_reset_type",
        torch.full_like(terrain_type_ids, -1),
      )
      .cpu()
      .tolist(),
      "plate_present": plate_present.cpu().tolist(),
      "plate_mass_kg": plate_mass.cpu().tolist(),
      "plate_friction": plate_friction.cpu().tolist(),
      "plate_longitudinal_offset_m": plate_longitudinal_offset.cpu().tolist(),
      "plate_lateral_offset_m": plate_lateral_offset.cpu().tolist(),
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
      "invalid_dynamics": invalid_dynamics.cpu().tolist(),
      "terrain_exit": terrain_exit.cpu().tolist(),
      "escape_first_step": escape_first.cpu().tolist(),
      "invalid_escape_setup": invalid_escape_setup.cpu().tolist(),
      "invalid_escape_contact": invalid_escape_contact.cpu().tolist(),
      "hand_support": hand_support.cpu().tolist(),
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
