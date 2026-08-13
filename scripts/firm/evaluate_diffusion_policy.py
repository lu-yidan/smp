"""Evaluate the FIRM action diffusion model in closed-loop simulation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from evaluate_expert import _aggregate, _masked_max
from mjlab.tasks.tracking.mdp.metrics import (
  compute_mpkpe,
  compute_root_relative_mpkpe,
)

import smp.rl.tasks  # noqa: F401
from smp.firm.action_diffusion import (
  denormalize_actions,
  load_action_diffusion_checkpoint,
  normalize_action_condition,
  sample_action_horizon,
)
from smp.firm.deterministic_actor import load_deterministic_actor_checkpoint
from smp.firm.expert_runtime import (
  actor_base_observation,
  create_expert_runtime,
  runtime_metadata,
  sha256_file,
)
from smp.firm.goal_adapter import (
  load_goal_adapter_checkpoint,
  retrieve_adapter_goal,
  retrieve_nearest_route_goal,
)
from smp.rl.tasks.firm.keyframe_env_cfg import MOTION_FILE

TASK_ID = "Firm-Keyframe-G1"


@dataclass(frozen=True)
class EvaluateDiffusionPolicyConfig:
  """Fixed-start closed-loop diffusion-policy evaluation configuration."""

  action_checkpoint_file: str | None = None
  deterministic_checkpoint_file: str | None = None
  hybrid_disagreement_threshold: float = 0.25
  """Use deterministic action when normalized first-action RMSE exceeds this."""
  adapter_checkpoint_file: str | None = None
  adapter_goal_refresh_steps: int = 5
  """Retrieve a codebook goal every N control steps when an adapter is used."""
  nearest_route_lookahead: int | None = None
  """Diagnostic pose-route selector; mutually exclusive with an adapter."""
  expert_checkpoint_file: str | None = None
  """Stage-0 expert checkpoint used only to construct the matched runtime."""
  expert_wandb_run_path: str | None = None
  expert_wandb_checkpoint_name: str | None = None
  motion_file: str = MOTION_FILE
  start_frame: int | None = None
  """Optional exact start frame; requires num_start_frames=1."""
  num_start_frames: int = 25
  episodes_per_frame: int = 32
  max_steps: int = 500
  standing_hold_steps: int = 25
  root_height_threshold: float = 0.65
  upright_threshold: float = 0.85
  root_linear_speed_threshold: float = 0.50
  root_angular_speed_threshold: float = 0.50
  observation_corruption: bool = True
  use_ema: bool = True
  action_execution_steps: int = 1
  """Number of sampled horizon actions executed before replanning."""
  num_action_samples: int = 1
  """Independent DDPM horizons averaged at each replanning step."""
  seed: int = 42
  device: str | None = None
  output_file: str | None = None
  log_root: str = "logs/rsl_rl"


def run_evaluation(cfg: EvaluateDiffusionPolicyConfig) -> dict:
  """Run receding-horizon diffusion inference, executing its first action."""
  if cfg.max_steps <= 0 or cfg.standing_hold_steps <= 0:
    raise ValueError("max_steps and standing_hold_steps must be positive")
  if cfg.num_action_samples <= 0:
    raise ValueError("num_action_samples must be positive")
  if cfg.adapter_goal_refresh_steps <= 0:
    raise ValueError("adapter_goal_refresh_steps must be positive")
  has_diffusion = cfg.action_checkpoint_file is not None
  has_deterministic = cfg.deterministic_checkpoint_file is not None
  if not has_diffusion and not has_deterministic:
    raise ValueError("provide an action checkpoint, a deterministic checkpoint, or both")
  hybrid = has_diffusion and has_deterministic
  if cfg.hybrid_disagreement_threshold < 0:
    raise ValueError("hybrid_disagreement_threshold must be non-negative")
  if has_deterministic and (
    cfg.adapter_checkpoint_file is not None
    or cfg.nearest_route_lookahead is not None
    or cfg.action_execution_steps != 1
  ):
    raise ValueError("deterministic and hybrid evaluation require a fixed one-step goal")
  if has_deterministic and not has_diffusion and cfg.num_action_samples != 1:
    raise ValueError("deterministic-only evaluation requires num_action_samples=1")
  if cfg.nearest_route_lookahead is not None:
    if cfg.nearest_route_lookahead < 0:
      raise ValueError("nearest_route_lookahead must be non-negative")
    if cfg.adapter_checkpoint_file is not None:
      raise ValueError(
        "nearest_route_lookahead is mutually exclusive with an adapter"
      )
  if cfg.start_frame is not None:
    if cfg.start_frame < 0:
      raise ValueError("start_frame must be non-negative")
    if cfg.num_start_frames != 1:
      raise ValueError("an exact start_frame requires num_start_frames=1")

  runtime = create_expert_runtime(
    task_id=TASK_ID,
    motion_file=cfg.motion_file,
    checkpoint_file=cfg.expert_checkpoint_file,
    wandb_run_path=cfg.expert_wandb_run_path,
    wandb_checkpoint_name=cfg.expert_wandb_checkpoint_name,
    log_root=cfg.log_root,
    num_start_frames=cfg.num_start_frames,
    episodes_per_frame=cfg.episodes_per_frame,
    seed=cfg.seed,
    device=cfg.device,
    observation_corruption=cfg.observation_corruption,
    start_frame_range=(
      (cfg.start_frame, cfg.start_frame) if cfg.start_frame is not None else None
    ),
  )
  env = runtime.env
  raw_env = env.unwrapped
  robot = raw_env.scene["robot"]
  command = runtime.command
  device = torch.device(env.device)
  diffusion_model = None
  deterministic_model = None
  scheduler = None
  diffusion_checkpoint = None
  deterministic_checkpoint = None
  diffusion_path = None
  deterministic_path = None
  statistics: dict[str, torch.Tensor] | None = None
  model_horizon = 1
  if has_diffusion:
    assert cfg.action_checkpoint_file is not None
    diffusion_model, scheduler, statistics, diffusion_checkpoint = (
      load_action_diffusion_checkpoint(
        cfg.action_checkpoint_file,
        device,
        use_ema=cfg.use_ema,
      )
    )
    model_horizon = diffusion_model.horizon
    diffusion_path = Path(cfg.action_checkpoint_file).expanduser().resolve()
  if has_deterministic:
    assert cfg.deterministic_checkpoint_file is not None
    deterministic_model, deterministic_statistics, deterministic_checkpoint = (
      load_deterministic_actor_checkpoint(cfg.deterministic_checkpoint_file, device)
    )
    deterministic_path = (
      Path(cfg.deterministic_checkpoint_file).expanduser().resolve()
    )
    if statistics is None:
      statistics = deterministic_statistics
    elif any(
      not torch.allclose(statistics[name], deterministic_statistics[name])
      for name in statistics
    ):
      env.close()
      raise ValueError("diffusion and deterministic normalization tensors differ")
  assert statistics is not None
  if not 1 <= cfg.action_execution_steps <= model_horizon:
    env.close()
    raise ValueError(f"action_execution_steps must be in [1, {model_horizon}]")
  adapter = None
  adapter_payload = None
  if cfg.adapter_checkpoint_file is not None:
    adapter, adapter_payload = load_goal_adapter_checkpoint(
      cfg.adapter_checkpoint_file, device
    )
    expected_action_hash = adapter_payload["artifacts"][
      "action_checkpoint_sha256"
    ]
    assert diffusion_path is not None
    actual_action_hash = sha256_file(diffusion_path)
    if expected_action_hash != actual_action_hash:
      env.close()
      raise ValueError(
        "adapter/action checkpoint mismatch: "
        f"expected {expected_action_hash}, got {actual_action_hash}"
      )
  route_goals = None
  if cfg.nearest_route_lookahead is not None:
    route_goals = command.motion.joint_pos[command.keyframe_indices].clone()
  selector_size = (
    len(adapter_payload["codebook_goals"]) if adapter_payload is not None
    else 0 if route_goals is None else len(route_goals)
  )
  n = env.num_envs

  active_steps = torch.zeros(n, dtype=torch.long, device=device)
  done = torch.zeros(n, dtype=torch.bool, device=device)
  success = torch.zeros_like(done)
  gate_activations = torch.zeros(n, dtype=torch.long, device=device)
  gate_decisions = torch.zeros(n, dtype=torch.long, device=device)
  gate_disagreement_sum = torch.zeros(n, device=device)
  gate_disagreement_max = torch.zeros(n, device=device)
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
  sampling_seconds = 0.0
  sampled_windows = 0
  normalized_horizon: torch.Tensor | None = None
  observation_history: torch.Tensor | None = None
  retrieved_goal = command.joint_pos.clone()
  retrieved_index = torch.full((n,), -1, dtype=torch.long, device=device)
  retrieval_score_sum = torch.zeros(n, device=device)
  retrieval_count = torch.zeros(n, dtype=torch.long, device=device)
  retrieval_switches = torch.zeros(n, dtype=torch.long, device=device)
  retrieval_histogram = torch.zeros(
    selector_size,
    dtype=torch.long,
    device=device,
  )

  obs = env.get_observations()
  try:
    for step in range(cfg.max_steps):
      active = ~done
      if not active.any():
        break

      state_observation = actor_base_observation(obs)
      if observation_history is None:
        history_steps = 50 if adapter is None else adapter.history_steps
        observation_history = state_observation[:, None, :].expand(
          -1, history_steps, -1
        ).clone()
      else:
        observation_history = torch.roll(observation_history, shifts=-1, dims=1)
        observation_history[:, -1] = state_observation
      if adapter is not None and step % cfg.adapter_goal_refresh_steps == 0:
        assert adapter_payload is not None
        new_goal, new_index, similarity = retrieve_adapter_goal(
          adapter, observation_history, adapter_payload
        )
        retrieval_switches += (
          active & (retrieved_index >= 0) & (new_index != retrieved_index)
        ).long()
        retrieved_goal = new_goal
        retrieved_index = new_index
        retrieval_score_sum += torch.where(active, similarity, 0.0)
        retrieval_count += active.long()
        retrieval_histogram.scatter_add_(0, new_index, active.long())
      elif (
        route_goals is not None
        and step % cfg.adapter_goal_refresh_steps == 0
      ):
        assert cfg.nearest_route_lookahead is not None
        new_goal, new_index, similarity = retrieve_nearest_route_goal(
          state_observation[:, 3:32],
          route_goals,
          statistics["joint_mean"],
          statistics["joint_std"],
          cfg.nearest_route_lookahead,
        )
        retrieval_switches += (
          active & (retrieved_index >= 0) & (new_index != retrieved_index)
        ).long()
        retrieved_goal = new_goal
        retrieved_index = new_index
        retrieval_score_sum += torch.where(active, similarity, 0.0)
        retrieval_count += active.long()
        retrieval_histogram.scatter_add_(0, new_index, active.long())

      action_index = step % cfg.action_execution_steps
      if action_index == 0:
        conditioning_goal = (
          retrieved_goal
          if adapter is not None or route_goals is not None
          else command.joint_pos
        )
        normalized_observation, current_joint, normalized_goal = (
          normalize_action_condition(state_observation, conditioning_goal, statistics)
        )
        if device.type == "cuda":
          torch.cuda.synchronize(device)
        sample_start = time.perf_counter()
        deterministic_action = None
        if deterministic_model is not None:
          deterministic_action = deterministic_model(
            normalized_observation, current_joint, normalized_goal
          )
        if diffusion_model is None:
          assert deterministic_action is not None
          normalized_horizon = deterministic_action[:, None, :]
        else:
          assert scheduler is not None
          diffusion_observation = normalized_observation
          diffusion_current_joint = current_joint
          diffusion_goal = normalized_goal
          if cfg.num_action_samples > 1:
            diffusion_observation = diffusion_observation.repeat_interleave(
              cfg.num_action_samples, dim=0
            )
            diffusion_current_joint = diffusion_current_joint.repeat_interleave(
              cfg.num_action_samples, dim=0
            )
            diffusion_goal = diffusion_goal.repeat_interleave(
              cfg.num_action_samples, dim=0
            )
          normalized_horizon = sample_action_horizon(
            diffusion_model,
            scheduler,
            diffusion_observation,
            diffusion_current_joint,
            diffusion_goal,
          )
          if cfg.num_action_samples > 1:
            normalized_horizon = normalized_horizon.view(
              n,
              cfg.num_action_samples,
              model_horizon,
              diffusion_model.action_dim,
            ).mean(dim=1)
          if deterministic_action is not None:
            disagreement = torch.sqrt(
              torch.mean(
                torch.square(normalized_horizon[:, 0] - deterministic_action),
                dim=-1,
              )
            )
            use_deterministic = active & (
              disagreement > cfg.hybrid_disagreement_threshold
            )
            normalized_horizon[:, 0] = torch.where(
              use_deterministic[:, None],
              deterministic_action,
              normalized_horizon[:, 0],
            )
            gate_activations += use_deterministic.long()
            gate_decisions += active.long()
            gate_disagreement_sum += torch.where(active, disagreement, 0.0)
            gate_disagreement_max = torch.maximum(
              gate_disagreement_max,
              torch.where(active, disagreement, 0.0),
            )
        if device.type == "cuda":
          torch.cuda.synchronize(device)
        sampling_seconds += time.perf_counter() - sample_start
        sampled_windows += n * cfg.num_action_samples
      assert normalized_horizon is not None
      actions = denormalize_actions(normalized_horizon[:, action_index], statistics)
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
      action_rate_sq_sum += torch.where(
        active, torch.mean(torch.square(actions - previous_action), dim=-1), 0.0
      )
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
    "policy": (
      "firm_diffusion_deterministic_gate"
      if hybrid
      else (
        "firm_action_diffusion_action_chunking"
        if has_diffusion
        else "firm_deterministic_actor"
      )
    ),
    "config": asdict(cfg),
    "artifacts": {
      **runtime_metadata(runtime),
      "action_checkpoint_file": (
        None if diffusion_path is None else str(diffusion_path)
      ),
      "action_checkpoint_sha256": (
        None if diffusion_path is None else sha256_file(diffusion_path)
      ),
      "action_checkpoint_epoch": (
        None if diffusion_checkpoint is None else int(diffusion_checkpoint["epoch"])
      ),
      "action_weights": (
        None if diffusion_checkpoint is None
        else "ema" if cfg.use_ema else "online"
      ),
      "deterministic_checkpoint_file": (
        None if deterministic_path is None else str(deterministic_path)
      ),
      "deterministic_checkpoint_sha256": (
        None if deterministic_path is None else sha256_file(deterministic_path)
      ),
      "deterministic_checkpoint_epoch": (
        None
        if deterministic_checkpoint is None
        else int(deterministic_checkpoint["epoch"])
      ),
    },
    "adapter": {
      "enabled": adapter is not None,
      "mode": (
        "adapter"
        if adapter is not None
        else "nearest_route" if route_goals is not None else "fixed"
      ),
      "checkpoint_file": cfg.adapter_checkpoint_file,
      "checkpoint_sha256": (
        sha256_file(cfg.adapter_checkpoint_file)
        if cfg.adapter_checkpoint_file is not None
        else None
      ),
      "goal_refresh_steps": cfg.adapter_goal_refresh_steps,
      "mean_retrieval_similarity": float(
        (retrieval_score_sum / retrieval_count.clamp(min=1)).mean()
      ),
      "mean_goal_switches": float(retrieval_switches.float().mean()),
      "goal_index_counts": retrieval_histogram.tolist(),
    },
    "hybrid_gate": {
      "enabled": hybrid,
      "threshold": cfg.hybrid_disagreement_threshold,
      "activation_rate": float(
        gate_activations.sum() / gate_decisions.sum().clamp(min=1)
      ),
      "mean_normalized_action_rmse": float(
        gate_disagreement_sum.sum() / gate_decisions.sum().clamp(min=1)
      ),
      "max_normalized_action_rmse": float(gate_disagreement_max.max()),
    },
    "inference": {
      "sampled_windows": sampled_windows,
      "sampling_seconds": sampling_seconds,
      "windows_per_second": sampled_windows / max(sampling_seconds, 1.0e-9),
      "ddpm_steps_per_window": None if scheduler is None else scheduler.num_timesteps,
      "executed_actions_per_window": cfg.action_execution_steps,
      "averaged_action_samples": cfg.num_action_samples,
    },
    **aggregates,
  }
  print(
    json.dumps(
      {"overall": result["overall"], "inference": result["inference"]}, indent=2
    )
  )
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[INFO] Evaluation written to {output_path}")
  return result


def main() -> None:
  run_evaluation(tyro.cli(EvaluateDiffusionPolicyConfig))


if __name__ == "__main__":
  main()
