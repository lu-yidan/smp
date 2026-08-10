"""Collect expert action chunks at on-policy diffusion failure states."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from collect_rollouts import ShardWriter, _cpu, _prepare_output

import smp.rl.tasks  # noqa: F401
from smp.firm.action_diffusion import (
  denormalize_actions,
  load_action_diffusion_checkpoint,
  normalize_action_condition,
  sample_action_horizon,
)
from smp.firm.expert_runtime import (
  actor_base_observation,
  create_expert_runtime,
  runtime_metadata,
  sha256_file,
)
from smp.rl.tasks.firm.keyframe_env_cfg import MOTION_FILE

TASK_ID = "Firm-Keyframe-G1"


@dataclass(frozen=True)
class CollectOnPolicyCorrectionsConfig:
  """SafeDAgger-style diffusion rollout and expert-intervention settings."""

  action_checkpoint_file: str
  expert_checkpoint_file: str | None = None
  expert_wandb_run_path: str | None = None
  expert_wandb_checkpoint_name: str | None = None
  motion_file: str = MOTION_FILE
  num_start_frames: int = 17
  start_frame_range: tuple[int, int] | None = (0, 324)
  episodes_per_frame: int = 4
  horizon: int = 12
  disagreement_threshold: float = 0.12
  """Trigger when diffusion/expert action RMSE reaches this value."""
  cooldown_steps: int = 20
  """Diffusion-only steps required after a completed intervention."""
  max_steps: int = 500
  max_windows: int = 0
  """Maximum saved corrections; zero keeps all available corrections."""
  observation_corruption: bool = True
  use_ema: bool = True
  num_action_samples: int = 1
  """Independent DDPM horizons averaged before disagreement is measured."""
  seed: int = 42
  device: str | None = None
  output_dir: str = "datasets/firm/rollouts/c003_onpolicy_corrections"
  shard_size: int = 50_000
  log_root: str = "logs/rsl_rl"


def _validate(cfg: CollectOnPolicyCorrectionsConfig) -> None:
  if cfg.horizon <= 0 or cfg.max_steps <= 0:
    raise ValueError("horizon and max_steps must be positive")
  if cfg.disagreement_threshold < 0.0:
    raise ValueError("disagreement_threshold must be non-negative")
  if cfg.cooldown_steps < 0 or cfg.max_windows < 0:
    raise ValueError("cooldown_steps and max_windows must be non-negative")
  if cfg.num_action_samples <= 0:
    raise ValueError("num_action_samples must be positive")


@torch.inference_mode()
def _diffusion_action(
  observation: torch.Tensor,
  goal: torch.Tensor,
  *,
  model: torch.nn.Module,
  scheduler: object,
  statistics: dict[str, torch.Tensor],
  num_samples: int,
) -> torch.Tensor:
  normalized_observation, current_joint, normalized_goal = normalize_action_condition(
    observation, goal, statistics
  )
  batch = observation.shape[0]
  if num_samples > 1:
    normalized_observation = normalized_observation.repeat_interleave(
      num_samples, dim=0
    )
    current_joint = current_joint.repeat_interleave(num_samples, dim=0)
    normalized_goal = normalized_goal.repeat_interleave(num_samples, dim=0)
  horizon = sample_action_horizon(
    model,
    scheduler,
    normalized_observation,
    current_joint,
    normalized_goal,
  )
  if num_samples > 1:
    horizon = horizon.view(batch, num_samples, model.horizon, model.action_dim).mean(
      dim=1
    )
  return denormalize_actions(horizon[:, 0], statistics)


def _select_triggers(
  candidates: torch.Tensor,
  disagreement: torch.Tensor,
  available: int | None,
) -> torch.Tensor:
  ids = torch.where(candidates)[0]
  if available is None or ids.numel() <= available:
    return ids
  if available <= 0:
    return ids[:0]
  order = torch.topk(disagreement[ids], k=available).indices
  return ids[order]


def run_collection(cfg: CollectOnPolicyCorrectionsConfig) -> dict:
  """Roll out diffusion and save coherent expert corrections at disagreements."""
  _validate(cfg)
  output_dir = Path(cfg.output_dir).expanduser()
  _prepare_output(output_dir)

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
    start_frame_range=cfg.start_frame_range,
  )
  env = runtime.env
  command = runtime.command
  device = torch.device(env.device)
  model, scheduler, statistics, action_checkpoint = load_action_diffusion_checkpoint(
    cfg.action_checkpoint_file,
    device,
    use_ema=cfg.use_ema,
  )
  if model.horizon != cfg.horizon:
    env.close()
    raise ValueError(
      f"checkpoint horizon={model.horizon} does not match horizon={cfg.horizon}"
    )

  n = env.num_envs
  h = cfg.horizon
  writer = ShardWriter(output_dir, cfg.shard_size)
  episode_done = torch.zeros(n, dtype=torch.bool, device=device)
  remaining = torch.zeros(n, dtype=torch.long, device=device)
  label_count = torch.zeros_like(remaining)
  cooldown = torch.zeros_like(remaining)
  label_observation = torch.zeros(n, 90, device=device)
  label_goal = torch.zeros(n, 29, device=device)
  label_actions = torch.zeros(n, h, 29, device=device)
  label_motion_frames = torch.zeros(n, h, dtype=torch.long, device=device)
  label_goal_frame = torch.zeros(n, dtype=torch.long, device=device)
  label_start_frame = torch.zeros(n, dtype=torch.long, device=device)
  label_disagreement = torch.zeros(n, device=device)

  records: list[dict[str, int | float | bool]] = []
  completed_windows = 0
  started_windows = 0
  discarded_partial = 0
  trigger_values: list[float] = []
  obs = env.get_observations()

  try:
    for step in range(cfg.max_steps):
      active = ~episode_done
      if not active.any():
        break
      state_observation = actor_base_observation(obs)
      goal = command.joint_pos.clone()
      diffusion_action = _diffusion_action(
        state_observation,
        goal,
        model=model,
        scheduler=scheduler,
        statistics=statistics,
        num_samples=cfg.num_action_samples,
      )
      expert_action = runtime.policy(obs)
      if env.clip_actions is not None:
        diffusion_action = diffusion_action.clamp(-env.clip_actions, env.clip_actions)
        expert_action = expert_action.clamp(-env.clip_actions, env.clip_actions)
      disagreement = torch.sqrt(
        torch.mean(torch.square(diffusion_action - expert_action), dim=-1)
      )

      idle = active & (remaining == 0)
      cooldown = torch.where(idle & (cooldown > 0), cooldown - 1, cooldown)
      stable_goal_steps = command.goal_steps - command.time_steps
      candidates = (
        idle
        & (cooldown == 0)
        & (stable_goal_steps >= h)
        & (disagreement >= cfg.disagreement_threshold)
      )
      available: int | None = None
      if cfg.max_windows > 0:
        reserved = completed_windows + int((remaining > 0).sum().item())
        available = max(cfg.max_windows - reserved, 0)
      trigger_ids = _select_triggers(candidates, disagreement, available)
      if trigger_ids.numel() > 0:
        label_observation[trigger_ids] = state_observation[trigger_ids]
        label_goal[trigger_ids] = goal[trigger_ids]
        label_goal_frame[trigger_ids] = command.goal_steps[trigger_ids]
        label_start_frame[trigger_ids] = runtime.env_start_frames[trigger_ids]
        label_disagreement[trigger_ids] = disagreement[trigger_ids]
        label_count[trigger_ids] = 0
        remaining[trigger_ids] = h
        started_windows += trigger_ids.numel()

      intervention_ids = torch.where(active & (remaining > 0))[0]
      actions = diffusion_action.clone()
      if intervention_ids.numel() > 0:
        indices = label_count[intervention_ids]
        label_actions[intervention_ids, indices] = expert_action[intervention_ids]
        label_motion_frames[intervention_ids, indices] = command.time_steps[
          intervention_ids
        ]
        actions[intervention_ids] = expert_action[intervention_ids]

      next_obs, _, dones, _ = env.step(actions)
      if intervention_ids.numel() > 0:
        label_count[intervention_ids] += 1
        remaining[intervention_ids] -= 1

      transition_done = dones.bool() & active
      completed = (label_count == h) & (remaining == 0) & ~transition_done
      completed_ids = torch.where(completed)[0]
      if completed_ids.numel() > 0:
        count = completed_ids.numel()
        episode_ids = torch.arange(
          completed_windows,
          completed_windows + count,
          dtype=torch.long,
          device=device,
        )
        repeated_episode_ids = episode_ids.repeat_interleave(h)
        episode_steps = torch.arange(h, device=device).repeat(count)
        done_flags = torch.zeros(count, h, dtype=torch.bool, device=device)
        done_flags[:, -1] = True
        zeros = torch.zeros(count * h, dtype=torch.bool, device=device)
        trigger_rmse = label_disagreement[completed_ids]
        writer.add(
          {
            "observation": _cpu(
              label_observation[completed_ids].repeat_interleave(h, dim=0),
              torch.float32,
            ),
            "goal": _cpu(
              label_goal[completed_ids].repeat_interleave(h, dim=0), torch.float32
            ),
            "action": _cpu(
              label_actions[completed_ids].reshape(count * h, 29), torch.float32
            ),
            "episode_id": _cpu(repeated_episode_ids, torch.int64),
            "episode_step": _cpu(episode_steps, torch.int32),
            "start_frame": _cpu(
              label_start_frame[completed_ids].repeat_interleave(h), torch.int32
            ),
            "source_env_id": _cpu(completed_ids.repeat_interleave(h), torch.int32),
            "motion_frame": _cpu(
              label_motion_frames[completed_ids].reshape(count * h), torch.int32
            ),
            "goal_frame": _cpu(
              label_goal_frame[completed_ids].repeat_interleave(h), torch.int32
            ),
            "trigger_action_rmse": _cpu(
              trigger_rmse.repeat_interleave(h), torch.float32
            ),
            "expert_intervention": np.ones(count * h, dtype=np.bool_),
            "transition_disturbed": _cpu(zeros, torch.bool),
            "done": _cpu(done_flags.reshape(-1), torch.bool),
            "unsafe": _cpu(zeros, torch.bool),
            "timeout": _cpu(zeros, torch.bool),
          }
        )
        for offset, env_id in enumerate(completed_ids.tolist()):
          correction_id = completed_windows + offset
          trigger_value = float(label_disagreement[env_id].item())
          trigger_values.append(trigger_value)
          records.append(
            {
              "episode_id": correction_id,
              "source_env_id": env_id,
              "start_frame": int(label_start_frame[env_id].item()),
              "motion_frame": int(label_motion_frames[env_id, 0].item()),
              "goal_frame": int(label_goal_frame[env_id].item()),
              "trigger_action_rmse": trigger_value,
              "samples": h,
              "completed": True,
              "success": True,
              "unsafe": False,
              "timeout": False,
            }
          )
        completed_windows += count
        cooldown[completed_ids] = cfg.cooldown_steps
        label_count[completed_ids] = 0

      interrupted = transition_done & (label_count > 0)
      if interrupted.any():
        discarded_partial += int(interrupted.sum().item())
      if transition_done.any():
        episode_done[transition_done] = True
        remaining[transition_done] = 0
        label_count[transition_done] = 0

      obs = next_obs
      if step % 50 == 0 or completed_ids.numel() > 0 or transition_done.any():
        print(
          f"[INFO] step={step:03d} active={int((~episode_done).sum())} "
          f"started={started_windows} saved={completed_windows} "
          f"partial_discarded={discarded_partial}"
        )
      cap_reached = cfg.max_windows > 0 and completed_windows >= cfg.max_windows
      if cap_reached and not (remaining > 0).any():
        break
  finally:
    env.close()
  writer.flush()

  action_path = Path(cfg.action_checkpoint_file).expanduser().resolve()
  trigger_array = np.asarray(trigger_values, dtype=np.float64)
  trigger_statistics = {
    "count": int(trigger_array.size),
    "mean": float(trigger_array.mean()) if trigger_array.size else None,
    "minimum": float(trigger_array.min()) if trigger_array.size else None,
    "maximum": float(trigger_array.max()) if trigger_array.size else None,
  }
  manifest = {
    "format_version": 1,
    "task_id": TASK_ID,
    "collector": "safe_dagger_expert_intervention",
    "config": asdict(cfg),
    "artifacts": {
      **runtime_metadata(runtime),
      "action_checkpoint_file": str(action_path),
      "action_checkpoint_sha256": sha256_file(action_path),
      "action_checkpoint_epoch": int(action_checkpoint["epoch"]),
      "action_weights": "ema" if cfg.use_ema else "online",
    },
    "layout": {
      "observation": {
        "shape": [90],
        "description": "diffusion on-policy state at intervention trigger",
      },
      "goal": {
        "shape": [29],
        "description": "constant sparse-keyframe goal at intervention trigger",
      },
      "action": {
        "shape": [29],
        "description": "expert action during the coherent intervention",
      },
      "ordering": "one independent horizon per episode_id",
    },
    "total_samples": writer.total,
    "episodes": completed_windows,
    "successful_episodes": completed_windows,
    "unsafe_episodes": 0,
    "disturbed_transitions": 0,
    "collection": {
      "started_interventions": started_windows,
      "saved_interventions": completed_windows,
      "discarded_partial_interventions": discarded_partial,
      "trigger_action_rmse": trigger_statistics,
    },
    "shards": writer.shards,
    "episode_records": records,
  }
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(
    f"[INFO] Correction manifest written to {manifest_path}: "
    f"{completed_windows} windows / {writer.total} samples"
  )
  return manifest


def main() -> None:
  run_collection(tyro.cli(CollectOnPolicyCorrectionsConfig))


if __name__ == "__main__":
  main()
