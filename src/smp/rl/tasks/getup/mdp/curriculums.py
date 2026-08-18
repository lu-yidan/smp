"""Curriculum terms for terrain recovery."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["terrain_levels_getup"]


def terrain_levels_getup(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stand_hold_steps: int = 25,
  success_radius: float = 1.50,
  minimum_episode_steps: int = 20,
  accept_completed_recovery_stage: bool = False,
) -> dict[str, torch.Tensor]:
  """Advance one terrain level after an anchored stable recovery.

  Failed timeouts or patch exits move down one level.  Flat-control episodes
  remain at level zero, preserving a stable replay cohort throughout training.
  Curriculum labels are training-only and never enter actor observations.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return {"mean": torch.tensor(0.0, device=env.device)}
  if env_ids.numel() == 0:
    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean(), "max": levels.max()}

  levels = terrain.terrain_levels
  episode_steps = env.episode_length_buf[env_ids]
  valid = episode_steps >= minimum_episode_steps
  stand_count = getattr(env, "_getup_stand_count", None)
  if stand_count is None:
    stable = torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device)
  else:
    stable = stand_count[env_ids] >= stand_hold_steps
  displacement = torch.linalg.vector_norm(
    env.scene["robot"].data.root_link_pos_w[env_ids, :2]
    - env.scene.env_origins[env_ids, :2],
    dim=-1,
  )
  stage = getattr(env, "_v4_recovery_stage", None)
  if accept_completed_recovery_stage and stage is not None:
    # Stage three is an episode-level achievement latch: reaching it already
    # required the seated, crouched, and standing holds in order.  Unlike the
    # instantaneous stand counter it survives exploratory action noise unless
    # the robot substantially falls again, so curriculum progress does not
    # depend on 25 additional noise-free actions at the reset boundary.
    completed_stage = stage[env_ids] >= 3
  else:
    completed_stage = torch.zeros_like(stable)
  success = valid & (stable | completed_stage) & (displacement <= success_radius)

  generator = terrain.cfg.terrain_generator
  assert generator is not None
  names = list(generator.sub_terrains)
  terrain_types = terrain.terrain_types[env_ids]
  flat_col = names.index("flat") if "flat" in names else -1
  nonflat = terrain_types != flat_col
  local_levels = levels[env_ids]
  move_up = success & nonflat & (local_levels < terrain.max_terrain_level - 1)
  move_down = valid & ~success & nonflat & (local_levels > 0) & ~move_up
  terrain.update_env_origins(env_ids, move_up, move_down)

  all_levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": all_levels.mean(),
    "max": all_levels.max(),
    "success": success.float().mean(),
    "stand_success": (valid & stable).float().mean(),
    "stage_success": (valid & completed_stage).float().mean(),
  }
  for index, name in enumerate(names):
    mask = terrain.terrain_types == index
    if mask.any():
      result[name] = all_levels[mask].mean()
  return result
