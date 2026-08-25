"""Curriculum terms for terrain recovery."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["terrain_levels_getup"]


def _minimum_terrain_replay_levels(
  env: ManagerBasedRlEnv,
  fractions: tuple[float, ...],
  flat_col: int,
) -> torch.Tensor:
  """Build deterministic per-family difficulty floors for anti-collapse replay."""
  cached = getattr(env, "_terrain_replay_floor", None)
  if cached is not None:
    return cached
  terrain = env.scene.terrain
  if terrain is None:
    raise RuntimeError("Terrain replay floors require a terrain entity")
  if len(fractions) != terrain.max_terrain_level:
    raise ValueError("minimum_level_fractions must have one entry per terrain level")
  weights = torch.tensor(fractions, device=env.device, dtype=torch.float)
  expected_total = torch.tensor(1.0, device=env.device)
  if torch.any(weights < 0) or not torch.isclose(weights.sum(), expected_total):
    raise ValueError("minimum_level_fractions must be non-negative and sum to 1")

  floors = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  cumulative = torch.cumsum(weights, dim=0)[:-1]
  for terrain_type in torch.unique(terrain.terrain_types):
    type_index = int(terrain_type)
    if type_index == flat_col:
      continue
    ids = torch.nonzero(terrain.terrain_types == type_index, as_tuple=False).flatten()
    positions = (torch.arange(ids.numel(), device=env.device) + 0.5) / ids.numel()
    floors[ids] = torch.bucketize(positions, cumulative)
  env._terrain_replay_floor = floors  # type: ignore[attr-defined]
  return floors


def terrain_levels_getup(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stand_hold_steps: int = 25,
  success_radius: float = 1.50,
  minimum_episode_steps: int = 20,
  accept_completed_recovery_stage: bool = False,
  minimum_level_fractions: tuple[float, ...] | None = None,
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
  replay_floor = torch.zeros_like(levels)
  if minimum_level_fractions is not None:
    replay_floor = _minimum_terrain_replay_levels(
      env, minimum_level_fractions, flat_col
    )
    local_below_floor = levels[env_ids] < replay_floor[env_ids]
    if local_below_floor.any():
      floor_env_ids = env_ids[local_below_floor]
      levels[floor_env_ids] = replay_floor[floor_env_ids]
      assert terrain.env_origins is not None
      terrain.env_origins[floor_env_ids] = terrain.terrain_origins[
        levels[floor_env_ids], terrain.terrain_types[floor_env_ids]
      ]

  local_levels = levels[env_ids]
  local_floor = replay_floor[env_ids]
  move_up = success & nonflat & (local_levels < terrain.max_terrain_level - 1)
  move_down = valid & ~success & nonflat & (local_levels > local_floor) & ~move_up
  terrain.update_env_origins(env_ids, move_up, move_down)

  all_levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": all_levels.mean(),
    "max": all_levels.max(),
    "success": success.float().mean(),
    "stand_success": (valid & stable).float().mean(),
    "stage_success": (valid & completed_stage).float().mean(),
    "replay_floor_mean": replay_floor.float().mean(),
  }
  for index, name in enumerate(names):
    mask = terrain.terrain_types == index
    if mask.any():
      result[name] = all_levels[mask].mean()
  return result
