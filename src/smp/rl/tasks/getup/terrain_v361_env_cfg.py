"""V3.6.1 terrain curriculum with exploration-robust progression."""

from __future__ import annotations

from smp.rl.tasks.getup.terrain_v36_env_cfg import (
  g1_getup_terrain_v36_smp_env_cfg,
)


def g1_getup_terrain_v361_smp_env_cfg(play: bool = False):
  """Build V3.6.1 while preserving V3.6 actor inputs and objectives.

  V3.6's deterministic policies recovered reliably at level zero, but the PPO
  rollout action standard deviation prevented its separate standing counter
  from remaining true for 25 consecutive steps.  Stage three already certifies
  the ordered seated-crouched-standing holds, so V3.6.1 accepts that persistent
  achievement latch when updating terrain levels at episode reset.
  """
  cfg = g1_getup_terrain_v36_smp_env_cfg(play=play)
  if not play:
    cfg.curriculum["terrain_levels"].params["accept_completed_recovery_stage"] = True
  return cfg


__all__ = ["g1_getup_terrain_v361_smp_env_cfg"]
