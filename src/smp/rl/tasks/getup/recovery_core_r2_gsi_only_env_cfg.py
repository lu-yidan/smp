"""Recovery-Core R2 reset-distribution ablation using only GSI states."""

from __future__ import annotations

from smp.rl.tasks.getup.recovery_core_r2_ordered_env_cfg import (
  g1_recovery_core_r2_ordered_smp_env_cfg,
)


def g1_recovery_core_r2_gsi_only_smp_env_cfg(play: bool = False):
  """Return ordered R2 with every reset retained from the GSI generator."""
  cfg = g1_recovery_core_r2_ordered_smp_env_cfg(play=play)
  cfg.events["mixed_fall_reset"].params["procedural_probability"] = 0.0
  return cfg


__all__ = ["g1_recovery_core_r2_gsi_only_smp_env_cfg"]
