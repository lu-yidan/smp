"""Recovery-Core R2: preserve the ordered support-to-stand route."""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.recovery_core_r1_env_cfg import (
  g1_recovery_core_r1_smp_env_cfg,
)

_STABLE_STAND_PARAMS = {
  "head_height": 1.10,
  "min_upright": 0.85,
  "max_linear_speed": 0.50,
  "max_angular_speed": 1.0,
  "relative_to_env_origin": True,
}


def g1_recovery_core_r2_ordered_smp_env_cfg(play: bool = False):
  """Return R1 with standing gated by ordered recovery-stage completion."""
  cfg = g1_recovery_core_r1_smp_env_cfg(play=play)

  if "scratch_stable_stand" in cfg.rewards:
    cfg.rewards["scratch_stable_stand"].func = mdp.ordered_stable_stand_metric
    cfg.rewards["scratch_stable_stand"].params = dict(_STABLE_STAND_PARAMS)
    cfg.rewards["scratch_unordered_stand"] = RewardTermCfg(
      func=mdp.unordered_stable_stand_metric,
      weight=-2.0,
      params=dict(_STABLE_STAND_PARAMS),
    )

  return cfg


__all__ = ["g1_recovery_core_r2_ordered_smp_env_cfg"]
