"""V3.4 plate-escape task with only base linear velocity removed from the actor."""

from __future__ import annotations

from smp.rl.tasks.getup.escape_v34_env_cfg import (
  g1_getup_escape_plate_v34_smp_env_cfg,
)


def g1_getup_escape_plate_v34_93d_smp_env_cfg(play: bool = False):
  """Build the matched 93D V3.4 observation ablation.

  All V3.4 reset, plate, reward, stage, replay, termination, timing, critic,
  and PPO contracts are inherited unchanged.  The sole environment change is
  removal of the three-dimensional true base linear velocity actor term.
  """
  cfg = g1_getup_escape_plate_v34_smp_env_cfg(play=play)
  actor = cfg.observations["actor"]
  removed = actor.terms.pop("base_lin_vel", None)
  if removed is None:
    raise RuntimeError("V3.4 actor no longer contains base_lin_vel")
  return cfg


__all__ = ["g1_getup_escape_plate_v34_93d_smp_env_cfg"]
