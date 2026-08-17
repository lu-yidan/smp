"""Supine-and-prone geometry-aware guided-board escape curriculum."""

from __future__ import annotations

import os

from mjlab.managers.metrics_manager import MetricsTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.escape_v33_env_cfg import (
  g1_getup_escape_plate_v33_smp_env_cfg,
)

_PLAY_RESET_POSE_ENV = "SMP_PLAY_ESCAPE_RESET_POSE"
_POSE_WEIGHTS = {
  "mixed": (1.0, 1.0, 0.0, 0.0),
  "supine": (1.0, 0.0, 0.0, 0.0),
  "prone": (0.0, 1.0, 0.0, 0.0),
}


def g1_getup_escape_plate_v34_smp_env_cfg(play: bool = False):
  """Build V3.4 by adding physically grounded supine plate resets to V3.3."""
  cfg = g1_getup_escape_plate_v33_smp_env_cfg(play=play)
  reset_pose = os.environ.get(_PLAY_RESET_POSE_ENV, "mixed") if play else "mixed"
  if reset_pose not in _POSE_WEIGHTS:
    choices = ", ".join(_POSE_WEIGHTS)
    raise ValueError(f"unknown escape reset pose {reset_pose!r}; choose {choices}")

  cfg.events["mixed_fall_reset"].params.update(
    {
      "procedural_probability": 1.0,
      "mode_weights": _POSE_WEIGHTS[reset_pose],
    }
  )
  cfg.events["reset_escape_obstacle"].params["eligible_reset_types"] = (1, 2)
  cfg.metrics["supine_reset"] = MetricsTermCfg(func=mdp.supine_reset_metric)
  return cfg
