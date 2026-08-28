"""V3.8.7.4 phase-balanced GSI and LAFAN milestone recovery curriculum."""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.plate_terrain_v3873_scratch_stage_bridge_env_cfg import (
  g1_getup_plate_terrain_v3873_scratch_stage_bridge_deploy_smp_env_cfg,
)

# mixed_fall_reset first keeps 50% GSI. Milestones then replace 20% of all
# resets, producing approximately 40% GSI, 40% procedural falls, 20% milestones.
MIXED_FALL_PROCEDURAL_PROBABILITY = 0.50
LAFAN_MILESTONE_PROBABILITY = 0.20
LAFAN_MILESTONE_STAGE_WEIGHTS = (0.45, 0.35, 0.20)


def _insert_after_event(cfg, after: str, name: str, term) -> None:
  reordered = {}
  inserted = False
  for event_name, event_term in cfg.events.items():
    reordered[event_name] = event_term
    if event_name == after:
      reordered[name] = term
      inserted = True
  if not inserted:
    raise RuntimeError(f"required reset event {after!r} is missing")
  cfg.events = reordered


def g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg(
  play: bool = False,
):
  """Bridge long-horizon exploration without adding actor privileges."""
  cfg = g1_getup_plate_terrain_v3873_scratch_stage_bridge_deploy_smp_env_cfg(
    play=play
  )
  if play:
    return cfg

  cfg.events["mixed_fall_reset"].params["procedural_probability"] = (
    MIXED_FALL_PROCEDURAL_PROBABILITY
  )
  _insert_after_event(
    cfg,
    "reset_recovery_stage",
    "lafan_milestone_reset",
    EventTermCfg(
      func=mdp.lafan_milestone_reset,
      mode="reset",
      params={
        "probability": LAFAN_MILESTONE_PROBABILITY,
        "stage_weights": LAFAN_MILESTONE_STAGE_WEIGHTS,
      },
    ),
  )

  # Remove stage-occupancy farming. Pay mostly for transitions and sustained
  # standing, while a two-sided knee band prevents permanent deep crouching.
  cfg.rewards["scratch_staged_pose"].func = mdp.staged_recovery_pose_band
  cfg.rewards["scratch_staged_pose"].weight = 1.20
  cfg.rewards["scratch_recovery_stage"].weight = 0.20
  cfg.rewards["scratch_stage_transition"] = RewardTermCfg(
    func=mdp.recovery_stage_transition_reward,
    weight=4.00,
  )
  cfg.rewards["scratch_stable_stand"].weight = 4.00
  cfg.rewards["recovery_initiation"].weight = 0.15

  # Success is a state to retain with the same policy, not an early reset or a
  # hand-off to a separate balance controller.
  cfg.terminations.pop("stood_up", None)

  cfg.metrics.update(
    {
      "gsi_reset": MetricsTermCfg(func=mdp.gsi_reset_metric),
      "lafan_milestone_reset": MetricsTermCfg(func=mdp.lafan_milestone_reset_metric),
      "lafan_milestone_stage": MetricsTermCfg(func=mdp.lafan_milestone_stage_metric),
    }
  )
  return cfg


__all__ = [
  "g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg",
]
