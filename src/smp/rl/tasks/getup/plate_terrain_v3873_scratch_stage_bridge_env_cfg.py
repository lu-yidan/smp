"""V3.8.7.3 seated-crouched-standing bridge for scratch recovery."""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.plate_terrain_v3872_scratch_s0_dense_env_cfg import (
  g1_getup_plate_terrain_v3872_scratch_s0_dense_deploy_smp_env_cfg,
)


def g1_getup_plate_terrain_v3873_scratch_stage_bridge_deploy_smp_env_cfg(
  play: bool = False,
):
  """Replace the upright-only shortcut with ordered recovery waypoints.

  V3.8.7.2 reached high torso uprightness with nearly straight knees and never
  entered the seated waypoint. This refinement directly rewards the existing
  stage-conditioned seated, crouched, and standing pose and velocity targets.
  """
  cfg = g1_getup_plate_terrain_v3872_scratch_s0_dense_deploy_smp_env_cfg(
    play=play
  )
  if play:
    return cfg

  # Remove the two shortcut terms that can be maximized without bending the
  # knees or completing the ordered recovery route.
  cfg.rewards["scratch_head_height_progress"].weight = 0.0
  cfg.rewards["scratch_upright_progress"].weight = 0.0
  cfg.rewards["recovery_initiation"].weight = 0.30

  cfg.rewards["scratch_staged_pose"] = RewardTermCfg(
    func=mdp.staged_recovery_pose,
    weight=1.20,
    params={"relative_to_env_origin": True},
  )
  cfg.rewards["scratch_staged_velocity"] = RewardTermCfg(
    func=mdp.staged_head_velocity_profile,
    weight=0.15,
    params={"relative_to_env_origin": True},
  )
  cfg.rewards["scratch_recovery_stage"].weight = 2.00
  cfg.rewards["scratch_stable_stand"].weight = 3.00

  # Keep contact-rich recovery possible while tightening the unsafe trend seen
  # in the frozen model-1998 audit.
  cfg.rewards["joint_speed_excess"].weight = -0.03
  cfg.rewards["joint_power_excess"].weight = -3.0e-6
  return cfg


__all__ = [
  "g1_getup_plate_terrain_v3873_scratch_stage_bridge_deploy_smp_env_cfg",
]
