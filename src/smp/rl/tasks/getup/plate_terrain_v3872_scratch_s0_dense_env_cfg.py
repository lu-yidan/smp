"""V3.8.7.2 task-first dense curriculum for deployable scratch recovery."""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.plate_terrain_v387_scratch_s0_env_cfg import (
  g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg,
)


def g1_getup_plate_terrain_v3872_scratch_s0_dense_deploy_smp_env_cfg(
  play: bool = False,
):
  """Learn task progress before restoring the full adversarial prior weight.

  V3.8.7.1 increased its SMP product while upright and recovery initiation
  collapsed. This gate keeps the same four-frame deployable actor interface,
  but prevents the motion prior from rewarding a natural-looking static fall.
  """
  cfg = g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg(play=play)
  if play:
    return cfg

  cfg.rewards["task_smp_product"].weight = 0.05
  cfg.rewards["recovery_initiation"].weight = 0.60
  cfg.rewards["prone_support_route"].weight = 0.08
  cfg.rewards["scratch_recovery_stage"].weight = 1.00
  cfg.rewards["scratch_stable_stand"].weight = 2.00
  cfg.rewards["scratch_head_height_progress"] = RewardTermCfg(
    func=mdp.track_head_height,
    weight=0.25,
    params={
      "target_height": 1.10,
      "scale": 1.5,
      "relative_to_env_origin": True,
    },
  )
  cfg.rewards["scratch_upright_progress"] = RewardTermCfg(
    func=mdp.upright_posture,
    weight=0.20,
    params={"power": 1.0},
  )
  return cfg


__all__ = [
  "g1_getup_plate_terrain_v3872_scratch_s0_dense_deploy_smp_env_cfg",
]
