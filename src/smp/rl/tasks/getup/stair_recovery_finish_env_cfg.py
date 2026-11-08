"""Stair adaptation that emphasizes completing recovery under the motion prior."""

from mjlab.managers.metrics_manager import MetricsTermCfg

from smp.rl.tasks.getup.mdp.rewards import _head_height, upright_posture
from smp.rl.tasks.getup.stair_recovery_93d_env_cfg import stair_recovery_93d_env_cfg


def final_posture_progress(env):
  """Fixed final target, independent of the intermediate recovery-stage label.

  Reaching a seated waypoint cannot saturate this score. It remains inside the
  existing SMP product rather than introducing a second prior evaluation.
  """
  height = _head_height(env, relative_to_env_origin=True)
  upright = upright_posture(env, power=1.0)
  height_progress = ((height - 0.65) / 0.50).clamp(0, 1).square()
  upright_progress = ((upright - 0.60) / 0.30).clamp(0, 1)
  return height_progress * upright_progress


def stair_recovery_finish_env_cfg(play=False):
  cfg = stair_recovery_93d_env_cfg(play)
  product = cfg.rewards["task_smp_product"]
  product.weight = 4.0
  product.params["task_terms"] = (
    *product.params["task_terms"],
    (final_posture_progress, 0.50, {}),
  )
  cfg.rewards["recovery_initiation"].weight = 0.03
  cfg.rewards["prone_support_route"].weight = 0.03
  cfg.metrics["final_posture_progress"] = MetricsTermCfg(func=final_posture_progress)
  return cfg
