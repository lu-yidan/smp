"""SMP baseline and robust get-up task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.getup.getup_env_cfg import g1_getup_smp_env_cfg
from smp.rl.tasks.getup.robust_env_cfg import g1_getup_robust_smp_env_cfg
from smp.rl.tasks.getup.safe_env_cfg import g1_getup_robust_safe_smp_env_cfg
from smp.rl.tasks.getup.smooth_env_cfg import g1_getup_robust_smooth_smp_env_cfg
from smp.rl.tasks.getup.staged_env_cfg import g1_getup_robust_staged_smp_env_cfg

_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_getup_rl.experiment_name = "smp_getup_g1"
_getup_rl.run_name = "smp_getup_g1"

register_mjlab_task(
  task_id="Smp-Getup-G1",
  env_cfg=g1_getup_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_smp_env_cfg(play=True),
  rl_cfg=_getup_rl,
)

_robust_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_robust_getup_rl.experiment_name = "smp_getup_robust_g1"
_robust_getup_rl.run_name = "smp_getup_robust_g1"

register_mjlab_task(
  task_id="Smp-Getup-Robust-G1",
  env_cfg=g1_getup_robust_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_robust_smp_env_cfg(play=True),
  rl_cfg=_robust_getup_rl,
)

_smooth_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_smooth_getup_rl.experiment_name = "smp_getup_robust_smooth_g1"
_smooth_getup_rl.run_name = "smp_getup_robust_smooth_g1"

register_mjlab_task(
  task_id="Smp-Getup-Robust-Smooth-G1",
  env_cfg=g1_getup_robust_smooth_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_robust_smooth_smp_env_cfg(play=True),
  rl_cfg=_smooth_getup_rl,
)

_staged_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_staged_getup_rl.experiment_name = "smp_getup_robust_smooth_v4_g1"
_staged_getup_rl.run_name = "smp_getup_robust_smooth_v4_g1"
_staged_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Robust-Smooth-V4-G1",
  env_cfg=g1_getup_robust_staged_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_robust_staged_smp_env_cfg(play=True),
  rl_cfg=_staged_getup_rl,
)

_safe_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_safe_getup_rl.experiment_name = "smp_getup_robust_smooth_v5_g1"
_safe_getup_rl.run_name = "smp_getup_robust_smooth_v5_g1"
_safe_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Robust-Smooth-V5-G1",
  env_cfg=g1_getup_robust_safe_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_robust_safe_smp_env_cfg(play=True),
  rl_cfg=_safe_getup_rl,
)

__all__ = [
  "g1_getup_robust_smp_env_cfg",
  "g1_getup_robust_safe_smp_env_cfg",
  "g1_getup_robust_smooth_smp_env_cfg",
  "g1_getup_robust_staged_smp_env_cfg",
  "g1_getup_smp_env_cfg",
]
