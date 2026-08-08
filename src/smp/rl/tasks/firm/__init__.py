"""FIRM-style sparse-keyframe expert task registration."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_firm_expert_ppo_runner_cfg
from smp.rl.tasks.firm.keyframe_env_cfg import g1_firm_keyframe_env_cfg
from smp.rl.tasks.firm.runner import LocalMotionOnPolicyRunner

_firm_keyframe_rl = unitree_g1_firm_expert_ppo_runner_cfg()
_firm_keyframe_rl.experiment_name = "firm_keyframe_g1_c003"
_firm_keyframe_rl.run_name = "firm_keyframe_g1_c003"

register_mjlab_task(
  task_id="Firm-Keyframe-G1",
  env_cfg=g1_firm_keyframe_env_cfg(play=False),
  play_env_cfg=g1_firm_keyframe_env_cfg(play=True),
  rl_cfg=_firm_keyframe_rl,
  runner_cls=LocalMotionOnPolicyRunner,
)

__all__ = ["g1_firm_keyframe_env_cfg"]
