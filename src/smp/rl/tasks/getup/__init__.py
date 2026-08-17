"""SMP baseline and robust get-up task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.getup.constrained_env_cfg import (
  g1_getup_constrained_smp_env_cfg,
)
from smp.rl.tasks.getup.escape_env_cfg import g1_getup_escape_smp_env_cfg
from smp.rl.tasks.getup.escape_v3_env_cfg import (
  g1_getup_escape_plate_v3_smp_env_cfg,
)
from smp.rl.tasks.getup.escape_v31_env_cfg import (
  g1_getup_escape_plate_v31_smp_env_cfg,
)
from smp.rl.tasks.getup.escape_v32_env_cfg import (
  g1_getup_escape_plate_v32_smp_env_cfg,
)
from smp.rl.tasks.getup.getup_env_cfg import g1_getup_smp_env_cfg
from smp.rl.tasks.getup.robust_env_cfg import g1_getup_robust_smp_env_cfg
from smp.rl.tasks.getup.safe_env_cfg import g1_getup_robust_safe_smp_env_cfg
from smp.rl.tasks.getup.smooth_env_cfg import g1_getup_robust_smooth_smp_env_cfg
from smp.rl.tasks.getup.staged_env_cfg import g1_getup_robust_staged_smp_env_cfg
from smp.rl.tasks.getup.v6_env_cfg import (
  g1_getup_v6_prior_smp_env_cfg,
  g1_getup_v6_smp_env_cfg,
)
from smp.rl.tasks.getup.v7_env_cfg import g1_getup_v7_route_smp_env_cfg
from smp.rl.tasks.getup.v8_env_cfg import g1_getup_v8_natural_smp_env_cfg

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

_v6_prior_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_v6_prior_getup_rl.experiment_name = "smp_getup_v6_prior_g1"
_v6_prior_getup_rl.run_name = "smp_getup_v6_prior_g1"
_v6_prior_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Robust-Smooth-V6-Prior-G1",
  env_cfg=g1_getup_v6_prior_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_v6_prior_smp_env_cfg(play=True),
  rl_cfg=_v6_prior_getup_rl,
)

_v6_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_v6_getup_rl.experiment_name = "smp_getup_v6_g1"
_v6_getup_rl.run_name = "smp_getup_v6_g1"
_v6_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Robust-Smooth-V6-G1",
  env_cfg=g1_getup_v6_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_v6_smp_env_cfg(play=True),
  rl_cfg=_v6_getup_rl,
)

_v7_route_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_v7_route_getup_rl.experiment_name = "smp_getup_v7_route_g1"
_v7_route_getup_rl.run_name = "smp_getup_v7_route_g1"
_v7_route_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Robust-Smooth-V7-Route-G1",
  env_cfg=g1_getup_v7_route_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_v7_route_smp_env_cfg(play=True),
  rl_cfg=_v7_route_getup_rl,
)

_v8_natural_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_v8_natural_getup_rl.experiment_name = "smp_getup_v8_natural_g1"
_v8_natural_getup_rl.run_name = "smp_getup_v8_natural_g1"
_v8_natural_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Robust-Smooth-V8-Natural-G1",
  env_cfg=g1_getup_v8_natural_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_v8_natural_smp_env_cfg(play=True),
  rl_cfg=_v8_natural_getup_rl,
)

_constrained_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_constrained_getup_rl.experiment_name = "smp_getup_constrained_g1"
_constrained_getup_rl.run_name = "smp_getup_constrained_g1"
_constrained_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Constrained-G1",
  env_cfg=g1_getup_constrained_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_constrained_smp_env_cfg(play=True),
  rl_cfg=_constrained_getup_rl,
)

_escape_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_escape_getup_rl.experiment_name = "smp_getup_escape_g1"
_escape_getup_rl.run_name = "smp_getup_escape_g1"
_escape_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Escape-G1",
  env_cfg=g1_getup_escape_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_escape_smp_env_cfg(play=True),
  rl_cfg=_escape_getup_rl,
)

_escape_plate_v3_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_escape_plate_v3_getup_rl.experiment_name = "smp_getup_escape_plate_v3_g1"
_escape_plate_v3_getup_rl.run_name = "smp_getup_escape_plate_v3_g1"
_escape_plate_v3_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Escape-Plate-V3-G1",
  env_cfg=g1_getup_escape_plate_v3_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_escape_plate_v3_smp_env_cfg(play=True),
  rl_cfg=_escape_plate_v3_getup_rl,
)

_escape_plate_v31_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_escape_plate_v31_getup_rl.experiment_name = "smp_getup_escape_plate_v31_g1"
_escape_plate_v31_getup_rl.run_name = "smp_getup_escape_plate_v31_g1"
_escape_plate_v31_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Escape-Plate-V31-G1",
  env_cfg=g1_getup_escape_plate_v31_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_escape_plate_v31_smp_env_cfg(play=True),
  rl_cfg=_escape_plate_v31_getup_rl,
)

_escape_plate_v32_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_escape_plate_v32_getup_rl.experiment_name = "smp_getup_escape_plate_v32_g1"
_escape_plate_v32_getup_rl.run_name = "smp_getup_escape_plate_v32_g1"
_escape_plate_v32_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Escape-Plate-V32-G1",
  env_cfg=g1_getup_escape_plate_v32_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_escape_plate_v32_smp_env_cfg(play=True),
  rl_cfg=_escape_plate_v32_getup_rl,
)

__all__ = [
  "g1_getup_robust_smp_env_cfg",
  "g1_getup_robust_safe_smp_env_cfg",
  "g1_getup_robust_smooth_smp_env_cfg",
  "g1_getup_robust_staged_smp_env_cfg",
  "g1_getup_smp_env_cfg",
  "g1_getup_constrained_smp_env_cfg",
  "g1_getup_escape_smp_env_cfg",
  "g1_getup_escape_plate_v3_smp_env_cfg",
  "g1_getup_escape_plate_v31_smp_env_cfg",
  "g1_getup_escape_plate_v32_smp_env_cfg",
  "g1_getup_v6_prior_smp_env_cfg",
  "g1_getup_v6_smp_env_cfg",
  "g1_getup_v7_route_smp_env_cfg",
  "g1_getup_v8_natural_smp_env_cfg",
]
