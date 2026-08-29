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
from smp.rl.tasks.getup.escape_v33_env_cfg import (
  g1_getup_escape_plate_v33_smp_env_cfg,
)
from smp.rl.tasks.getup.escape_v34_env_cfg import (
  g1_getup_escape_plate_v34_smp_env_cfg,
)
from smp.rl.tasks.getup.getup_env_cfg import g1_getup_smp_env_cfg
from smp.rl.tasks.getup.plate_terrain_v38_env_cfg import (
  g1_getup_plate_terrain_v38_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v381_env_cfg import (
  g1_getup_plate_terrain_v381_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v382_env_cfg import (
  g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg,
  g1_getup_plate_terrain_v382_h10_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v383_env_cfg import (
  g1_getup_plate_terrain_v383_right_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v384_env_cfg import (
  g1_getup_plate_terrain_v384_symmetric_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v386_scratch_env_cfg import (
  g1_getup_plate_terrain_v386_scratch_s1_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v387_scratch_s0_env_cfg import (
  g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v3872_scratch_s0_dense_env_cfg import (
  g1_getup_plate_terrain_v3872_scratch_s0_dense_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v3873_scratch_stage_bridge_env_cfg import (
  g1_getup_plate_terrain_v3873_scratch_stage_bridge_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.plate_terrain_v3874_gsi_milestone_env_cfg import (
  g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg,
)
from smp.rl.tasks.getup.recovery_core_r1_env_cfg import (
  g1_recovery_core_r1_smp_env_cfg,
)
from smp.rl.tasks.getup.recovery_core_r2_gsi_only_env_cfg import (
  g1_recovery_core_r2_gsi_only_smp_env_cfg,
)
from smp.rl.tasks.getup.recovery_core_r2_ordered_env_cfg import (
  g1_recovery_core_r2_ordered_smp_env_cfg,
)
from smp.rl.tasks.getup.robust_env_cfg import g1_getup_robust_smp_env_cfg
from smp.rl.tasks.getup.safe_env_cfg import g1_getup_robust_safe_smp_env_cfg
from smp.rl.tasks.getup.scratch_causal_ablation_env_cfg import (
  g1_scratch_a0_f2s2_gsi_env_cfg,
  g1_scratch_a1_v7_gsi_env_cfg,
  g1_scratch_a2_f2s2_mix_strict_env_cfg,
  g1_scratch_a3_v7_mix_strict_env_cfg,
  g1_scratch_a4_f2s2_mix_reset_aware_env_cfg,
  g1_scratch_a5_v7_mix_reset_aware_env_cfg,
  g1_scratch_a6_f2s2_mix_bridge_env_cfg,
  g1_scratch_a7_v7_mix_bridge_env_cfg,
)
from smp.rl.tasks.getup.smooth_env_cfg import g1_getup_robust_smooth_smp_env_cfg
from smp.rl.tasks.getup.smp_observation_factorial_env_cfg import (
  g1_getup_obs_f1_nolinvel_smp_env_cfg,
  g1_getup_obs_f4_nolinvel_smp_env_cfg,
  g1_getup_obs_f4_vel_smp_env_cfg,
)
from smp.rl.tasks.getup.staged_env_cfg import g1_getup_robust_staged_smp_env_cfg
from smp.rl.tasks.getup.terrain_v35_env_cfg import (
  g1_getup_terrain_v35_smp_env_cfg,
)
from smp.rl.tasks.getup.terrain_v36_env_cfg import (
  g1_getup_terrain_v36_smp_env_cfg,
)
from smp.rl.tasks.getup.terrain_v37_env_cfg import (
  g1_getup_terrain_v37_deploy_smp_env_cfg,
  g1_getup_terrain_v37_smp_env_cfg,
)
from smp.rl.tasks.getup.terrain_v361_env_cfg import (
  g1_getup_terrain_v361_smp_env_cfg,
)
from smp.rl.tasks.getup.terrain_v362_env_cfg import (
  g1_getup_terrain_v362_deploy_smp_env_cfg,
  g1_getup_terrain_v362_smp_env_cfg,
)
from smp.rl.tasks.getup.terrain_v363_env_cfg import (
  g1_getup_terrain_v363_deploy_smp_env_cfg,
  g1_getup_terrain_v363_smp_env_cfg,
)
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


def _observation_factorial_runner(experiment_name: str):
  cfg = unitree_g1_smp_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.run_name = experiment_name
  cfg.save_interval = 500
  return cfg


register_mjlab_task(
  task_id="Smp-Getup-Obs-F1-NoLinVel-G1",
  env_cfg=g1_getup_obs_f1_nolinvel_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_obs_f1_nolinvel_smp_env_cfg(play=True),
  rl_cfg=_observation_factorial_runner("smp_getup_obs_f1_nolinvel_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Obs-F4-Vel-G1",
  env_cfg=g1_getup_obs_f4_vel_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_obs_f4_vel_smp_env_cfg(play=True),
  rl_cfg=_observation_factorial_runner("smp_getup_obs_f4_vel_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Obs-F4-NoLinVel-G1",
  env_cfg=g1_getup_obs_f4_nolinvel_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_obs_f4_nolinvel_smp_env_cfg(play=True),
  rl_cfg=_observation_factorial_runner("smp_getup_obs_f4_nolinvel_g1"),
)


def _scratch_causal_runner(experiment_name: str):
  cfg = unitree_g1_smp_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.run_name = experiment_name
  cfg.save_interval = 1000
  return cfg


register_mjlab_task(
  task_id="Smp-Getup-Scratch-A0-F2S2-GSI-G1",
  env_cfg=g1_scratch_a0_f2s2_gsi_env_cfg(play=False),
  play_env_cfg=g1_scratch_a0_f2s2_gsi_env_cfg(play=True),
  rl_cfg=_scratch_causal_runner("smp_scratch_a0_f2s2_gsi_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Scratch-A1-V7-GSI-G1",
  env_cfg=g1_scratch_a1_v7_gsi_env_cfg(play=False),
  play_env_cfg=g1_scratch_a1_v7_gsi_env_cfg(play=True),
  rl_cfg=_scratch_causal_runner("smp_scratch_a1_v7_gsi_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Scratch-A2-F2S2-Mix-Strict-G1",
  env_cfg=g1_scratch_a2_f2s2_mix_strict_env_cfg(play=False),
  play_env_cfg=g1_scratch_a2_f2s2_mix_strict_env_cfg(play=True),
  rl_cfg=_scratch_causal_runner("smp_scratch_a2_f2s2_mix_strict_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Scratch-A3-V7-Mix-Strict-G1",
  env_cfg=g1_scratch_a3_v7_mix_strict_env_cfg(play=False),
  play_env_cfg=g1_scratch_a3_v7_mix_strict_env_cfg(play=True),
  rl_cfg=_scratch_causal_runner("smp_scratch_a3_v7_mix_strict_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Scratch-A4-F2S2-Mix-ResetAware-G1",
  env_cfg=g1_scratch_a4_f2s2_mix_reset_aware_env_cfg(play=False),
  play_env_cfg=g1_scratch_a4_f2s2_mix_reset_aware_env_cfg(play=True),
  rl_cfg=_scratch_causal_runner("smp_scratch_a4_f2s2_mix_reset_aware_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Scratch-A5-V7-Mix-ResetAware-G1",
  env_cfg=g1_scratch_a5_v7_mix_reset_aware_env_cfg(play=False),
  play_env_cfg=g1_scratch_a5_v7_mix_reset_aware_env_cfg(play=True),
  rl_cfg=_scratch_causal_runner("smp_scratch_a5_v7_mix_reset_aware_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Scratch-A6-F2S2-Mix-Bridge-G1",
  env_cfg=g1_scratch_a6_f2s2_mix_bridge_env_cfg(play=False),
  play_env_cfg=g1_scratch_a6_f2s2_mix_bridge_env_cfg(play=True),
  rl_cfg=_scratch_causal_runner("smp_scratch_a6_f2s2_mix_bridge_g1"),
)

register_mjlab_task(
  task_id="Smp-Getup-Scratch-A7-V7-Mix-Bridge-G1",
  env_cfg=g1_scratch_a7_v7_mix_bridge_env_cfg(play=False),
  play_env_cfg=g1_scratch_a7_v7_mix_bridge_env_cfg(play=True),
  rl_cfg=_scratch_causal_runner("smp_scratch_a7_v7_mix_bridge_g1"),
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

_escape_plate_v33_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_escape_plate_v33_getup_rl.experiment_name = "smp_getup_escape_plate_v33_g1"
_escape_plate_v33_getup_rl.run_name = "smp_getup_escape_plate_v33_g1"
_escape_plate_v33_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Escape-Plate-V33-G1",
  env_cfg=g1_getup_escape_plate_v33_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_escape_plate_v33_smp_env_cfg(play=True),
  rl_cfg=_escape_plate_v33_getup_rl,
)

_escape_plate_v34_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_escape_plate_v34_getup_rl.experiment_name = "smp_getup_escape_plate_v34_g1"
_escape_plate_v34_getup_rl.run_name = "smp_getup_escape_plate_v34_g1"
_escape_plate_v34_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Escape-Plate-V34-G1",
  env_cfg=g1_getup_escape_plate_v34_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_escape_plate_v34_smp_env_cfg(play=True),
  rl_cfg=_escape_plate_v34_getup_rl,
)

_terrain_v35_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_terrain_v35_getup_rl.experiment_name = "smp_getup_terrain_v35_g1"
_terrain_v35_getup_rl.run_name = "smp_getup_terrain_v35_g1"
_terrain_v35_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V35-G1",
  env_cfg=g1_getup_terrain_v35_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v35_smp_env_cfg(play=True),
  rl_cfg=_terrain_v35_getup_rl,
)

_terrain_v36_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_terrain_v36_getup_rl.experiment_name = "smp_getup_terrain_v36_g1"
_terrain_v36_getup_rl.run_name = "smp_getup_terrain_v36_g1"
_terrain_v36_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V36-G1",
  env_cfg=g1_getup_terrain_v36_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v36_smp_env_cfg(play=True),
  rl_cfg=_terrain_v36_getup_rl,
)

_terrain_v361_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_terrain_v361_getup_rl.experiment_name = "smp_getup_terrain_v361_g1"
_terrain_v361_getup_rl.run_name = "smp_getup_terrain_v361_g1"
_terrain_v361_getup_rl.save_interval = 1000

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V361-G1",
  env_cfg=g1_getup_terrain_v361_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v361_smp_env_cfg(play=True),
  rl_cfg=_terrain_v361_getup_rl,
)

_terrain_v362_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_terrain_v362_getup_rl.experiment_name = "smp_getup_terrain_v362_g1"
_terrain_v362_getup_rl.run_name = "smp_getup_terrain_v362_g1"
_terrain_v362_getup_rl.save_interval = 250

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V362-G1",
  env_cfg=g1_getup_terrain_v362_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v362_smp_env_cfg(play=True),
  rl_cfg=_terrain_v362_getup_rl,
)

_terrain_v362_deploy_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_terrain_v362_deploy_getup_rl.experiment_name = "smp_getup_terrain_v362_deploy_g1"
_terrain_v362_deploy_getup_rl.run_name = "smp_getup_terrain_v362_deploy_g1"
_terrain_v362_deploy_getup_rl.save_interval = 250

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V362-Deploy-G1",
  env_cfg=g1_getup_terrain_v362_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v362_deploy_smp_env_cfg(play=True),
  rl_cfg=_terrain_v362_deploy_getup_rl,
)


def _terrain_v363_runner_cfg(experiment_name: str):
  cfg = unitree_g1_smp_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.run_name = experiment_name
  cfg.save_interval = 50
  cfg.algorithm.learning_rate = 1.0e-5
  cfg.algorithm.clip_param = 0.10
  cfg.algorithm.desired_kl = 0.005
  cfg.algorithm.num_learning_epochs = 2
  cfg.algorithm.max_grad_norm = 0.5
  return cfg


_terrain_v363_getup_rl = _terrain_v363_runner_cfg("smp_getup_terrain_v363_g1")

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V363-G1",
  env_cfg=g1_getup_terrain_v363_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v363_smp_env_cfg(play=True),
  rl_cfg=_terrain_v363_getup_rl,
)

_terrain_v363_deploy_getup_rl = _terrain_v363_runner_cfg(
  "smp_getup_terrain_v363_deploy_g1"
)

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V363-Deploy-G1",
  env_cfg=g1_getup_terrain_v363_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v363_deploy_smp_env_cfg(play=True),
  rl_cfg=_terrain_v363_deploy_getup_rl,
)


def _terrain_v37_runner_cfg(experiment_name: str):
  cfg = _terrain_v363_runner_cfg(experiment_name)
  cfg.save_interval = 10
  cfg.algorithm.learning_rate = 5.0e-6
  return cfg


_terrain_v37_getup_rl = _terrain_v37_runner_cfg("smp_getup_terrain_v37_g1")

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V37-G1",
  env_cfg=g1_getup_terrain_v37_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v37_smp_env_cfg(play=True),
  rl_cfg=_terrain_v37_getup_rl,
)

_terrain_v37_deploy_getup_rl = _terrain_v37_runner_cfg(
  "smp_getup_terrain_v37_deploy_g1"
)

register_mjlab_task(
  task_id="Smp-Getup-Terrain-V37-Deploy-G1",
  env_cfg=g1_getup_terrain_v37_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_terrain_v37_deploy_smp_env_cfg(play=True),
  rl_cfg=_terrain_v37_deploy_getup_rl,
)


def _plate_terrain_v38_runner_cfg():
  cfg = _terrain_v363_runner_cfg("smp_getup_plate_terrain_v38_deploy_g1")
  cfg.save_interval = 100
  cfg.algorithm.learning_rate = 5.0e-6
  return cfg


_plate_terrain_v38_getup_rl = _plate_terrain_v38_runner_cfg()

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V38-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v38_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v38_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v38_getup_rl,
)


def _plate_terrain_v381_runner_cfg():
  cfg = _terrain_v363_runner_cfg("smp_getup_plate_terrain_v381_deploy_g1")
  cfg.save_interval = 100
  cfg.algorithm.learning_rate = 2.0e-6
  return cfg


_plate_terrain_v381_getup_rl = _plate_terrain_v381_runner_cfg()

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V381-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v381_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v381_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v381_getup_rl,
)


def _plate_terrain_v382_runner_cfg(experiment_name: str):
  cfg = _terrain_v363_runner_cfg(experiment_name)
  cfg.save_interval = 25
  cfg.algorithm.learning_rate = 1.0e-6
  return cfg


_plate_terrain_v382_h4_getup_rl = _plate_terrain_v382_runner_cfg(
  "smp_getup_plate_terrain_v382_h4_deploy_g1"
)
_plate_terrain_v382_h10_getup_rl = _plate_terrain_v382_runner_cfg(
  "smp_getup_plate_terrain_v382_h10_deploy_g1"
)

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V382-H4-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v382_h4_getup_rl,
)

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V382-H10-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v382_h10_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v382_h10_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v382_h10_getup_rl,
)


def _plate_terrain_v383_runner_cfg():
  cfg = _terrain_v363_runner_cfg("smp_getup_plate_terrain_v383_right_deploy_g1")
  # A 25-update diagnostic has useful snapshots at 0, 12, and 24 without
  # producing the dense checkpoint stream used by early experiments.
  cfg.save_interval = 12
  cfg.algorithm.learning_rate = 5.0e-7
  return cfg


_plate_terrain_v383_getup_rl = _plate_terrain_v383_runner_cfg()

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V383-Right-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v383_right_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v383_right_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v383_getup_rl,
)


def _plate_terrain_v384_runner_cfg():
  cfg = _terrain_v363_runner_cfg("smp_getup_plate_terrain_v384_symmetric_deploy_g1")
  cfg.save_interval = 12
  cfg.algorithm.learning_rate = 5.0e-7
  cfg.algorithm.symmetry_cfg = {
    "data_augmentation_func": "smp.rl.symmetry:g1_sagittal_data_augmentation",
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.05,
  }
  return cfg


_plate_terrain_v384_getup_rl = _plate_terrain_v384_runner_cfg()

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V384-Symmetric-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v384_symmetric_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v384_symmetric_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v384_getup_rl,
)


def _plate_terrain_v386_scratch_s1_runner_cfg():
  cfg = _terrain_v363_runner_cfg("smp_getup_plate_terrain_v386_scratch_s1_deploy_g1")
  cfg.save_interval = 100
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.symmetry_cfg = {
    "data_augmentation_func": "smp.rl.symmetry:g1_sagittal_data_augmentation",
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.02,
  }
  return cfg


_plate_terrain_v386_scratch_s1_getup_rl = _plate_terrain_v386_scratch_s1_runner_cfg()

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V386-Scratch-S1-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v386_scratch_s1_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v386_scratch_s1_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v386_scratch_s1_getup_rl,
)


def _plate_terrain_v387_scratch_s0_runner_cfg():
  cfg = _terrain_v363_runner_cfg("smp_getup_plate_terrain_v387_scratch_s0_deploy_g1")
  cfg.save_interval = 25
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.num_learning_epochs = 5
  cfg.algorithm.num_mini_batches = 8
  cfg.algorithm.symmetry_cfg = {
    "data_augmentation_func": "smp.rl.symmetry:g1_sagittal_data_augmentation",
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.02,
  }
  return cfg


_plate_terrain_v387_scratch_s0_getup_rl = _plate_terrain_v387_scratch_s0_runner_cfg()

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V387-Scratch-S0-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v387_scratch_s0_getup_rl,
)


def _plate_terrain_v3871_scratch_s0_runner_cfg():
  """Fast scratch gate with a non-collapsing PPO learning rate."""
  cfg = _plate_terrain_v387_scratch_s0_runner_cfg()
  cfg.experiment_name = "smp_getup_plate_terrain_v3871_scratch_s0_fixed_deploy_g1"
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.algorithm.schedule = "fixed"
  cfg.save_interval = 50
  return cfg


_plate_terrain_v3871_scratch_s0_getup_rl = (
  _plate_terrain_v3871_scratch_s0_runner_cfg()
)

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V3871-Scratch-S0-Fixed-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg(play=True),
  rl_cfg=_plate_terrain_v3871_scratch_s0_getup_rl,
)


def _plate_terrain_v3872_scratch_s0_dense_runner_cfg():
  cfg = _plate_terrain_v3871_scratch_s0_runner_cfg()
  cfg.experiment_name = "smp_getup_plate_terrain_v3872_scratch_s0_dense_deploy_g1"
  return cfg


_plate_terrain_v3872_scratch_s0_dense_getup_rl = (
  _plate_terrain_v3872_scratch_s0_dense_runner_cfg()
)

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V3872-Scratch-S0-Dense-Deploy-G1",
  env_cfg=(
    g1_getup_plate_terrain_v3872_scratch_s0_dense_deploy_smp_env_cfg(play=False)
  ),
  play_env_cfg=(
    g1_getup_plate_terrain_v3872_scratch_s0_dense_deploy_smp_env_cfg(play=True)
  ),
  rl_cfg=_plate_terrain_v3872_scratch_s0_dense_getup_rl,
)


def _plate_terrain_v3873_scratch_stage_bridge_runner_cfg():
  cfg = _plate_terrain_v3872_scratch_s0_dense_runner_cfg()
  cfg.experiment_name = "smp_getup_plate_terrain_v3873_scratch_stage_bridge_deploy_g1"
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.save_interval = 250
  return cfg


_plate_terrain_v3873_scratch_stage_bridge_getup_rl = (
  _plate_terrain_v3873_scratch_stage_bridge_runner_cfg()
)

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V3873-Scratch-Stage-Bridge-Deploy-G1",
  env_cfg=(
    g1_getup_plate_terrain_v3873_scratch_stage_bridge_deploy_smp_env_cfg(False)
  ),
  play_env_cfg=(
    g1_getup_plate_terrain_v3873_scratch_stage_bridge_deploy_smp_env_cfg(True)
  ),
  rl_cfg=_plate_terrain_v3873_scratch_stage_bridge_getup_rl,
)


def _plate_terrain_v3874_gsi_milestone_runner_cfg():
  cfg = _plate_terrain_v3873_scratch_stage_bridge_runner_cfg()
  cfg.experiment_name = "smp_getup_plate_terrain_v3874_gsi_milestone_deploy_g1"
  cfg.save_interval = 250
  return cfg


_plate_terrain_v3874_gsi_milestone_getup_rl = (
  _plate_terrain_v3874_gsi_milestone_runner_cfg()
)

register_mjlab_task(
  task_id="Smp-Getup-Plate-Terrain-V3874-GSI-Milestone-Deploy-G1",
  env_cfg=g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg(False),
  play_env_cfg=g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg(True),
  rl_cfg=_plate_terrain_v3874_gsi_milestone_getup_rl,
)


def _recovery_core_r1_runner_cfg():
  cfg = _plate_terrain_v3874_gsi_milestone_runner_cfg()
  cfg.experiment_name = "smp_recovery_core_r1_g1"
  cfg.save_interval = 250
  return cfg


_recovery_core_r1_rl = _recovery_core_r1_runner_cfg()

register_mjlab_task(
  task_id="Smp-Recovery-Core-R1-G1",
  env_cfg=g1_recovery_core_r1_smp_env_cfg(False),
  play_env_cfg=g1_recovery_core_r1_smp_env_cfg(True),
  rl_cfg=_recovery_core_r1_rl,
)


def _recovery_core_r2_ordered_runner_cfg():
  cfg = _recovery_core_r1_runner_cfg()
  cfg.experiment_name = "smp_recovery_core_r2_ordered_g1"
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.save_interval = 250
  return cfg


_recovery_core_r2_ordered_rl = _recovery_core_r2_ordered_runner_cfg()

register_mjlab_task(
  task_id="Smp-Recovery-Core-R2-Ordered-G1",
  env_cfg=g1_recovery_core_r2_ordered_smp_env_cfg(False),
  play_env_cfg=g1_recovery_core_r2_ordered_smp_env_cfg(True),
  rl_cfg=_recovery_core_r2_ordered_rl,
)


def _recovery_core_r2_gsi_only_runner_cfg():
  cfg = _recovery_core_r2_ordered_runner_cfg()
  cfg.experiment_name = "smp_recovery_core_r2_gsi_only_g1"
  return cfg


_recovery_core_r2_gsi_only_rl = _recovery_core_r2_gsi_only_runner_cfg()

register_mjlab_task(
  task_id="Smp-Recovery-Core-R2-GSI-Only-G1",
  env_cfg=g1_recovery_core_r2_gsi_only_smp_env_cfg(False),
  play_env_cfg=g1_recovery_core_r2_gsi_only_smp_env_cfg(True),
  rl_cfg=_recovery_core_r2_gsi_only_rl,
)

__all__ = [
  "g1_getup_plate_terrain_v38_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v381_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v382_h4_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v382_h10_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v383_right_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v384_symmetric_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v386_scratch_s1_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v387_scratch_s0_deploy_smp_env_cfg",
  "g1_getup_plate_terrain_v3874_gsi_milestone_deploy_smp_env_cfg",
  "g1_getup_robust_smp_env_cfg",
  "g1_getup_robust_safe_smp_env_cfg",
  "g1_getup_robust_smooth_smp_env_cfg",
  "g1_getup_robust_staged_smp_env_cfg",
  "g1_recovery_core_r1_smp_env_cfg",
  "g1_recovery_core_r2_gsi_only_smp_env_cfg",
  "g1_recovery_core_r2_ordered_smp_env_cfg",
  "g1_getup_terrain_v35_smp_env_cfg",
  "g1_getup_terrain_v36_smp_env_cfg",
  "g1_getup_terrain_v361_smp_env_cfg",
  "g1_getup_terrain_v362_deploy_smp_env_cfg",
  "g1_getup_terrain_v362_smp_env_cfg",
  "g1_getup_terrain_v363_deploy_smp_env_cfg",
  "g1_getup_terrain_v363_smp_env_cfg",
  "g1_getup_terrain_v37_deploy_smp_env_cfg",
  "g1_getup_terrain_v37_smp_env_cfg",
  "g1_getup_smp_env_cfg",
  "g1_getup_constrained_smp_env_cfg",
  "g1_getup_escape_smp_env_cfg",
  "g1_getup_escape_plate_v3_smp_env_cfg",
  "g1_getup_escape_plate_v31_smp_env_cfg",
  "g1_getup_escape_plate_v32_smp_env_cfg",
  "g1_getup_escape_plate_v33_smp_env_cfg",
  "g1_getup_escape_plate_v34_smp_env_cfg",
  "g1_getup_v6_prior_smp_env_cfg",
  "g1_getup_v6_smp_env_cfg",
  "g1_getup_v7_route_smp_env_cfg",
  "g1_getup_v8_natural_smp_env_cfg",
]
