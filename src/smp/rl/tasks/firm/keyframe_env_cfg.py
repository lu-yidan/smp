"""Paper-faithful first-stage FIRM sparse-keyframe expert for G1."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from smp.rl.tasks.firm import mdp

MOTION_FILE = "datasets/firm/lafan/fallAndGetUp2_subject2_candidate_003_validated.npz"


def _firm_observations(play: bool) -> dict[str, ObservationGroupCfg]:
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
    "keyframe_joint_error": ObservationTermCfg(
      func=mdp.keyframe_joint_error,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "phase": ObservationTermCfg(
      func=mdp.motion_phase,
      params={"command_name": "motion"},
    ),
  }
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=env_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
    "keyframe_joint_error": ObservationTermCfg(
      func=mdp.keyframe_joint_error,
      params={"command_name": "motion"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.motion_phase,
      params={"command_name": "motion"},
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def _firm_rewards() -> dict[str, RewardTermCfg]:
  robot = SceneEntityCfg("robot", joint_names=(".*",))
  return {
    "rigid_body_position": RewardTermCfg(
      func=tracking_mdp.motion_relative_body_position_error_exp,
      weight=1.25,
      params={"command_name": "motion", "std": 0.3},
    ),
    "rigid_body_orientation": RewardTermCfg(
      func=tracking_mdp.motion_relative_body_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "rigid_body_linear_velocity": RewardTermCfg(
      func=tracking_mdp.motion_global_body_linear_velocity_error_exp,
      weight=0.125,
      params={"command_name": "motion", "std": 1.0},
    ),
    "rigid_body_angular_velocity": RewardTermCfg(
      func=tracking_mdp.motion_global_body_angular_velocity_error_exp,
      weight=0.125,
      params={"command_name": "motion", "std": 3.14},
    ),
    "joint_position": RewardTermCfg(
      func=mdp.keyframe_joint_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.5},
    ),
    "joint_velocity": RewardTermCfg(
      func=mdp.keyframe_joint_velocity_error_exp,
      weight=0.125,
      params={"command_name": "motion", "std": 5.0},
    ),
    "joint_position_limit": RewardTermCfg(
      func=env_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": robot},
    ),
    "joint_velocity_limit": RewardTermCfg(
      func=mdp.joint_velocity_limit,
      weight=-5.0,
      params={"asset_cfg": robot, "velocity_limit": 12.0},
    ),
    "action_rate": RewardTermCfg(
      func=env_mdp.action_rate_l2,
      weight=-1.0e-3,
    ),
    "torques": RewardTermCfg(
      func=env_mdp.joint_torques_l2,
      weight=-1.0e-6,
      params={"asset_cfg": robot},
    ),
    "joint_acceleration": RewardTermCfg(
      func=env_mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": robot},
    ),
    "body_collision": RewardTermCfg(
      func=tracking_mdp.self_collision_cost,
      weight=-1.0e-7,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }


def g1_firm_keyframe_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build candidate 003's first sparse-keyframe augmentation expert."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  base_motion = cfg.commands["motion"]
  cfg.commands["motion"] = mdp.SparseKeyframeCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    pose_range={} if play else base_motion.pose_range,
    velocity_range={} if play else base_motion.velocity_range,
    joint_position_range=(0.0, 0.0) if play else (-0.1, 0.1),
    sampling_mode="start" if play else "uniform",
    motion_file=MOTION_FILE,
    anchor_body_name=base_motion.anchor_body_name,
    body_names=base_motion.body_names,
  )
  cfg.observations = _firm_observations(play)
  cfg.rewards = _firm_rewards()
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "unsafe_velocity": TerminationTermCfg(
      func=mdp.unsafe_velocity,
      params={
        "joint_speed_limit": 30.0,
        "root_linear_speed_limit": 15.0,
        "root_angular_speed_limit": 20.0,
      },
    ),
  }

  # Stage 0 learns the keyframe objective without automatic pushes or actuator
  # dropout. These robustness curricula are enabled only after the expert can
  # track the seed, so their contribution can be measured independently.
  cfg.events.pop("push_robot", None)
  cfg.sim.nconmax = 64
  cfg.episode_length_s = int(1e9) if play else 10.0
  return cfg
