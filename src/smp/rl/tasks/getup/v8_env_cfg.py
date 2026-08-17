"""Natural prone-support refinement on top of the V7 route prior."""

from __future__ import annotations

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v7_env_cfg import g1_getup_v7_route_smp_env_cfg


def g1_getup_v8_natural_smp_env_cfg(play: bool = False):
  """Build V8: hand/knee-supported prone recovery without a task obstacle."""
  cfg = g1_getup_v7_route_smp_env_cfg(play=play)

  # Hand/knee support adds enough simultaneous contacts to exceed the robust
  # base task's 64-contact allocation during prone recovery. Keep headroom for
  # interactive dragging and highly folded poses so MuJoCo does not discard
  # broadphase contacts during evaluation.
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 2500

  hand_ground = ContactSensorCfg(
    name="natural_hand_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"(left|right)_hand_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  knee_ground = ContactSensorCfg(
    name="natural_knee_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"(left|right)_(shin|linkage_brace)_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (hand_ground, knee_ground)

  # The support term stays saturated after the first ordered waypoint, so the
  # policy cannot gain more return by remaining on hands and knees forever.
  cfg.rewards.update(
    {
      "prone_support_route": RewardTermCfg(
        func=mdp.prone_support_route,
        weight=0.12,
        params={
          "hand_sensor_name": "natural_hand_ground_contact",
          "knee_sensor_name": "natural_knee_ground_contact",
        },
      ),
      "prone_leg_splay": RewardTermCfg(
        func=mdp.prone_leg_splay_excess_l2,
        weight=-0.06,
        params={"hip_roll_limit": 0.65, "hip_yaw_limit": 0.75},
      ),
    }
  )

  # Tighten only the large low-stage bursts. Static support torque remains
  # available, while joint speed and mechanical power shortcuts become less
  # attractive than the demonstrated hand/knee route.
  cfg.rewards["joint_speed_excess"].params["speed_limits"] = (5.0, 4.5, 4.0, 3.5)
  cfg.rewards["joint_power_excess"].params["power_limits"] = (
    120.0,
    100.0,
    90.0,
    75.0,
  )

  cfg.metrics.update(
    {
      "prone_hand_support": MetricsTermCfg(
        func=mdp.ground_support_contact_metric,
        params={"sensor_name": "natural_hand_ground_contact"},
      ),
      "prone_knee_support": MetricsTermCfg(
        func=mdp.ground_support_contact_metric,
        params={"sensor_name": "natural_knee_ground_contact"},
      ),
      "prone_support_route": MetricsTermCfg(
        func=mdp.prone_support_route,
        params={
          "hand_sensor_name": "natural_hand_ground_contact",
          "knee_sensor_name": "natural_knee_ground_contact",
        },
      ),
      "prone_leg_splay": MetricsTermCfg(func=mdp.prone_leg_splay_excess_l2),
    }
  )
  return cfg


__all__ = ["g1_getup_v8_natural_smp_env_cfg"]
