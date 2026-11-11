"""Short stair-support refinement: gather the feet without freezing foot placement."""

import math

import mujoco
import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.tasks.getup.mdp.rewards import _head_height
from smp.rl.tasks.getup.stair_recovery_93d_env_cfg import (
  HEIGHTS,
  place_on_stairs,
  surface_height,
)
from smp.rl.tasks.getup.stair_recovery_v34_speed_env_cfg import (
  stair_recovery_v34_speed_env_cfg,
)


def heading_components(delta, quat):
  """Return signed fore-aft and lateral separation in root yaw coordinates."""
  w, x, y, z = quat.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
  forward = delta[:, 0] * yaw.cos() + delta[:, 1] * yaw.sin()
  lateral = -delta[:, 0] * yaw.sin() + delta[:, 1] * yaw.cos()
  return forward, lateral


def support_gate(height, upright):
  a = ((height - 0.65) / 0.40).clamp(0, 1)
  b = ((upright - 0.55) / 0.30).clamp(0, 1)
  return a.square() * (3 - 2 * a) * b.square() * (3 - 2 * b)


def stance_cost(forward, lateral, gate, nominal_width):
  # Allow more room while crouched; avoid forcing feet together or suppressing
  # the fore-aft offset needed when supporting on adjacent treads.
  upper = nominal_width + 0.20 - 0.08 * gate
  lower = max(0.08, nominal_width - 0.10)
  wide = ((lateral - upper) / 0.15).clamp(0, 2).square()
  crossed = ((lower - lateral) / 0.15).clamp(0, 2).square()
  long = ((forward.abs() - 0.55) / 0.20).clamp(0, 2).square()
  return gate * (wide + crossed + 0.25 * long)


def descending_support(start, current, step_height):
  """Ground elevation loss, not pelvis height change during crouching."""
  loss = (start - current).clamp_min(0)
  threshold = torch.maximum(step_height * 0.5, step_height.new_tensor(0.025))
  return loss, loss > threshold


def calibrate_nominal_width(env):
  if hasattr(env, "_support_nominal_width"):
    return
  robot = env.scene["robot"]
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  data.qpos[robot.indexing.free_joint_q_adr.cpu().numpy()] = np.array(
    (0, 0, 20, 1, 0, 0, 0), dtype=float
  )
  data.qpos[robot.indexing.joint_q_adr.cpu().numpy()] = (
    robot.data.default_joint_pos[0].cpu().numpy()
  )
  mujoco.mj_forward(model, data)
  site_ids = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]
  global_ids = robot.indexing.site_ids[site_ids].cpu().numpy()
  width = float(data.site_xpos[global_ids[0], 1] - data.site_xpos[global_ids[1], 1])
  if not 0.10 < width < 0.40:
    raise ValueError(f"Unexpected default stance width: {width}")
  env._support_nominal_width = width


@torch.no_grad()
def reset_support(env, env_ids=None):
  place_on_stairs(env, env_ids)
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  calibrate_nominal_width(env)
  if not hasattr(env, "_support_start_ground"):
    env._support_start_ground = torch.zeros(env.num_envs, device=env.device)
  xy = env.scene["robot"].data.root_link_pos_w[ids, None, :2]
  env._support_start_ground[ids] = surface_height(env, ids, xy).squeeze(-1)
  env._support_cache_step = None


def support_values(env):
  step = env.common_step_counter
  if getattr(env, "_support_cache_step", None) == step:
    return env._support_values
  robot = env.scene["robot"]
  feet = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]
  xy = robot.data.site_pos_w[:, feet, :2]
  forward, lateral = heading_components(
    xy[:, 0] - xy[:, 1], robot.data.root_link_quat_w
  )
  upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
  gate = support_gate(_head_height(env, relative_to_env_origin=True), upright)
  ids = torch.arange(env.num_envs, device=env.device)
  # Eight terrain probes 6 cm around each foot site measure nearby downward
  # edges. This is a clearance proxy, not a full sole-contact polygon.
  angle = torch.arange(8, device=env.device) * (math.pi / 4)
  offsets = 0.06 * torch.stack((angle.cos(), angle.sin()), -1)
  probes = xy[:, :, None] + offsets
  center_h = surface_height(env, ids, xy)
  probe_h = surface_height(env, ids, probes.reshape(env.num_envs, 16, 2)).reshape(
    -1, 2, 8
  )
  edge_fraction = ((center_h[:, :, None] - probe_h) > 0.02).float().mean(-1)
  forces = []
  for side in ("left", "right"):
    force = env.scene[f"support_{side}_foot"].data.force
    # Contact frame sign depends on geom ordering; use vertical load magnitude.
    forces.append(force[..., 2].abs().reshape(env.num_envs, -1).amax(-1))
  load = torch.stack(forces, -1)
  load_gate = ((load - 10) / 40).clamp(0, 1)
  ground = surface_height(env, ids, robot.data.root_link_pos_w[:, None, :2]).squeeze(-1)
  heights = ground.new_tensor(HEIGHTS)[env.scene.terrain.terrain_levels]
  descent, _ = descending_support(env._support_start_ground, ground, heights)
  values = {
    "stance": stance_cost(forward, lateral, gate, env._support_nominal_width),
    "edge": gate * (edge_fraction * load_gate).mean(-1),
    "descent": (0.25 + 0.75 * gate) * ((descent - 0.02) / heights).clamp(0, 2).square(),
    "lateral_width": lateral,
    "fore_aft": forward.abs(),
    "edge_loaded": (edge_fraction * load_gate).mean(-1),
    "ground_loss": descent,
    "gate": gate,
  }
  env._support_values, env._support_cache_step = values, step
  return values


def support_term(env, field):
  return support_values(env)[field]


def stair_support_env_cfg(play=False):
  cfg = stair_recovery_v34_speed_env_cfg(play)
  cfg.events["ground_procedural_fall_on_terrain"].func = reset_support
  cfg.scene.sensors += tuple(
    ContactSensorCfg(
      name=f"support_{side}_foot",
      primary=ContactMatch(
        mode="geom", pattern=rf"{side}_foot[1-7]_collision$", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force", "normal", "tangent"),
      reduce="maxforce",
      num_slots=1,
      global_frame=True,
    )
    for side in ("left", "right")
  )
  cfg.rewards.pop("terrain_stance_width")
  for field, weight in (("stance", -0.20), ("edge", -0.15), ("descent", -0.20)):
    cfg.rewards["support_" + field] = RewardTermCfg(
      func=support_term, weight=weight, params={"field": field}
    )
  for field in ("lateral_width", "fore_aft", "edge_loaded", "ground_loss", "gate"):
    cfg.metrics["support_" + field] = MetricsTermCfg(
      func=support_term, params={"field": field}
    )
  return cfg
