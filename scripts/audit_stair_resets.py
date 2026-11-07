"""Inspect multi-position stair resets and partial-reset isolation in MuJoCo."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

import smp.rl.tasks  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
cfg = load_env_cfg("Smp-Getup-Stairs-V36-93D-G1")
cfg.scene.num_envs = 256
cfg.seed = 20261109
cfg.events["init_smp_state"].params.update(
  compile_model=False, gsi_buffer_size=64, gsi_batch_size=64
)
env = ManagerBasedRlEnv(cfg, device="cuda:0")
wrapped = RslRlVecEnvWrapper(env)
# Check both the initial mixture and late difficult-stair mixture.
reports = []
for mixture, counter in [("initial", 0), ("late", 48000)]:
  env.common_step_counter = counter
  env.reset()
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  terrain_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  qpos = env.sim.data.qpos.cpu().numpy()
  qvel = env.sim.data.qvel.cpu().numpy()
  min_terrain = 0.0
  min_self = 0.0
  bad_terrain = []
  bad_self = []
  records = []
  for i in range(env.num_envs):
    d.qpos[:] = qpos[i]
    d.qvel[:] = qvel[i]
    mujoco.mj_forward(m, d)
    terrain_dist = []
    self_dist = []
    for c in d.contact:
      bodies = m.geom_bodyid[[c.geom1, c.geom2]]
      (terrain_dist if terrain_body in bodies else self_dist).append(float(c.dist))
    t = min(terrain_dist, default=0.0)
    s = min(self_dist, default=0.0)
    min_terrain = min(min_terrain, t)
    min_self = min(min_self, s)
    if t < -0.001:
      bad_terrain.append(i)
    if s < -0.0011:
      bad_self.append(i)
    records.append(
      dict(
        env=i,
        terrain_type=int(env.scene.terrain.terrain_types[i]),
        level=int(env.scene.terrain.terrain_levels[i]),
        region=int(env._stair_region[i]),
        pose=int(env._robust_reset_type[i]),
        terrain_contact_min_m=t,
        self_contact_min_m=s,
      )
    )
  # Query conservative placement constraints directly, independently of contact margin.
  from smp.rl.tasks.getup.mdp.events import _collision_vertical_geometry
  from smp.rl.tasks.getup.stair_recovery_93d_env_cfg import _boxes

  ids = torch.arange(env.num_envs, device=env.device)
  pos, extent, center, half, _ = _collision_vertical_geometry(
    env, ids, r".*_collision$"
  )
  boxes = _boxes(env, ids)
  overlap = (
    (center[:, :, None, :2] - boxes[:, None, :, :2]).abs()
    <= half[:, :, None, :2] + boxes[:, None, :, 3:5]
  ).all(-1)
  gap = (pos[:, :, 2] - extent)[:, :, None] - (boxes[:, :, 2] + boxes[:, :, 5])[
    :, None, :
  ]
  min_gap = torch.where(overlap, gap, 10000).amin((1, 2))
  r = {
    "mixture": mixture,
    "terrain_min_m": min_terrain,
    "self_min_m": min_self,
    "bad_terrain_ids": bad_terrain,
    "bad_self_ids": bad_self,
    "aabb_min_gap_m": float(min_gap.min()),
    "levels": torch.bincount(env.scene.terrain.terrain_levels, minlength=4)
    .cpu()
    .tolist(),
    "poses": torch.bincount(env._robust_reset_type, minlength=5).cpu().tolist(),
    "regions": torch.bincount(env._stair_region, minlength=4).cpu().tolist(),
  }
  reports.append(r)
  (a.output / f"{mixture}_records.json").write_text(json.dumps(records, indent=2))
  if mixture == "late":
    # Eight cells show all four poses and reset regions, including harder levels.
    selected = []
    for pose in range(1, 5):
      for region in (1, 2):
        candidates = [
          x["env"]
          for x in records
          if x["terrain_type"] > 0
          and x["level"] >= 2
          and x["pose"] == pose
          and x["region"] == region
        ]
        if candidates:
          selected.append(candidates[0])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    renderer = mujoco.Renderer(m, height=400, width=600)
    for ax, i in zip(axes.flat, selected, strict=False):
      d.qpos[:] = qpos[i]
      d.qvel[:] = qvel[i]
      mujoco.mj_forward(m, d)
      cam = mujoco.MjvCamera()
      mujoco.mjv_defaultCamera(cam)
      cam.lookat[:] = env.scene["robot"].data.root_link_pos_w[i].cpu().numpy()
      cam.lookat[2] += 0.2
      cam.distance = 2.9
      cam.elevation = -25
      cam.azimuth = 135
      renderer.update_scene(d, camera=cam)
      ax.imshow(renderer.render())
      ax.axis("off")
      x = records[i]
      ax.set_title(
        f"{['', 'prone', 'supine', 'left', 'right'][x['pose']]} | {['top', 'tread/edge', 'bottom', 'corner'][x['region']]} | {[5, 10, 15, 20][x['level']]} cm"
      )
    fig.savefig(a.output / "stair_reset_examples.png", dpi=120)
    plt.close(fig)
    renderer.close()
  before = env.sim.data.qpos.clone()
  old_origin = env.scene.env_origins.clone()
  old_anchor = env._recovery_start_xy.clone()
  subset = torch.arange(0, env.num_envs, 4, device=env.device)
  env.reset(env_ids=subset)
  other = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  other[subset] = False
  torch.testing.assert_close(env.sim.data.qpos[other], before[other], rtol=0, atol=0)
  torch.testing.assert_close(
    env.scene.env_origins[other], old_origin[other], rtol=0, atol=0
  )
  torch.testing.assert_close(
    env._recovery_start_xy[other], old_anchor[other], rtol=0, atol=0
  )
(a.output / "audit.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports), flush=True)
assert not any(r["bad_terrain_ids"] for r in reports)
assert not any(r["bad_self_ids"] for r in reports)
env.close()
