"""Record unselected, fixed-seed stair examples with a robot-following camera."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from PIL import Image, ImageDraw, ImageFont

import smp.rl.tasks  # noqa: F401


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--level", type=int, choices=range(4), required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("artifacts/v36_projected/model_102000_zero_velocity_93d.pt"),
  )
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  configure_torch_backends()
  task = "Smp-Getup-Stairs-V36-93D-G1"
  cfg, agent = load_env_cfg(task, play=True), load_rl_cfg(task)
  cfg.scene.num_envs = 256
  cfg.seed = 20260818
  cfg.events["init_smp_state"].params.update(
    compile_model=False, gsi_buffer_size=64, gsi_batch_size=64
  )
  cfg.events["sample_stair_level"].params["fixed_level"] = args.level
  cfg.terminations, cfg.metrics, cfg.rewards = {}, {}, {}
  raw = ManagerBasedRlEnv(cfg, device="cuda:0")
  raw.scene.terrain.terrain_types[:] = 1 + torch.arange(256, device="cuda:0") % 2
  env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent), device="cuda:0")
  runner.load(
    str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
  )
  policy = runner.get_inference_policy(device="cuda:0")
  obs = env.get_observations()
  names = [
    "Prone | tread / edge",
    "Supine | upper platform",
    "Left side | bottom approach",
    "Right side | corner",
  ]
  selected = []
  for pose, region in ((1, 1), (2, 0), (3, 2), (4, 3)):
    matches = torch.nonzero(
      (raw._robust_reset_type == pose) & (raw._stair_region == region)
    ).flatten()
    if not matches.numel():
      raise RuntimeError(f"No fixed-seed example for pose={pose}, region={region}")
    selected.append(int(matches[0]))
  robot = raw.scene["robot"]
  start = robot.data.root_link_pos_w[:, :2].clone()
  head = robot.find_sites(["head"], preserve_order=True)[0][0]
  hold = torch.zeros(256, device="cuda:0", dtype=torch.long)
  success = torch.zeros(256, device="cuda:0", dtype=torch.bool)
  peak_distance = torch.zeros(256, device="cuda:0")
  model, data = raw.sim.mj_model, mujoco.MjData(raw.sim.mj_model)
  model.vis.global_.offwidth = 640
  model.vis.global_.offheight = 400
  renderer = mujoco.Renderer(model, width=640, height=400)
  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(camera)
  camera.distance, camera.elevation, camera.azimuth = 2.8, -23, 135
  font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
  try:
    font = ImageFont.truetype(font_path, 21)
  except OSError:
    font = ImageFont.load_default(size=21)
  height_cm = (5, 10, 15, 20)[args.level]
  metadata = {
    "checkpoint": str(args.checkpoint),
    "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "seed": cfg.seed,
    "step_height_cm": height_cm,
    "selected_env_ids": selected,
    "labels": names,
    "selection": "first environment matching each predefined pose/region, before rollout; no success selection",
    "fps": 25,
    "duration_s": 23,
    "speed": "1x",
    "termination": "disabled for uninterrupted viewing; no automatic reset or replay",
    "scene": "new multi-position stair reset protocol, not historical center reset",
    "observation_corruption": cfg.observations["actor"].enable_corruption,
  }
  writer = imageio.get_writer(
    str(args.output),
    fps=25,
    codec="libx264",
    quality=8,
    macro_block_size=1,
    ffmpeg_params=["-pix_fmt", "yuv420p"],
  )
  for step in range(1150):
    if step % 2 == 0:
      positions = robot.data.root_link_pos_w[selected].cpu().numpy()
      qpos = raw.sim.data.qpos[selected].cpu().numpy()
      qvel = raw.sim.data.qvel[selected].cpu().numpy()
      distance = (robot.data.root_link_pos_w[:, :2] - start).norm(dim=-1)
      canvas = Image.new("RGB", (1280, 966), (16, 20, 28))
      draw = ImageDraw.Draw(canvas)
      draw.text(
        (16, 10),
        f"V36 original 93D projection | {height_cm} cm stairs | 1x | t={step * 0.02:.2f}s",
        font=font,
        fill="white",
      )
      for tile, idx in enumerate(selected):
        data.qpos[:] = qpos[tile]
        data.qvel[:] = qvel[tile]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = positions[tile]
        camera.lookat[2] += 0.25
        renderer.update_scene(data, camera=camera)
        x, y = tile % 2 * 640, 46 + tile // 2 * 460
        canvas.paste(Image.fromarray(renderer.render()), (x, y))
        draw.text((x + 12, y + 403), names[tile], font=font, fill="white")
        draw.text(
          (x + 12, y + 429),
          f"stand hold: {float(hold[idx]) * 0.02:.1f}s | drift: {float(distance[idx]):.2f}m",
          font=font,
          fill=(205, 218, 230),
        )
      writer.append_data(np.asarray(canvas))
      if step in (0, 500, 1000):
        canvas.save(args.output.with_name(args.output.stem + f"_{step:04d}.jpg"))
    with torch.inference_mode():
      obs, _, _, _ = env.step(policy(obs))
    head_height = robot.data.site_pos_w[:, head, 2] - raw._recovery_support_height
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
    stable = (
      (head_height >= 1.10)
      & (upright >= 0.85)
      & (robot.data.root_link_lin_vel_w.norm(dim=-1) < 0.5)
      & (robot.data.root_link_ang_vel_w.norm(dim=-1) < 1.0)
    )
    hold = torch.where(stable, hold + 1, 0)
    distance = (robot.data.root_link_pos_w[:, :2] - start).norm(dim=-1)
    peak_distance = torch.maximum(peak_distance, distance)
    success |= (hold >= 150) & (step + 1 - hold <= 1000) & (peak_distance <= 1.5)
    if not torch.isfinite(raw.sim.data.qpos[selected]).all():
      raise RuntimeError("Nonfinite selected rollout; do not publish incomplete video")
    if step % 250 == 0:
      print("record step", step, flush=True)
  writer.close()
  renderer.close()
  metadata["selected_sr3"] = success[selected].cpu().tolist()
  metadata["selected_peak_displacement_m"] = peak_distance[selected].cpu().tolist()
  args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
  raw.close()
  print("VIDEO_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
  main()
