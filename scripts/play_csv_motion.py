"""Interactively inspect a retargeted G1 motion CSV in MuJoCo.

This viewer reuses the same ``MotionLoader`` and G1 model as ``csv_to_npz.py``.
It is intentionally a kinematic data viewer: no policy acts on the robot and
no physics integration changes the recorded pose.
"""

from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import torch
import tyro
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
)
from mjlab.viewer.native.keys import (
  KEY_DOWN,
  KEY_END,
  KEY_HOME,
  KEY_L,
  KEY_LEFT,
  KEY_RIGHT,
  KEY_SPACE,
  KEY_UP,
)

from csv_to_npz import JOINT_NAMES


@dataclass
class Cfg:
  input_file: str
  """Headerless 36-column G1 CSV to display."""
  input_fps: int = 30
  output_fps: int = 50
  """Interpolation and playback rate used by SMP conversion."""
  device: str = "cpu"
  start_frame: int = 0
  """Zero-based input CSV frame at which playback begins."""
  end_frame: int = -1
  """Exclusive input CSV end frame; -1 uses the complete motion."""
  speed: float = 1.0
  loop: bool = True
  paused: bool = False
  manifest: str = ""
  """Optional route manifest. Empty auto-detects a sibling manifest.json."""
  dry_run: bool = False
  """Validate and print the motion summary without opening a window."""


def _load_stage_spans(
  input_file: Path,
  manifest_arg: str,
) -> tuple[list[dict[str, object]], Path | None]:
  if manifest_arg:
    manifest_path = Path(manifest_arg).expanduser().resolve()
  else:
    candidate = input_file.parent / "manifest.json"
    manifest_path = candidate if candidate.exists() else None
  if manifest_path is None:
    return [], None
  payload = json.loads(manifest_path.read_text())
  entry = next(
    (clip for clip in payload.get("clips", []) if clip.get("output") == input_file.name),
    None,
  )
  if entry is None:
    return [], manifest_path
  return list(entry.get("stage_spans", [])), manifest_path


def _stage_at_frame(spans: list[dict[str, object]], frame: int) -> str:
  for span in spans:
    start, end = span["frame_span"]
    if int(start) <= frame < int(end):
      return str(span["name"])
  return "unlabelled"


def _build_sim(device: str, output_fps: int) -> tuple[Simulation, Scene]:
  if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"CUDA device requested but unavailable: {device}")
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _input_to_output_frame(frame: int, input_fps: int, output_fps: int) -> int:
  return round(frame * output_fps / input_fps)


def main(cfg: Cfg) -> None:
  input_file = Path(cfg.input_file).expanduser().resolve()
  if not input_file.is_file():
    raise FileNotFoundError(input_file)
  if cfg.input_fps <= 0 or cfg.output_fps <= 0:
    raise ValueError("input_fps and output_fps must be positive")
  if cfg.start_frame < 0:
    raise ValueError("start_frame must be non-negative")
  if cfg.speed <= 0.0:
    raise ValueError("speed must be positive")

  motion = MotionLoader(
    motion_file=str(input_file),
    input_fps=cfg.input_fps,
    output_fps=cfg.output_fps,
    device=cfg.device,
  )
  if motion.output_frames < 3:
    raise ValueError("motion must contain at least three interpolated frames")

  end_input = motion.input_frames if cfg.end_frame < 0 else cfg.end_frame
  if end_input <= cfg.start_frame or end_input > motion.input_frames:
    raise ValueError(
      f"invalid input frame range [{cfg.start_frame}, {end_input}) for "
      f"{motion.input_frames} frames"
    )
  start_output = _input_to_output_frame(
    cfg.start_frame, cfg.input_fps, cfg.output_fps
  )
  end_output = min(
    motion.output_frames,
    _input_to_output_frame(end_input, cfg.input_fps, cfg.output_fps),
  )
  if end_output - start_output < 2:
    raise ValueError("selected frame range is too short")

  stage_spans, manifest_path = _load_stage_spans(input_file, cfg.manifest)
  duration_s = (end_output - start_output) / cfg.output_fps
  print(f"Motion: {input_file}")
  print(
    f"Input: {motion.input_frames} frames at {cfg.input_fps} Hz | "
    f"interpolated: {motion.output_frames} frames at {cfg.output_fps} Hz"
  )
  print(
    f"Selection: input [{cfg.start_frame}, {end_input}) | "
    f"duration {duration_s:.2f} s"
  )
  if manifest_path is not None:
    print(f"Manifest: {manifest_path} | stage labels: {len(stage_spans)}")
  else:
    print("Manifest: none (stage overlay disabled)")

  sim, scene = _build_sim(cfg.device, cfg.output_fps)
  robot: Entity = scene["robot"]
  joint_ids = robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0]
  actions: queue.SimpleQueue[str] = queue.SimpleQueue()

  def key_callback(key: int) -> None:
    mapping = {
      KEY_SPACE: "pause",
      KEY_RIGHT: "next",
      KEY_LEFT: "previous",
      KEY_UP: "faster",
      KEY_DOWN: "slower",
      KEY_HOME: "home",
      KEY_END: "end",
      KEY_L: "loop",
    }
    if key in mapping:
      actions.put(mapping[key])

  current = start_output
  paused = cfg.paused
  loop = cfg.loop
  speed = cfg.speed
  dirty = True

  def write_frame(index: int) -> None:
    root = robot.data.default_root_state.clone()
    root[:, :3] = motion.motion_base_poss[index : index + 1]
    root[:, :2] += scene.env_origins[:, :2]
    root[:, 3:7] = motion.motion_base_rots[index : index + 1]
    root[:, 7:10] = motion.motion_base_lin_vels[index : index + 1]
    root[:, 10:13] = motion.motion_base_ang_vels[index : index + 1]
    robot.write_root_state_to_sim(root)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = motion.motion_dof_poss[index : index + 1]
    joint_vel[:, joint_ids] = motion.motion_dof_vels[index : index + 1]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    sim.mj_data.qpos[:] = sim.data.qpos[0].cpu().numpy()
    sim.mj_data.qvel[:] = sim.data.qvel[0].cpu().numpy()
    mujoco.mj_forward(sim.mj_model, sim.mj_data)

  write_frame(current)
  if cfg.dry_run:
    print("MuJoCo validation: first selected frame forwarded successfully")
    return
  viewer = mujoco.viewer.launch_passive(
    sim.mj_model,
    sim.mj_data,
    key_callback=key_callback,
    show_left_ui=False,
    show_right_ui=False,
  )
  if viewer is None:
    raise RuntimeError("failed to launch MuJoCo viewer")
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
  viewer.cam.trackbodyid = robot.indexing.root_body_id
  viewer.cam.fixedcamid = -1
  viewer.cam.distance = 2.6
  viewer.cam.elevation = -10.0
  viewer.cam.azimuth = 90.0

  print("Keys: Space pause | Left/Right step | Up/Down speed | Home/End | L loop")
  accumulator = 0.0
  last_time = time.monotonic()
  try:
    while viewer.is_running():
      while True:
        try:
          action = actions.get_nowait()
        except queue.Empty:
          break
        if action == "pause":
          paused = not paused
        elif action == "next":
          paused = True
          current = min(current + 1, end_output - 1)
          dirty = True
        elif action == "previous":
          paused = True
          current = max(current - 1, start_output)
          dirty = True
        elif action == "faster":
          speed = min(speed * 2.0, 8.0)
        elif action == "slower":
          speed = max(speed / 2.0, 1.0 / 8.0)
        elif action == "home":
          current = start_output
          dirty = True
        elif action == "end":
          current = end_output - 1
          paused = True
          dirty = True
        elif action == "loop":
          loop = not loop

      now = time.monotonic()
      elapsed = min(now - last_time, 0.25)
      last_time = now
      if not paused:
        accumulator += elapsed * cfg.output_fps * speed
        advance = int(accumulator)
        if advance:
          accumulator -= advance
          next_frame = current + advance
          if next_frame >= end_output:
            if loop:
              next_frame = start_output + (next_frame - start_output) % (
                end_output - start_output
              )
            else:
              next_frame = end_output - 1
              paused = True
          current = next_frame
          dirty = True

      if dirty:
        write_frame(current)
        dirty = False
      input_frame = round(current * cfg.input_fps / cfg.output_fps)
      stage = _stage_at_frame(stage_spans, input_frame)
      viewer.set_texts(
        (
          mujoco.mjtFontScale.mjFONTSCALE_150.value,
          mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
          "File frame\nTime\nStage\nStatus\nSpeed\nLoop",
          f"{input_frame}/{motion.input_frames - 1}\n"
          f"{input_frame / cfg.input_fps:.2f} s\n"
          f"{stage}\n"
          f"{'PAUSED' if paused else 'RUNNING'}\n"
          f"{speed:g}x\n"
          f"{'on' if loop else 'off'}",
        )
      )
      viewer.sync()
      time.sleep(0.005)
  finally:
    viewer.close()


if __name__ == "__main__":
  main(tyro.cli(Cfg))
