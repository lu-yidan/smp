"""Physically validate one selected LAFAN1 fall/recovery candidate.

The script uses MJLab's quaternion interpolation and the repository's 29-DoF G1
model. It produces a git-ignored state artifact and keyframe contact sheet, plus
a tracked JSON validation summary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

from smp.firm.data import G1_JOINT_NAMES
from smp.utils import detect_device


@dataclass
class Args:
  manifest_file: Path = Path("configs/firm/lafan/fallAndGetUp2_subject2.json")
  dataset_root: Path = Path("/home/d080/workspace/LAFAN1_Retargeting_Dataset/g1")
  artifact_dir: Path = Path("datasets/firm/lafan")
  validation_file: Path = Path(
    "configs/firm/lafan/fallAndGetUp2_subject2_candidate_003_validation.json"
  )
  output_fps: int = 50
  ground_clearance: float = 0.05
  stand_start_source_frame: int = 1597
  device: str = ""
  render_contact_sheet: bool = True


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _selected_candidate(manifest: dict[str, Any]) -> dict[str, Any]:
  selected_id = manifest["selected_candidate_id"]
  for candidate in manifest["candidates"]:
    if candidate["candidate_id"] == selected_id:
      return candidate
  msg = f"Selected candidate {selected_id} is missing from manifest"
  raise ValueError(msg)


def _setup_sim(device: str, output_fps: int) -> tuple[Simulation, Scene, Entity]:
  sim_cfg = SimulationCfg(nconmax=64)
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene, scene["robot"]


def _write_state(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  joint_indexes: list[int],
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> None:
  root_state = robot.data.default_root_state.clone()
  root_state[:, :3] = root_pos
  root_state[:, :2] += scene.env_origins[:, :2]
  root_state[:, 3:7] = root_quat
  root_state[:, 7:10] = root_lin_vel
  root_state[:, 10:13] = root_ang_vel
  robot.write_root_state_to_sim(root_state)

  full_joint_pos = robot.data.default_joint_pos.clone()
  full_joint_vel = robot.data.default_joint_vel.clone()
  full_joint_pos[:, joint_indexes] = joint_pos
  full_joint_vel[:, joint_indexes] = joint_vel
  robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel)
  sim.forward()
  scene.update(sim.mj_model.opt.timestep)


def _render_contact_sheet(
  images: list[np.ndarray],
  source_indices: np.ndarray,
  output: Path,
) -> None:
  columns = 5
  rows = int(np.ceil(len(images) / columns))
  figure, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows))
  for axis in np.asarray(axes).reshape(-1):
    axis.axis("off")
  for axis, image, source_idx in zip(
    np.asarray(axes).reshape(-1), images, source_indices, strict=False
  ):
    axis.imshow(image)
    axis.set_title(f"source frame {source_idx}")
    axis.axis("off")
  figure.tight_layout()
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=140)
  plt.close(figure)


@torch.no_grad()
def main(args: Args) -> None:
  if args.output_fps <= 0:
    msg = f"output_fps must be positive, got {args.output_fps}"
    raise ValueError(msg)
  if args.ground_clearance < 0.0:
    msg = f"ground_clearance must be non-negative, got {args.ground_clearance}"
    raise ValueError(msg)

  manifest = json.loads(args.manifest_file.read_text(encoding="utf-8"))
  candidate = _selected_candidate(manifest)
  source = args.dataset_root / manifest["source_file"]
  if _sha256(source) != manifest["source_sha256"]:
    msg = f"{source}: SHA256 does not match manifest"
    raise ValueError(msg)

  frame_start = int(candidate["frame_start"])
  frame_end = int(candidate["frame_end_exclusive"])
  device = args.device or detect_device()
  motion = MotionLoader(
    motion_file=str(source),
    input_fps=int(manifest["fps"]),
    output_fps=args.output_fps,
    device=device,
    line_range=(frame_start + 1, frame_end),
  )

  sim, scene, robot = _setup_sim(device, args.output_fps)
  joint_indexes = robot.find_joints(list(G1_JOINT_NAMES), preserve_order=True)[0]

  ground_shifts: list[torch.Tensor] = []
  scene.reset()
  for frame in range(motion.output_frames):
    _write_state(
      sim,
      scene,
      robot,
      joint_indexes,
      motion.motion_base_poss[frame : frame + 1],
      motion.motion_base_rots[frame : frame + 1],
      motion.motion_base_lin_vels[frame : frame + 1],
      motion.motion_base_ang_vels[frame : frame + 1],
      motion.motion_dof_poss[frame : frame + 1],
      motion.motion_dof_vels[frame : frame + 1],
    )
    lowest_body_z = robot.data.body_link_pos_w[0, :, 2].min()
    ground_shifts.append(args.ground_clearance - lowest_body_z)

  shift = torch.stack(ground_shifts)
  aligned_root_pos = motion.motion_base_poss.clone()
  aligned_root_pos[:, 2] += shift
  output_dt = 1.0 / args.output_fps
  aligned_root_lin_vel = torch.gradient(aligned_root_pos, spacing=output_dt, dim=0)[0]

  if not frame_start < args.stand_start_source_frame < frame_end:
    msg = (
      "stand_start_source_frame must lie inside the selected clip, got "
      f"{args.stand_start_source_frame} for [{frame_start}, {frame_end})"
    )
    raise ValueError(msg)
  motion_keyframe_count = int(manifest["num_keyframes"]) - 1
  motion_keyframe_end = round(
    (args.stand_start_source_frame - frame_start)
    / int(manifest["fps"])
    * args.output_fps
  )
  motion_keyframe_end = min(motion_keyframe_end, motion.output_frames - 2)
  motion_keyframes = torch.round(
    torch.linspace(
      0,
      motion_keyframe_end,
      motion_keyframe_count,
      device=device,
    )
  ).long()
  keyframe_indices = torch.cat(
    [motion_keyframes, motion_keyframes.new_tensor([motion.output_frames - 1])]
  )
  keyframe_set = set(keyframe_indices.cpu().tolist())

  renderer = None
  if args.render_contact_sheet:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

  body_pos: list[np.ndarray] = []
  body_quat: list[np.ndarray] = []
  body_lin_vel: list[np.ndarray] = []
  body_ang_vel: list[np.ndarray] = []
  sim_joint_pos: list[np.ndarray] = []
  sim_joint_vel: list[np.ndarray] = []
  aligned_lowest_body_z: list[float] = []
  images: list[np.ndarray] = []

  scene.reset()
  for frame in range(motion.output_frames):
    _write_state(
      sim,
      scene,
      robot,
      joint_indexes,
      aligned_root_pos[frame : frame + 1],
      motion.motion_base_rots[frame : frame + 1],
      aligned_root_lin_vel[frame : frame + 1],
      motion.motion_base_ang_vels[frame : frame + 1],
      motion.motion_dof_poss[frame : frame + 1],
      motion.motion_dof_vels[frame : frame + 1],
    )
    body_pos.append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    body_quat.append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    body_lin_vel.append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    body_ang_vel.append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())
    sim_joint_pos.append(robot.data.joint_pos[0].cpu().numpy().copy())
    sim_joint_vel.append(robot.data.joint_vel[0].cpu().numpy().copy())
    aligned_lowest_body_z.append(
      float(robot.data.body_link_pos_w[0, :, 2].min().item())
    )
    if renderer is not None and frame in keyframe_set:
      renderer.update(sim.data)
      images.append(renderer.render())

  body_pos_array = np.stack(body_pos)
  body_quat_array = np.stack(body_quat)
  body_lin_vel_array = np.stack(body_lin_vel)
  body_ang_vel_array = np.stack(body_ang_vel)
  sim_joint_pos_array = np.stack(sim_joint_pos)
  sim_joint_vel_array = np.stack(sim_joint_vel)
  joint_pos = motion.motion_dof_poss
  joint_vel = motion.motion_dof_vels
  joint_acc = torch.gradient(joint_vel, spacing=output_dt, dim=0)[0]

  limits = robot.data.joint_pos_limits[0, joint_indexes]
  violation = torch.maximum(
    torch.clamp(limits[:, 0] - joint_pos, min=0.0),
    torch.clamp(joint_pos - limits[:, 1], min=0.0),
  )
  shift_velocity = torch.gradient(shift, spacing=output_dt, dim=0)[0]
  source_keyframes = frame_start + np.rint(
    keyframe_indices.cpu().numpy() / args.output_fps * int(manifest["fps"])
  ).astype(np.int64)
  source_keyframes = np.clip(source_keyframes, frame_start, frame_end - 1)

  args.artifact_dir.mkdir(parents=True, exist_ok=True)
  stem = f"{source.stem}_candidate_{int(candidate['candidate_id']):03d}_validated"
  artifact_file = args.artifact_dir / f"{stem}.npz"
  np.savez_compressed(
    artifact_file,
    fps=np.int64(args.output_fps),
    body_names=np.asarray(robot.body_names),
    root_pos=aligned_root_pos.cpu().numpy().astype(np.float32),
    root_quat_wxyz=motion.motion_base_rots.cpu().numpy().astype(np.float32),
    root_lin_vel=aligned_root_lin_vel.cpu().numpy().astype(np.float32),
    root_ang_vel=motion.motion_base_ang_vels.cpu().numpy().astype(np.float32),
    joint_pos=sim_joint_pos_array.astype(np.float32),
    joint_vel=sim_joint_vel_array.astype(np.float32),
    body_pos_w=body_pos_array.astype(np.float32),
    body_quat_w=body_quat_array.astype(np.float32),
    body_lin_vel_w=body_lin_vel_array.astype(np.float32),
    body_ang_vel_w=body_ang_vel_array.astype(np.float32),
    keyframe_indices=keyframe_indices.cpu().numpy(),
    keyframe_source_indices=source_keyframes,
  )

  contact_sheet_file = args.artifact_dir / f"{stem}_keyframes.png"
  if images:
    _render_contact_sheet(images, source_keyframes, contact_sheet_file)

  summary = {
    "schema_version": 1,
    "source_file": source.name,
    "source_sha256": manifest["source_sha256"],
    "candidate_id": candidate["candidate_id"],
    "source_frame_start": frame_start,
    "source_frame_end_exclusive": frame_end,
    "input_fps": manifest["fps"],
    "output_fps": args.output_fps,
    "output_frames": motion.output_frames,
    "duration_s": motion.duration,
    "num_keyframes": len(keyframe_indices),
    "motion_keyframe_count": motion_keyframe_count,
    "motion_keyframe_end_output_frame": motion_keyframe_end,
    "stand_start_source_frame": args.stand_start_source_frame,
    "keyframe_output_indices": keyframe_indices.cpu().tolist(),
    "keyframe_source_indices": source_keyframes.tolist(),
    "artifact_sha256": _sha256(artifact_file),
    "direction": candidate["direction"],
    "csv_joint_names": list(G1_JOINT_NAMES),
    "sim_joint_names": list(robot.joint_names),
    "csv_to_sim_joint_indexes": joint_indexes,
    "max_joint_limit_violation_rad": float(violation.max().item()),
    "joint_limit_violation_samples": int((violation > 1e-6).sum().item()),
    "max_abs_joint_speed_rad_s": float(joint_vel.abs().max().item()),
    "p95_abs_joint_speed_rad_s": float(torch.quantile(joint_vel.abs(), 0.95).item()),
    "max_abs_joint_acc_rad_s2": float(joint_acc.abs().max().item()),
    "p95_abs_joint_acc_rad_s2": float(torch.quantile(joint_acc.abs(), 0.95).item()),
    "ground_shift_min_m": float(shift.min().item()),
    "ground_shift_max_m": float(shift.max().item()),
    "max_abs_ground_shift_speed_m_s": float(shift_velocity.abs().max().item()),
    "aligned_lowest_body_z_min_m": min(aligned_lowest_body_z),
    "aligned_lowest_body_z_max_m": max(aligned_lowest_body_z),
    "artifact_file": str(artifact_file),
    "contact_sheet_file": str(contact_sheet_file) if images else None,
  }
  args.validation_file.parent.mkdir(parents=True, exist_ok=True)
  args.validation_file.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(Args))
