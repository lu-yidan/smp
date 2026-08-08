"""Data utilities for a paper-faithful FIRM reimplementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

G1_NUM_JOINTS = 29
LAFAN_G1_FRAME_DIM = 3 + 4 + G1_NUM_JOINTS
G1_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
assert len(G1_JOINT_NAMES) == G1_NUM_JOINTS


@dataclass(frozen=True)
class LafanG1Motion:
  """A retargeted LAFAN1 G1 motion.

  The source CSV layout is root position, root quaternion in xyzw order, then
  29 joint positions. The dataset contains kinematic trajectories only.
  """

  source: Path
  frames: np.ndarray
  fps: int = 30

  @property
  def root_pos(self) -> np.ndarray:
    return self.frames[:, :3]

  @property
  def root_quat_xyzw(self) -> np.ndarray:
    return self.frames[:, 3:7]

  @property
  def root_quat_wxyz(self) -> np.ndarray:
    return self.root_quat_xyzw[:, [3, 0, 1, 2]]

  @property
  def joint_pos(self) -> np.ndarray:
    return self.frames[:, 7:]

  @property
  def duration_s(self) -> float:
    return len(self.frames) / self.fps

  @property
  def torso_up_z(self) -> np.ndarray:
    """World-z component of the root-local z axis."""
    qx = self.root_quat_xyzw[:, 0]
    qy = self.root_quat_xyzw[:, 1]
    return 1.0 - 2.0 * (qx * qx + qy * qy)

  @property
  def joint_vel(self) -> np.ndarray:
    return np.gradient(self.joint_pos, axis=0) * self.fps


@dataclass(frozen=True)
class FallCandidate:
  """A candidate fall/recovery interval detected from a long recording."""

  candidate_id: int
  frame_start: int
  frame_end_exclusive: int
  fallen_start: int
  fallen_end_exclusive: int
  direction: str
  duration_s: float
  min_root_height: float
  min_torso_up_z: float
  pre_root_height_mean: float
  pre_torso_up_z_mean: float
  post_root_height_mean: float
  post_torso_up_z_mean: float
  joint_speed_p95: float
  selected: bool = False

  def to_dict(self) -> dict[str, int | float | str | bool]:
    return asdict(self)


def load_lafan_g1_csv(path: str | Path, fps: int = 30) -> LafanG1Motion:
  """Load and validate a headerless retargeted LAFAN1 G1 CSV."""
  source = Path(path)
  if not source.is_file():
    raise FileNotFoundError(source)
  if fps <= 0:
    msg = f"fps must be positive, got {fps}"
    raise ValueError(msg)

  frames = np.loadtxt(source, delimiter=",", dtype=np.float64)
  if frames.ndim != 2 or frames.shape[1] != LAFAN_G1_FRAME_DIM:
    msg = f"{source}: expected shape (T, {LAFAN_G1_FRAME_DIM}), got {frames.shape}"
    raise ValueError(msg)
  if len(frames) < 2:
    msg = f"{source}: expected at least 2 frames, got {len(frames)}"
    raise ValueError(msg)
  if not np.isfinite(frames).all():
    msg = f"{source}: contains NaN or infinity"
    raise ValueError(msg)

  quat_norm = np.linalg.norm(frames[:, 3:7], axis=1)
  max_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
  if max_quat_error > 5e-3:
    msg = f"{source}: quaternion norm error is too large ({max_quat_error:.6f})"
    raise ValueError(msg)

  return LafanG1Motion(source=source, frames=frames, fps=fps)


def _true_intervals(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
  padded = np.pad(mask.astype(np.int8), (1, 1))
  edges = np.flatnonzero(np.diff(padded))
  return [
    (int(start), int(end))
    for start, end in zip(edges[::2], edges[1::2], strict=True)
    if end - start >= min_frames
  ]


def _fill_short_false_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
  if max_gap_frames <= 0:
    return mask
  filled = mask.copy()
  false_runs = _true_intervals(~mask, min_frames=1)
  for start, end in false_runs:
    bounded = start > 0 and end < len(mask)
    if bounded and end - start <= max_gap_frames:
      filled[start:end] = True
  return filled


def _root_yaw_xyzw(quat: np.ndarray) -> float:
  qx, qy, qz, qw = quat
  return float(
    np.arctan2(
      2.0 * (qw * qz + qx * qy),
      1.0 - 2.0 * (qy * qy + qz * qz),
    )
  )


def _estimate_fall_direction(
  motion: LafanG1Motion,
  frame_start: int,
  fallen_start: int,
  fallen_end: int,
) -> str:
  fallen_slice = motion.root_pos[fallen_start:fallen_end]
  min_height_offset = int(np.argmin(fallen_slice[:, 2]))
  impact_frame = fallen_start + min_height_offset
  displacement = motion.root_pos[impact_frame, :2] - motion.root_pos[frame_start, :2]

  yaw = _root_yaw_xyzw(motion.root_quat_xyzw[frame_start])
  cos_yaw = np.cos(yaw)
  sin_yaw = np.sin(yaw)
  local_x = cos_yaw * displacement[0] + sin_yaw * displacement[1]
  local_y = -sin_yaw * displacement[0] + cos_yaw * displacement[1]

  if np.hypot(local_x, local_y) < 0.05:
    return "unknown"
  if abs(local_x) >= abs(local_y):
    return "forward" if local_x >= 0.0 else "backward"
  return "left" if local_y >= 0.0 else "right"


def detect_fall_candidates(
  motion: LafanG1Motion,
  *,
  root_height_threshold: float = 0.50,
  torso_up_z_threshold: float = 0.50,
  min_fallen_s: float = 0.30,
  max_gap_s: float = 0.20,
  pre_roll_s: float = 2.0,
  post_roll_s: float = 3.0,
  selected_candidate: int | None = None,
) -> list[FallCandidate]:
  """Detect long-recording regions that likely contain fall and recovery.

  Detection is intentionally permissive. The returned candidates are for
  visualization and manual review, not automatic acceptance as demonstrations.
  """
  if min_fallen_s <= 0.0:
    msg = f"min_fallen_s must be positive, got {min_fallen_s}"
    raise ValueError(msg)
  if pre_roll_s < 0.0 or post_roll_s < 0.0:
    msg = "pre_roll_s and post_roll_s must be non-negative"
    raise ValueError(msg)

  fallen = (motion.root_pos[:, 2] < root_height_threshold) | (
    motion.torso_up_z < torso_up_z_threshold
  )
  fallen = _fill_short_false_gaps(fallen, max_gap_frames=round(max_gap_s * motion.fps))
  intervals = _true_intervals(
    fallen, min_frames=max(1, round(min_fallen_s * motion.fps))
  )

  pre_frames = round(pre_roll_s * motion.fps)
  post_frames = round(post_roll_s * motion.fps)
  joint_vel = motion.joint_vel
  candidates: list[FallCandidate] = []
  for candidate_id, (fallen_start, fallen_end) in enumerate(intervals):
    frame_start = max(0, fallen_start - pre_frames)
    frame_end = min(len(motion.frames), fallen_end + post_frames)
    pre_slice = slice(frame_start, fallen_start)
    post_slice = slice(fallen_end, frame_end)

    candidates.append(
      FallCandidate(
        candidate_id=candidate_id,
        frame_start=frame_start,
        frame_end_exclusive=frame_end,
        fallen_start=fallen_start,
        fallen_end_exclusive=fallen_end,
        direction=_estimate_fall_direction(
          motion, frame_start, fallen_start, fallen_end
        ),
        duration_s=(frame_end - frame_start) / motion.fps,
        min_root_height=float(motion.root_pos[fallen_start:fallen_end, 2].min()),
        min_torso_up_z=float(motion.torso_up_z[fallen_start:fallen_end].min()),
        pre_root_height_mean=float(motion.root_pos[pre_slice, 2].mean()),
        pre_torso_up_z_mean=float(motion.torso_up_z[pre_slice].mean()),
        post_root_height_mean=float(motion.root_pos[post_slice, 2].mean()),
        post_torso_up_z_mean=float(motion.torso_up_z[post_slice].mean()),
        joint_speed_p95=float(
          np.percentile(np.abs(joint_vel[frame_start:frame_end]), 95)
        ),
        selected=candidate_id == selected_candidate,
      )
    )
  return candidates
