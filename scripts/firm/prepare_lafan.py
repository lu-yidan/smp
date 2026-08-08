"""Detect and export FIRM seed candidates from a long retargeted LAFAN1 CSV.

The committed JSON manifest contains source frame indices and quality statistics.
Derived NPZ clips are written below datasets/ and remain git-ignored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro

from smp.firm.data import FallCandidate, detect_fall_candidates, load_lafan_g1_csv


@dataclass
class Args:
  input_file: Path = Path(
    "/home/d080/workspace/LAFAN1_Retargeting_Dataset/g1/fallAndGetUp2_subject2.csv"
  )
  """Headerless 30 Hz retargeted LAFAN1 G1 CSV."""

  manifest_file: Path = Path("configs/firm/lafan/fallAndGetUp2_subject2.json")
  """Tracked JSON manifest to write."""

  artifact_dir: Path = Path("datasets/firm/lafan")
  """Git-ignored directory for the selected NPZ clip."""

  selected_candidate: int = 3
  """Candidate to mark and export. Use -1 to select none."""

  fps: int = 30
  root_height_threshold: float = 0.50
  torso_up_z_threshold: float = 0.50
  min_fallen_s: float = 0.30
  max_gap_s: float = 0.20
  pre_roll_s: float = 2.0
  post_roll_s: float = 3.0
  num_keyframes: int = 25
  write_artifact: bool = True


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _candidate_by_id(
  candidates: list[FallCandidate], candidate_id: int
) -> FallCandidate:
  for candidate in candidates:
    if candidate.candidate_id == candidate_id:
      return candidate
  msg = (
    f"Candidate {candidate_id} does not exist; valid range is 0..{len(candidates) - 1}"
  )
  raise ValueError(msg)


def _write_selected_artifact(
  args: Args,
  motion_frames: np.ndarray,
  candidate: FallCandidate,
) -> Path:
  start = candidate.frame_start
  end = candidate.frame_end_exclusive
  clip = motion_frames[start:end]
  local_keyframes = np.rint(np.linspace(0, len(clip) - 1, args.num_keyframes)).astype(
    np.int64
  )
  if len(np.unique(local_keyframes)) != args.num_keyframes:
    msg = (
      f"Clip has {len(clip)} frames, too few for {args.num_keyframes} unique keyframes"
    )
    raise ValueError(msg)

  args.artifact_dir.mkdir(parents=True, exist_ok=True)
  output = (
    args.artifact_dir
    / f"{args.input_file.stem}_candidate_{candidate.candidate_id:03d}.npz"
  )
  np.savez_compressed(
    output,
    frames=clip.astype(np.float32),
    root_pos=clip[:, :3].astype(np.float32),
    root_quat_xyzw=clip[:, 3:7].astype(np.float32),
    root_quat_wxyz=clip[:, [6, 3, 4, 5]].astype(np.float32),
    joint_pos=clip[:, 7:].astype(np.float32),
    keyframes=clip[local_keyframes].astype(np.float32),
    keyframe_local_indices=local_keyframes,
    keyframe_source_indices=local_keyframes + start,
    source_frame_start=np.int64(start),
    source_frame_end_exclusive=np.int64(end),
    fps=np.int64(args.fps),
  )
  return output


def main(args: Args) -> None:
  if args.num_keyframes <= 1:
    msg = f"num_keyframes must be greater than 1, got {args.num_keyframes}"
    raise ValueError(msg)

  selected = args.selected_candidate if args.selected_candidate >= 0 else None
  motion = load_lafan_g1_csv(args.input_file, fps=args.fps)
  candidates = detect_fall_candidates(
    motion,
    root_height_threshold=args.root_height_threshold,
    torso_up_z_threshold=args.torso_up_z_threshold,
    min_fallen_s=args.min_fallen_s,
    max_gap_s=args.max_gap_s,
    pre_roll_s=args.pre_roll_s,
    post_roll_s=args.post_roll_s,
    selected_candidate=selected,
  )
  if not candidates:
    msg = f"No fall candidates detected in {args.input_file}"
    raise RuntimeError(msg)

  selected_record = (
    _candidate_by_id(candidates, selected) if selected is not None else None
  )
  manifest = {
    "schema_version": 1,
    "source_dataset": "LAFAN1_Retargeting_Dataset/g1",
    "source_file": args.input_file.name,
    "source_sha256": _sha256(args.input_file),
    "license": "CC-BY-NC-ND-4.0",
    "frame_layout": "root_xyz,root_quat_xyzw,29_joint_positions",
    "fps": args.fps,
    "num_source_frames": len(motion.frames),
    "source_duration_s": motion.duration_s,
    "detector": {
      key: value
      for key, value in asdict(args).items()
      if key
      in {
        "root_height_threshold",
        "torso_up_z_threshold",
        "min_fallen_s",
        "max_gap_s",
        "pre_roll_s",
        "post_roll_s",
      }
    },
    "selected_candidate_id": selected,
    "num_keyframes": args.num_keyframes,
    "candidates": [candidate.to_dict() for candidate in candidates],
  }

  args.manifest_file.parent.mkdir(parents=True, exist_ok=True)
  args.manifest_file.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(f"Wrote {len(candidates)} candidates to {args.manifest_file}")

  if args.write_artifact and selected_record is not None:
    output = _write_selected_artifact(args, motion.frames, selected_record)
    print(
      f"Wrote selected candidate {selected_record.candidate_id:03d} "
      f"({selected_record.duration_s:.2f}s, {selected_record.direction}) to {output}"
    )


if __name__ == "__main__":
  main(tyro.cli(Args))
