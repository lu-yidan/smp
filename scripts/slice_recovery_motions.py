"""Extract balanced fall-to-stand clips from LAFAN G1 recovery CSV files.

The source CSV layout is the same 36-column, headerless format consumed by
scripts/csv_to_npz.py: root xyz, root quaternion xyzw, and 29 G1 joints.
Candidates start shortly before a sustained fallen interval and end after the
next sustained standing interval. Very long waits on the ground are cropped so
they do not dominate diffusion-prior pretraining.

The tool writes one CSV per recovery, optional sagittal mirrors, a reproducible
manifest with source hashes, and per-source diagnostic plots.

Example:
  uv run scripts/slice_recovery_motions.py \
    --input-dir /path/to/LAFAN1_Retargeting_Dataset/g1 \
    --output-dir datasets/csv/getup_lafan6_sliced
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

from csv_to_npz import JOINT_NAMES


@dataclass(frozen=True)
class FrameSpan:
  start: int
  end: int

  @property
  def length(self) -> int:
    return self.end - self.start


@dataclass
class Cfg:
  input_dir: str
  """Directory containing fallAndGetUp*.csv files."""
  output_dir: str = "datasets/csv/getup_lafan6_sliced"
  """Destination for sliced CSVs, manifest, and diagnostic plots."""
  glob: str = "fallAndGetUp*.csv"
  fps: int = 30
  fallen_root_height: float = 0.52
  fallen_upright: float = 0.50
  standing_root_height: float = 0.70
  standing_upright: float = 0.80
  min_fallen_s: float = 0.50
  min_standing_s: float = 1.00
  pre_fall_s: float = 0.35
  post_stand_s: float = 1.00
  max_clip_s: float = 12.0
  max_transition_s: float = 3.0
  """Maximum gap between leaving the fallen set and sustained standing.

  Longer gaps usually pair an unrelated early fall with a later standing
  segment and inject long non-recovery motion into the prior dataset.
  """
  mirror: bool = True
  write_plots: bool = True
  overwrite: bool = False


def _runs(mask: np.ndarray, minimum: int) -> list[FrameSpan]:
  edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
  starts = np.flatnonzero(edges == 1)
  ends = np.flatnonzero(edges == -1)
  return [
    FrameSpan(int(start), int(end))
    for start, end in zip(starts, ends, strict=True)
    if end - start >= minimum
  ]


def _root_metrics(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  quat = data[:, 3:7].copy()  # xyzw
  quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-8)
  x, y, z, w = (quat[:, i] for i in range(4))

  # Body-frame gravity is R^T * [0, 0, -1]. Its negative z component is one
  # when the root is upright and zero when it lies on its front/back/side.
  r20 = 2.0 * (x * z - w * y)
  r21 = 2.0 * (y * z + w * x)
  r22 = 1.0 - 2.0 * (x * x + y * y)
  projected_gravity = np.stack((-r20, -r21, -r22), axis=-1)
  upright = np.clip(-projected_gravity[:, 2], 0.0, 1.0)
  return data[:, 2], upright, projected_gravity


def _find_recoveries(data: np.ndarray, cfg: Cfg) -> list[dict]:
  root_z, upright, projected_gravity = _root_metrics(data)
  fallen_mask = (root_z < cfg.fallen_root_height) | (upright < cfg.fallen_upright)
  standing_mask = (root_z > cfg.standing_root_height) & (upright > cfg.standing_upright)
  fallen_runs = _runs(fallen_mask, round(cfg.min_fallen_s * cfg.fps))
  standing_runs = _runs(standing_mask, round(cfg.min_standing_s * cfg.fps))

  pre_frames = round(cfg.pre_fall_s * cfg.fps)
  post_frames = round(cfg.post_stand_s * cfg.fps)
  max_clip_frames = round(cfg.max_clip_s * cfg.fps)
  max_transition_frames = round(cfg.max_transition_s * cfg.fps)
  recoveries: list[dict] = []
  consumed_until = -1

  for fallen in fallen_runs:
    if fallen.start < consumed_until:
      continue
    next_standing = next(
      (
        standing
        for standing in standing_runs
        if standing.start > fallen.end
        and standing.start - fallen.end <= max_transition_frames
      ),
      None,
    )
    if next_standing is None:
      continue

    end = min(len(data), next_standing.start + post_frames)
    start = max(0, fallen.start - pre_frames, end - max_clip_frames)
    if end - start < round(2.0 * cfg.fps):
      continue

    fallen_slice = projected_gravity[fallen.start : fallen.end]
    representative = np.median(fallen_slice, axis=0)
    dominant_axis = int(np.argmax(np.abs(representative[:2])))
    axis = ("pitch", "roll")[dominant_axis]
    sign = "pos" if representative[dominant_axis] >= 0.0 else "neg"
    recoveries.append(
      {
        "clip": FrameSpan(start, end),
        "fallen": fallen,
        "standing": next_standing,
        "lying_orientation": f"{axis}_{sign}",
        "projected_gravity_median": representative.tolist(),
        "transition_gap_s": (next_standing.start - fallen.end) / cfg.fps,
        "truncated_before_fallen_s": max(0, start - fallen.start) / cfg.fps,
        "max_clip_truncated": start > max(0, fallen.start - pre_frames),
      }
    )
    consumed_until = next_standing.end

  return recoveries


def _joint_mirror_spec() -> tuple[np.ndarray, np.ndarray]:
  name_to_index = {name: i for i, name in enumerate(JOINT_NAMES)}
  mirrored_indices = np.empty(len(JOINT_NAMES), dtype=np.int64)
  signs = np.ones(len(JOINT_NAMES), dtype=np.float32)
  negate_tokens = ("_roll_joint", "_yaw_joint")

  for dst, name in enumerate(JOINT_NAMES):
    source_name = name
    if name.startswith("left_"):
      source_name = "right_" + name.removeprefix("left_")
    elif name.startswith("right_"):
      source_name = "left_" + name.removeprefix("right_")
    mirrored_indices[dst] = name_to_index[source_name]
    if any(token in name for token in negate_tokens):
      signs[dst] = -1.0

  if sorted(mirrored_indices.tolist()) != list(range(len(JOINT_NAMES))):
    raise RuntimeError("Joint mirror mapping is not a permutation")
  return mirrored_indices, signs


def _mirror_motion(data: np.ndarray) -> np.ndarray:
  """Mirror a G1 CSV motion across the sagittal x-z plane."""
  mirrored = data.copy()
  mirrored[:, 1] *= -1.0

  # Quaternion reflection M R M for M=diag(1,-1,1). In xyzw form its proper
  # rotation quaternion is (-x, y, -z, w).
  mirrored[:, 3] *= -1.0
  mirrored[:, 5] *= -1.0

  indices, signs = _joint_mirror_spec()
  mirrored[:, 7:] = data[:, 7:][:, indices] * signs[None, :]
  return mirrored


def _validate_mirror(data: np.ndarray) -> None:
  """Check algebraic invariants before emitting augmented motion."""
  twice = _mirror_motion(_mirror_motion(data))
  if not np.allclose(twice, data, atol=1.0e-6, rtol=1.0e-6):
    raise RuntimeError("Motion mirror is not an involution")
  original_norm = np.linalg.norm(data[:, 3:7], axis=-1)
  mirrored_norm = np.linalg.norm(_mirror_motion(data)[:, 3:7], axis=-1)
  if not np.allclose(mirrored_norm, original_norm, atol=1.0e-6, rtol=1.0e-6):
    raise RuntimeError("Motion mirror changed root quaternion norms")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_plot(
  source: Path,
  data: np.ndarray,
  recoveries: list[dict],
  plot_path: Path,
  cfg: Cfg,
) -> None:
  root_z, upright, _ = _root_metrics(data)
  time = np.arange(len(data)) / cfg.fps
  fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
  axes[0].plot(time, root_z, linewidth=0.8)
  axes[0].axhline(cfg.fallen_root_height, color="tab:red", linestyle="--")
  axes[0].axhline(cfg.standing_root_height, color="tab:green", linestyle="--")
  axes[0].set_ylabel("root z (m)")
  axes[1].plot(time, upright, linewidth=0.8)
  axes[1].axhline(cfg.fallen_upright, color="tab:red", linestyle="--")
  axes[1].axhline(cfg.standing_upright, color="tab:green", linestyle="--")
  axes[1].set_ylabel("root upright")
  axes[1].set_xlabel("time (s)")
  for index, recovery in enumerate(recoveries):
    span = recovery["clip"]
    for axis in axes:
      axis.axvspan(
        span.start / cfg.fps,
        span.end / cfg.fps,
        alpha=0.16,
        color=f"C{index % 10}",
      )
      axis.text(
        span.start / cfg.fps,
        0.02,
        str(index),
        transform=axis.get_xaxis_transform(),
      )
  fig.suptitle(f"{source.name}: {len(recoveries)} recovery candidates")
  fig.tight_layout()
  fig.savefig(plot_path, dpi=160)
  plt.close(fig)


def _write_csv(path: Path, data: np.ndarray, overwrite: bool) -> None:
  if path.exists() and not overwrite:
    raise FileExistsError(f"{path} exists; pass --overwrite True to replace it")
  np.savetxt(path, data, delimiter=",", fmt="%.6f")


def main(cfg: Cfg) -> None:
  input_dir = Path(cfg.input_dir).expanduser().resolve()
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  source_files = sorted(input_dir.glob(cfg.glob))
  if not source_files:
    raise FileNotFoundError(f"No {cfg.glob!r} files under {input_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)
  existing_outputs = list(output_dir.glob("*.csv"))
  if (
    existing_outputs or (output_dir / "manifest.json").exists()
  ) and not cfg.overwrite:
    raise FileExistsError(
      f"{output_dir} already contains a generated dataset; "
      "choose an empty directory or pass --overwrite True"
    )
  plot_dir = output_dir / "diagnostics"
  if cfg.write_plots:
    plot_dir.mkdir(parents=True, exist_ok=True)

  manifest: dict = {
    "format_version": 2,
    "config": asdict(cfg),
    "input_dir": str(input_dir),
    "clips": [],
    "sources": [],
  }
  orientation_counts: dict[str, int] = {}

  for source in source_files:
    data = np.loadtxt(source, delimiter=",", dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 7 + len(JOINT_NAMES):
      raise ValueError(
        f"{source}: expected (*, {7 + len(JOINT_NAMES)}) CSV, got {data.shape}"
      )
    recoveries = _find_recoveries(data, cfg)
    if cfg.mirror:
      _validate_mirror(data)
    print(
      f"{source.name}: frames={len(data)} seconds={len(data) / cfg.fps:.1f} "
      f"recoveries={len(recoveries)}"
    )
    source_hash = _sha256(source)
    manifest["sources"].append(
      {
        "file": source.name,
        "sha256": source_hash,
        "num_frames": len(data),
        "num_recoveries": len(recoveries),
      }
    )

    if cfg.write_plots:
      _write_plot(
        source,
        data,
        recoveries,
        plot_dir / f"{source.stem}.png",
        cfg,
      )

    for index, recovery in enumerate(recoveries):
      span: FrameSpan = recovery["clip"]
      orientation = recovery["lying_orientation"]
      orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1
      clip = data[span.start : span.end]
      stem = f"{source.stem}__recovery_{index:03d}"
      clip_path = output_dir / f"{stem}.csv"
      _write_csv(clip_path, clip, cfg.overwrite)
      record = {
        "name": stem,
        "source": source.name,
        "source_sha256": source_hash,
        "source_frame_span": [span.start, span.end],
        "fallen_frame_span": [
          recovery["fallen"].start,
          recovery["fallen"].end,
        ],
        "standing_frame_span": [
          recovery["standing"].start,
          recovery["standing"].end,
        ],
        "num_frames": len(clip),
        "duration_s": len(clip) / cfg.fps,
        "lying_orientation": orientation,
        "projected_gravity_median": recovery["projected_gravity_median"],
        "transition_gap_s": recovery["transition_gap_s"],
        "truncated_before_fallen_s": recovery["truncated_before_fallen_s"],
        "max_clip_truncated": recovery["max_clip_truncated"],
        "mirrored": False,
        "output": clip_path.name,
      }
      manifest["clips"].append(record)
      print(
        f"  {stem}: frames=[{span.start},{span.end}) "
        f"duration={len(clip) / cfg.fps:.2f}s orientation={orientation}"
      )

      if cfg.mirror:
        mirrored_path = output_dir / f"{stem}__mirror.csv"
        _write_csv(mirrored_path, _mirror_motion(clip), cfg.overwrite)
        mirrored_record = {
          **record,
          "name": f"{stem}__mirror",
          "mirrored": True,
          "output": mirrored_path.name,
        }
        manifest["clips"].append(mirrored_record)

  manifest["summary"] = {
    "num_sources": len(source_files),
    "num_output_clips": len(manifest["clips"]),
    "num_unmirrored_clips": sum(not clip["mirrored"] for clip in manifest["clips"]),
    "total_output_frames": sum(clip["num_frames"] for clip in manifest["clips"]),
    "orientation_counts_unmirrored": orientation_counts,
  }
  manifest_path = output_dir / "manifest.json"
  if manifest_path.exists() and not cfg.overwrite:
    raise FileExistsError(
      f"{manifest_path} exists; pass --overwrite True to replace it"
    )
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  print(json.dumps(manifest["summary"], indent=2))
  print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
