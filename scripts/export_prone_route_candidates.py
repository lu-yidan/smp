"""Export auditable prone-recovery candidates for visual review.

The source recovery dataset remains untouched. Candidate CSVs are copied into a
new directory with an automatic stage manifest and flat JSONL review records.
Automatic labels are diagnostic only; the output is deliberately marked as not
approved for training until a human watches the clips.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass(frozen=True)
class Span:
  start: int
  end: int


@dataclass
class Cfg:
  input_dir: str = "datasets/csv/getup_lafan6_sliced"
  """Existing V6 sliced CSV directory with manifest.json."""
  npz_dir: str = "datasets/npz/getup_lafan6_sliced"
  """Existing 50 Hz NPZ windows generated from input_dir."""
  output_dir: str = "datasets/csv/getup_lafan_prone_routes_v7_candidates"
  """New review-only candidate directory. It must not already contain files."""
  lying_orientation: str = "pitch_pos"
  include_mirrors: bool = True
  input_fps: int = 30
  output_fps: int = 50
  max_clip_s: float = 8.0
  max_kneel_to_crouch_s: float = 3.0
  min_foot_height: float = -0.05


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _runs(mask: np.ndarray, minimum: int) -> list[Span]:
  edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
  starts = np.flatnonzero(edges == 1)
  ends = np.flatnonzero(edges == -1)
  return [
    Span(int(start), int(end))
    for start, end in zip(starts, ends, strict=True)
    if end - start >= minimum
  ]


def _first_run(mask: np.ndarray, minimum: int, after: int) -> Span | None:
  return next((span for span in _runs(mask, minimum) if span.start >= after), None)


def _output_to_input_frame(
  endpoint_index: int,
  window_size: int,
  input_fps: int,
  output_fps: int,
  num_input_frames: int,
) -> int:
  output_frame = endpoint_index + window_size - 1
  return min(num_input_frames, round(output_frame * input_fps / output_fps))


def _stage_analysis(
  npz_path: Path,
  num_input_frames: int,
  cfg: Cfg,
) -> dict[str, object]:
  archive = np.load(npz_path)
  endpoints = archive["windows"][:, -1, :]
  window_size = int(archive["window_size"][0])

  root_z = endpoints[:, 2]
  gravity_x = -endpoints[:, 5]
  upright = np.clip(endpoints[:, 8], 0.0, 1.0)
  left_knee = endpoints[:, 12]
  right_knee = endpoints[:, 18]
  left_foot_z = root_z + endpoints[:, 40]
  right_foot_z = root_z + endpoints[:, 43]
  left_wrist_z = root_z + endpoints[:, 49]
  right_wrist_z = root_z + endpoints[:, 52]

  prone = (gravity_x > 0.45) & (root_z < 0.62) & (upright < 0.55)
  upper_body_supported = (
    (root_z < 0.78)
    & (upright > 0.35)
    & (np.maximum(left_wrist_z, right_wrist_z) < 0.38)
  )
  kneeling = (
    (root_z > 0.38)
    & (root_z < 0.78)
    & (upright > 0.45)
    & (np.minimum(left_foot_z, right_foot_z) < 0.25)
    & (np.minimum(left_knee, right_knee) > 0.40)
  )
  crouch = (
    (root_z > 0.48)
    & (root_z < 0.86)
    & (upright > 0.68)
    & (np.maximum(left_foot_z, right_foot_z) < 0.20)
    & (np.minimum(left_knee, right_knee) > 0.45)
  )
  standing = (root_z > 0.70) & (upright > 0.80)

  prone_runs = _runs(prone, minimum=3)
  # Clips commonly begin with a short pre-fall standing interval. Select the
  # dominant prone dwell first, then search for the recovery stand after it.
  prone_run = max(prone_runs, key=lambda span: span.end - span.start, default=None)
  after_prone = prone_run.end if prone_run is not None else 0
  stand_run = _first_run(standing, minimum=10, after=after_prone)
  kneel_run = _first_run(kneeling, minimum=3, after=after_prone)
  after_kneel = kneel_run.start if kneel_run is not None else after_prone
  crouch_run = _first_run(crouch, minimum=3, after=after_kneel)
  after_crouch = crouch_run.start if crouch_run is not None else after_kneel
  if stand_run is None or stand_run.start < after_crouch:
    stand_run = _first_run(standing, minimum=10, after=after_crouch)

  output_boundaries = {
    "prone_start": prone_run.start if prone_run else -1,
    "support_start": prone_run.end if prone_run else -1,
    "kneeling_start": kneel_run.start if kneel_run else -1,
    "crouched_start": crouch_run.start if crouch_run else -1,
    "standing_start": stand_run.start if stand_run else -1,
  }
  input_boundaries = {
    name: (
      _output_to_input_frame(
        value, window_size, cfg.input_fps, cfg.output_fps, num_input_frames
      )
      if value >= 0
      else -1
    )
    for name, value in output_boundaries.items()
  }

  flags: list[str] = []
  if any(value < 0 for value in input_boundaries.values()):
    flags.append("missing_auto_stage")
  ordered = [value for value in input_boundaries.values() if value >= 0]
  if len(ordered) == len(input_boundaries) and ordered != sorted(ordered):
    flags.append("non_monotonic_stages")
  duration_s = num_input_frames / cfg.input_fps
  if duration_s > cfg.max_clip_s:
    flags.append("long_clip")
  kneel_to_crouch_s = -1.0
  if input_boundaries["kneeling_start"] >= 0 and input_boundaries["crouched_start"] >= 0:
    kneel_to_crouch_s = (
      input_boundaries["crouched_start"] - input_boundaries["kneeling_start"]
    ) / cfg.input_fps
    if kneel_to_crouch_s > cfg.max_kneel_to_crouch_s:
      flags.append("long_kneel_to_crouch")
  minimum_foot_z = float(min(left_foot_z.min(), right_foot_z.min()))
  if minimum_foot_z < cfg.min_foot_height:
    flags.append("foot_below_review_threshold")

  stage_spans: list[dict[str, object]] = []
  if not any(value < 0 for value in input_boundaries.values()):
    p0 = input_boundaries["prone_start"]
    p1 = input_boundaries["support_start"]
    p2 = input_boundaries["kneeling_start"]
    p3 = input_boundaries["crouched_start"]
    p4 = input_boundaries["standing_start"]
    stage_spans = [
      {"name": "falling", "frame_span": [0, p0]},
      {"name": "prone", "frame_span": [p0, p1]},
      {"name": "upper_body_supported", "frame_span": [p1, p2]},
      {"name": "kneeling_or_half_kneeling", "frame_span": [p2, p3]},
      {"name": "crouched", "frame_span": [p3, p4]},
      {"name": "standing", "frame_span": [p4, num_input_frames]},
    ]

  return {
    "auto_boundaries": input_boundaries,
    "stage_spans": stage_spans,
    "duration_s": duration_s,
    "kneel_to_crouch_s": kneel_to_crouch_s,
    "minimum_foot_z": minimum_foot_z,
    "recommended_for_visual_review": not flags,
    "review_flags": flags,
    "diagnostic_counts": {
      "prone_endpoints": int(prone.sum()),
      "upper_body_supported_endpoints": int(upper_body_supported.sum()),
      "kneeling_endpoints": int(kneeling.sum()),
      "crouch_endpoints": int(crouch.sum()),
      "standing_endpoints": int(standing.sum()),
    },
  }


def _count_rows(path: Path) -> int:
  with path.open("rb") as stream:
    return sum(1 for _ in stream)


def main(cfg: Cfg) -> None:
  input_dir = Path(cfg.input_dir).expanduser().resolve()
  npz_dir = Path(cfg.npz_dir).expanduser().resolve()
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  source_manifest_path = input_dir / "manifest.json"
  if not source_manifest_path.is_file():
    raise FileNotFoundError(source_manifest_path)
  if not npz_dir.is_dir():
    raise FileNotFoundError(npz_dir)
  if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(
      f"{output_dir} is not empty; choose a new directory to preserve prior reviews"
    )
  output_dir.mkdir(parents=True, exist_ok=True)

  source_manifest = json.loads(source_manifest_path.read_text())
  candidates = [
    clip
    for clip in source_manifest["clips"]
    if clip["lying_orientation"] == cfg.lying_orientation
    and (cfg.include_mirrors or not clip["mirrored"])
  ]
  if not candidates:
    raise RuntimeError("no clips matched the requested orientation")

  output_manifest: dict[str, object] = {
    "format_version": 1,
    "dataset_kind": "prone_route_visual_review_candidates",
    "approved_for_training": False,
    "warning": "Automatic stage labels require visual review before training.",
    "config": asdict(cfg),
    "source_manifest": str(source_manifest_path),
    "source_manifest_sha256": _sha256(source_manifest_path),
    "clips": [],
  }
  review_rows: list[dict[str, object]] = []
  for clip in candidates:
    source_csv = input_dir / clip["output"]
    source_npz = npz_dir / f"{source_csv.stem}.npz"
    if not source_csv.is_file() or not source_npz.is_file():
      raise FileNotFoundError(f"missing CSV/NPZ pair: {source_csv}, {source_npz}")
    num_frames = _count_rows(source_csv)
    analysis = _stage_analysis(source_npz, num_frames, cfg)
    destination = output_dir / source_csv.name
    shutil.copy2(source_csv, destination)
    entry = {
      **clip,
      **analysis,
      "candidate_sha256": _sha256(destination),
      "human_review": "pending",
    }
    output_manifest["clips"].append(entry)
    boundaries = analysis["auto_boundaries"]
    review_rows.append(
      {
        "clip_name": clip["name"],
        "source": clip["source"],
        "source_start_frame": clip["source_frame_span"][0],
        "source_end_frame": clip["source_frame_span"][1],
        "mirrored": clip["mirrored"],
        "num_frames": num_frames,
        "duration_s": round(float(analysis["duration_s"]), 3),
        **boundaries,
        "kneel_to_crouch_s": round(float(analysis["kneel_to_crouch_s"]), 3),
        "minimum_foot_z": round(float(analysis["minimum_foot_z"]), 4),
        "recommended_for_visual_review": analysis[
          "recommended_for_visual_review"
        ],
        "review_flags": ";".join(analysis["review_flags"]),
        "human_review": "pending",
        "output": source_csv.name,
      }
    )

  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
  review_path = output_dir / "review.jsonl"
  review_path.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in review_rows)
  )

  recommended = sum(
    bool(row["recommended_for_visual_review"]) for row in review_rows
  )
  print(f"Exported {len(review_rows)} candidates to {output_dir}")
  print(f"Automatically clean enough to inspect first: {recommended}")
  print(f"Review table: {review_path}")
  print("This candidate dataset is NOT approved for training.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
