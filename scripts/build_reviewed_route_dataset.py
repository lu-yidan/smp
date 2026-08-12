"""Build a training-ready route dataset from visually reviewed candidates.

The candidate CSVs and source manifest remain untouched. This tool accepts an
explicit, version-controlled review specification, crops locomotion after the
first sustained stand, and writes a new dataset with complete lineage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro


@dataclass
class Cfg:
  review_spec: str = "configs/data/getup_lafan_prone_routes_v7.json"
  candidate_dir: str = "datasets/csv/getup_lafan_prone_routes_v7_candidates"
  output_dir: str = "datasets/csv/getup_lafan_prone_routes_v7"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _read_csv_lines(path: Path) -> list[str]:
  lines = path.read_text(encoding="utf-8").splitlines()
  if not lines:
    raise ValueError(f"empty CSV: {path}")
  for index, line in enumerate(lines):
    fields = line.split(",")
    if len(fields) != 36:
      raise ValueError(f"{path}:{index + 1}: expected 36 columns, got {len(fields)}")
    try:
      [float(value) for value in fields]
    except ValueError as exc:
      raise ValueError(f"{path}:{index + 1}: non-numeric CSV value") from exc
  return lines


def _target_names(spec: dict[str, object]) -> list[str]:
  bases = spec.get("approved_base_stems")
  if (
    not isinstance(bases, list)
    or not bases
    or not all(isinstance(x, str) for x in bases)
  ):
    raise ValueError("review spec needs a non-empty approved_base_stems string list")
  if len(set(bases)) != len(bases):
    raise ValueError("approved_base_stems contains duplicates")
  names = list(bases)
  if spec.get("include_mirrors", False):
    names.extend(f"{base}__mirror" for base in bases)
  return names


def main(cfg: Cfg) -> None:
  review_spec_path = Path(cfg.review_spec).expanduser().resolve()
  candidate_dir = Path(cfg.candidate_dir).expanduser().resolve()
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  candidate_manifest_path = candidate_dir / "manifest.json"

  spec = json.loads(review_spec_path.read_text(encoding="utf-8"))
  if spec.get("review_status") != "approved":
    raise ValueError("review spec must explicitly set review_status=approved")
  manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
  if manifest.get("approved_for_training") is not False:
    raise ValueError("expected a review-only candidate manifest")
  if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(f"{output_dir} is not empty; refusing to overwrite it")
  output_dir.mkdir(parents=True, exist_ok=True)

  input_fps = int(spec["input_fps"])
  post_stand_s = float(spec["post_stand_s"])
  if input_fps <= 0 or post_stand_s < 0:
    raise ValueError("input_fps must be positive and post_stand_s non-negative")
  post_stand_frames = round(input_fps * post_stand_s)

  by_name = {clip["name"]: clip for clip in manifest["clips"]}
  target_names = _target_names(spec)
  missing = sorted(set(target_names) - set(by_name))
  if missing:
    raise ValueError(f"approved clips are absent from candidate manifest: {missing}")

  output_clips: list[dict[str, object]] = []
  for name in target_names:
    candidate = by_name[name]
    source_path = candidate_dir / candidate["output"]
    candidate_sha256 = _sha256(source_path)
    if candidate_sha256 != candidate["candidate_sha256"]:
      raise ValueError(f"{name}: candidate CSV hash does not match its manifest")
    lines = _read_csv_lines(source_path)
    source_start = int(candidate["source_frame_span"][0])
    standing_start = int(candidate["auto_boundaries"]["standing_start"])
    if standing_start < 0 or standing_start >= len(lines):
      raise ValueError(f"{name}: invalid automatic standing_start={standing_start}")
    crop_end = min(len(lines), standing_start + post_stand_frames)

    output_path = output_dir / source_path.name
    output_path.write_text("\n".join(lines[:crop_end]) + "\n", encoding="utf-8")
    if output_path.read_text(encoding="utf-8").splitlines() != lines[:crop_end]:
      raise RuntimeError(f"{name}: output verification failed")

    stage_spans = []
    for stage in candidate.get("stage_spans", []):
      start, end = stage["frame_span"]
      if start >= crop_end:
        continue
      stage_spans.append(
        {"name": stage["name"], "frame_span": [start, min(end, crop_end)]}
      )
    output_clips.append(
      {
        "name": name,
        "output": output_path.name,
        "mirrored": bool(candidate["mirrored"]),
        "source": candidate["source"],
        "candidate_source_frame_span": candidate["source_frame_span"],
        "output_source_frame_span": [source_start, source_start + crop_end],
        "candidate_sha256": candidate_sha256,
        "output_sha256": _sha256(output_path),
        "original_num_frames": len(lines),
        "num_frames": crop_end,
        "removed_tail_frames": len(lines) - crop_end,
        "duration_s": crop_end / input_fps,
        "standing_start": standing_start,
        "post_stand_frames": crop_end - standing_start,
        "stage_spans": stage_spans,
        "human_review": "approved",
      }
    )
    print(
      f"{name}: {len(lines)} -> {crop_end} frames "
      f"(removed {len(lines) - crop_end}, stand={standing_start})"
    )

  output_manifest = {
    "format_version": 1,
    "dataset_kind": "reviewed_prone_recovery_routes",
    "approved_for_training": True,
    "config": asdict(cfg),
    "review_spec": str(review_spec_path),
    "review_spec_sha256": _sha256(review_spec_path),
    "candidate_manifest": str(candidate_manifest_path),
    "candidate_manifest_sha256": _sha256(candidate_manifest_path),
    "input_fps": input_fps,
    "post_stand_s": post_stand_s,
    "clips": output_clips,
    "summary": {
      "num_distinct_routes": len(spec["approved_base_stems"]),
      "num_output_clips": len(output_clips),
      "num_mirrored_clips": sum(bool(clip["mirrored"]) for clip in output_clips),
      "total_frames": sum(int(clip["num_frames"]) for clip in output_clips),
      "total_duration_s": sum(float(clip["duration_s"]) for clip in output_clips),
      "removed_tail_frames": sum(
        int(clip["removed_tail_frames"]) for clip in output_clips
      ),
    },
  }
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(
    json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(output_manifest["summary"], indent=2))
  print(f"Training-ready manifest: {manifest_path}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
