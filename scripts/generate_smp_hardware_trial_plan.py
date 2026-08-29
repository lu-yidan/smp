"""Generate the immutable randomized 80-trial real-G1 recovery plan."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tyro

_POSE_COUNTS = {
  "prone": 15,
  "supine": 15,
  "left_side": 15,
  "right_side": 15,
  "random_fall_state": 20,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class HardwareTrialPlanCfg:
  output_json: Path
  block_id: str
  frozen_before_trial_utc: str
  randomization_seed: int
  policy_seed: int
  checkpoint_sha256: str
  onnx_sha256: str
  deploy_git_commit: str
  robot_id: str
  surface: str
  condition: str = "flat_core"


def _validate(cfg: HardwareTrialPlanCfg) -> None:
  for field in ("block_id", "robot_id", "surface", "condition"):
    if not str(getattr(cfg, field)).strip():
      raise ValueError(f"{field} must not be empty")
  if cfg.condition != "flat_core":
    raise ValueError("the v3 plan generator supports only condition=flat_core")
  for field in ("checkpoint_sha256", "onnx_sha256"):
    if not _SHA256.fullmatch(getattr(cfg, field)):
      raise ValueError(f"{field} must be a SHA-256 digest")
  if not _COMMIT.fullmatch(cfg.deploy_git_commit):
    raise ValueError("deploy_git_commit is not a valid Git commit")
  normalized = cfg.frozen_before_trial_utc.strip().replace("Z", "+00:00")
  try:
    timestamp = datetime.fromisoformat(normalized)
  except ValueError as error:
    raise ValueError("frozen_before_trial_utc must be an ISO-8601 timestamp") from error
  if timestamp.tzinfo is None or timestamp.utcoffset() is None:
    raise ValueError("frozen_before_trial_utc must include a timezone")


def generate_plan(cfg: HardwareTrialPlanCfg) -> dict[str, Any]:
  _validate(cfg)
  poses = [
    pose
    for pose, count in _POSE_COUNTS.items()
    for _ in range(count)
  ]
  random.Random(cfg.randomization_seed).shuffle(poses)
  return {
    "schema_version": 1,
    "protocol": "real_g1_flat_core_v3",
    "frozen_before_trial_utc": cfg.frozen_before_trial_utc,
    "provenance": {
      "block_id": cfg.block_id,
      "randomization_seed": cfg.randomization_seed,
      "policy_seed": cfg.policy_seed,
      "checkpoint_sha256": cfg.checkpoint_sha256,
      "onnx_sha256": cfg.onnx_sha256,
      "deploy_git_commit": cfg.deploy_git_commit,
      "robot_id": cfg.robot_id,
      "condition": cfg.condition,
      "surface": cfg.surface,
    },
    "assignments": [
      {"planned_slot": slot, "initial_pose": pose}
      for slot, pose in enumerate(poses)
    ],
  }


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def main(cfg: HardwareTrialPlanCfg) -> None:
  plan = generate_plan(cfg)
  content = json.dumps(plan, indent=2, sort_keys=True) + "\n"
  _atomic_write(cfg.output_json, content)
  digest = hashlib.sha256(content.encode()).hexdigest()
  print(f"TRIAL_PLAN_FROZEN: {cfg.output_json.resolve()} sha256={digest}")


if __name__ == "__main__":
  main(tyro.cli(HardwareTrialPlanCfg))
