"""Run a resumable, exactly stratified T/P frozen evaluation matrix."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_POSES = ("prone", "supine", "left_side", "right_side")
_TERRAINS = ("flat", "slope", "stairs", "rough")
_LEVELS = (0, 1, 2)
_EDGE_COHORTS = ("center", "near_edge", "straddle", "lower_tread")
_PLATE_MASSES = (4.0, 8.0, 12.0)
_BASE_EVALUATION_SCHEMA_VERSION = 2
_SPECIALIST_MATRIX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpecialistMatrixCfg:
  manifest: Path
  output_dir: Path
  devices: tuple[str, ...] = (
    "cuda:0",
    "cuda:1",
    "cuda:2",
    "cuda:3",
    "cuda:4",
    "cuda:5",
    "cuda:6",
    "cuda:7",
  )
  eval_seed: int = 20260910
  num_envs: int = 256
  steps: int = 750
  include_per_env: bool = True
  overwrite: bool = False
  dry_run: bool = False


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _slug(value: str) -> str:
  return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  payload = json.loads(path.read_text())
  runs = payload.get("runs")
  if not isinstance(runs, list) or len(runs) != 1:
    raise ValueError("specialist manifest must contain exactly one run")
  phase = payload.get("phase")
  if phase not in ("T", "P") or runs[0].get("phase") != phase:
    raise ValueError("specialist manifest has inconsistent T/P phase")
  checkpoint = Path(runs[0]["checkpoint"])
  if not checkpoint.is_file() or _sha256(checkpoint) != runs[0].get(
    "checkpoint_sha256"
  ):
    raise ValueError(f"specialist checkpoint changed: {checkpoint}")
  return payload, runs[0]


def terrain_strata() -> list[dict[str, Any]]:
  strata = []
  for terrain in _TERRAINS:
    levels = (0,) if terrain == "flat" else _LEVELS
    cohorts = _EDGE_COHORTS if terrain == "stairs" else (None,)
    for level in levels:
      for cohort in cohorts:
        for pose in _POSES:
          stratum = {
            "evaluation_profile": "terrain",
            "terrain_type": terrain,
            "terrain_level": level,
            "stair_edge_cohort": cohort,
            "fall_pose": pose,
            "plate_mode": None,
            "plate_present": False,
            "plate_mass_kg": None,
          }
          strata.append(stratum)
  return strata


def plate_strata() -> list[dict[str, Any]]:
  strata = []
  for pose in _POSES:
    strata.append(
      {
        "evaluation_profile": "plate",
        "terrain_type": None,
        "terrain_level": None,
        "stair_edge_cohort": None,
        "fall_pose": pose,
        "plate_mode": "unpinned",
        "plate_present": False,
        "plate_mass_kg": None,
      }
    )
  for pose in ("prone", "supine"):
    for mass in _PLATE_MASSES:
      strata.append(
        {
          "evaluation_profile": "plate",
          "terrain_type": None,
          "terrain_level": None,
          "stair_edge_cohort": None,
          "fall_pose": pose,
          "plate_mode": "pinned",
          "plate_present": True,
          "plate_mass_kg": mass,
        }
      )
  return strata


def _stratum_id(stratum: dict[str, Any]) -> str:
  if stratum["evaluation_profile"] == "terrain":
    parts = [
      "terrain",
      stratum["terrain_type"],
      f"l{stratum['terrain_level']}",
      stratum["stair_edge_cohort"] or "not_edge",
      stratum["fall_pose"],
    ]
  else:
    parts = ["plate", stratum["plate_mode"], stratum["fall_pose"]]
    if stratum["plate_present"]:
      parts.append(f"{int(stratum['plate_mass_kg'])}kg")
  return _slug("__".join(parts))


def _strata_hash(strata: list[dict[str, Any]]) -> str:
  return hashlib.sha256(json.dumps(strata, sort_keys=True).encode()).hexdigest()


def _valid_result(path: Path, expected: dict[str, Any]) -> bool:
  try:
    result = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError):
    return False
  return all(result.get(key) == value for key, value in expected.items())


def _command(
  evaluator: Path,
  run: dict[str, Any],
  stratum: dict[str, Any],
  output: Path,
  cfg: SpecialistMatrixCfg,
) -> list[str]:
  command = [
    sys.executable,
    str(evaluator),
    "--checkpoint",
    str(Path(run["checkpoint"]).resolve()),
    "--task",
    run["task"],
    "--reset-mode",
    stratum["fall_pose"],
    "--num-envs",
    str(cfg.num_envs),
    "--steps",
    str(cfg.steps),
    "--seed",
    str(cfg.eval_seed),
    "--policy-seed",
    str(run["policy_seed"]),
    "--evaluation-profile",
    stratum["evaluation_profile"],
    "--output",
    str(output),
  ]
  if stratum["evaluation_profile"] == "terrain":
    command.extend(
      (
        "--terrain-type",
        stratum["terrain_type"],
        "--terrain-level",
        str(stratum["terrain_level"]),
      )
    )
    if stratum["stair_edge_cohort"] is not None:
      command.extend(("--stair-edge-cohort", stratum["stair_edge_cohort"]))
  else:
    command.extend(("--plate-mode", stratum["plate_mode"]))
    if stratum["plate_present"]:
      command.extend(("--plate-mass-kg", str(stratum["plate_mass_kg"])))
  if cfg.include_per_env:
    command.append("--include-per-env")
  return command


def _assign(
  commands: list[list[str]], devices: tuple[str, ...]
) -> list[tuple[str, list[list[str]]]]:
  if not devices or len(set(devices)) != len(devices):
    raise ValueError("specialist evaluation requires unique non-empty devices")
  buckets: list[list[list[str]]] = [[] for _ in devices]
  for index, command in enumerate(commands):
    device_index = index % len(devices)
    buckets[device_index].append(command + ["--device", devices[device_index]])
  return [
    (device, bucket) for device, bucket in zip(devices, buckets, strict=True) if bucket
  ]


def _run_bucket(device: str, commands: list[list[str]]) -> None:
  for command in commands:
    print(f"[RUN {device}] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_matrix(cfg: SpecialistMatrixCfg) -> dict[str, Any]:
  manifest, run = _load_manifest(cfg.manifest)
  phase = manifest["phase"]
  strata = terrain_strata() if phase == "T" else plate_strata()
  expected_count = 76 if phase == "T" else 10
  if len(strata) != expected_count or len({_stratum_id(row) for row in strata}) != len(
    strata
  ):
    raise RuntimeError("specialist stratum catalog is incomplete or duplicated")
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  evaluator = Path(__file__).with_name("evaluate_smp_baseline.py").resolve()
  commands = []
  result_paths = []
  for stratum in strata:
    identifier = _stratum_id(stratum)
    output = cfg.output_dir / f"{identifier}.json"
    result_paths.append((output, identifier, stratum))
    expected = {
      "evaluation_schema_version": _BASE_EVALUATION_SCHEMA_VERSION,
      "checkpoint_path": str(Path(run["checkpoint"]).resolve()),
      "task": run["task"],
      "policy_seed": run["policy_seed"],
      "seed": cfg.eval_seed,
      "num_envs": cfg.num_envs,
      "steps": cfg.steps,
      "evaluation_profile": stratum["evaluation_profile"],
      "stratum": stratum,
    }
    if not cfg.overwrite and _valid_result(output, expected):
      print(f"[SKIP] {output.name}")
      continue
    commands.append(_command(evaluator, run, stratum, output, cfg))

  assignments = _assign(commands, cfg.devices) if commands else []
  if cfg.dry_run:
    for device, bucket in assignments:
      for command in bucket:
        print(f"[DRY RUN {device}] " + " ".join(command))
    return {
      "status": "DRY_RUN",
      "phase": phase,
      "stratum_count": len(strata),
      "command_count": len(commands),
    }
  if assignments:
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
      futures = [
        executor.submit(_run_bucket, device, bucket) for device, bucket in assignments
      ]
      for future in futures:
        future.result()

  rows = []
  for path, identifier, stratum in result_paths:
    result = json.loads(path.read_text())
    if result.get("stratum") != stratum:
      raise ValueError(f"completed result has wrong stratum: {path}")
    result.pop("per_env", None)
    result["stratum_id"] = identifier
    rows.append(result)
  summary = {
    "schema_version": _SPECIALIST_MATRIX_SCHEMA_VERSION,
    "phase": phase,
    "manifest": str(cfg.manifest.resolve()),
    "manifest_sha256": _sha256(cfg.manifest),
    "policy_seed": run["policy_seed"],
    "checkpoint_step": manifest["checkpoint_step"],
    "strata_sha256": _strata_hash(strata),
    "evaluations": rows,
  }
  _atomic_json(cfg.output_dir / "summary.json", summary)
  complete = {
    "specialist_matrix_schema_version": _SPECIALIST_MATRIX_SCHEMA_VERSION,
    "base_evaluation_schema_version": _BASE_EVALUATION_SCHEMA_VERSION,
    "phase": phase,
    "manifest": str(cfg.manifest.resolve()),
    "manifest_sha256": _sha256(cfg.manifest),
    "policy_seed": run["policy_seed"],
    "checkpoint_step": manifest["checkpoint_step"],
    "strata_sha256": _strata_hash(strata),
    "stratum_count": len(strata),
    "result_count": len(rows),
    "eval_seed": cfg.eval_seed,
    "num_envs_per_stratum": cfg.num_envs,
    "steps": cfg.steps,
    "devices": list(cfg.devices),
  }
  _atomic_json(cfg.output_dir / "_COMPLETE.json", complete)
  return {"status": "COMPLETE", **complete}


def main(cfg: SpecialistMatrixCfg) -> None:
  result = run_matrix(cfg)
  print(
    f"{result['status']}: phase={result['phase']}, strata={result['stratum_count']}"
  )


if __name__ == "__main__":
  main(tyro.cli(SpecialistMatrixCfg))
