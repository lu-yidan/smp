"""Run and summarize a resumable frozen SMP evaluation matrix."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_DEFAULT_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")
_EVALUATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MatrixCfg:
  manifest: Path
  output_dir: Path
  modes: tuple[str, ...] = _DEFAULT_MODES
  eval_seeds: tuple[int, ...] = (20260829,)
  num_envs: int = 512
  steps: int = 500
  device: str = "cuda:0"
  devices: tuple[str, ...] = ()
  include_per_env: bool = True
  overwrite: bool = False
  dry_run: bool = False


def _slug(value: str) -> str:
  return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  data = json.loads(path.read_text())
  if isinstance(data, list):
    metadata: dict[str, Any] = {}
    runs = data
  elif isinstance(data, dict) and isinstance(data.get("runs"), list):
    metadata = {key: value for key, value in data.items() if key != "runs"}
    runs = data["runs"]
  else:
    raise ValueError("manifest must be a list or an object containing a runs list")

  required = {"name", "task", "checkpoint"}
  for index, run in enumerate(runs):
    missing = required - set(run)
    if missing:
      raise ValueError(f"manifest run {index} is missing: {sorted(missing)}")
    checkpoint = Path(run["checkpoint"])
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
  return metadata, runs


def _valid_result(path: Path, expected: dict[str, Any]) -> bool:
  try:
    result = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError):
    return False
  return all(result.get(key) == value for key, value in expected.items())


def _atomic_write(path: Path, content: str) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def _assign_commands(
  commands: list[list[str]], devices: tuple[str, ...]
) -> list[tuple[str, list[list[str]]]]:
  if not devices:
    raise ValueError("at least one evaluation device is required")
  if len(set(devices)) != len(devices):
    raise ValueError(f"evaluation devices must be unique, got {devices}")
  buckets: list[list[list[str]]] = [[] for _ in devices]
  for index, command in enumerate(commands):
    device_index = index % len(devices)
    buckets[device_index].append(command + ["--device", devices[device_index]])
  return [
    (device, bucket)
    for device, bucket in zip(devices, buckets, strict=True)
    if bucket
  ]


def _run_bucket(device: str, commands: list[list[str]]) -> None:
  for command in commands:
    print(f"[RUN {device}] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _write_summary(
  output_dir: Path,
  metadata: dict[str, Any],
  rows: list[dict[str, Any]],
) -> None:
  grouped: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
  for row in rows:
    key = (row["arm"], row["checkpoint"], row.get("policy_seed"))
    grouped.setdefault(key, []).append(row)

  summary_rows: list[dict[str, Any]] = []
  for (arm, checkpoint, policy_seed), group in sorted(grouped.items()):
    rates_by_mode: dict[str, list[float]] = {}
    for row in group:
      rates_by_mode.setdefault(row["reset_mode"], []).append(
        row["strict_success_rate"]
      )
    mode_means = {
      mode: sum(values) / len(values) for mode, values in rates_by_mode.items()
    }
    fixed = [mode_means[mode] for mode in _DEFAULT_MODES[1:] if mode in mode_means]
    summary_rows.append(
      {
        "arm": arm,
        "checkpoint": checkpoint,
        "policy_seed": policy_seed,
        "native_gsi": mode_means.get("native_gsi"),
        "prone": mode_means.get("prone"),
        "supine": mode_means.get("supine"),
        "left_side": mode_means.get("left_side"),
        "right_side": mode_means.get("right_side"),
        "fixed_macro": sum(fixed) / len(fixed) if fixed else None,
        "fixed_worst": min(fixed) if fixed else None,
      }
    )

  payload = {"metadata": metadata, "evaluations": rows, "summary": summary_rows}
  _atomic_write(
    output_dir / "summary.json",
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
  )
  fields = list(summary_rows[0]) if summary_rows else ["arm", "checkpoint"]
  csv_temporary = output_dir / "summary.csv.tmp"
  with csv_temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary_rows)
  csv_temporary.replace(output_dir / "summary.csv")


def main(cfg: MatrixCfg) -> None:
  metadata, runs = _load_manifest(cfg.manifest)
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  evaluator = Path(__file__).with_name("evaluate_smp_baseline.py")
  result_paths: list[Path] = []
  commands: list[list[str]] = []

  for run in runs:
    checkpoint = Path(run["checkpoint"]).resolve()
    policy_seed = run.get("policy_seed")
    for mode in cfg.modes:
      for eval_seed in cfg.eval_seeds:
        filename = "__".join(
          (
            _slug(run["name"]),
            _slug(checkpoint.stem),
            _slug(mode),
            f"eval{eval_seed}",
          )
        ) + ".json"
        output = cfg.output_dir / filename
        expected = {
          "evaluation_schema_version": _EVALUATION_SCHEMA_VERSION,
          "checkpoint_path": str(checkpoint),
          "task": run["task"],
          "reset_mode": mode,
          "seed": eval_seed,
          "num_envs": cfg.num_envs,
          "steps": cfg.steps,
          "policy_seed": policy_seed,
        }
        result_paths.append(output)
        if not cfg.overwrite and _valid_result(output, expected):
          print(f"[SKIP] {output.name}")
          continue

        command = [
          sys.executable,
          str(evaluator),
          "--checkpoint",
          str(checkpoint),
          "--task",
          run["task"],
          "--reset-mode",
          mode,
          "--num-envs",
          str(cfg.num_envs),
          "--steps",
          str(cfg.steps),
          "--seed",
          str(eval_seed),
          "--output",
          str(output),
        ]
        if policy_seed is not None:
          command.extend(("--policy-seed", str(policy_seed)))
        if cfg.include_per_env:
          command.append("--include-per-env")
        commands.append(command)

  devices = cfg.devices or (cfg.device,)
  assignments = _assign_commands(commands, devices) if commands else []
  if cfg.dry_run:
    for device, bucket in assignments:
      for command in bucket:
        print(f"[DRY RUN {device}] " + " ".join(command))
    return

  if assignments:
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
      futures = [
        executor.submit(_run_bucket, device, bucket)
        for device, bucket in assignments
      ]
      for future in futures:
        future.result()

  rows: list[dict[str, Any]] = []
  for output in result_paths:
    result = json.loads(output.read_text())
    result.pop("per_env", None)
    result["arm"] = output.name.split("__", maxsplit=1)[0]
    rows.append(result)
  _write_summary(cfg.output_dir, metadata, rows)
  complete = {
    "evaluation_schema_version": _EVALUATION_SCHEMA_VERSION,
    "manifest": str(cfg.manifest.resolve()),
    "result_count": len(rows),
    "modes": list(cfg.modes),
    "eval_seeds": list(cfg.eval_seeds),
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "devices": list(devices),
  }
  _atomic_write(
    cfg.output_dir / "_COMPLETE.json",
    json.dumps(complete, indent=2, sort_keys=True) + "\n",
  )


if __name__ == "__main__":
  main(tyro.cli(MatrixCfg))
