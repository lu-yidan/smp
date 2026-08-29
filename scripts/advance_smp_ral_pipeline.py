"""Advance the frozen SMP evidence pipeline without competing with training."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from build_smp_causal_manifest import ManifestCfg, build_manifest
from launch_smp_policy_seed_confirmation import (
  ConfirmationCfg,
  launch_confirmation,
)
from monitor_smp_training_health import HealthCfg, inspect
from select_smp_stable_arm import SelectionCfg, write_selection

_EVALUATION_SCHEMA_VERSION = 2
_LOCKED_MANIFEST_HASHES = {
  8000: "64506f71e85b69b58bb5579621b10a8aa6969a172428c2d213115fb54a08c333",
}


@dataclass(frozen=True)
class PipelineCfg:
  control_dir: Path
  evidence_dir: Path
  state: Path
  gates: tuple[int, ...] = (8000, 15000, 25000, 29999)
  expected_jobs: int = 8
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
  modes: tuple[str, ...] = (
    "native_gsi",
    "prone",
    "supine",
    "left_side",
    "right_side",
  )
  eval_seed: int = 20260829
  policy_seed: int = 42
  environment_seed: int = 42
  num_envs: int = 512
  steps: int = 500
  launch_when_ready: bool = False
  launch_confirmation_when_ready: bool = False
  confirmation_control_dir: Path = Path(
    "run_control/scratch_causal_policy_seed_confirmation"
  )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _gpu_processes() -> list[str]:
  result = subprocess.run(
    (
      "nvidia-smi",
      "--query-compute-apps=pid",
      "--format=csv,noheader,nounits",
    ),
    check=True,
    capture_output=True,
    text=True,
  )
  return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
  if path.exists():
    return
  _atomic_json(path, payload)


def _validate_manifest(
  path: Path, gate: int, policy_seed: int, environment_seed: int
) -> str:
  payload = json.loads(path.read_text())
  runs = payload.get("runs", [])
  if payload.get("checkpoint_step") != gate or len(runs) != 8:
    raise ValueError(f"invalid frozen manifest: {path}")
  if (
    payload.get("policy_seed") != policy_seed
    or payload.get("environment_seed") != environment_seed
  ):
    raise ValueError(f"manifest has incorrect effective seeds: {path}")
  names = {run.get("name") for run in runs}
  checkpoints = {run.get("checkpoint") for run in runs}
  if len(names) != 8 or len(checkpoints) != 8:
    raise ValueError(f"manifest does not contain eight unique arms: {path}")
  for run in runs:
    if (
      run.get("policy_seed") != policy_seed
      or run.get("environment_seed") != environment_seed
    ):
      raise ValueError(f"run has incorrect effective seeds: {run.get('name')}")
    checkpoint = Path(run["checkpoint"])
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
    if _sha256(checkpoint) != run.get("checkpoint_sha256"):
      raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
  digest = _sha256(path)
  locked = _LOCKED_MANIFEST_HASHES.get(gate)
  if locked is not None and digest != locked:
    raise ValueError(
      f"locked gate {gate} manifest hash changed: expected {locked}, got {digest}"
    )
  return digest


def _ensure_manifests(cfg: PipelineCfg) -> tuple[list[dict[str, Any]], list[int]]:
  manifests = []
  pending = []
  manifest_dir = cfg.evidence_dir / "manifests"
  for gate in cfg.gates:
    path = manifest_dir / f"gate_{gate}.json"
    if not path.exists():
      try:
        payload = build_manifest(
          ManifestCfg(
            checkpoint_step=gate,
            output=path,
            policy_seed=cfg.policy_seed,
            environment_seed=cfg.environment_seed,
          )
        )
      except FileNotFoundError:
        pending.append(gate)
        continue
      _write_manifest(path, payload)
    manifests.append(
      {
        "gate": gate,
        "path": str(path.resolve()),
        "sha256": _validate_manifest(path, gate, cfg.policy_seed, cfg.environment_seed),
      }
    )
  return manifests, pending


def _analysis_complete(cfg: PipelineCfg, output_dir: Path, manifest: Path) -> bool:
  required = ("_COMPLETE.json", "summary.json", "analysis.json", "analysis.md")
  if not all((output_dir / name).is_file() for name in required):
    return False
  try:
    complete = json.loads((output_dir / "_COMPLETE.json").read_text())
    analysis = json.loads((output_dir / "analysis.json").read_text())
  except (json.JSONDecodeError, OSError):
    return False
  expected_count = 8 * len(cfg.modes)
  return (
    complete.get("evaluation_schema_version") == _EVALUATION_SCHEMA_VERSION
    and complete.get("manifest") == str(manifest.resolve())
    and complete.get("result_count") == expected_count
    and complete.get("modes") == list(cfg.modes)
    and complete.get("eval_seeds") == [cfg.eval_seed]
    and complete.get("num_envs") == cfg.num_envs
    and complete.get("steps") == cfg.steps
    and analysis.get("status") in {"NO_PROMOTION", "SCREEN_PASS_NOT_FINAL"}
  )


def _active_eval(cfg: PipelineCfg) -> dict[str, Any] | None:
  launch_state = cfg.evidence_dir / "active_evaluation.json"
  if not launch_state.exists():
    return None
  payload = json.loads(launch_state.read_text())
  pid = int(payload["pid"])
  if _pid_alive(pid):
    return payload
  launch_state.unlink()
  return None


def _launch_evaluation(
  cfg: PipelineCfg, manifest: Path, gate: int, output_dir: Path
) -> dict[str, Any]:
  matrix = Path(__file__).with_name("run_smp_frozen_eval_matrix.py").resolve()
  analyzer = Path(__file__).with_name("analyze_smp_frozen_matrix.py").resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  matrix_command = [
    sys.executable,
    str(matrix),
    "--manifest",
    str(manifest.resolve()),
    "--output-dir",
    str(output_dir.resolve()),
    "--devices",
    *cfg.devices,
    "--modes",
    *cfg.modes,
    "--eval-seeds",
    str(cfg.eval_seed),
    "--num-envs",
    str(cfg.num_envs),
    "--steps",
    str(cfg.steps),
  ]
  analysis_command = [
    sys.executable,
    str(analyzer),
    "--summary",
    str((output_dir / "summary.json").resolve()),
    "--output-json",
    str((output_dir / "analysis.json").resolve()),
    "--output-markdown",
    str((output_dir / "analysis.md").resolve()),
  ]
  log = output_dir / "evaluation.log"
  command = " ".join(shlex.quote(part) for part in matrix_command)
  command += " && " + " ".join(shlex.quote(part) for part in analysis_command)
  with log.open("a") as stream:
    process = subprocess.Popen(
      ("bash", "-lc", command),
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  payload = {
    "gate": gate,
    "pid": process.pid,
    "manifest": str(manifest.resolve()),
    "output_dir": str(output_dir.resolve()),
    "log": str(log.resolve()),
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(cfg.evidence_dir / "active_evaluation.json", payload)
  return payload


def advance(cfg: PipelineCfg) -> dict[str, Any]:
  health = inspect(
    HealthCfg(
      control_dir=cfg.control_dir,
      output=cfg.state.with_name("training_health_latest.json"),
      expected_jobs=cfg.expected_jobs,
    )
  )
  _atomic_json(cfg.state.with_name("training_health_latest.json"), health)
  manifests, pending_gates = _ensure_manifests(cfg)
  completed_training = bool(health["jobs"]) and all(
    job["completed"] for job in health["jobs"]
  )
  active = _active_eval(cfg)
  evaluations = []
  next_gate = None
  for manifest_info in manifests:
    gate = int(manifest_info["gate"])
    manifest = Path(manifest_info["path"])
    output_dir = cfg.evidence_dir / f"gate_{gate}"
    complete = _analysis_complete(cfg, output_dir, manifest)
    evaluations.append(
      {
        "gate": gate,
        "output_dir": str(output_dir.resolve()),
        "analysis_complete": complete,
      }
    )
    if not complete and next_gate is None:
      next_gate = gate

  gpu_processes = _gpu_processes()
  stable_selection = None
  confirmation = None
  confirmation_waiting_for_gpu = False
  if completed_training and next_gate is None:
    stable_selection = write_selection(SelectionCfg(evidence_dir=cfg.evidence_dir))
    wants_confirmation = (
      cfg.launch_confirmation_when_ready
      and stable_selection["status"] == "PROMOTE_FOR_POLICY_SEEDS"
    )
    confirmation_waiting_for_gpu = wants_confirmation and bool(gpu_processes)
    if wants_confirmation and not gpu_processes:
      confirmation = launch_confirmation(
        ConfirmationCfg(
          selection=cfg.evidence_dir / "stable_selection.json",
          control_dir=cfg.confirmation_control_dir,
          launch=True,
        )
      )
  status = "TRAINING_ACTIVE"
  action = "Continue health monitoring; do not contend with training GPUs."
  launched = None
  if not health["healthy"]:
    status = "TRAINING_ALERT"
    action = "Inspect health alerts before changing or stopping any job."
  elif active is not None:
    status = "EVAL_RUNNING"
    action = "Wait for the resumable frozen matrix and analyzer to finish."
  elif completed_training and next_gate is None:
    if confirmation is not None:
      alive = [job for job in confirmation["jobs"] if _pid_alive(int(job["pid"]))]
      if alive:
        status = "CONFIRMATION_TRAINING"
        action = (
          f"Policy-seed confirmation is running: {len(alive)}/"
          f"{len(confirmation['jobs'])} processes alive."
        )
      else:
        confirmation_health = inspect(
          HealthCfg(
            control_dir=cfg.confirmation_control_dir,
            output=cfg.state.with_name("confirmation_health_latest.json"),
            expected_jobs=len(confirmation["jobs"]),
          )
        )
        _atomic_json(
          cfg.state.with_name("confirmation_health_latest.json"),
          confirmation_health,
        )
        if confirmation_health["jobs"] and all(
          job["completed"] for job in confirmation_health["jobs"]
        ):
          status = "CONFIRMATION_COMPLETE"
          action = (
            "Policy-seed training is complete; freeze seed-level evaluation manifests."
          )
        else:
          status = "CONFIRMATION_ALERT"
          action = (
            "Confirmation processes exited before every job completed; inspect logs."
          )
    elif confirmation_waiting_for_gpu:
      status = "WAITING_FREE_GPU"
      action = "Stable candidates are ready; wait for all GPU processes to exit."
    elif stable_selection["status"] == "PROMOTE_FOR_POLICY_SEEDS":
      status = "ANALYSIS_COMPLETE"
      action = (
        "Frozen cross-gate selection is complete; launch independent policy "
        "seeds for the promoted configurations."
      )
    else:
      status = "ANALYSIS_COMPLETE"
      action = (
        "Frozen cross-gate selection found no eligible arm; do not relax "
        "thresholds post hoc."
      )
  elif completed_training and gpu_processes:
    status = "WAITING_FREE_GPU"
    action = "Training is complete; wait for remaining compute processes to exit."
  elif completed_training and next_gate is not None:
    status = "READY_FOR_EVAL"
    action = f"Run the frozen matrix for gate {next_gate}."
    if cfg.launch_when_ready:
      manifest = cfg.evidence_dir / "manifests" / f"gate_{next_gate}.json"
      launched = _launch_evaluation(
        cfg, manifest, next_gate, cfg.evidence_dir / f"gate_{next_gate}"
      )
      status = "EVAL_RUNNING"
      action = f"Launched frozen matrix and analysis for gate {next_gate}."

  return {
    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "action": action,
    "training_healthy": health["healthy"],
    "completed_training_jobs": sum(job["completed"] for job in health["jobs"]),
    "observed_training_jobs": len(health["jobs"]),
    "iterations": {job["log"]: job["iteration"] for job in health["jobs"]},
    "gpu_process_count": len(gpu_processes),
    "manifests": manifests,
    "pending_manifest_gates": pending_gates,
    "evaluations": evaluations,
    "active_evaluation": active or launched,
    "stable_selection": (
      {
        "status": stable_selection["status"],
        "promoted_candidates": stable_selection["promoted_candidates"],
        "path": str((cfg.evidence_dir / "stable_selection.json").resolve()),
      }
      if stable_selection is not None
      else None
    ),
    "confirmation": (
      {
        "status": confirmation["status"],
        "plan_id": confirmation["plan_id"],
        "job_count": len(confirmation["jobs"]),
        "path": str((cfg.confirmation_control_dir / "launch_manifest.json").resolve()),
      }
      if confirmation is not None
      else None
    ),
  }


def main(cfg: PipelineCfg) -> None:
  state = advance(cfg)
  _atomic_json(cfg.state, state)
  print(f"{state['status']}: {state['action']}")


if __name__ == "__main__":
  main(tyro.cli(PipelineCfg))
