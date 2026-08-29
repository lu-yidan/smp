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

from aggregate_smp_policy_seeds import AggregateCfg, write_aggregate
from aggregate_smp_specialist_seeds import (
  SpecialistAggregateCfg,
)
from aggregate_smp_specialist_seeds import (
  write_aggregate as write_specialist_aggregate,
)
from analyze_smp_native_baseline_effects import (
  NativeEffectCfg,
)
from analyze_smp_native_baseline_effects import (
  write_analysis as write_native_effect_analysis,
)
from audit_smp_baseline_registry import audit as audit_baseline_registry
from bind_smp_native_eval_banks import EvalBindingCfg, write_bindings
from build_smp_causal_manifest import ManifestCfg, build_manifest
from build_smp_confirmation_manifests import (
  ConfirmationManifestCfg,
)
from build_smp_confirmation_manifests import (
  write_manifests as write_confirmation_manifests,
)
from build_smp_native_baseline_manifests import (
  NativeBaselineManifestCfg,
)
from build_smp_native_baseline_manifests import (
  write_manifests as write_native_baseline_manifests,
)
from build_smp_specialist_manifests import (
  SpecialistManifestCfg,
)
from build_smp_specialist_manifests import (
  write_manifests as write_specialist_manifests,
)
from generate_smp_matched_eval_banks import EvalBankCfg
from generate_smp_matched_eval_banks import generate as generate_eval_banks
from generate_smp_matched_reset_bank import (
  ResetBankCfg,
)
from generate_smp_matched_reset_bank import (
  build_plan as build_reset_bank_plan,
)
from launch_smp_native_baselines import (
  NativeBaselineLaunchCfg,
  _job_log_completed,
  launch_baselines,
)
from launch_smp_policy_seed_confirmation import (
  ConfirmationCfg,
  launch_confirmation,
)
from launch_smp_tp_specialists import SpecialistLaunchCfg, launch_specialists
from monitor_smp_training_health import HealthCfg, inspect
from select_smp_confirmed_flat_arm import (
  FlatPromotionCfg,
)
from select_smp_confirmed_flat_arm import (
  write_selection as write_flat_selection,
)
from select_smp_stable_arm import SelectionCfg, write_selection
from select_smp_unified_prerequisites import (
  UnifiedPrerequisiteCfg,
)
from select_smp_unified_prerequisites import (
  write_selection as write_unified_selection,
)

_EVALUATION_SCHEMA_VERSION = 2
_LOCKED_MANIFEST_HASHES = {
  8000: "64506f71e85b69b58bb5579621b10a8aa6969a172428c2d213115fb54a08c333",
  15000: "2e99d3af1bc6a3f9d5c01bf17f15d3588b73b673853af78a6dd711820382f7d9",
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
  launch_specialists_when_ready: bool = False
  launch_baseline_bank_when_ready: bool = False
  launch_native_eval_bank_when_ready: bool = False
  launch_native_baselines_when_ready: bool = False
  launch_native_evaluations_when_ready: bool = False
  confirmation_control_dir: Path = Path(
    "run_control/scratch_causal_policy_seed_confirmation"
  )
  confirmation_evidence_dir: Path = Path("run_control/scratch_causal_policy_seed_eval")
  specialist_control_dir: Path = Path("run_control/ral_tp_specialists")
  specialist_evidence_dir: Path = Path("run_control/ral_tp_specialist_eval")
  specialist_smoke_work_dir: Path = Path("run_control/ral_tp_smoke")
  specialist_smoke_output: Path = Path("run_control/ral_tp_smoke/result.json")
  progression_protocol: Path = Path("docs/ral_terrain_plate_protocol.json")
  baseline_control_dir: Path = Path("run_control/ral_baselines")
  baseline_registry_template: Path = Path("docs/ral_baseline_registry.json")
  baseline_runtime_registry: Path = Path("run_control/ral_baselines/registry.json")
  baseline_bank_output: Path = Path("run_control/ral_baselines/matched_reset_bank.pt")
  baseline_bank_manifest: Path = Path(
    "run_control/ral_baselines/matched_reset_bank.json"
  )
  baseline_bank_device: str = "cuda:6"
  baseline_native_control_dir: Path = Path("run_control/ral_baselines/native_training")
  baseline_native_manifest_dir: Path = Path(
    "run_control/ral_baselines/native_checkpoint_manifests"
  )
  baseline_eval_bank_output_dir: Path = Path(
    "run_control/ral_baselines/held_out_eval_banks"
  )
  baseline_eval_bank_manifest: Path = Path(
    "run_control/ral_baselines/held_out_eval_banks.json"
  )
  baseline_formal_manifest_dir: Path = Path(
    "run_control/ral_baselines/formal_manifests"
  )
  baseline_evidence_dir: Path = Path("run_control/ral_baselines/native_eval")
  baseline_effect_output: Path = Path(
    "run_control/ral_baselines/native_eval/paired_effects.json"
  )
  specialist_eval_seed: int = 20260910
  specialist_eval_num_envs: int = 256
  specialist_eval_steps: int = 750


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
    manifest_payload = json.loads(manifest.read_text())
  except (json.JSONDecodeError, OSError):
    return False
  runs = manifest_payload.get("runs")
  if not isinstance(runs, list) or not runs:
    return False
  expected_count = len(runs) * len(cfg.modes)
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


def _active_eval(evidence_dir: Path) -> dict[str, Any] | None:
  launch_state = evidence_dir / "active_evaluation.json"
  if not launch_state.exists():
    return None
  payload = json.loads(launch_state.read_text())
  pid = int(payload["pid"])
  payload["process_alive"] = _pid_alive(pid)
  return payload


def _reconcile_active_eval(
  evidence_dir: Path,
  active: dict[str, Any] | None,
  evaluations: list[dict[str, Any]],
  *,
  identity_key: str,
) -> tuple[dict[str, Any] | None, bool]:
  """Clear a successful worker marker or retain a dead incomplete worker."""
  if active is None or active.get("process_alive", True):
    return active, False
  identity = int(active["gate"])
  complete = any(
    int(row[identity_key]) == identity and row["analysis_complete"]
    for row in evaluations
  )
  if complete:
    (evidence_dir / "active_evaluation.json").unlink()
    return None, False
  return active, True


def _launch_evaluation(
  cfg: PipelineCfg,
  manifest: Path,
  gate: int,
  output_dir: Path,
  state_dir: Path | None = None,
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
  _atomic_json((state_dir or cfg.evidence_dir) / "active_evaluation.json", payload)
  return payload


def _active_tp_smoke(cfg: PipelineCfg) -> dict[str, Any] | None:
  state = cfg.specialist_smoke_work_dir / "active_smoke.json"
  if not state.is_file():
    return None
  payload = json.loads(state.read_text())
  if _pid_alive(int(payload["pid"])):
    return payload
  state.unlink()
  return None


def _launch_tp_smoke(cfg: PipelineCfg, promotion: Path) -> dict[str, Any]:
  script = Path(__file__).with_name("run_smp_tp_physics_smoke.py").resolve()
  cfg.specialist_smoke_work_dir.mkdir(parents=True, exist_ok=True)
  log = cfg.specialist_smoke_work_dir / "physics_smoke_launcher.log"
  command = [
    sys.executable,
    str(script),
    "--promotion",
    str(promotion.resolve()),
    "--output",
    str(cfg.specialist_smoke_output.resolve()),
    "--work-dir",
    str(cfg.specialist_smoke_work_dir.resolve()),
    "--protocol",
    str(cfg.progression_protocol.resolve()),
    "--run",
  ]
  with log.open("a") as stream:
    process = subprocess.Popen(
      command,
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  payload = {
    "pid": process.pid,
    "promotion": str(promotion.resolve()),
    "output": str(cfg.specialist_smoke_output.resolve()),
    "log": str(log.resolve()),
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(cfg.specialist_smoke_work_dir / "active_smoke.json", payload)
  return payload


def _reset_bank_cfg(
  cfg: PipelineCfg, promotion: Path, *, run: bool = False
) -> ResetBankCfg:
  return ResetBankCfg(
    promotion=promotion,
    output=cfg.baseline_bank_output,
    manifest=cfg.baseline_bank_manifest,
    registry_template=cfg.baseline_registry_template,
    runtime_registry=cfg.baseline_runtime_registry,
    device=cfg.baseline_bank_device,
    run=run,
  )


def _validate_reset_bank_artifacts(
  cfg: PipelineCfg, promotion: Path
) -> dict[str, Any] | None:
  paths = (
    cfg.baseline_bank_output,
    cfg.baseline_bank_manifest,
    cfg.baseline_runtime_registry,
  )
  exists = tuple(path.is_file() for path in paths)
  if not any(exists):
    return None
  if not all(exists):
    raise ValueError("partial matched reset-bank artifacts exist")
  plan = build_reset_bank_plan(_reset_bank_cfg(cfg, promotion))
  manifest = json.loads(cfg.baseline_bank_manifest.read_text())
  if (
    manifest.get("status") != "READY"
    or manifest.get("plan_id") != plan["plan_id"]
    or manifest.get("promotion_id") != plan["promotion_id"]
    or manifest.get("generator_code_sha256") != plan["generator_code_sha256"]
    or manifest.get("bank") != str(cfg.baseline_bank_output.resolve())
    or manifest.get("bank_sha256") != _sha256(cfg.baseline_bank_output)
  ):
    raise ValueError("matched reset-bank manifest conflicts with frozen plan")
  runtime_registry = json.loads(cfg.baseline_runtime_registry.read_text())
  readiness = audit_baseline_registry(runtime_registry, cfg.baseline_runtime_registry)
  if not readiness["reset_bank_ready"]:
    raise ValueError("runtime baseline registry did not validate the reset bank")
  return {"manifest": manifest, "readiness": readiness}


def _active_reset_bank(cfg: PipelineCfg, promotion: Path) -> dict[str, Any] | None:
  state = cfg.baseline_control_dir / "active_generation.json"
  if not state.is_file():
    return None
  payload = json.loads(state.read_text())
  if _pid_alive(int(payload["pid"])):
    return payload
  try:
    complete = _validate_reset_bank_artifacts(cfg, promotion)
  except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    return {**payload, "failed": True, "error": str(error)}
  if complete is None:
    return {**payload, "failed": True, "error": "generator exited without artifacts"}
  state.unlink()
  return None


def _launch_reset_bank(cfg: PipelineCfg, promotion: Path) -> dict[str, Any]:
  script = Path(__file__).with_name("generate_smp_matched_reset_bank.py").resolve()
  plan = build_reset_bank_plan(_reset_bank_cfg(cfg, promotion))
  cfg.baseline_control_dir.mkdir(parents=True, exist_ok=True)
  log = cfg.baseline_control_dir / "matched_reset_bank.log"
  command = [
    sys.executable,
    str(script),
    "--promotion",
    str(promotion.resolve()),
    "--output",
    str(cfg.baseline_bank_output.resolve()),
    "--manifest",
    str(cfg.baseline_bank_manifest.resolve()),
    "--registry-template",
    str(cfg.baseline_registry_template.resolve()),
    "--runtime-registry",
    str(cfg.baseline_runtime_registry.resolve()),
    "--device",
    cfg.baseline_bank_device,
    "--run",
  ]
  with log.open("a") as stream:
    process = subprocess.Popen(
      command,
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  payload = {
    "pid": process.pid,
    "plan_id": plan["plan_id"],
    "promotion": str(promotion.resolve()),
    "output": str(cfg.baseline_bank_output.resolve()),
    "manifest": str(cfg.baseline_bank_manifest.resolve()),
    "runtime_registry": str(cfg.baseline_runtime_registry.resolve()),
    "log": str(log.resolve()),
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(cfg.baseline_control_dir / "active_generation.json", payload)
  return payload


def _specialist_analysis_complete(
  cfg: PipelineCfg, output_dir: Path, manifest_path: Path
) -> bool:
  required = ("_COMPLETE.json", "summary.json", "analysis.json", "analysis.md")
  if not all((output_dir / name).is_file() for name in required):
    return False
  try:
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads((output_dir / "_COMPLETE.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    analysis = json.loads((output_dir / "analysis.json").read_text())
  except (json.JSONDecodeError, OSError):
    return False
  phase = manifest.get("phase")
  expected_strata = 76 if phase == "T" else 10 if phase == "P" else -1
  return (
    complete.get("specialist_matrix_schema_version") == 1
    and complete.get("base_evaluation_schema_version") == _EVALUATION_SCHEMA_VERSION
    and complete.get("phase") == phase
    and complete.get("manifest") == str(manifest_path.resolve())
    and complete.get("manifest_sha256") == _sha256(manifest_path)
    and complete.get("policy_seed") == manifest.get("policy_seed")
    and complete.get("checkpoint_step") == manifest.get("checkpoint_step")
    and complete.get("stratum_count") == expected_strata
    and complete.get("result_count") == expected_strata
    and complete.get("eval_seed") == cfg.specialist_eval_seed
    and complete.get("num_envs_per_stratum") == cfg.specialist_eval_num_envs
    and complete.get("steps") == cfg.specialist_eval_steps
    and summary.get("manifest_sha256") == _sha256(manifest_path)
    and len(summary.get("evaluations", [])) == expected_strata
    and analysis.get("status") in {"PASS", "NO_PROMOTION"}
    and analysis.get("summary") == str((output_dir / "summary.json").resolve())
    and analysis.get("summary_sha256") == _sha256(output_dir / "summary.json")
    and analysis.get("manifest_sha256") == _sha256(manifest_path)
  )


def _active_specialist_eval(cfg: PipelineCfg) -> dict[str, Any] | None:
  state = cfg.specialist_evidence_dir / "active_evaluation.json"
  if not state.is_file():
    return None
  payload = json.loads(state.read_text())
  if _pid_alive(int(payload["pid"])):
    return payload
  output_dir = Path(payload["output_dir"])
  if all(
    (output_dir / name).is_file()
    for name in ("_COMPLETE.json", "summary.json", "analysis.json", "analysis.md")
  ):
    state.unlink()
    return None
  return {**payload, "failed": True}


def _launch_specialist_evaluation(
  cfg: PipelineCfg, manifest: Path, output_dir: Path
) -> dict[str, Any]:
  matrix = Path(__file__).with_name("run_smp_specialist_eval_matrix.py").resolve()
  analyzer = Path(__file__).with_name("analyze_smp_specialist_matrix.py").resolve()
  manifest_payload = json.loads(manifest.read_text())
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
    "--eval-seed",
    str(cfg.specialist_eval_seed),
    "--num-envs",
    str(cfg.specialist_eval_num_envs),
    "--steps",
    str(cfg.specialist_eval_steps),
  ]
  analysis_command = [
    sys.executable,
    str(analyzer),
    "--summary",
    str((output_dir / "summary.json").resolve()),
    "--protocol",
    str(cfg.progression_protocol.resolve()),
    "--output-json",
    str((output_dir / "analysis.json").resolve()),
    "--output-markdown",
    str((output_dir / "analysis.md").resolve()),
  ]
  command = " ".join(shlex.quote(part) for part in matrix_command)
  command += " && " + " ".join(shlex.quote(part) for part in analysis_command)
  log = output_dir / "evaluation.log"
  with log.open("a") as stream:
    process = subprocess.Popen(
      ("bash", "-lc", command),
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  payload = {
    "pid": process.pid,
    "phase": manifest_payload["phase"],
    "policy_seed": manifest_payload["policy_seed"],
    "checkpoint_step": manifest_payload["checkpoint_step"],
    "manifest": str(manifest.resolve()),
    "manifest_sha256": _sha256(manifest),
    "output_dir": str(output_dir.resolve()),
    "log": str(log.resolve()),
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(cfg.specialist_evidence_dir / "active_evaluation.json", payload)
  return payload


def _advance_specialist_evaluations(
  cfg: PipelineCfg, index: dict[str, Any], gpu_processes: list[str]
) -> dict[str, Any]:
  active = _active_specialist_eval(cfg)
  evaluations = []
  next_row = None
  rows = sorted(
    index["manifests"],
    key=lambda row: (
      int(row["checkpoint_step"]),
      str(row["phase"]),
      int(row["policy_seed"]),
    ),
  )
  for row in rows:
    phase = str(row["phase"])
    seed = int(row["policy_seed"])
    gate = int(row["checkpoint_step"])
    manifest = Path(row["path"])
    output_dir = (
      cfg.specialist_evidence_dir / phase.lower() / f"seed_{seed}" / f"gate_{gate}"
    )
    complete = _specialist_analysis_complete(cfg, output_dir, manifest)
    evaluations.append(
      {
        "phase": phase,
        "policy_seed": seed,
        "checkpoint_step": gate,
        "output_dir": str(output_dir.resolve()),
        "analysis_complete": complete,
      }
    )
    if not complete and next_row is None:
      next_row = (row, manifest, output_dir)
  if active is not None:
    if active.get("failed"):
      return {
        "status": "TP_SPECIALIST_EVAL_ALERT",
        "action": (
          "Specialist evaluator exited without complete evidence; inspect its log "
          "before an explicit resumable retry."
        ),
        "evaluations": evaluations,
        "active_evaluation": active,
        "terrain_aggregate": None,
        "plate_aggregate": None,
        "unified_prerequisite": None,
      }
    return {
      "status": "TP_SPECIALIST_EVAL_RUNNING",
      "action": "Wait for the active resumable specialist matrix and analyzer.",
      "evaluations": evaluations,
      "active_evaluation": active,
      "terrain_aggregate": None,
      "plate_aggregate": None,
      "unified_prerequisite": None,
    }
  if next_row is not None:
    row, manifest, output_dir = next_row
    if gpu_processes:
      return {
        "status": "WAITING_FREE_GPU",
        "action": "Specialist checkpoints are ready; wait for free evaluation GPUs.",
        "evaluations": evaluations,
        "active_evaluation": None,
        "terrain_aggregate": None,
        "plate_aggregate": None,
        "unified_prerequisite": None,
      }
    if not cfg.launch_when_ready:
      return {
        "status": "TP_SPECIALIST_READY_FOR_EVAL",
        "action": (
          f"Run {row['phase']} seed {row['policy_seed']} gate "
          f"{row['checkpoint_step']} frozen matrix."
        ),
        "evaluations": evaluations,
        "active_evaluation": None,
        "terrain_aggregate": None,
        "plate_aggregate": None,
        "unified_prerequisite": None,
      }
    launched = _launch_specialist_evaluation(cfg, manifest, output_dir)
    return {
      "status": "TP_SPECIALIST_EVAL_RUNNING",
      "action": (
        f"Launched {row['phase']} seed {row['policy_seed']} gate "
        f"{row['checkpoint_step']} frozen matrix."
      ),
      "evaluations": evaluations,
      "active_evaluation": launched,
      "terrain_aggregate": None,
      "plate_aggregate": None,
      "unified_prerequisite": None,
    }

  aggregates = {}
  for phase in ("T", "P"):
    analyses = tuple(
      cfg.specialist_evidence_dir
      / phase.lower()
      / f"seed_{seed}"
      / "gate_19999"
      / "analysis.json"
      for seed in index["policy_seeds"]
    )
    output = cfg.specialist_evidence_dir / phase.lower() / "policy_seed_aggregate.json"
    aggregates[phase] = write_specialist_aggregate(
      SpecialistAggregateCfg(
        analyses=analyses,
        output_json=output,
        protocol=cfg.progression_protocol,
      )
    )
  unified_path = cfg.specialist_evidence_dir / "unified_prerequisite.json"
  unified = write_unified_selection(
    UnifiedPrerequisiteCfg(
      terrain_aggregate=cfg.specialist_evidence_dir
      / "t"
      / "policy_seed_aggregate.json",
      plate_aggregate=cfg.specialist_evidence_dir / "p" / "policy_seed_aggregate.json",
      output=unified_path,
      protocol=cfg.progression_protocol,
    )
  )
  return {
    "status": (
      "U_PREREQUISITES_MET"
      if unified["status"] == "PROMOTE_U"
      else "TP_SPECIALIST_NO_PROMOTION"
    ),
    "action": (
      "T and P matched-seed final gates passed; implement and smoke the frozen U task."
      if unified["status"] == "PROMOTE_U"
      else "At least one frozen T/P phase gate failed; do not launch U or relax thresholds."
    ),
    "evaluations": evaluations,
    "active_evaluation": None,
    "terrain_aggregate": aggregates["T"],
    "plate_aggregate": aggregates["P"],
    "unified_prerequisite": unified,
  }


def _advance_specialists(cfg: PipelineCfg, gpu_processes: list[str]) -> dict[str, Any]:
  aggregate = cfg.confirmation_evidence_dir / "policy_seed_aggregate.json"
  manifest_index = cfg.confirmation_evidence_dir / "manifests" / "index.json"
  promotion_path = cfg.confirmation_evidence_dir / "flat_promotion.json"
  promotion = write_flat_selection(
    FlatPromotionCfg(
      aggregate=aggregate,
      confirmation_manifest_index=manifest_index,
      protocol=cfg.progression_protocol,
      output=promotion_path,
    )
  )
  if promotion["status"] != "PROMOTE_TP_SPECIALISTS":
    return {
      "status": "TP_NO_PROMOTION",
      "action": "Three-seed flat evidence failed frozen T/P prerequisites.",
      "promotion": promotion,
      "smoke": None,
      "launch": None,
    }

  active_bank = _active_reset_bank(cfg, promotion_path)
  if active_bank is not None:
    return {
      "status": (
        "RESET_BANK_ALERT" if active_bank.get("failed") else "RESET_BANK_RUNNING"
      ),
      "action": (
        "Matched reset-bank generator exited without valid immutable artifacts."
        if active_bank.get("failed")
        else "Wait for the promotion-bound matched reset bank."
      ),
      "promotion": promotion,
      "reset_bank": active_bank,
      "smoke": None,
      "launch": None,
    }
  try:
    reset_bank = _validate_reset_bank_artifacts(cfg, promotion_path)
  except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    return {
      "status": "RESET_BANK_ALERT",
      "action": f"Matched reset-bank validation failed: {error}",
      "promotion": promotion,
      "reset_bank": None,
      "smoke": None,
      "launch": None,
    }
  if reset_bank is None:
    if gpu_processes:
      return {
        "status": "WAITING_FREE_GPU",
        "action": "Flat promotion is ready; wait for an idle GPU before reset-bank generation.",
        "promotion": promotion,
        "reset_bank": None,
        "smoke": None,
        "launch": None,
      }
    if not cfg.launch_baseline_bank_when_ready:
      return {
        "status": "RESET_BANK_READY",
        "action": "Generate the frozen matched reset bank before T/P smoke or Tier-A baselines.",
        "promotion": promotion,
        "reset_bank": None,
        "smoke": None,
        "launch": None,
      }
    launched_bank = _launch_reset_bank(cfg, promotion_path)
    return {
      "status": "RESET_BANK_RUNNING",
      "action": "Launched promotion-bound matched reset-bank generation.",
      "promotion": promotion,
      "reset_bank": launched_bank,
      "smoke": None,
      "launch": None,
    }

  launch_manifest = cfg.specialist_control_dir / "launch_manifest.json"
  if launch_manifest.is_file():
    launched = json.loads(launch_manifest.read_text())
    alive = [
      job
      for job in launched.get("jobs", [])
      if job.get("pid") is not None and _pid_alive(int(job["pid"]))
    ]
    if alive:
      return {
        "status": "TP_SPECIALIST_TRAINING",
        "action": (
          f"T/P specialist training is running: {len(alive)}/{len(launched['jobs'])}."
        ),
        "promotion": promotion,
        "smoke": None,
        "launch": launched,
        "health": None,
        "manifest_index": None,
      }
    health = inspect(
      HealthCfg(
        control_dir=cfg.specialist_control_dir,
        output=cfg.state.with_name("tp_specialist_health_latest.json"),
        expected_jobs=len(launched["jobs"]),
      )
    )
    _atomic_json(cfg.state.with_name("tp_specialist_health_latest.json"), health)
    if not health["jobs"] or not all(job["completed"] for job in health["jobs"]):
      return {
        "status": "TP_SPECIALIST_ALERT",
        "action": "T/P processes exited before every specialist completed; inspect logs.",
        "promotion": promotion,
        "smoke": None,
        "launch": launched,
        "health": health,
        "manifest_index": None,
      }
    index = write_specialist_manifests(
      SpecialistManifestCfg(
        launch_manifest=launch_manifest,
        output_dir=cfg.specialist_evidence_dir / "manifests",
      )
    )
    progress = _advance_specialist_evaluations(cfg, index, gpu_processes)
    return {
      "status": progress["status"],
      "action": progress["action"],
      "promotion": promotion,
      "smoke": None,
      "launch": launched,
      "health": health,
      "manifest_index": index,
      "evaluation_progress": progress,
    }

  active_smoke = _active_tp_smoke(cfg)
  if active_smoke is not None:
    return {
      "status": "TP_SMOKE_RUNNING",
      "action": "Wait for sequential T/P MuJoCo physics smoke tests.",
      "promotion": promotion,
      "smoke": active_smoke,
      "launch": None,
      "health": None,
      "manifest_index": None,
    }
  smoke = None
  if cfg.specialist_smoke_output.is_file():
    smoke = json.loads(cfg.specialist_smoke_output.read_text())
    if smoke.get("status") != "PASS":
      return {
        "status": "TP_SMOKE_ALERT",
        "action": "T/P smoke artifact exists but is not PASS.",
        "promotion": promotion,
        "smoke": smoke,
        "launch": None,
        "health": None,
        "manifest_index": None,
      }
  if smoke is None:
    if gpu_processes:
      return {
        "status": "WAITING_FREE_GPU",
        "action": "Flat promotion is ready; wait for a free GPU for T/P smoke.",
        "promotion": promotion,
        "smoke": None,
        "launch": None,
        "health": None,
        "manifest_index": None,
      }
    if not cfg.launch_specialists_when_ready:
      return {
        "status": "TP_READY_FOR_SMOKE",
        "action": "Run physical T/P smoke before specialist launch.",
        "promotion": promotion,
        "smoke": None,
        "launch": None,
        "health": None,
        "manifest_index": None,
      }
    active_smoke = _launch_tp_smoke(cfg, promotion_path)
    return {
      "status": "TP_SMOKE_RUNNING",
      "action": "Launched sequential T/P MuJoCo physics smoke tests.",
      "promotion": promotion,
      "smoke": active_smoke,
      "launch": None,
      "health": None,
      "manifest_index": None,
    }

  if gpu_processes:
    return {
      "status": "WAITING_FREE_GPU",
      "action": "T/P smoke passed; wait for all GPUs before six-job launch.",
      "promotion": promotion,
      "smoke": smoke,
      "launch": None,
      "health": None,
      "manifest_index": None,
    }
  planned = launch_specialists(
    SpecialistLaunchCfg(
      promotion=promotion_path,
      control_dir=cfg.specialist_control_dir,
      smoke_test=cfg.specialist_smoke_output,
      protocol=cfg.progression_protocol,
      launch=cfg.launch_specialists_when_ready,
    )
  )
  return {
    "status": "TP_SPECIALIST_TRAINING"
    if planned["status"] == "LAUNCHED"
    else "TP_READY_FOR_LAUNCH",
    "action": (
      "Launched matched-seed T and P specialists."
      if planned["status"] == "LAUNCHED"
      else "T/P smoke passed; specialist launch plan is ready."
    ),
    "promotion": promotion,
    "smoke": smoke,
    "launch": planned,
    "health": None,
    "manifest_index": None,
  }


def _eval_bank_cfg(
  cfg: PipelineCfg, promotion: Path, *, run: bool = False
) -> EvalBankCfg:
  return EvalBankCfg(
    promotion=promotion,
    training_bank=cfg.baseline_bank_output,
    training_bank_manifest=cfg.baseline_bank_manifest,
    registry=cfg.baseline_registry_template,
    output_dir=cfg.baseline_eval_bank_output_dir,
    manifest=cfg.baseline_eval_bank_manifest,
    num_states_per_mode=cfg.num_envs,
    seed=cfg.eval_seed,
    device=cfg.baseline_bank_device,
    run=run,
  )


def _validate_eval_bank_artifacts(
  cfg: PipelineCfg, promotion: Path
) -> dict[str, Any] | None:
  result = generate_eval_banks(_eval_bank_cfg(cfg, promotion))
  if result.get("status") == "PLANNED":
    return None
  if result.get("status") != "READY":
    raise ValueError(f"inadmissible held-out bank status: {result.get('status')}")
  return result


def _active_eval_bank(cfg: PipelineCfg, promotion: Path) -> dict[str, Any] | None:
  state = cfg.baseline_control_dir / "active_held_out_generation.json"
  if not state.is_file():
    return None
  payload = json.loads(state.read_text())
  if _pid_alive(int(payload["pid"])):
    return payload
  try:
    complete = _validate_eval_bank_artifacts(cfg, promotion)
  except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    return {**payload, "failed": True, "error": str(error)}
  if complete is None:
    return {
      **payload,
      "failed": True,
      "error": "held-out generator exited without complete immutable artifacts",
    }
  state.unlink()
  return None


def _launch_eval_bank(cfg: PipelineCfg, promotion: Path) -> dict[str, Any]:
  planned = generate_eval_banks(_eval_bank_cfg(cfg, promotion))
  if planned.get("status") != "PLANNED":
    raise ValueError("held-out bank launch expected a PLANNED artifact set")
  script = Path(__file__).with_name("generate_smp_matched_eval_banks.py").resolve()
  cfg.baseline_control_dir.mkdir(parents=True, exist_ok=True)
  log = cfg.baseline_control_dir / "held_out_eval_bank.log"
  command = [
    sys.executable,
    str(script),
    "--promotion",
    str(promotion.resolve()),
    "--training-bank",
    str(cfg.baseline_bank_output.resolve()),
    "--training-bank-manifest",
    str(cfg.baseline_bank_manifest.resolve()),
    "--registry",
    str(cfg.baseline_registry_template.resolve()),
    "--output-dir",
    str(cfg.baseline_eval_bank_output_dir.resolve()),
    "--manifest",
    str(cfg.baseline_eval_bank_manifest.resolve()),
    "--num-states-per-mode",
    str(cfg.num_envs),
    "--seed",
    str(cfg.eval_seed),
    "--device",
    cfg.baseline_bank_device,
    "--run",
  ]
  with log.open("a") as stream:
    process = subprocess.Popen(
      command,
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  payload = {
    "pid": process.pid,
    "plan_id": planned["plan_id"],
    "promotion": str(promotion.resolve()),
    "manifest": str(cfg.baseline_eval_bank_manifest.resolve()),
    "log": str(log.resolve()),
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(cfg.baseline_control_dir / "active_held_out_generation.json", payload)
  return payload


def _active_native_evaluation(cfg: PipelineCfg) -> dict[str, Any] | None:
  state = cfg.baseline_evidence_dir / "active_evaluation.json"
  if not state.is_file():
    return None
  payload = json.loads(state.read_text())
  if _pid_alive(int(payload["pid"])):
    return payload
  manifest = Path(payload["manifest"])
  output_dir = Path(payload["output_dir"])
  if _analysis_complete(cfg, output_dir, manifest):
    state.unlink()
    return None
  return {
    **payload,
    "failed": True,
    "error": "native evaluator exited without complete matrix and analysis",
  }


def _advance_native_baselines(
  cfg: PipelineCfg, gpu_processes: list[str], promotion: Path
) -> dict[str, Any]:
  active_bank = _active_eval_bank(cfg, promotion)
  if active_bank is not None:
    return {
      "status": (
        "NATIVE_HELD_OUT_ALERT"
        if active_bank.get("failed")
        else "NATIVE_HELD_OUT_RUNNING"
      ),
      "action": (
        "Held-out generator failed closed; inspect its immutable state and log."
        if active_bank.get("failed")
        else "Wait for the disjoint held-out reset banks to finish."
      ),
      "held_out_bank": active_bank,
      "launch": None,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }
  try:
    held_out = _validate_eval_bank_artifacts(cfg, promotion)
  except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    return {
      "status": "NATIVE_HELD_OUT_ALERT",
      "action": f"Held-out reset-bank validation failed: {error}",
      "held_out_bank": None,
      "launch": None,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }
  if held_out is None:
    if gpu_processes:
      status = "WAITING_FREE_GPU"
      action = "Wait for free GPUs before freezing held-out baseline states."
      launched = None
    elif not cfg.launch_native_eval_bank_when_ready:
      status = "NATIVE_HELD_OUT_READY"
      action = "Generate the preregistered disjoint held-out baseline states."
      launched = None
    else:
      launched = _launch_eval_bank(cfg, promotion)
      status = "NATIVE_HELD_OUT_RUNNING"
      action = "Launched preregistered held-out reset-bank generation."
    return {
      "status": status,
      "action": action,
      "held_out_bank": launched,
      "launch": None,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }

  launch_cfg = NativeBaselineLaunchCfg(
    promotion=promotion,
    runtime_registry=cfg.baseline_runtime_registry,
    control_dir=cfg.baseline_native_control_dir,
    launch=cfg.launch_native_baselines_when_ready,
  )
  launch_path = cfg.baseline_native_control_dir / "launch_manifest.json"
  if not launch_path.is_file() and gpu_processes:
    return {
      "status": "WAITING_FREE_GPU",
      "action": "Held-out states are frozen; wait for free GPUs before Tier-A training.",
      "held_out_bank": held_out,
      "launch": None,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }
  try:
    launch = launch_baselines(launch_cfg)
  except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
    return {
      "status": "NATIVE_BASELINE_ALERT",
      "action": f"Native Tier-A training failed closed: {error}",
      "held_out_bank": held_out,
      "launch": None,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }
  if launch.get("status") != "LAUNCHED":
    return {
      "status": "NATIVE_BASELINE_READY",
      "action": "Held-out states are frozen; launch the nine native Tier-A jobs.",
      "held_out_bank": held_out,
      "launch": launch,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }
  alive = [
    worker
    for worker in launch["workers"]
    if worker.get("pid") is not None and _pid_alive(int(worker["pid"]))
  ]
  completed = [
    job
    for job in launch["jobs"]
    if _job_log_completed(Path(job["log"]), int(launch["max_updates"]))
  ]
  if alive:
    return {
      "status": "NATIVE_BASELINE_TRAINING",
      "action": (
        f"Native Tier-A training is running: {len(completed)}/9 jobs complete, "
        f"{len(alive)} workers alive."
      ),
      "held_out_bank": held_out,
      "launch": launch,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }
  if len(completed) != len(launch["jobs"]):
    return {
      "status": "NATIVE_BASELINE_ALERT",
      "action": "All native workers exited before all nine immutable jobs completed.",
      "held_out_bank": held_out,
      "launch": launch,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }

  try:
    checkpoint_index = write_native_baseline_manifests(
      NativeBaselineManifestCfg(
        launch_manifest=launch_path,
        output_dir=cfg.baseline_native_manifest_dir,
      )
    )
    formal_index = write_bindings(
      EvalBindingCfg(
        checkpoint_index=cfg.baseline_native_manifest_dir / "index.json",
        eval_bank_manifest=cfg.baseline_eval_bank_manifest,
        output_dir=cfg.baseline_formal_manifest_dir,
      )
    )
  except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
    return {
      "status": "NATIVE_BINDING_ALERT",
      "action": f"Native checkpoint or held-out binding failed closed: {error}",
      "held_out_bank": held_out,
      "launch": launch,
      "manifest_index": None,
      "formal_index": None,
      "evaluations": [],
      "aggregates": None,
    }

  active = _active_native_evaluation(cfg)
  evaluations = []
  next_row = None
  for row in sorted(
    formal_index["manifests"],
    key=lambda item: (int(item["checkpoint_step"]), int(item["policy_seed"])),
  ):
    gate = int(row["checkpoint_step"])
    seed = int(row["policy_seed"])
    manifest = Path(row["path"])
    output_dir = cfg.baseline_evidence_dir / f"gate_{gate}" / f"seed_{seed}"
    complete = _analysis_complete(cfg, output_dir, manifest)
    evaluations.append(
      {
        "checkpoint_step": gate,
        "policy_seed": seed,
        "manifest": str(manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "analysis_complete": complete,
      }
    )
    if not complete and next_row is None:
      next_row = (gate, seed, manifest, output_dir)
  if active is not None:
    return {
      "status": (
        "NATIVE_BASELINE_EVAL_ALERT"
        if active.get("failed")
        else "NATIVE_BASELINE_EVAL_RUNNING"
      ),
      "action": (
        "Native evaluator failed closed; inspect its log before an explicit retry."
        if active.get("failed")
        else "Wait for the active native Tier-A frozen matrix."
      ),
      "held_out_bank": held_out,
      "launch": launch,
      "manifest_index": checkpoint_index,
      "formal_index": formal_index,
      "evaluations": evaluations,
      "active_evaluation": active,
      "aggregates": None,
    }
  if next_row is not None:
    gate, seed, manifest, output_dir = next_row
    if gpu_processes:
      status = "WAITING_FREE_GPU"
      action = "Native manifests are bound; wait for free GPUs for formal evaluation."
      launched = None
    elif not cfg.launch_native_evaluations_when_ready:
      status = "NATIVE_BASELINE_READY_FOR_EVAL"
      action = f"Run native Tier-A gate {gate}, policy seed {seed}."
      launched = None
    else:
      launched = _launch_evaluation(
        cfg,
        manifest,
        gate,
        output_dir,
        state_dir=cfg.baseline_evidence_dir,
      )
      status = "NATIVE_BASELINE_EVAL_RUNNING"
      action = f"Launched native Tier-A gate {gate}, policy seed {seed}."
    return {
      "status": status,
      "action": action,
      "held_out_bank": held_out,
      "launch": launch,
      "manifest_index": checkpoint_index,
      "formal_index": formal_index,
      "evaluations": evaluations,
      "active_evaluation": launched,
      "aggregates": None,
    }

  aggregates = {}
  seeds = tuple(int(seed) for seed in formal_index["policy_seeds"])
  try:
    for gate in (8000, 15000, 25000, 29999):
      output = cfg.baseline_evidence_dir / f"gate_{gate}" / "policy_seed_aggregate.json"
      aggregate = write_aggregate(
        AggregateCfg(
          summaries=tuple(
            cfg.baseline_evidence_dir / f"gate_{gate}" / f"seed_{seed}" / "summary.json"
            for seed in seeds
          ),
          output_json=output,
        )
      )
      if aggregate.get("status") != "MINIMUM_POLICY_SEEDS_MET":
        raise ValueError(f"native gate {gate} lacks three policy seeds")
      aggregates[str(gate)] = {
        "path": str(output.resolve()),
        "sha256": _sha256(output),
        "status": aggregate["status"],
      }
  except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    return {
      "status": "NATIVE_AGGREGATE_ALERT",
      "action": f"Native policy-seed aggregation failed closed: {error}",
      "held_out_bank": held_out,
      "launch": launch,
      "manifest_index": checkpoint_index,
      "formal_index": formal_index,
      "evaluations": evaluations,
      "active_evaluation": None,
      "aggregates": None,
    }
  try:
    paired_effects = write_native_effect_analysis(
      NativeEffectCfg(
        evidence_dir=cfg.baseline_evidence_dir,
        registry=cfg.baseline_registry_template,
        output_json=cfg.baseline_effect_output,
      )
    )
  except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    return {
      "status": "NATIVE_PAIRED_EFFECT_ALERT",
      "action": f"Native paired-effect analysis failed closed: {error}",
      "held_out_bank": held_out,
      "launch": launch,
      "manifest_index": checkpoint_index,
      "formal_index": formal_index,
      "evaluations": evaluations,
      "active_evaluation": None,
      "aggregates": aggregates,
      "paired_effects": None,
    }
  supported = paired_effects["status"] == "PROPOSED_PAIRED_ADVANTAGE_SUPPORTED"
  return {
    "status": "NATIVE_BASELINE_EVIDENCE_COMPLETE",
    "action": (
      "Native paired evidence supports the preregistered Proposed SMP advantage; "
      "external adapter baselines remain required."
      if supported
      else "Native paired evidence is complete without the preregistered advantage; "
      "report the null result and do not tune thresholds post hoc."
    ),
    "held_out_bank": held_out,
    "launch": launch,
    "manifest_index": checkpoint_index,
    "formal_index": formal_index,
    "evaluations": evaluations,
    "active_evaluation": None,
    "aggregates": aggregates,
    "paired_effects": {
      "status": paired_effects["status"],
      "path": str(cfg.baseline_effect_output.resolve()),
      "sha256": _sha256(cfg.baseline_effect_output),
      "support_rule_checks": paired_effects["support_rule_checks"],
    },
  }


def _advance_confirmation(
  cfg: PipelineCfg,
  confirmation: dict[str, Any],
  gpu_processes: list[str],
) -> dict[str, Any]:
  alive = [job for job in confirmation["jobs"] if _pid_alive(int(job["pid"]))]
  if alive:
    return {
      "status": "CONFIRMATION_TRAINING",
      "action": (
        f"Policy-seed confirmation is running: {len(alive)}/"
        f"{len(confirmation['jobs'])} processes alive."
      ),
      "health": None,
      "manifest_index": None,
      "evaluations": [],
      "active_evaluation": None,
      "aggregate": None,
    }

  health = inspect(
    HealthCfg(
      control_dir=cfg.confirmation_control_dir,
      output=cfg.state.with_name("confirmation_health_latest.json"),
      expected_jobs=len(confirmation["jobs"]),
    )
  )
  _atomic_json(cfg.state.with_name("confirmation_health_latest.json"), health)
  if not health["jobs"] or not all(job["completed"] for job in health["jobs"]):
    return {
      "status": "CONFIRMATION_ALERT",
      "action": "Confirmation processes exited before every job completed; inspect logs.",
      "health": health,
      "manifest_index": None,
      "evaluations": [],
      "active_evaluation": None,
      "aggregate": None,
    }

  index = write_confirmation_manifests(
    ConfirmationManifestCfg(
      launch_manifest=cfg.confirmation_control_dir / "launch_manifest.json",
      output_dir=cfg.confirmation_evidence_dir / "manifests",
    )
  )
  active = _active_eval(cfg.confirmation_evidence_dir)
  evaluations = []
  next_seed = None
  for row in index["manifests"]:
    seed = int(row["policy_seed"])
    manifest = Path(row["path"])
    output_dir = cfg.confirmation_evidence_dir / f"seed_{seed}"
    complete = _analysis_complete(cfg, output_dir, manifest)
    evaluations.append(
      {
        "policy_seed": seed,
        "output_dir": str(output_dir.resolve()),
        "analysis_complete": complete,
      }
    )
    if not complete and next_seed is None:
      next_seed = seed

  active, evaluation_alert = _reconcile_active_eval(
    cfg.confirmation_evidence_dir,
    active,
    evaluations,
    identity_key="policy_seed",
  )

  if evaluation_alert:
    return {
      "status": "CONFIRMATION_EVAL_ALERT",
      "action": (
        "Confirmation evaluator exited before a complete valid matrix and "
        "analysis; inspect the log and partial results before any explicit retry."
      ),
      "health": health,
      "manifest_index": index,
      "evaluations": evaluations,
      "active_evaluation": active,
      "aggregate": None,
    }
  if active is not None:
    return {
      "status": "CONFIRMATION_EVAL_RUNNING",
      "action": "Wait for the current confirmation-seed matrix to finish.",
      "health": health,
      "manifest_index": index,
      "evaluations": evaluations,
      "active_evaluation": active,
      "aggregate": None,
    }
  if next_seed is not None and gpu_processes:
    return {
      "status": "WAITING_FREE_GPU",
      "action": "Confirmation checkpoints are ready; wait for free evaluation GPUs.",
      "health": health,
      "manifest_index": index,
      "evaluations": evaluations,
      "active_evaluation": None,
      "aggregate": None,
    }
  if next_seed is not None:
    launched = None
    status = "CONFIRMATION_READY_FOR_EVAL"
    action = f"Run the frozen confirmation matrix for policy seed {next_seed}."
    if cfg.launch_when_ready:
      manifest = cfg.confirmation_evidence_dir / "manifests" / f"seed_{next_seed}.json"
      launched = _launch_evaluation(
        cfg,
        manifest,
        next_seed,
        cfg.confirmation_evidence_dir / f"seed_{next_seed}",
        state_dir=cfg.confirmation_evidence_dir,
      )
      status = "CONFIRMATION_EVAL_RUNNING"
      action = f"Launched frozen confirmation matrix for policy seed {next_seed}."
    return {
      "status": status,
      "action": action,
      "health": health,
      "manifest_index": index,
      "evaluations": evaluations,
      "active_evaluation": launched,
      "aggregate": None,
    }

  summaries = tuple(
    cfg.confirmation_evidence_dir / f"seed_{seed}" / "summary.json"
    for seed in index["policy_seeds"]
  )
  aggregate_path = cfg.confirmation_evidence_dir / "policy_seed_aggregate.json"
  aggregate = write_aggregate(
    AggregateCfg(summaries=summaries, output_json=aggregate_path)
  )
  return {
    "status": "CONFIRMATION_ANALYSIS_COMPLETE",
    "action": "Three-seed frozen aggregate is complete; audit promotion evidence.",
    "health": health,
    "manifest_index": index,
    "evaluations": evaluations,
    "active_evaluation": None,
    "aggregate": {
      "status": aggregate["status"],
      "policy_seeds": aggregate["policy_seeds"],
      "path": str(aggregate_path.resolve()),
    },
  }


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
  active = _active_eval(cfg.evidence_dir)
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
  active, evaluation_alert = _reconcile_active_eval(
    cfg.evidence_dir,
    active,
    evaluations,
    identity_key="gate",
  )

  gpu_processes = _gpu_processes()
  stable_selection = None
  confirmation = None
  confirmation_progress = None
  specialist_progress = None
  native_baseline_progress = None
  confirmation_waiting_for_gpu = False
  if completed_training and next_gate is None:
    stable_selection = write_selection(SelectionCfg(evidence_dir=cfg.evidence_dir))
    wants_confirmation = (
      cfg.launch_confirmation_when_ready
      and stable_selection["status"] == "PROMOTE_FOR_POLICY_SEEDS"
    )
    confirmation_manifest_exists = (
      cfg.confirmation_control_dir / "launch_manifest.json"
    ).is_file()
    confirmation_waiting_for_gpu = (
      wants_confirmation and bool(gpu_processes) and not confirmation_manifest_exists
    )
    if wants_confirmation and (confirmation_manifest_exists or not gpu_processes):
      confirmation = launch_confirmation(
        ConfirmationCfg(
          selection=cfg.evidence_dir / "stable_selection.json",
          control_dir=cfg.confirmation_control_dir,
          launch=True,
        )
      )
      confirmation_progress = _advance_confirmation(cfg, confirmation, gpu_processes)
      if confirmation_progress["status"] == "CONFIRMATION_ANALYSIS_COMPLETE":
        aggregate = confirmation_progress.get("aggregate") or {}
        if aggregate.get("status") == "MINIMUM_POLICY_SEEDS_MET":
          specialist_progress = _advance_specialists(cfg, gpu_processes)
          if specialist_progress["status"] in {
            "U_PREREQUISITES_MET",
            "TP_SPECIALIST_NO_PROMOTION",
          }:
            native_baseline_progress = _advance_native_baselines(
              cfg,
              gpu_processes,
              cfg.confirmation_evidence_dir / "flat_promotion.json",
            )
  status = "TRAINING_ACTIVE"
  action = "Continue health monitoring; do not contend with training GPUs."
  launched = None
  if not health["healthy"]:
    status = "TRAINING_ALERT"
    action = "Inspect health alerts before changing or stopping any job."
  elif evaluation_alert:
    status = "EVAL_ALERT"
    action = (
      "Frozen evaluation process exited before a complete valid matrix and "
      "analysis; inspect its log and partial results before any explicit retry."
    )
  elif active is not None:
    status = "EVAL_RUNNING"
    action = "Wait for the resumable frozen matrix and analyzer to finish."
  elif completed_training and next_gate is None:
    if native_baseline_progress is not None:
      status = native_baseline_progress["status"]
      action = native_baseline_progress["action"]
    elif specialist_progress is not None:
      status = specialist_progress["status"]
      action = specialist_progress["action"]
    elif confirmation is not None:
      status = confirmation_progress["status"]
      action = confirmation_progress["action"]
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
        "progress": confirmation_progress,
      }
      if confirmation is not None
      else None
    ),
    "specialists": specialist_progress,
    "native_baselines": native_baseline_progress,
  }


def main(cfg: PipelineCfg) -> None:
  state = advance(cfg)
  _atomic_json(cfg.state, state)
  print(f"{state['status']}: {state['action']}")


if __name__ == "__main__":
  main(tyro.cli(PipelineCfg))
