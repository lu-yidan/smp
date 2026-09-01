"""Fail-closed manifest, evaluation, and analysis for the A10 reset-only canary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import tyro

_STUDY_ID = "smp-reset-only-warmstart-canary-v1"
_STUDY_ROLE = "ENGINEERING_CANARY_NOT_PROMOTION_OR_PERFORMANCE_EVIDENCE"
_PLAN_ID = "ec0d18c3f23390d2b4ab0814671064fa9e8c950cfa07c9f461ea53fce531794e"
_PROTOCOL_SHA256 = "7416457635c9fbb9c30880b07aebfc865f2a0d19e2e0581737069ffec2e84dbf"
_TRAINING_COMMIT = "d61797cd15b7fe68070495abd8d6328f10393a1f"
_MINIMUM_COMMIT = "d61797c"
_TASK = "Smp-Getup-Scratch-A10-F2S2-Physical-Reset-G1"
_RUN_NAME = "reset_only_v1_a10_5k_seed20261201"
_SOURCE_SHA256 = "533fdb6c2072fc9f3436c3e33593c36b29a9c33df7a38667c593845922f016fc"
_FINAL_SHA256 = "97230e57e449338cefc1d8315fe3b2609bf711fafa065840df6cca1f50083f7e"
_POLICY_SEED = 20261201
_EVAL_SEED = 20261210
_GATES = (0, 1000, 3000, 4999)
_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")
_FIXED_MODES = _MODES[1:]
_NUM_ENVS = 512
_STEPS = 500
_FAILURE_CODEBOOK = {
  "0": "success",
  "1": "nonfinite_action",
  "2": "invalid_dynamics",
  "3": "terrain_patch_exit",
  "4": "invalid_escape_setup",
  "5": "invalid_escape_contact",
  "6": "plate_not_escaped",
  "7": "never_reached_head_height",
  "8": "never_upright_while_high",
  "9": "never_low_linear_speed_while_upright",
  "10": "never_low_angular_speed_while_pose_stable",
  "11": "strict_candidate_hold_too_short",
}
_FAILURE_NAMES = tuple(_FAILURE_CODEBOOK.values())
_PER_ENV_REQUIRED = (
  "strict_first_step",
  "strict_failure_reason_code",
  "head_height_gate_first_step",
  "upright_while_high_gate_first_step",
  "low_linear_speed_gate_first_step",
  "low_angular_speed_gate_first_step",
  "strict_candidate_first_step",
  "strict_candidate_max_hold_steps",
  "finite_action",
  "invalid_dynamics",
  "physical_gsi_rejected",
  "physical_procedural_reset",
)
_SAFETY_MAX = (
  "contact_foot_slip_p95_m_s",
  "post_success_root_drift_p95_m",
  "secondary_fall_rate_after_success",
  "foot_separation_at_success_p95_m",
  "action_delta_rms_p95",
  "action_second_difference_rms_p95",
  "max_power_mean_w",
  "max_joint_speed_p95_rad_s",
)
_ERROR = re.compile(
  r"traceback|cuda out of memory|outofmemoryerror|physical_reset_alert|fatal|"
  r"segmentation fault|(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)",
  re.IGNORECASE | re.MULTILINE,
)
_GIB = 1024**3


@dataclass(frozen=True)
class ResetOnlyCanaryCfg:
  protocol: Path = Path("docs/ral_reset_only_warmstart_v1.json")
  training_control_dir: Path = Path("run_control/reset_only_warmstart_v1/training")
  logs_root: Path = Path("logs/rsl_rl/smp_scratch_a10_f2s2_physical_reset_g1")
  manifest_dir: Path = Path("run_control/reset_only_warmstart_v1/eval/manifests")
  evaluation_root: Path = Path("run_control/reset_only_warmstart_v1/eval/formal")
  analysis_json: Path = Path("run_control/reset_only_warmstart_v1/eval/analysis.json")
  analysis_markdown: Path = Path("run_control/reset_only_warmstart_v1/eval/analysis.md")
  state: Path = Path("run_control/automation_state/reset_only_canary_latest.json")
  devices: tuple[str, ...] = tuple(f"cuda:{index}" for index in range(8))
  launch_evaluations_when_ready: bool = False


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
  encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if path.is_file():
    if path.read_text() != encoded:
      raise ValueError(f"immutable artifact drifted: {path}")
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(encoded)
  temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ("git", *args), cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _gpu_processes() -> list[str]:
  result = subprocess.run(
    (
      "nvidia-smi",
      "--query-compute-apps=pid,gpu_uuid,process_name",
      "--format=csv,noheader",
    ),
    check=True,
    capture_output=True,
    text=True,
  )
  return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _disk_preflight(path: Path) -> dict[str, float]:
  usage = shutil.disk_usage(path)
  stats = os.statvfs(path)
  inode_free_fraction = stats.f_favail / stats.f_files if stats.f_files else 0.0
  free_gib = usage.free / _GIB
  if free_gib < 100.0 or inode_free_fraction < 0.10:
    raise RuntimeError(
      f"DISK_SPACE_ALERT: free_gib={free_gib:.1f}, "
      f"inode_free_fraction={inode_free_fraction:.3f}"
    )
  return {"free_gib": free_gib, "inode_free_fraction": inode_free_fraction}


def _tensor_integrity(path: Path, expected_iteration: int) -> dict[str, Any]:
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  if checkpoint.get("iter") != expected_iteration:
    raise ValueError(f"{path} embedded iteration is not {expected_iteration}")
  actor = checkpoint.get("actor_state_dict")
  critic = checkpoint.get("critic_state_dict")
  if not isinstance(actor, dict) or not isinstance(critic, dict):
    raise ValueError(f"{path} lacks actor/critic state dictionaries")
  if tuple(actor["mlp.0.weight"].shape) != (512, 93):
    raise ValueError(f"{path} actor is not 93D single-frame")
  if tuple(critic["mlp.0.weight"].shape) != (512, 960):
    raise ValueError(f"{path} critic dimension drifted")

  def collect(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
      return [value]
    if isinstance(value, dict):
      return [tensor for item in value.values() for tensor in collect(item)]
    if isinstance(value, (tuple, list)):
      return [tensor for item in value for tensor in collect(item)]
    return []

  tensors = collect(checkpoint)
  if not tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
    raise ValueError(f"{path} contains missing or nonfinite tensors")
  return {
    "embedded_iteration": expected_iteration,
    "tensor_count": len(tensors),
    "tensor_elements": sum(tensor.numel() for tensor in tensors),
    "all_tensors_finite": True,
    "actor_observation_dim": 93,
    "critic_observation_dim": 960,
  }


def _validate_training(cfg: ResetOnlyCanaryCfg) -> tuple[dict[str, Any], Path, dict[int, dict[str, Any]]]:
  if not cfg.protocol.is_file() or _sha256(cfg.protocol) != _PROTOCOL_SHA256:
    raise ValueError("reset-only protocol SHA-256 drifted")
  launch = _load(cfg.training_control_dir / "launch_manifest.json")
  expected = {
    "status": "LAUNCHED",
    "study_id": _STUDY_ID,
    "study_role": _STUDY_ROLE,
    "plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
    "code_commit": _TRAINING_COMMIT,
    "task": _TASK,
    "source_checkpoint_sha256": _SOURCE_SHA256,
    "policy_seed": _POLICY_SEED,
    "environment_seed": _POLICY_SEED,
    "num_envs": 4096,
    "max_iterations": 5000,
    "save_interval": 500,
    "learning_rate": 0.0001,
    "run_name": _RUN_NAME,
  }
  for name, value in expected.items():
    if launch.get(name) != value:
      raise ValueError(f"training launch {name} drifted")
  source = Path(launch["source_checkpoint"])
  if not source.is_file() or _sha256(source) != _SOURCE_SHA256:
    raise ValueError("warm-start source checkpoint drifted")
  matches = [path for path in cfg.logs_root.glob(f"*_{_RUN_NAME}") if path.is_dir()]
  if len(matches) != 1:
    raise ValueError(f"expected one A10 run, found {len(matches)}")
  run = matches[0]
  agent = run / "params/agent.yaml"
  environment = run / "params/env.yaml"
  if not agent.is_file() or not environment.is_file():
    raise ValueError("A10 saved configuration is incomplete")
  agent_text = agent.read_text(errors="replace")
  env_text = environment.read_text(errors="replace")
  for pattern in (
    r"^seed: 20261201$",
    r"^num_steps_per_env: 24$",
    r"^max_iterations: 5000$",
    r"^save_interval: 500$",
    r"^resume: true$",
    r"^load_run: \^reset_only_source_a6_seed20261102_gate15000\$$",
    r"^load_checkpoint: \^model_15000\.pt\$$",
    r"^  learning_rate: 0\.0001$",
  ):
    if re.search(pattern, agent_text, re.MULTILINE) is None:
      raise ValueError(f"A10 agent config drifted: {pattern}")
  for pattern in (r"^  num_envs: 4096$", r"^seed: 20261201$"):
    if re.search(pattern, env_text, re.MULTILINE) is None:
      raise ValueError(f"A10 environment config drifted: {pattern}")
  log = Path(launch["log"])
  log_text = log.read_text(errors="replace") if log.is_file() else ""
  if "Learning iteration 4999/5000" not in log_text or _ERROR.search(log_text):
    raise ValueError("A10 training log is incomplete or contains a fatal condition")
  checkpoints: dict[int, dict[str, Any]] = {}
  for gate in _GATES:
    checkpoint = run / f"model_{gate}.pt"
    if not checkpoint.is_file():
      raise ValueError(f"A10 checkpoint missing: {checkpoint}")
    digest = _sha256(checkpoint)
    if gate == 4999 and digest != _FINAL_SHA256:
      raise ValueError("A10 final checkpoint SHA-256 drifted")
    checkpoints[gate] = {
      "path": str(checkpoint.resolve()),
      "sha256": digest,
      "integrity": _tensor_integrity(checkpoint, gate),
    }
  return launch, run, checkpoints


def _manifest_material(
  launch: dict[str, Any],
  run: Path,
  gate: int,
  checkpoint: dict[str, Any],
  evaluation_commit: str,
) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "status": "READY_FOR_FROZEN_EVALUATION",
    "evaluation_status": "READY_FOR_FROZEN_EVALUATION",
    "study_id": _STUDY_ID,
    "study_role": _STUDY_ROLE,
    "claim_boundary": launch["claim_boundary"],
    "training_plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
    "training_code_commit": _TRAINING_COMMIT,
    "evaluation_code_commit": evaluation_commit,
    "checkpoint_step": gate,
    "policy_seed": _POLICY_SEED,
    "environment_seed": _POLICY_SEED,
    "run_dir": str(run.resolve()),
    "saved_agent_config": {
      "path": str((run / "params/agent.yaml").resolve()),
      "sha256": _sha256(run / "params/agent.yaml"),
    },
    "saved_environment_config": {
      "path": str((run / "params/env.yaml").resolve()),
      "sha256": _sha256(run / "params/env.yaml"),
    },
    "checkpoint_integrity": checkpoint["integrity"],
    "evaluation_protocol": {
      "modes": list(_MODES),
      "eval_seed": _EVAL_SEED,
      "num_envs": _NUM_ENVS,
      "steps": _STEPS,
      "schema_version": 2,
      "include_per_env": True,
      "physical_reset_validation": True,
    },
    "runs": [
      {
        "name": "a10_physical_reset",
        "task": _TASK,
        "checkpoint": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "policy_seed": _POLICY_SEED,
        "physical_reset_validation": True,
      }
    ],
  }


def _write_manifests(
  cfg: ResetOnlyCanaryCfg,
  launch: dict[str, Any],
  run: Path,
  checkpoints: dict[int, dict[str, Any]],
  repo: Path,
) -> dict[str, Any]:
  rows = []
  for gate in _GATES:
    path = cfg.manifest_dir / f"gate_{gate}.json"
    evaluation_commit = _git(repo, "rev-parse", "HEAD")
    if path.is_file():
      existing_commit = _load(path).get("evaluation_code_commit")
      if not isinstance(existing_commit, str):
        raise ValueError(f"immutable manifest lacks evaluation commit: {path}")
      _git(repo, "merge-base", "--is-ancestor", existing_commit, "HEAD")
      evaluation_commit = existing_commit
    material = _manifest_material(
      launch, run, gate, checkpoints[gate], evaluation_commit
    )
    manifest_id = hashlib.sha256(
      json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {**material, "manifest_id": manifest_id}
    _immutable_json(path, manifest)
    rows.append(
      {
        "checkpoint_step": gate,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "manifest_id": manifest_id,
        "checkpoint_sha256": checkpoints[gate]["sha256"],
      }
    )
  material = {
    "schema_version": 1,
    "status": "READY_FOR_FROZEN_EVALUATION",
    "study_id": _STUDY_ID,
    "study_role": _STUDY_ROLE,
    "training_plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
    "manifests": rows,
  }
  index = {
    **material,
    "index_id": hashlib.sha256(
      json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
  }
  _immutable_json(cfg.manifest_dir / "index.json", index)
  return index


def _finite(value: Any, name: str) -> float:
  number = float(value)
  if not math.isfinite(number):
    raise ValueError(f"nonfinite {name}: {value}")
  return number


def _audit_raw(path: Path, manifest: dict[str, Any], mode: str) -> dict[str, Any]:
  result = _load(path)
  run = manifest["runs"][0]
  expected = {
    "evaluation_schema_version": 2,
    "policy_kind": "rsl_rl",
    "checkpoint_path": run["checkpoint"],
    "checkpoint_sha256": run["checkpoint_sha256"],
    "task": _TASK,
    "reset_mode": mode,
    "policy_seed": _POLICY_SEED,
    "seed": _EVAL_SEED,
    "num_envs": _NUM_ENVS,
    "steps": _STEPS,
    "actor_observation_dim": 93,
    "critic_observation_dim": 960,
    "physical_reset_validation": True,
  }
  for name, value in expected.items():
    if result.get(name) != value:
      raise ValueError(f"{path} {name} drifted: {result.get(name)} != {value}")
  diagnosis = result.get("strict_failure_diagnosis")
  if not isinstance(diagnosis, dict) or diagnosis.get("schema_version") != 1:
    raise ValueError(f"{path} lacks strict failure telemetry schema 1")
  if diagnosis.get("does_not_change_strict_success") is not True:
    raise ValueError(f"{path} failure telemetry changed strict success")
  if diagnosis.get("reason_codebook") != _FAILURE_CODEBOOK:
    raise ValueError(f"{path} failure reason codebook drifted")
  counts = diagnosis.get("reason_counts")
  if not isinstance(counts, dict) or set(counts) != set(_FAILURE_NAMES):
    raise ValueError(f"{path} failure reason counts are incomplete")
  if sum(int(value) for value in counts.values()) != _NUM_ENVS:
    raise ValueError(f"{path} failure counts do not sum to {_NUM_ENVS}")
  strict_successes = int(result.get("strict_successes", -1))
  if int(counts["success"]) != strict_successes:
    raise ValueError(f"{path} success count disagrees with strict successes")
  per_env = result.get("per_env")
  if not isinstance(per_env, dict):
    raise ValueError(f"{path} lacks per-environment telemetry")
  for name, values in per_env.items():
    if not isinstance(values, list) or len(values) != _NUM_ENVS:
      raise ValueError(f"{path} per_env.{name} does not have {_NUM_ENVS} rows")
  for name in _PER_ENV_REQUIRED:
    if name not in per_env:
      raise ValueError(f"{path} lacks per_env.{name}")
  observed = Counter(
    _FAILURE_CODEBOOK[str(int(code))]
    for code in per_env["strict_failure_reason_code"]
  )
  if {name: observed.get(name, 0) for name in _FAILURE_NAMES} != {
    name: int(counts[name]) for name in _FAILURE_NAMES
  }:
    raise ValueError(f"{path} failure codes disagree with counts")
  if not all(bool(value) for value in per_env["finite_action"]):
    raise ValueError(f"{path} contains a nonfinite action")
  if any(bool(value) for value in per_env["invalid_dynamics"]):
    raise ValueError(f"{path} contains invalid dynamics")
  procedural = [bool(value) for value in per_env["physical_procedural_reset"]]
  rejected = [bool(value) for value in per_env["physical_gsi_rejected"]]
  procedural_rate = sum(procedural) / _NUM_ENVS
  rejected_rate = sum(rejected) / _NUM_ENVS
  if not math.isclose(
    _finite(result.get("physical_procedural_reset_rate"), "procedural rate"),
    procedural_rate,
  ):
    raise ValueError(f"{path} procedural reset rate disagrees with per-env rows")
  if not math.isclose(
    _finite(result.get("physical_gsi_rejection_rate"), "GSI rejection rate"),
    rejected_rate,
  ):
    raise ValueError(f"{path} GSI rejection rate disagrees with per-env rows")
  if mode in _FIXED_MODES and not all(procedural):
    raise ValueError(f"{path} fixed pose did not use grounded procedural reset")
  if mode == "native_gsi" and procedural != rejected:
    raise ValueError(f"{path} native GSI fallback does not equal rejected GSI")
  if _finite(result.get("finite_action_rate"), "finite action rate") != 1.0:
    raise ValueError(f"{path} finite action rate is not one")
  if _finite(result.get("invalid_dynamics_rate"), "invalid dynamics rate") != 0.0:
    raise ValueError(f"{path} invalid dynamics rate is not zero")
  if not math.isclose(
    _finite(result.get("strict_success_rate"), "strict success rate"),
    strict_successes / _NUM_ENVS,
  ):
    raise ValueError(f"{path} strict success rate disagrees with count")
  return result


def _audit_matrix(cfg: ResetOnlyCanaryCfg, row: dict[str, Any]) -> dict[str, Any]:
  manifest_path = Path(row["path"])
  if not manifest_path.is_file() or _sha256(manifest_path) != row["sha256"]:
    raise ValueError("canary manifest missing or drifted")
  manifest = _load(manifest_path)
  stable = {key: value for key, value in manifest.items() if key != "manifest_id"}
  expected_id = hashlib.sha256(
    json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()
  if manifest.get("manifest_id") != expected_id or expected_id != row["manifest_id"]:
    raise ValueError("canary manifest identity drifted")
  gate = int(row["checkpoint_step"])
  matrix_dir = cfg.evaluation_root / f"gate_{gate}"
  complete = _load(matrix_dir / "_COMPLETE.json")
  expected_complete = {
    "evaluation_schema_version": 2,
    "manifest": str(manifest_path.resolve()),
    "result_count": 5,
    "modes": list(_MODES),
    "eval_seeds": [_EVAL_SEED],
    "num_envs": _NUM_ENVS,
    "steps": _STEPS,
  }
  for name, value in expected_complete.items():
    if complete.get(name) != value:
      raise ValueError(f"gate {gate} _COMPLETE {name} drifted")
  summary = _load(matrix_dir / "summary.json")
  if summary.get("metadata", {}).get("manifest_id") != row["manifest_id"]:
    raise ValueError(f"gate {gate} summary lineage drifted")
  raws: dict[str, dict[str, Any]] = {}
  artifacts = []
  for mode in _MODES:
    matches = list(matrix_dir.glob(f"a10_physical_reset__model_{gate}__{mode}__eval{_EVAL_SEED}.json"))
    if len(matches) != 1:
      raise ValueError(f"gate {gate} mode {mode} raw result count is {len(matches)}")
    result = _audit_raw(matches[0], manifest, mode)
    raws[mode] = result
    artifacts.append({"mode": mode, "path": str(matches[0].resolve()), "sha256": _sha256(matches[0])})
  rates = {mode: float(raws[mode]["strict_success_rate"]) for mode in _MODES}
  fixed = [rates[mode] for mode in _FIXED_MODES]
  return {
    "checkpoint_step": gate,
    "manifest_id": row["manifest_id"],
    "modes": rates,
    "gsi": rates["native_gsi"],
    "fixed_macro": sum(fixed) / len(fixed),
    "fixed_worst": min(fixed),
    "physical_gsi_rejection_rate": {
      mode: float(raws[mode]["physical_gsi_rejection_rate"]) for mode in _MODES
    },
    "physical_procedural_reset_rate": {
      mode: float(raws[mode]["physical_procedural_reset_rate"]) for mode in _MODES
    },
    "safety_pareto": {
      name: max(float(raws[mode][name]) for mode in _MODES) for name in _SAFETY_MAX
    },
    "failure_reason_counts": {
      name: sum(int(raws[mode]["strict_failure_diagnosis"]["reason_counts"][name]) for mode in _MODES)
      for name in _FAILURE_NAMES
    },
    "raw_artifacts": artifacts,
  }


def _write_analysis(cfg: ResetOnlyCanaryCfg, rows: list[dict[str, Any]], index: dict[str, Any]) -> dict[str, Any]:
  baseline = rows[0]
  payload = {
    "schema_version": 1,
    "status": "RESET_ONLY_CANARY_COMPLETE_ENGINEERING_ONLY",
    "study_id": _STUDY_ID,
    "study_role": _STUDY_ROLE,
    "automatic_action": "STOP_AND_REVIEW_BEFORE_ANY_NEW_PREREGISTERED_STUDY",
    "promotion": None,
    "paper_evidence": False,
    "training_plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
    "manifest_index_id": index["index_id"],
    "policy_seed_sampling_unit_count": 1,
    "gates": rows,
    "changes_from_gate_0": {
      str(row["checkpoint_step"]): {
        "gsi": row["gsi"] - baseline["gsi"],
        "fixed_macro": row["fixed_macro"] - baseline["fixed_macro"],
        "fixed_worst": row["fixed_worst"] - baseline["fixed_worst"],
      }
      for row in rows[1:]
    },
    "interpretation_limit": (
      "A single warm-start lineage can diagnose whether the reset-only stage changes "
      "behavior, but cannot establish reproducibility, promotion, or RA-L evidence."
    ),
  }
  _immutable_json(cfg.analysis_json, payload)
  lines = [
    "# A10 reset-only warm-start canary",
    "",
    "Status: `RESET_ONLY_CANARY_COMPLETE_ENGINEERING_ONLY`.",
    "",
    "This is one warm-start lineage and is not promotion or paper evidence.",
    "",
    "| Gate | GSI | Fixed macro | Fixed worst | Native GSI rejection |",
    "|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    lines.append(
      f"| {row['checkpoint_step']} | {row['gsi']:.1%} | "
      f"{row['fixed_macro']:.1%} | {row['fixed_worst']:.1%} | "
      f"{row['physical_gsi_rejection_rate']['native_gsi']:.1%} |"
    )
  lines.extend(("", "Automatic action: stop and review before a new preregistered study.", ""))
  markdown = "\n".join(lines)
  if cfg.analysis_markdown.is_file() and cfg.analysis_markdown.read_text() != markdown:
    raise ValueError("immutable canary Markdown analysis drifted")
  cfg.analysis_markdown.parent.mkdir(parents=True, exist_ok=True)
  if not cfg.analysis_markdown.exists():
    cfg.analysis_markdown.write_text(markdown)
  return payload


def _launch_matrix(
  cfg: ResetOnlyCanaryCfg, row: dict[str, Any], preflight: dict[str, float]
) -> dict[str, Any]:
  gate = int(row["checkpoint_step"])
  output = cfg.evaluation_root / f"gate_{gate}"
  output.mkdir(parents=True)
  command = [
    sys.executable,
    str(Path(__file__).with_name("run_smp_frozen_eval_matrix.py").resolve()),
    "--manifest",
    row["path"],
    "--output-dir",
    str(output.resolve()),
    "--devices",
    *cfg.devices,
    "--modes",
    *_MODES,
    "--eval-seeds",
    str(_EVAL_SEED),
    "--num-envs",
    str(_NUM_ENVS),
    "--steps",
    str(_STEPS),
    "--include-per-env",
  ]
  log = output / "evaluation.log"
  with log.open("a") as stream:
    process = subprocess.Popen(
      command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
    )
  marker = {
    "schema_version": 1,
    "status": "ACTIVE",
    "study_id": _STUDY_ID,
    "training_plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
    "manifest_id": row["manifest_id"],
    "checkpoint_step": gate,
    "manifest": row["path"],
    "output_dir": str(output.resolve()),
    "log": str(log.resolve()),
    "command": command,
    "devices": list(cfg.devices),
    "pid": process.pid,
    "attempt": 1,
    "resource_preflight": preflight,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  _atomic_json(cfg.evaluation_root / "active_evaluation.json", marker)
  return marker


def _base(repo: Path) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "OBSERVING",
    "study_id": _STUDY_ID,
    "study_role": _STUDY_ROLE,
    "training_plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
    "code_commit": _git(repo, "rev-parse", "HEAD"),
  }


def advance(cfg: ResetOnlyCanaryCfg) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  state = _base(repo)
  _git(repo, "merge-base", "--is-ancestor", _MINIMUM_COMMIT, "HEAD")
  launch, run, checkpoints = _validate_training(cfg)
  state["training"] = {
    "status": "COMPLETE",
    "run_dir": str(run.resolve()),
    "pid": int(launch["pid"]),
    "process_alive": _pid_alive(int(launch["pid"])),
    "checkpoint_sha256": {str(gate): checkpoints[gate]["sha256"] for gate in _GATES},
  }
  if state["training"]["process_alive"]:
    return {**state, "status": "RESET_ONLY_TRAINING_FINALIZING"}
  gpu = _gpu_processes()
  state["gpu_compute_processes"] = gpu
  if gpu:
    return {**state, "status": "RESET_ONLY_WAITING_GPU_IDLE"}
  index = _write_manifests(cfg, launch, run, checkpoints, repo)
  state["manifests"] = {
    "status": index["status"],
    "index": str((cfg.manifest_dir / "index.json").resolve()),
    "index_id": index["index_id"],
    "manifest_count": len(index["manifests"]),
  }
  marker_path = cfg.evaluation_root / "active_evaluation.json"
  if marker_path.is_file():
    marker = _load(marker_path)
    if (
      marker.get("training_plan_id") != _PLAN_ID
      or marker.get("protocol_sha256") != _PROTOCOL_SHA256
      or marker.get("attempt") != 1
    ):
      return {**state, "status": "RESET_ONLY_EVAL_ALERT", "error": "active marker drifted"}
    if _pid_alive(int(marker["pid"])):
      return {**state, "status": "RESET_ONLY_EVALUATION_ACTIVE", "active_evaluation": marker}
    row = next(
      item for item in index["manifests"]
      if int(item["checkpoint_step"]) == int(marker["checkpoint_step"])
    )
    try:
      _audit_matrix(cfg, row)
    except Exception as error:
      return {
        **state,
        "status": "RESET_ONLY_EVAL_ALERT",
        "error": f"dead evaluator left invalid evidence: {type(error).__name__}: {error}",
        "active_evaluation": marker,
        "automatic_restart_forbidden": True,
      }
    marker_path.unlink()
  complete_rows = []
  for row in index["manifests"]:
    gate = int(row["checkpoint_step"])
    matrix_dir = cfg.evaluation_root / f"gate_{gate}"
    if (matrix_dir / "_COMPLETE.json").is_file():
      try:
        complete_rows.append(_audit_matrix(cfg, row))
      except Exception as error:
        return {
          **state,
          "status": "RESET_ONLY_EVAL_ALERT",
          "error": f"invalid completed matrix: {type(error).__name__}: {error}",
        }
      continue
    if matrix_dir.exists() and any(matrix_dir.iterdir()):
      return {
        **state,
        "status": "RESET_ONLY_EVAL_ALERT",
        "error": f"partial matrix exists without active evaluator: {matrix_dir}",
        "automatic_restart_forbidden": True,
      }
    state["evaluation"] = {
      "completed_matrix_count": len(complete_rows),
      "required_matrix_count": len(_GATES),
      "next_checkpoint_step": gate,
    }
    if not cfg.launch_evaluations_when_ready:
      return {**state, "status": "RESET_ONLY_READY_FOR_FROZEN_EVALUATION"}
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
      return {**state, "status": "CODE_SYNC_ALERT", "error": "tracked worktree is dirty"}
    gpu = _gpu_processes()
    if gpu:
      return {**state, "status": "RESET_ONLY_WAITING_GPU_IDLE", "gpu_compute_processes": gpu}
    try:
      preflight = _disk_preflight(cfg.evaluation_root.parent)
    except RuntimeError as error:
      return {**state, "status": "DISK_SPACE_ALERT", "error": str(error)}
    marker = _launch_matrix(cfg, row, preflight)
    return {**state, "status": "RESET_ONLY_EVALUATION_LAUNCHED", "active_evaluation": marker}
  analysis = _write_analysis(cfg, complete_rows, index)
  return {
    **state,
    "status": analysis["status"],
    "evaluation": {"completed_matrix_count": 4, "required_matrix_count": 4},
    "analysis": str(cfg.analysis_json.resolve()),
    "promotion": None,
  }


def main(cfg: ResetOnlyCanaryCfg) -> None:
  repo = Path(__file__).resolve().parents[1]
  try:
    result = advance(cfg)
  except Exception as error:
    result = {
      **_base(repo),
      "status": "RESET_ONLY_AUTOMATION_ALERT",
      "error": f"{type(error).__name__}: {error}",
      "automatic_restart_forbidden": True,
    }
  _atomic_json(cfg.state, result)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(ResetOnlyCanaryCfg))
