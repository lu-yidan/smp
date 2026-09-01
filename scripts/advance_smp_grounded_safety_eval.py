"""Fail-closed frozen evaluation for the A11 grounded safety fine-tune."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

import advance_smp_reset_only_canary as base

_STUDY_ID = "smp-grounded-safety-finetune-v1"
_STUDY_ROLE = "ENGINEERING_SAFETY_FINETUNE_NOT_PROMOTION_OR_RAL_EVIDENCE"
_PLAN_ID = "77a3a70ad46e0308d089c4a3e44e29a86dfacbd10591f3007e51aa4945e0631d"
_PROTOCOL_SHA256 = "02516a35e2f5f4d3eb09e41fa548883b9e56b33a95c8e912f530ec1549fddb3d"
_TRAINING_COMMIT = "dd273f2d709ce38768568867cd047bc2a6b9e8dc"
_MINIMUM_COMMIT = "dd273f2"
_TASK = "Smp-Getup-Scratch-A11-F2S2-Grounded-Safety-G1"
_RUN_NAME = "a11_grounded_safety_3k_seed20261301"
_RUN_LABEL = "a11_grounded_safety"
_SOURCE_SHA256 = "74582383da1189529ab233f409cc0bb65062a3caaaa5cb6cd28fe07b27cae51b"
_FINAL_SHA256 = "2cc7a58f85142a52ce29d8c68ca3918af7acb6599ecc565b5a5ec0e8e3e9dbc3"
_POLICY_SEED = 20261301
_EVAL_SEED = 20261210
_GATES = (0, 500, 1000, 2000, 2999)
_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")
_FIXED_MODES = _MODES[1:]
_NUM_ENVS = 512
_STEPS = 500
_REFERENCE_ANALYSIS = Path("run_control/reset_only_warmstart_v1/eval/analysis.json")
_REFERENCE_ANALYSIS_SHA256 = (
  "6651cc0cd98e5718e72314d43c33dcc084276d008792a8d218bb998c7350db6f"
)
_REFERENCE_GATE = 1000
_ERROR = re.compile(
  r"traceback|cuda out of memory|outofmemoryerror|physical_reset_alert|fatal|"
  r"segmentation fault|(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)",
  re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class GroundedSafetyEvalCfg:
  protocol: Path = Path("docs/ral_grounded_safety_finetune_v1.json")
  training_control_dir: Path = Path("run_control/grounded_safety_finetune_v1/training")
  logs_root: Path = Path("logs/rsl_rl/smp_scratch_a11_f2s2_grounded_safety_g1")
  manifest_dir: Path = Path("run_control/grounded_safety_finetune_v1/eval/manifests")
  evaluation_root: Path = Path("run_control/grounded_safety_finetune_v1/eval/formal")
  analysis_json: Path = Path(
    "run_control/grounded_safety_finetune_v1/eval/analysis.json"
  )
  analysis_markdown: Path = Path(
    "run_control/grounded_safety_finetune_v1/eval/analysis.md"
  )
  state: Path = Path("run_control/automation_state/grounded_safety_eval_latest.json")
  devices: tuple[str, ...] = tuple(f"cuda:{index}" for index in range(8))
  launch_evaluations_when_ready: bool = False


def _configure_base() -> None:
  for name, value in {
    "_STUDY_ID": _STUDY_ID,
    "_STUDY_ROLE": _STUDY_ROLE,
    "_PLAN_ID": _PLAN_ID,
    "_PROTOCOL_SHA256": _PROTOCOL_SHA256,
    "_TRAINING_COMMIT": _TRAINING_COMMIT,
    "_MINIMUM_COMMIT": _MINIMUM_COMMIT,
    "_TASK": _TASK,
    "_RUN_NAME": _RUN_NAME,
    "_SOURCE_SHA256": _SOURCE_SHA256,
    "_FINAL_SHA256": _FINAL_SHA256,
    "_POLICY_SEED": _POLICY_SEED,
    "_EVAL_SEED": _EVAL_SEED,
    "_GATES": _GATES,
    "_MODES": _MODES,
    "_FIXED_MODES": _FIXED_MODES,
    "_NUM_ENVS": _NUM_ENVS,
    "_STEPS": _STEPS,
  }.items():
    setattr(base, name, value)
  base._validate_training = _validate_training
  base._manifest_material = _manifest_material
  base._audit_matrix = _audit_matrix
  base._write_analysis = _write_analysis


def _validate_training(
  cfg: GroundedSafetyEvalCfg,
) -> tuple[dict[str, Any], Path, dict[int, dict[str, Any]]]:
  if not cfg.protocol.is_file() or base._sha256(cfg.protocol) != _PROTOCOL_SHA256:
    raise ValueError("A11 grounded-safety protocol SHA-256 drifted")
  launch = base._load(cfg.training_control_dir / "launch_manifest.json")
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
    "max_iterations": 3000,
    "save_interval": 500,
    "learning_rate": 1.0e-5,
    "run_name": _RUN_NAME,
  }
  for name, value in expected.items():
    if launch.get(name) != value:
      raise ValueError(f"A11 training launch {name} drifted")
  source = Path(launch["source_checkpoint"])
  if not source.is_file() or base._sha256(source) != _SOURCE_SHA256:
    raise ValueError("A11 warm-start source checkpoint drifted")
  matches = [path for path in cfg.logs_root.glob(f"*_{_RUN_NAME}") if path.is_dir()]
  if len(matches) != 1:
    raise ValueError(f"expected one A11 run, found {len(matches)}")
  run = matches[0]
  agent = run / "params/agent.yaml"
  environment = run / "params/env.yaml"
  if not agent.is_file() or not environment.is_file():
    raise ValueError("A11 saved configuration is incomplete")
  agent_text = agent.read_text(errors="replace")
  env_text = environment.read_text(errors="replace")
  for pattern in (
    r"^seed: 20261301$",
    r"^num_steps_per_env: 24$",
    r"^max_iterations: 3000$",
    r"^save_interval: 500$",
    r"^resume: true$",
    r"^load_run: \^a11_source_a10_seed20261201_gate1000\$$",
    r"^load_checkpoint: \^model_1000\.pt\$$",
    r"^  learning_rate: 1\.0e-05$",
  ):
    if re.search(pattern, agent_text, re.MULTILINE) is None:
      raise ValueError(f"A11 agent config drifted: {pattern}")
  for pattern in (r"^  num_envs: 4096$", r"^seed: 20261301$"):
    if re.search(pattern, env_text, re.MULTILINE) is None:
      raise ValueError(f"A11 environment config drifted: {pattern}")
  log = Path(launch["log"])
  log_text = log.read_text(errors="replace") if log.is_file() else ""
  if (
    "Learning iteration 2999/3000" not in log_text
    or "Total steps: 294912000" not in log_text
    or _ERROR.search(log_text)
  ):
    raise ValueError("A11 training log is incomplete or contains a fatal condition")
  checkpoints: dict[int, dict[str, Any]] = {}
  for gate in _GATES:
    checkpoint = run / f"model_{gate}.pt"
    if not checkpoint.is_file():
      raise ValueError(f"A11 checkpoint missing: {checkpoint}")
    digest = base._sha256(checkpoint)
    if gate == 2999 and digest != _FINAL_SHA256:
      raise ValueError("A11 final checkpoint SHA-256 drifted")
    checkpoints[gate] = {
      "path": str(checkpoint.resolve()),
      "sha256": digest,
      "integrity": base._tensor_integrity(checkpoint, gate),
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
      "sha256": base._sha256(run / "params/agent.yaml"),
    },
    "saved_environment_config": {
      "path": str((run / "params/env.yaml").resolve()),
      "sha256": base._sha256(run / "params/env.yaml"),
    },
    "checkpoint_integrity": checkpoint["integrity"],
    "evaluation_protocol": {
      "modes": list(_MODES),
      "eval_seed": _EVAL_SEED,
      "num_envs": _NUM_ENVS,
      "steps": _STEPS,
      "schema_version": 2,
      "include_per_env": True,
      "failure_diagnosis_schema_version": 1,
      "physical_reset_validation": True,
      "comparison_reference": {
        "study": "A10 reset-only canary",
        "checkpoint_step": _REFERENCE_GATE,
        "analysis_sha256": _REFERENCE_ANALYSIS_SHA256,
      },
    },
    "runs": [
      {
        "name": _RUN_LABEL,
        "task": _TASK,
        "checkpoint": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "policy_seed": _POLICY_SEED,
        "physical_reset_validation": True,
      }
    ],
  }


def _audit_matrix(cfg: GroundedSafetyEvalCfg, row: dict[str, Any]) -> dict[str, Any]:
  manifest_path = Path(row["path"])
  if not manifest_path.is_file() or base._sha256(manifest_path) != row["sha256"]:
    raise ValueError("A11 manifest missing or drifted")
  manifest = base._load(manifest_path)
  stable = {key: value for key, value in manifest.items() if key != "manifest_id"}
  expected_id = hashlib.sha256(
    json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()
  if manifest.get("manifest_id") != expected_id or expected_id != row["manifest_id"]:
    raise ValueError("A11 manifest identity drifted")
  gate = int(row["checkpoint_step"])
  matrix_dir = cfg.evaluation_root / f"gate_{gate}"
  complete = base._load(matrix_dir / "_COMPLETE.json")
  for name, value in {
    "evaluation_schema_version": 2,
    "manifest": str(manifest_path.resolve()),
    "result_count": len(_MODES),
    "modes": list(_MODES),
    "eval_seeds": [_EVAL_SEED],
    "num_envs": _NUM_ENVS,
    "steps": _STEPS,
  }.items():
    if complete.get(name) != value:
      raise ValueError(f"gate {gate} _COMPLETE {name} drifted")
  summary = base._load(matrix_dir / "summary.json")
  if summary.get("metadata", {}).get("manifest_id") != row["manifest_id"]:
    raise ValueError(f"gate {gate} summary lineage drifted")
  raws: dict[str, dict[str, Any]] = {}
  artifacts = []
  for mode in _MODES:
    matches = list(
      matrix_dir.glob(f"{_RUN_LABEL}__model_{gate}__{mode}__eval{_EVAL_SEED}.json")
    )
    if len(matches) != 1:
      raise ValueError(f"gate {gate} mode {mode} raw result count is {len(matches)}")
    result = base._audit_raw(matches[0], manifest, mode)
    raws[mode] = result
    artifacts.append(
      {
        "mode": mode,
        "path": str(matches[0].resolve()),
        "sha256": base._sha256(matches[0]),
      }
    )
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
      name: max(float(raws[mode][name]) for mode in _MODES) for name in base._SAFETY_MAX
    },
    "failure_reason_counts": {
      name: sum(
        int(raws[mode]["strict_failure_diagnosis"]["reason_counts"][name])
        for mode in _MODES
      )
      for name in base._FAILURE_NAMES
    },
    "raw_artifacts": artifacts,
  }


def _reference_gate(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  path = repo / _REFERENCE_ANALYSIS
  if not path.is_file() or base._sha256(path) != _REFERENCE_ANALYSIS_SHA256:
    raise ValueError("A10 gate-1000 reference analysis missing or drifted")
  analysis = base._load(path)
  if (
    analysis.get("status") != "RESET_ONLY_CANARY_COMPLETE_ENGINEERING_ONLY"
    or analysis.get("protocol_sha256")
    != "7416457635c9fbb9c30880b07aebfc865f2a0d19e2e0581737069ffec2e84dbf"
  ):
    raise ValueError("A10 reference lineage drifted")
  matches = [
    row
    for row in analysis.get("gates", [])
    if int(row.get("checkpoint_step", -1)) == _REFERENCE_GATE
  ]
  if len(matches) != 1:
    raise ValueError("A10 reference gate count drifted")
  gate = matches[0]
  for artifact in gate.get("raw_artifacts", []):
    raw = Path(artifact["path"])
    if not raw.is_file() or base._sha256(raw) != artifact["sha256"]:
      raise ValueError("A10 reference raw artifact missing or drifted")
  return gate, {
    "path": str(path.resolve()),
    "sha256": _REFERENCE_ANALYSIS_SHA256,
    "checkpoint_step": _REFERENCE_GATE,
    "manifest_id": gate["manifest_id"],
  }


def _write_analysis(
  cfg: GroundedSafetyEvalCfg,
  rows: list[dict[str, Any]],
  index: dict[str, Any],
) -> dict[str, Any]:
  repo = Path(__file__).resolve().parents[1]
  reference, reference_lineage = _reference_gate(repo)
  first = rows[0]
  payload = {
    "schema_version": 1,
    "status": "GROUNDED_SAFETY_EVAL_COMPLETE_ENGINEERING_ONLY",
    "study_id": _STUDY_ID,
    "study_role": _STUDY_ROLE,
    "automatic_action": "STOP_AND_REVIEW_NO_AUTOMATIC_PROMOTION",
    "promotion": None,
    "paper_evidence": False,
    "training_plan_id": _PLAN_ID,
    "protocol_sha256": _PROTOCOL_SHA256,
    "manifest_index_id": index["index_id"],
    "policy_seed_sampling_unit_count": 1,
    "comparison_reference": reference_lineage,
    "reference_gate": reference,
    "gates": rows,
    "changes_from_a11_gate_0": {
      str(row["checkpoint_step"]): {
        "gsi": row["gsi"] - first["gsi"],
        "fixed_macro": row["fixed_macro"] - first["fixed_macro"],
        "fixed_worst": row["fixed_worst"] - first["fixed_worst"],
      }
      for row in rows[1:]
    },
    "changes_from_a10_gate_1000": {
      str(row["checkpoint_step"]): {
        "gsi": row["gsi"] - reference["gsi"],
        "fixed_macro": row["fixed_macro"] - reference["fixed_macro"],
        "fixed_worst": row["fixed_worst"] - reference["fixed_worst"],
        "safety_pareto": {
          name: row["safety_pareto"][name] - reference["safety_pareto"][name]
          for name in base._SAFETY_MAX
        },
      }
      for row in rows
    },
    "interpretation_limit": (
      "A11 is one safety-finetuned warm-start lineage. Negative safety deltas "
      "favor A11, but no gate can establish reproducibility, promotion, RA-L "
      "evidence, or real-robot readiness."
    ),
  }
  base._immutable_json(cfg.analysis_json, payload)
  lines = [
    "# A11 grounded-safety fine-tune",
    "",
    "Status: GROUNDED_SAFETY_EVAL_COMPLETE_ENGINEERING_ONLY.",
    "",
    "A10 gate 1000 is the hash-locked comparison reference.",
    "",
    "| Gate | GSI | Fixed macro | Fixed worst | Slip P95 | Root drift P95 | Action d1 P95 | Action d2 P95 | Power max-mean | qvel P95 |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    safety = row["safety_pareto"]
    lines.append(
      f"| {row['checkpoint_step']} | {row['gsi']:.1%} | "
      f"{row['fixed_macro']:.1%} | {row['fixed_worst']:.1%} | "
      f"{safety['contact_foot_slip_p95_m_s']:.3f} | "
      f"{safety['post_success_root_drift_p95_m']:.3f} | "
      f"{safety['action_delta_rms_p95']:.3f} | "
      f"{safety['action_second_difference_rms_p95']:.3f} | "
      f"{safety['max_power_mean_w']:.1f} | "
      f"{safety['max_joint_speed_p95_rad_s']:.1f} |"
    )
  lines.extend(
    (
      "",
      "Automatic action: stop and review; there is no automatic promotion.",
      "",
    )
  )
  markdown = "\n".join(lines)
  if cfg.analysis_markdown.is_file() and cfg.analysis_markdown.read_text() != markdown:
    raise ValueError("immutable A11 Markdown analysis drifted")
  cfg.analysis_markdown.parent.mkdir(parents=True, exist_ok=True)
  if not cfg.analysis_markdown.exists():
    cfg.analysis_markdown.write_text(markdown)
  return payload


def _translate_status(result: dict[str, Any]) -> dict[str, Any]:
  status = str(result.get("status", ""))
  if status.startswith("RESET_ONLY_"):
    result = {
      **result,
      "status": status.replace("RESET_ONLY_", "GROUNDED_SAFETY_", 1),
    }
  if result.get("status") == "GROUNDED_SAFETY_EVAL_COMPLETE_ENGINEERING_ONLY":
    result["evaluation"] = {
      "completed_matrix_count": len(_GATES),
      "required_matrix_count": len(_GATES),
    }
  return result


def main(cfg: GroundedSafetyEvalCfg) -> None:
  repo = Path(__file__).resolve().parents[1]
  _configure_base()
  try:
    result = _translate_status(base.advance(cfg))
  except Exception as error:
    result = {
      **base._base(repo),
      "status": "GROUNDED_SAFETY_AUTOMATION_ALERT",
      "study_id": _STUDY_ID,
      "study_role": _STUDY_ROLE,
      "training_plan_id": _PLAN_ID,
      "protocol_sha256": _PROTOCOL_SHA256,
      "error": f"{type(error).__name__}: {error}",
      "automatic_restart_forbidden": True,
    }
  base._atomic_json(cfg.state, result)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(GroundedSafetyEvalCfg))
