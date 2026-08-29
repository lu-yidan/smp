"""Apply preregistered per-seed gates to one frozen T or P matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from run_smp_specialist_eval_matrix import _stratum_id, plate_strata, terrain_strata

_SAFETY_METRICS = (
  "max_joint_speed_p95_rad_s",
  "max_power_mean_w",
  "contact_foot_slip_p95_m_s",
  "action_delta_rms_p95",
  "action_second_difference_rms_p95",
)
_POSES = ("prone", "supine", "left_side", "right_side")


@dataclass(frozen=True)
class SpecialistAnalysisCfg:
  summary: Path
  protocol: Path = Path("docs/ral_terrain_plate_protocol.json")
  output_json: Path | None = None
  output_markdown: Path | None = None


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(path)
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def _mean(rows: list[dict[str, Any]], key: str = "strict_success_rate") -> float:
  if not rows:
    raise ValueError(f"cannot average an empty specialist stratum for {key}")
  return sum(float(row[key]) for row in rows) / len(rows)


def _worst_pose(rows: list[dict[str, Any]]) -> float:
  return min(
    _mean([row for row in rows if row["reset_mode"] == pose]) for pose in _POSES
  )


def _terrain_macro(
  rows: list[dict[str, Any]], level: int
) -> tuple[float, float, dict[str, float]]:
  by_terrain = {}
  level_rows = [
    row
    for row in rows
    if row["stratum"]["terrain_level"] == level
    and row["stratum"]["terrain_type"] != "flat"
  ]
  for terrain in ("slope", "stairs", "rough"):
    selected = [row for row in level_rows if row["stratum"]["terrain_type"] == terrain]
    by_terrain[terrain] = _mean(selected)
  pose_means = []
  for pose in _POSES:
    per_terrain = []
    for terrain in ("slope", "stairs", "rough"):
      selected = [
        row
        for row in level_rows
        if row["stratum"]["terrain_type"] == terrain and row["reset_mode"] == pose
      ]
      per_terrain.append(_mean(selected))
    pose_means.append(sum(per_terrain) / len(per_terrain))
  return (
    sum(by_terrain.values()) / len(by_terrain),
    min(pose_means),
    by_terrain,
  )


def _flat_safety(manifest: dict[str, Any], arm: str) -> dict[str, float]:
  path = Path(manifest["flat_summary"])
  if not path.is_file() or _sha256(path) != manifest.get("flat_summary_sha256"):
    raise ValueError("matched flat summary changed")
  payload = _load(path)
  rows = [row for row in payload.get("evaluations", []) if row.get("arm") == arm]
  if not rows:
    raise ValueError(f"flat summary does not contain selected arm {arm}")
  return {metric: max(float(row[metric]) for row in rows) for metric in _SAFETY_METRICS}


def _safety_ratios(
  rows: list[dict[str, Any]], baseline: dict[str, float]
) -> dict[str, float]:
  ratios = {}
  for metric in _SAFETY_METRICS:
    specialist = max(float(row[metric]) for row in rows)
    ratios[metric] = specialist / max(float(baseline[metric]), 1.0e-6)
  return ratios


def _terrain_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
  flat = [row for row in rows if row["stratum"]["terrain_type"] == "flat"]
  level0_macro, level0_worst, level0_terrains = _terrain_macro(rows, 0)
  level1_macro, level1_worst, level1_terrains = _terrain_macro(rows, 1)
  stair_edge = {}
  for level in (0, 1):
    selected = [
      row
      for row in rows
      if row["stratum"]["terrain_type"] == "stairs"
      and row["stratum"]["terrain_level"] == level
      and row["stratum"]["stair_edge_cohort"] != "center"
    ]
    stair_edge[level] = _mean(selected)
  return {
    "flat_macro": _mean(flat),
    "flat_worst_pose": _worst_pose(flat),
    "level0_nonflat_macro": level0_macro,
    "level0_nonflat_worst_pose": level0_worst,
    "level0_each_terrain_macro": level0_terrains,
    "level1_nonflat_macro": level1_macro,
    "level1_nonflat_worst_pose": level1_worst,
    "level1_each_terrain_macro": level1_terrains,
    "level0_stair_edge_macro": stair_edge[0],
    "level1_stair_edge_macro": stair_edge[1],
    "terrain_exit_rate_max": max(float(row["terrain_exit_rate"]) for row in rows),
    "invalid_dynamics_rate_max": max(
      float(row["invalid_dynamics_rate"]) for row in rows
    ),
    "secondary_fall_rate_max": max(
      max(0.0, float(row["secondary_fall_rate_after_success"])) for row in rows
    ),
  }


def _plate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
  unpinned = [row for row in rows if row["stratum"]["plate_mode"] == "unpinned"]
  pinned = [row for row in rows if row["stratum"]["plate_mode"] == "pinned"]
  light = [row for row in pinned if row["stratum"]["plate_mass_kg"] in (4.0, 8.0)]
  heavy = [row for row in pinned if row["stratum"]["plate_mass_kg"] == 12.0]
  pose_means = {
    pose: _mean([row for row in pinned if row["reset_mode"] == pose])
    for pose in ("prone", "supine")
  }
  return {
    "unpinned_flat_macro": _mean(unpinned),
    "unpinned_flat_worst_pose": _worst_pose(unpinned),
    "plate_4_8kg_escape_and_stand_macro": _mean(light),
    "plate_12kg_escape_and_stand_macro": _mean(heavy),
    "plate_worst_pose": min(pose_means.values()),
    "plate_pose_macro": pose_means,
    "invalid_setup_rate_max": max(
      float(row["invalid_escape_setup_rate"]) for row in pinned
    ),
    "invalid_dynamics_rate_max": max(
      float(row["invalid_dynamics_rate"]) for row in rows
    ),
    "secondary_fall_rate_max": max(
      max(0.0, float(row["secondary_fall_rate_after_success"])) for row in rows
    ),
    "hand_support_rate_macro": _mean(pinned, "hand_support_rate"),
  }


def analyze(cfg: SpecialistAnalysisCfg) -> dict[str, Any]:
  summary = _load(cfg.summary)
  protocol = _load(cfg.protocol)
  phase = summary.get("phase")
  rows = summary.get("evaluations")
  expected_count = 76 if phase == "T" else 10 if phase == "P" else None
  if not isinstance(rows, list) or len(rows) != expected_count:
    raise ValueError(f"{phase} matrix must contain exactly {expected_count} strata")
  expected_strata = terrain_strata() if phase == "T" else plate_strata()
  expected_ids = {_stratum_id(stratum) for stratum in expected_strata}
  observed_ids = [_stratum_id(row["stratum"]) for row in rows]
  if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != expected_ids:
    raise ValueError("specialist summary strata are duplicated or incomplete")
  if {row.get("evaluation_schema_version") for row in rows} != {2}:
    raise ValueError("specialist summary uses an incompatible evaluator schema")
  if {row.get("actor_observation_dim") for row in rows} != {93}:
    raise ValueError("specialist actor observation contract is not 93D")
  if {row.get("policy_seed") for row in rows} != {summary.get("policy_seed")}:
    raise ValueError("specialist summary mixes policy-training seeds")
  manifest_path = Path(summary["manifest"])
  if not manifest_path.is_file() or _sha256(manifest_path) != summary.get(
    "manifest_sha256"
  ):
    raise ValueError("specialist matrix manifest changed")
  manifest = _load(manifest_path)
  run = manifest["runs"][0]
  if (
    phase != manifest.get("phase")
    or summary.get("policy_seed") != manifest.get("policy_seed")
    or summary.get("checkpoint_step") != manifest.get("checkpoint_step")
  ):
    raise ValueError("specialist matrix provenance is inconsistent")
  baseline = _flat_safety(manifest, run["arm"])
  safety_ratios = _safety_ratios(rows, baseline)
  safety_ratio_max = max(safety_ratios.values())
  gates = protocol["phases"][phase]["promotion_gates"]
  metrics = _terrain_metrics(rows) if phase == "T" else _plate_metrics(rows)
  checks = {}
  if phase == "T":
    checks = {
      "flat_macro": metrics["flat_macro"] >= gates["flat_macro_success_min"],
      "flat_worst_pose": metrics["flat_worst_pose"]
      >= gates["flat_worst_pose_success_min"],
      "level0_nonflat_macro": metrics["level0_nonflat_macro"]
      >= gates["level_0_nonflat_macro_success_min"],
      "level0_nonflat_worst_pose": metrics["level0_nonflat_worst_pose"]
      >= gates["level_0_nonflat_worst_pose_success_min"],
      "level1_nonflat_macro": metrics["level1_nonflat_macro"]
      >= gates["level_1_nonflat_macro_success_min"],
      "level1_each_terrain_macro": min(metrics["level1_each_terrain_macro"].values())
      >= gates["level_1_each_terrain_macro_success_min"],
      "level0_stair_edge_macro": metrics["level0_stair_edge_macro"]
      >= gates["level_0_stair_edge_macro_success_min"],
      "level1_stair_edge_macro": metrics["level1_stair_edge_macro"]
      >= gates["level_1_stair_edge_macro_success_min"],
      "terrain_exit": metrics["terrain_exit_rate_max"]
      <= gates["terrain_exit_rate_max"],
    }
  else:
    checks = {
      "unpinned_flat_macro": metrics["unpinned_flat_macro"]
      >= gates["unpinned_flat_macro_success_min"],
      "unpinned_flat_worst_pose": metrics["unpinned_flat_worst_pose"]
      >= gates["unpinned_flat_worst_pose_success_min"],
      "plate_4_8kg": metrics["plate_4_8kg_escape_and_stand_macro"]
      >= gates["plate_4_8kg_escape_and_stand_macro_min"],
      "plate_12kg": metrics["plate_12kg_escape_and_stand_macro"]
      >= gates["plate_12kg_escape_and_stand_macro_min"],
      "plate_worst_pose": metrics["plate_worst_pose"]
      >= gates["plate_worst_pose_success_min"],
      "invalid_setup": metrics["invalid_setup_rate_max"]
      <= gates["invalid_setup_rate_max"],
    }
  checks.update(
    {
      "invalid_dynamics": metrics["invalid_dynamics_rate_max"]
      <= gates["invalid_dynamics_rate"],
      "secondary_fall": metrics["secondary_fall_rate_max"]
      <= gates["secondary_fall_rate_max"],
      "safety_relative_to_flat": safety_ratio_max
      <= gates["safety_p95_relative_to_flat_seed_max"],
      "finite_actions": min(float(row["finite_action_rate"]) for row in rows) == 1.0,
    }
  )
  passed = all(checks.values())
  return {
    "schema_version": 1,
    "status": "PASS" if passed else "NO_PROMOTION",
    "phase": phase,
    "policy_seed": summary["policy_seed"],
    "checkpoint_step": summary["checkpoint_step"],
    "arm": run["arm"],
    "summary": str(cfg.summary.resolve()),
    "summary_sha256": _sha256(cfg.summary),
    "manifest": str(manifest_path.resolve()),
    "manifest_sha256": _sha256(manifest_path),
    "protocol": str(cfg.protocol.resolve()),
    "protocol_sha256": _sha256(cfg.protocol),
    "metrics": metrics,
    "safety_baseline": baseline,
    "safety_ratios": safety_ratios,
    "safety_ratio_max": safety_ratio_max,
    "gates": checks,
    "failures": [name for name, value in checks.items() if not value],
    "claim_boundary": (
      "A per-seed checkpoint pass is not a phase promotion. Three matched policy "
      "seeds and the frozen final-checkpoint aggregate remain mandatory."
    ),
  }


def _markdown(result: dict[str, Any]) -> str:
  lines = [
    f"# Specialist {result['phase']} seed {result['policy_seed']} gate {result['checkpoint_step']}",
    "",
    f"Status: **{result['status']}**",
    "",
    f"Safety ratio max versus matched flat seed: {result['safety_ratio_max']:.3f}",
    "",
    "## Frozen gates",
    "",
  ]
  lines.extend(
    f"- {'PASS' if passed else 'FAIL'}: `{name}`"
    for name, passed in result["gates"].items()
  )
  lines.extend(("", result["claim_boundary"], ""))
  return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def write_analysis(cfg: SpecialistAnalysisCfg) -> dict[str, Any]:
  result = analyze(cfg)
  output_json = cfg.output_json or cfg.summary.with_name("analysis.json")
  output_markdown = cfg.output_markdown or cfg.summary.with_name("analysis.md")
  _atomic_write(output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
  _atomic_write(output_markdown, _markdown(result))
  return result


def main(cfg: SpecialistAnalysisCfg) -> None:
  result = write_analysis(cfg)
  print(f"{result['status']}: {result['failures']}")


if __name__ == "__main__":
  main(tyro.cli(SpecialistAnalysisCfg))
