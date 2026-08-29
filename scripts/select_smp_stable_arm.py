"""Select checkpoint-stable SMP arms from complete frozen gate analyses."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_FIXED_MODES = ("prone", "supine", "left_side", "right_side")
_ACCEPTED_GATE_STATUSES = {"NO_PROMOTION", "SCREEN_PASS_NOT_FINAL"}


@dataclass(frozen=True)
class SelectionCfg:
  evidence_dir: Path
  output_json: Path | None = None
  output_markdown: Path | None = None
  gates: tuple[int, ...] = (8000, 15000, 25000, 29999)
  stable_gates: tuple[int, ...] = (15000, 25000, 29999)
  min_gsi_success: float = 0.95
  min_fixed_macro: float = 0.80
  min_fixed_worst: float = 0.60
  max_late_regression: float = 0.10
  max_candidates: int = 2


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _wilson(successes: int, total: int) -> tuple[float, float]:
  if total <= 0:
    return (0.0, 0.0)
  z = 1.959963984540054
  rate = successes / total
  denominator = 1.0 + z**2 / total
  center = (rate + z**2 / (2.0 * total)) / denominator
  radius = (
    z * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) / denominator
  )
  return (
    min(rate, max(0.0, center - radius)),
    max(rate, min(1.0, center + radius)),
  )


def _fixed_intervals(metrics: dict[str, Any]) -> dict[str, float]:
  modes = metrics["modes"]
  missing = [mode for mode in _FIXED_MODES if mode not in modes]
  if missing:
    raise ValueError(f"fixed reset modes missing from analysis: {missing}")
  successes = sum(int(modes[mode]["successes"]) for mode in _FIXED_MODES)
  total = sum(int(modes[mode]["total"]) for mode in _FIXED_MODES)
  macro_low, macro_high = _wilson(successes, total)
  worst_mode = min(_FIXED_MODES, key=lambda mode: float(modes[mode]["rate"]))
  return {
    "fixed_macro_ci95_low": macro_low,
    "fixed_macro_ci95_high": macro_high,
    "fixed_worst_ci95_low": float(modes[worst_mode]["ci95_low"]),
    "fixed_worst_ci95_high": float(modes[worst_mode]["ci95_high"]),
  }


def _load_analyses(cfg: SelectionCfg) -> tuple[dict[int, dict[str, Any]], list[dict]]:
  analyses: dict[int, dict[str, Any]] = {}
  sources = []
  for gate in cfg.gates:
    path = cfg.evidence_dir / f"gate_{gate}" / "analysis.json"
    if not path.is_file():
      raise FileNotFoundError(f"required frozen gate analysis is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") not in _ACCEPTED_GATE_STATUSES:
      raise ValueError(
        f"inadmissible analysis status at gate {gate}: {payload.get('status')}"
      )
    if payload.get("checkpoint") != f"model_{gate}.pt":
      raise ValueError(
        f"gate {gate} analysis has checkpoint {payload.get('checkpoint')!r}"
      )
    if not isinstance(payload.get("arms"), dict) or not payload["arms"]:
      raise ValueError(f"gate {gate} analysis has no arms")
    analyses[gate] = payload
    sources.append({"gate": gate, "path": str(path.resolve()), "sha256": _sha256(path)})
  return analyses, sources


def _intervals_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
  return max(first[0], second[0]) <= min(first[1], second[1])


def select(cfg: SelectionCfg) -> dict[str, Any]:
  if len(cfg.gates) < 2 or not cfg.stable_gates:
    raise ValueError("selection requires multiple gates and at least one stable gate")
  if not set(cfg.stable_gates).issubset(cfg.gates):
    raise ValueError("stable gates must be a subset of required gates")
  if cfg.max_candidates < 1:
    raise ValueError("max_candidates must be positive")
  analyses, sources = _load_analyses(cfg)
  arm_sets = {tuple(sorted(payload["arms"])) for payload in analyses.values()}
  if len(arm_sets) != 1:
    raise ValueError("frozen gates do not contain the same arm set")
  policy_seeds = {payload.get("policy_seed") for payload in analyses.values()}
  if len(policy_seeds) != 1 or None in policy_seeds:
    raise ValueError("frozen gates do not contain one consistent policy seed")

  arms: dict[str, Any] = {}
  final_gate = max(cfg.gates)
  previous_gate = sorted(gate for gate in cfg.gates if gate < final_gate)[-1]
  for arm in next(iter(arm_sets)):
    gate_metrics = {}
    for gate in cfg.gates:
      metrics = analyses[gate]["arms"][arm]
      gate_metrics[str(gate)] = {
        "gsi": float(metrics["gsi"]),
        "fixed_macro": float(metrics["fixed_macro"]),
        "fixed_worst": float(metrics["fixed_worst"]),
        "finite_action_rate_min": float(metrics["finite_action_rate_min"]),
        "screen_pass": bool(metrics["screen_pass"]),
      }
    final = analyses[final_gate]["arms"][arm]
    previous = analyses[previous_gate]["arms"][arm]
    intervals = _fixed_intervals(final)
    eligibility = {
      "passes_all_stable_rapid_gates": all(
        bool(analyses[gate]["arms"][arm]["screen_pass"]) for gate in cfg.stable_gates
      ),
      "final_gsi": float(final["gsi"]) >= cfg.min_gsi_success,
      "final_fixed_macro": float(final["fixed_macro"]) >= cfg.min_fixed_macro,
      "final_fixed_worst": float(final["fixed_worst"]) >= cfg.min_fixed_worst,
      "finite_actions": float(final["finite_action_rate_min"]) >= 1.0,
      "macro_late_regression": (
        float(previous["fixed_macro"]) - float(final["fixed_macro"])
        <= cfg.max_late_regression
      ),
      "worst_late_regression": (
        float(previous["fixed_worst"]) - float(final["fixed_worst"])
        <= cfg.max_late_regression
      ),
    }
    arms[arm] = {
      "gate_metrics": gate_metrics,
      "eligibility": eligibility,
      "eligible": all(eligibility.values()),
      "ranking_metrics": {
        **intervals,
        "secondary_fall_rate_after_success_max": float(
          final["secondary_fall_rate_after_success_max"]
        ),
        "post_success_root_drift_p95_m": float(final["post_success_root_drift_p95_m"]),
        "contact_foot_slip_p95_m_s": float(final["contact_foot_slip_p95_m_s"]),
        "max_power_mean_w": float(final["max_power_mean_w"]),
        "max_joint_speed_p95_rad_s": float(final["max_joint_speed_p95_rad_s"]),
      },
    }

  ranked = [arm for arm, payload in arms.items() if payload["eligible"]]
  ranked.sort(
    key=lambda arm: (
      -arms[arm]["ranking_metrics"]["fixed_worst_ci95_low"],
      -arms[arm]["ranking_metrics"]["fixed_macro_ci95_low"],
      arms[arm]["ranking_metrics"]["secondary_fall_rate_after_success_max"],
      arms[arm]["ranking_metrics"]["post_success_root_drift_p95_m"],
      arms[arm]["ranking_metrics"]["contact_foot_slip_p95_m_s"],
      arms[arm]["ranking_metrics"]["max_power_mean_w"],
      arms[arm]["ranking_metrics"]["max_joint_speed_p95_rad_s"],
      arm,
    )
  )
  promoted = ranked[: cfg.max_candidates]
  ranking_resolved = None
  if len(promoted) >= 2:
    first = arms[promoted[0]]["ranking_metrics"]
    second = arms[promoted[1]]["ranking_metrics"]
    worst_overlap = _intervals_overlap(
      (first["fixed_worst_ci95_low"], first["fixed_worst_ci95_high"]),
      (second["fixed_worst_ci95_low"], second["fixed_worst_ci95_high"]),
    )
    macro_overlap = _intervals_overlap(
      (first["fixed_macro_ci95_low"], first["fixed_macro_ci95_high"]),
      (second["fixed_macro_ci95_low"], second["fixed_macro_ci95_high"]),
    )
    ranking_resolved = not (worst_overlap and macro_overlap)

  status = "PROMOTE_FOR_POLICY_SEEDS" if promoted else "NO_PROMOTION"
  return {
    "schema_version": 1,
    "status": status,
    "policy_seed": policy_seeds.pop(),
    "required_gates": list(cfg.gates),
    "stable_gates": list(cfg.stable_gates),
    "final_gate": final_gate,
    "late_regression_reference_gate": previous_gate,
    "thresholds": {
      "min_gsi_success": cfg.min_gsi_success,
      "min_fixed_macro": cfg.min_fixed_macro,
      "min_fixed_worst": cfg.min_fixed_worst,
      "max_late_regression": cfg.max_late_regression,
      "max_candidates": cfg.max_candidates,
    },
    "sources": sources,
    "arms": arms,
    "eligible_ranked": ranked,
    "promoted_candidates": promoted,
    "top_candidate_ranking_resolved": ranking_resolved,
    "next_action": (
      "Train each promoted configuration with at least three independent policy seeds."
      if promoted
      else "Do not relax thresholds post hoc; preregister a new causal experiment."
    ),
    "limitations": [
      "This resource-allocation decision contains one policy-training seed.",
      "Rollout Wilson intervals do not estimate policy-training variance.",
      "A promoted candidate is not a RAL result until independent policy seeds pass.",
    ],
  }


def _markdown(result: dict[str, Any]) -> str:
  lines = [
    "# Checkpoint-stable SMP selection",
    "",
    f"Decision: **{result['status']}**",
    "",
    "| Arm | Eligible | Final GSI | Final macro | Final worst | Worst CI low | Macro CI low |",
    "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  final_gate = str(result["final_gate"])
  for arm, payload in result["arms"].items():
    final = payload["gate_metrics"][final_gate]
    rank = payload["ranking_metrics"]
    lines.append(
      f"| {arm} | {'yes' if payload['eligible'] else 'no'} | "
      f"{final['gsi']:.1%} | {final['fixed_macro']:.1%} | "
      f"{final['fixed_worst']:.1%} | "
      f"{rank['fixed_worst_ci95_low']:.1%} | "
      f"{rank['fixed_macro_ci95_low']:.1%} |"
    )
  lines.extend(("", "## Promoted for independent policy seeds", ""))
  if result["promoted_candidates"]:
    for index, arm in enumerate(result["promoted_candidates"], 1):
      lines.append(f"{index}. `{arm}`")
    if result["top_candidate_ranking_resolved"] is False:
      lines.append("")
      lines.append(
        "The leading rollout intervals overlap; these candidates are unresolved, not a claimed winner."
      )
  else:
    lines.append("None. Frozen thresholds yield `NO_PROMOTION`.")
  lines.extend(("", "## Next action", "", result["next_action"], ""))
  return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def write_selection(cfg: SelectionCfg) -> dict[str, Any]:
  result = select(cfg)
  output_json = cfg.output_json or cfg.evidence_dir / "stable_selection.json"
  output_markdown = cfg.output_markdown or cfg.evidence_dir / "stable_selection.md"
  _atomic_write(output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
  _atomic_write(output_markdown, _markdown(result))
  return result


def main(cfg: SelectionCfg) -> None:
  result = write_selection(cfg)
  print(f"{result['status']}: {result['promoted_candidates']}")


if __name__ == "__main__":
  main(tyro.cli(SelectionCfg))
