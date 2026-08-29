"""Apply frozen promotion gates and causal contrasts to an SMP eval matrix."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_FIXED_MODES = ("prone", "supine", "left_side", "right_side")
_EVALUATION_SCHEMA_VERSION = 2
_CONTRASTS = (
  ("prior_gsi", "a1_v7_gsi", "a0_f2s2_gsi"),
  ("prior_mix_strict", "a3_v7_mix_strict", "a2_f2s2_mix_strict"),
  (
    "prior_mix_reset_aware",
    "a5_v7_mix_reset_aware",
    "a4_f2s2_mix_reset_aware",
  ),
  ("prior_mix_bridge", "a7_v7_mix_bridge", "a6_f2s2_mix_bridge"),
  ("procedural_reset_f2s2", "a2_f2s2_mix_strict", "a0_f2s2_gsi"),
  ("procedural_reset_v7", "a3_v7_mix_strict", "a1_v7_gsi"),
  (
    "reset_aware_termination_f2s2",
    "a4_f2s2_mix_reset_aware",
    "a2_f2s2_mix_strict",
  ),
  (
    "reset_aware_termination_v7",
    "a5_v7_mix_reset_aware",
    "a3_v7_mix_strict",
  ),
  ("reward_bridge_f2s2", "a6_f2s2_mix_bridge", "a4_f2s2_mix_reset_aware"),
  ("reward_bridge_v7", "a7_v7_mix_bridge", "a5_v7_mix_reset_aware"),
)


@dataclass(frozen=True)
class AnalysisCfg:
  summary: Path
  output_json: Path | None = None
  output_markdown: Path | None = None
  min_gsi_success: float = 0.95
  min_fixed_macro: float = 0.40
  min_fixed_worst: float = 0.20
  min_finite_action_rate: float = 1.0


def _wilson(successes: int, total: int) -> tuple[float, float]:
  if total <= 0:
    return (0.0, 0.0)
  z = 1.959963984540054
  rate = successes / total
  denominator = 1.0 + z**2 / total
  center = (rate + z**2 / (2.0 * total)) / denominator
  radius = (
    z
    * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2))
    / denominator
  )
  return (
    min(rate, max(0.0, center - radius)),
    max(rate, min(1.0, center + radius)),
  )


def _pool(rows: list[dict[str, Any]]) -> dict[str, float | int]:
  successes = sum(int(row["strict_successes"]) for row in rows)
  total = sum(int(row["num_envs"]) for row in rows)
  rate = successes / total if total else 0.0
  low, high = _wilson(successes, total)
  return {
    "successes": successes,
    "total": total,
    "rate": rate,
    "ci95_low": low,
    "ci95_high": high,
  }


def _arm_metrics(rows: list[dict[str, Any]], cfg: AnalysisCfg) -> dict[str, Any]:
  by_mode: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    by_mode.setdefault(row["reset_mode"], []).append(row)
  modes = {mode: _pool(group) for mode, group in by_mode.items()}
  fixed_rates = [
    float(modes[mode]["rate"]) for mode in _FIXED_MODES if mode in modes
  ]
  fixed_complete = len(fixed_rates) == len(_FIXED_MODES)
  fixed_macro = sum(fixed_rates) / len(fixed_rates) if fixed_rates else 0.0
  fixed_worst = min(fixed_rates) if fixed_complete else 0.0
  finite_min = min(float(row["finite_action_rate"]) for row in rows)
  gsi = float(modes.get("native_gsi", {}).get("rate", 0.0))
  gates = {
    "gsi": gsi >= cfg.min_gsi_success,
    "fixed_modes_complete": fixed_complete,
    "fixed_macro": fixed_macro >= cfg.min_fixed_macro,
    "fixed_worst": fixed_worst >= cfg.min_fixed_worst,
    "finite_actions": finite_min >= cfg.min_finite_action_rate,
  }
  return {
    "modes": modes,
    "gsi": gsi,
    "fixed_macro": fixed_macro,
    "fixed_worst": fixed_worst,
    "finite_action_rate_min": finite_min,
    "max_joint_speed_p95_rad_s": max(
      float(row["max_joint_speed_p95_rad_s"]) for row in rows
    ),
    "max_power_mean_w": max(float(row["max_power_mean_w"]) for row in rows),
    "max_torque_mean_nm": max(float(row["max_torque_mean_nm"]) for row in rows),
    "contact_foot_slip_p95_m_s": max(
      float(row["contact_foot_slip_p95_m_s"]) for row in rows
    ),
    "post_success_root_drift_p95_m": max(
      float(row["post_success_root_drift_p95_m"]) for row in rows
    ),
    "secondary_fall_rate_after_success_max": max(
      float(row["secondary_fall_rate_after_success"]) for row in rows
    ),
    "foot_separation_at_success_p95_m": max(
      float(row["foot_separation_at_success_p95_m"]) for row in rows
    ),
    "action_delta_rms_p95": max(float(row["action_delta_rms_p95"]) for row in rows),
    "action_second_difference_rms_p95": max(
      float(row["action_second_difference_rms_p95"]) for row in rows
    ),
    "gates": gates,
    "screen_pass": all(gates.values()),
  }


def _contrasts(arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  output: list[dict[str, Any]] = []
  for name, numerator, denominator in _CONTRASTS:
    if numerator not in arms or denominator not in arms:
      continue
    modes = {}
    for mode in ("native_gsi", *_FIXED_MODES):
      num = arms[numerator]["modes"].get(mode)
      den = arms[denominator]["modes"].get(mode)
      if num is None or den is None:
        continue
      delta = float(num["rate"]) - float(den["rate"])
      low = float(num["ci95_low"]) - float(den["ci95_high"])
      high = float(num["ci95_high"]) - float(den["ci95_low"])
      modes[mode] = {
        "delta": delta,
        "conservative_ci95_low": low,
        "conservative_ci95_high": high,
        "rollout_interval_excludes_zero": low > 0.0 or high < 0.0,
      }
    output.append(
      {
        "name": name,
        "numerator": numerator,
        "denominator": denominator,
        "modes": modes,
      }
    )
  return output


def analyze(payload: dict[str, Any], cfg: AnalysisCfg) -> dict[str, Any]:
  evaluations = payload.get("evaluations")
  if not isinstance(evaluations, list) or not evaluations:
    raise ValueError("summary does not contain evaluation rows")
  schema_versions = {row.get("evaluation_schema_version") for row in evaluations}
  if schema_versions != {_EVALUATION_SCHEMA_VERSION}:
    raise ValueError(
      "frozen matrix has incompatible evaluation schema versions: "
      f"{sorted(str(value) for value in schema_versions)}"
    )
  checkpoints = {row["checkpoint"] for row in evaluations}
  if len(checkpoints) != 1:
    raise ValueError(f"expected one frozen checkpoint, got: {sorted(checkpoints)}")
  policy_seeds = {row.get("policy_seed") for row in evaluations}
  if len(policy_seeds) != 1:
    raise ValueError("one matrix must contain exactly one policy-training seed")

  rows_by_arm: dict[str, list[dict[str, Any]]] = {}
  for row in evaluations:
    rows_by_arm.setdefault(row["arm"], []).append(row)
  arms = {
    arm: _arm_metrics(rows, cfg) for arm, rows in sorted(rows_by_arm.items())
  }
  candidates = [arm for arm, metrics in arms.items() if metrics["screen_pass"]]
  candidates.sort(
    key=lambda arm: (
      -arms[arm]["fixed_worst"],
      -arms[arm]["fixed_macro"],
      arms[arm]["secondary_fall_rate_after_success_max"],
      arms[arm]["contact_foot_slip_p95_m_s"],
      arms[arm]["post_success_root_drift_p95_m"],
      arms[arm]["max_power_mean_w"],
      arms[arm]["max_joint_speed_p95_rad_s"],
    )
  )
  status = "SCREEN_PASS_NOT_FINAL" if candidates else "NO_PROMOTION"
  next_action = (
    "Retain passing arms for later frozen gates; do not promote to terrain or "
    "plate until checkpoint stability and three policy seeds are complete."
    if candidates
    else "Continue to the next frozen gate without changing multiple factors; "
    "no arm satisfies the breadth screen yet."
  )
  return {
    "checkpoint": checkpoints.pop(),
    "policy_seed": policy_seeds.pop(),
    "status": status,
    "screen_candidates_ranked": candidates,
    "thresholds": {
      "min_gsi_success": cfg.min_gsi_success,
      "min_fixed_macro": cfg.min_fixed_macro,
      "min_fixed_worst": cfg.min_fixed_worst,
      "min_finite_action_rate": cfg.min_finite_action_rate,
    },
    "arms": arms,
    "causal_contrasts": _contrasts(arms),
    "next_action": next_action,
    "limitations": [
      "This screen contains one policy-training seed and cannot estimate training variance.",
      "Rollout confidence intervals do not justify a causal training claim by themselves.",
      "Safety metrics are reported for Pareto ranking but are not optimized in this screen.",
    ],
  }


def _markdown(analysis: dict[str, Any]) -> str:
  lines = [
    f"# Frozen analysis: {analysis['checkpoint']}",
    "",
    f"Decision: **{analysis['status']}**",
    "",
    "| Arm | GSI | Fixed macro | Worst | Pass | Foot slip p95 | Post-stand drift p95 | Secondary fall | Peak power |",
    "| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
  ]
  for arm, metrics in analysis["arms"].items():
    lines.append(
      f"| {arm} | {metrics['gsi']:.1%} | {metrics['fixed_macro']:.1%} | "
      f"{metrics['fixed_worst']:.1%} | "
      f"{'yes' if metrics['screen_pass'] else 'no'} | "
      f"{metrics['contact_foot_slip_p95_m_s']:.2f} m/s | "
      f"{metrics['post_success_root_drift_p95_m']:.2f} m | "
      f"{metrics['secondary_fall_rate_after_success_max']:.1%} | "
      f"{metrics['max_power_mean_w']:.1f} W |"
    )
  lines.extend(("", "## Ranked screen candidates", ""))
  if analysis["screen_candidates_ranked"]:
    for index, arm in enumerate(analysis["screen_candidates_ranked"], 1):
      lines.append(f"{index}. `{arm}`")
  else:
    lines.append("None. This checkpoint is `NO_PROMOTION`.")
  lines.extend(("", "## Next action", "", analysis["next_action"], ""))
  lines.append(
    "Rollout intervals are not policy-seed uncertainty; three independently "
    "trained policy seeds remain mandatory before a RAL claim."
  )
  return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def main(cfg: AnalysisCfg) -> None:
  payload = json.loads(cfg.summary.read_text())
  analysis = analyze(payload, cfg)
  output_json = cfg.output_json or cfg.summary.with_name("analysis.json")
  output_markdown = cfg.output_markdown or cfg.summary.with_name("analysis.md")
  _atomic_write(output_json, json.dumps(analysis, indent=2, sort_keys=True) + "\n")
  _atomic_write(output_markdown, _markdown(analysis))
  print(f"{analysis['status']}: {analysis['screen_candidates_ranked']}")


if __name__ == "__main__":
  main(tyro.cli(AnalysisCfg))
