"""Aggregate frozen SMP evaluations with policy seed as the sampling unit."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_FIXED_MODES = ("prone", "supine", "left_side", "right_side")
_EVALUATION_SCHEMA_VERSION = 2
_SAFETY_METRICS = (
  "max_joint_speed_p95_rad_s",
  "max_power_mean_w",
  "contact_foot_slip_p95_m_s",
  "post_success_root_drift_p95_m",
  "secondary_fall_rate_after_success",
  "foot_separation_at_success_p95_m",
  "action_delta_rms_p95",
  "action_second_difference_rms_p95",
)
_MIN_METRICS = ("finite_action_rate",)


@dataclass(frozen=True)
class AggregateCfg:
  summaries: tuple[Path, ...]
  output_json: Path
  output_markdown: Path | None = None
  bootstrap_replicates: int = 20000
  bootstrap_seed: int = 20260829
  minimum_policy_seeds: int = 3


def _quantile(sorted_values: list[float], q: float) -> float:
  if not sorted_values:
    raise ValueError("cannot take a quantile of an empty list")
  index = q * (len(sorted_values) - 1)
  lower = int(index)
  upper = min(lower + 1, len(sorted_values) - 1)
  fraction = index - lower
  return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean(
  values: list[float], replicates: int, rng: random.Random
) -> dict[str, float]:
  if not values:
    raise ValueError("cannot bootstrap an empty metric")
  means = []
  count = len(values)
  for _ in range(replicates):
    means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
  means.sort()
  return {
    "mean": sum(values) / count,
    "ci95_low": _quantile(means, 0.025),
    "ci95_high": _quantile(means, 0.975),
    "policy_seed_values": values,
  }


def _load_policy_summary(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
  payload = json.loads(path.read_text())
  evaluations = payload.get("evaluations")
  if not isinstance(evaluations, list) or not evaluations:
    raise ValueError(f"{path} has no evaluation rows")
  schema_versions = {row.get("evaluation_schema_version") for row in evaluations}
  if schema_versions != {_EVALUATION_SCHEMA_VERSION}:
    raise ValueError(f"{path} has incompatible evaluation schemas: {schema_versions}")
  policy_seeds = {row.get("policy_seed") for row in evaluations}
  if None in policy_seeds or len(policy_seeds) != 1:
    raise ValueError(f"{path} must contain exactly one non-null policy seed")
  policy_seed = int(policy_seeds.pop())
  by_arm_mode: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for row in evaluations:
    by_arm_mode.setdefault((row["arm"], row["reset_mode"]), []).append(row)

  arms: dict[str, dict[str, Any]] = {}
  for (arm, mode), rows in by_arm_mode.items():
    successes = sum(int(row["strict_successes"]) for row in rows)
    total = sum(int(row["num_envs"]) for row in rows)
    arm_metrics = arms.setdefault(arm, {"modes": {}, "safety": {}})
    arm_metrics["modes"][mode] = successes / total if total else 0.0
    for metric in _SAFETY_METRICS:
      arm_metrics["safety"].setdefault(metric, {})[mode] = max(
        float(row[metric]) for row in rows
      )
    for metric in _MIN_METRICS:
      arm_metrics["safety"].setdefault(metric, {})[mode] = min(
        float(row[metric]) for row in rows
      )
  for arm_metrics in arms.values():
    fixed = [arm_metrics["modes"].get(mode, 0.0) for mode in _FIXED_MODES]
    arm_metrics["fixed_macro"] = sum(fixed) / len(fixed)
    arm_metrics["fixed_worst"] = min(fixed)
    arm_metrics["gsi"] = arm_metrics["modes"].get("native_gsi", 0.0)
    arm_metrics["safety_worst"] = {
      metric: max(values.values()) for metric, values in arm_metrics["safety"].items()
    }
    for metric in _MIN_METRICS:
      arm_metrics["safety_worst"][metric] = min(arm_metrics["safety"][metric].values())
  return policy_seed, arms


def aggregate(payloads: list[tuple[int, dict[str, dict[str, Any]]]], cfg: AggregateCfg):
  if not payloads:
    raise ValueError("at least one policy summary is required")
  if cfg.bootstrap_replicates <= 0:
    raise ValueError("bootstrap_replicates must be positive")
  seeds = [seed for seed, _ in payloads]
  if len(set(seeds)) != len(seeds):
    raise ValueError(f"policy seeds must be unique, got {seeds}")
  shared_arms = set.intersection(*(set(arms) for _, arms in payloads))
  if not shared_arms:
    raise ValueError("summaries do not share an arm")
  rng = random.Random(cfg.bootstrap_seed)
  arms_output = {}
  for arm in sorted(shared_arms):
    metrics: dict[str, list[float]] = {
      "gsi": [],
      "fixed_macro": [],
      "fixed_worst": [],
    }
    metrics.update({metric: [] for metric in _SAFETY_METRICS})
    metrics.update({metric: [] for metric in _MIN_METRICS})
    mode_values = {mode: [] for mode in ("native_gsi", *_FIXED_MODES)}
    for _, arms in payloads:
      arm_data = arms[arm]
      metrics["gsi"].append(arm_data["gsi"])
      metrics["fixed_macro"].append(arm_data["fixed_macro"])
      metrics["fixed_worst"].append(arm_data["fixed_worst"])
      for metric in _SAFETY_METRICS:
        metrics[metric].append(arm_data["safety_worst"][metric])
      for metric in _MIN_METRICS:
        metrics[metric].append(arm_data["safety_worst"][metric])
      for mode in mode_values:
        mode_values[mode].append(arm_data["modes"].get(mode, 0.0))
    arms_output[arm] = {
      "metrics": {
        name: _bootstrap_mean(values, cfg.bootstrap_replicates, rng)
        for name, values in metrics.items()
      },
      "modes": {
        mode: _bootstrap_mean(values, cfg.bootstrap_replicates, rng)
        for mode, values in mode_values.items()
      },
    }
  enough = len(seeds) >= cfg.minimum_policy_seeds
  return {
    "status": ("MINIMUM_POLICY_SEEDS_MET" if enough else "INSUFFICIENT_POLICY_SEEDS"),
    "policy_seeds": sorted(seeds),
    "policy_seed_count": len(seeds),
    "minimum_policy_seeds": cfg.minimum_policy_seeds,
    "bootstrap_replicates": cfg.bootstrap_replicates,
    "bootstrap_seed": cfg.bootstrap_seed,
    "sampling_unit": "independently trained policy seed",
    "arms": arms_output,
    "limitations": [
      "With only three policy seeds, percentile bootstrap intervals are coarse.",
      "Parallel rollout environments are pooled within a policy seed, not treated as training replicates.",
      "Meeting the minimum seed count does not by itself establish terrain, plate, or real-robot validity.",
    ],
  }


def _metric_cell(metric: dict[str, Any], name: str, suffix: str = "") -> str:
  value = metric[name]
  return (
    f"{value['mean']:.3f} [{value['ci95_low']:.3f}, {value['ci95_high']:.3f}]{suffix}"
  )


def _markdown(result: dict[str, Any]) -> str:
  lines = [
    "# SMP policy-seed aggregate",
    "",
    f"Status: **{result['status']}**",
    "",
    f"Policy seeds: `{result['policy_seeds']}`",
    "",
    "| Arm | GSI mean [95% CI] | Fixed macro | Fixed worst | Foot slip p95 | Post-stand drift p95 |",
    "| --- | --- | --- | --- | --- | --- |",
  ]
  for arm, data in result["arms"].items():
    metric = data["metrics"]
    lines.append(
      f"| {arm} | {_metric_cell(metric, 'gsi')} | "
      f"{_metric_cell(metric, 'fixed_macro')} | "
      f"{_metric_cell(metric, 'fixed_worst')} | "
      f"{_metric_cell(metric, 'contact_foot_slip_p95_m_s', ' m/s')} | "
      f"{_metric_cell(metric, 'post_success_root_drift_p95_m', ' m')} |"
    )
  lines.extend(
    (
      "",
      "The bootstrap sampling unit is an independently trained policy seed. "
      "Rollout environments are not counted as independent training replicates.",
      "",
    )
  )
  return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_aggregate(cfg: AggregateCfg) -> dict[str, Any]:
  payloads = [_load_policy_summary(path) for path in cfg.summaries]
  result = aggregate(payloads, cfg)
  result["source_summaries"] = [
    {"path": str(path.resolve()), "sha256": _sha256(path)} for path in cfg.summaries
  ]
  output_markdown = cfg.output_markdown or cfg.output_json.with_suffix(".md")
  _atomic_write(cfg.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
  _atomic_write(output_markdown, _markdown(result))
  return result


def main(cfg: AggregateCfg) -> None:
  result = write_aggregate(cfg)
  print(f"{result['status']}: seeds={result['policy_seeds']}")


if __name__ == "__main__":
  main(tyro.cli(AggregateCfg))
