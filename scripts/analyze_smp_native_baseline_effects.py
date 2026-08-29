"""Analyze matched native Tier-A baselines with policy-seed paired effects."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from audit_smp_baseline_registry import audit as audit_registry

_METHODS = (
  "task_only_ppo",
  "original_product_smp",
  "proposed_smp_recovery",
)
_MODES = ("native_gsi", "prone", "supine", "left_side", "right_side")
_FIXED_MODES = _MODES[1:]
_GATES = (8000, 15000, 25000, 29999)
_SEEDS = (20260901, 20260902, 20260903)
_CONTRASTS = (
  (
    "proposed_smp_recovery_minus_original_product_smp",
    "proposed_smp_recovery",
    "original_product_smp",
  ),
  (
    "proposed_smp_recovery_minus_task_only_ppo",
    "proposed_smp_recovery",
    "task_only_ppo",
  ),
  (
    "original_product_smp_minus_task_only_ppo",
    "original_product_smp",
    "task_only_ppo",
  ),
)
_SUCCESS_METRICS = ("native_gsi", "fixed_macro", "fixed_worst")
_LOWER_SAFETY_METRICS = (
  "max_joint_speed_p95_rad_s",
  "max_power_mean_w",
  "contact_foot_slip_p95_m_s",
  "post_success_root_drift_p95_m",
  "secondary_fall_rate_after_success",
  "foot_separation_at_success_p95_m",
  "action_delta_rms_p95",
  "action_second_difference_rms_p95",
)


@dataclass(frozen=True)
class NativeEffectCfg:
  evidence_dir: Path
  registry: Path = Path("docs/ral_baseline_registry.json")
  output_json: Path = Path("run_control/ral_baselines/native_eval/paired_effects.json")
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


def _quantile(values: list[float], q: float) -> float:
  ordered = sorted(values)
  position = q * (len(ordered) - 1)
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  fraction = position - lower
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_interval(values: list[float], replicates: int, seed: int) -> dict[str, Any]:
  if len(values) != 3 or not all(isinstance(value, float) for value in values):
    raise ValueError("paired effects require exactly three finite policy-seed values")
  if not all(float("-inf") < value < float("inf") for value in values):
    raise ValueError("paired effects contain non-finite values")
  rng = random.Random(seed)
  means = [
    sum(values[rng.randrange(len(values))] for _ in values) / len(values)
    for _ in range(replicates)
  ]
  return {
    "mean": sum(values) / len(values),
    "ci95_low": _quantile(means, 0.025),
    "ci95_high": _quantile(means, 0.975),
    "policy_seed_deltas": values,
    "positive_seed_count": sum(value > 0.0 for value in values),
    "negative_seed_count": sum(value < 0.0 for value in values),
  }


def _load_summary(
  path: Path, gate: int, expected_seed: int
) -> tuple[dict[str, dict[str, float]], str]:
  payload = _load(path)
  rows = payload.get("evaluations")
  if not isinstance(rows, list) or len(rows) != len(_METHODS) * len(_MODES):
    raise ValueError(f"{path} must contain the exact three-method five-mode matrix")
  keys = {(row.get("arm"), row.get("reset_mode")) for row in rows}
  if keys != {(method, mode) for method in _METHODS for mode in _MODES}:
    raise ValueError(f"{path} method/mode factorial drifted")
  manifest_hashes = {row.get("matched_eval_manifest_sha256") for row in rows}
  if len(manifest_hashes) != 1:
    raise ValueError(f"{path} mixes held-out evaluation manifests")
  held_out_hash = manifest_hashes.pop()
  if not isinstance(held_out_hash, str) or len(held_out_hash) != 64:
    raise ValueError(f"{path} lacks held-out evaluation provenance")

  by_method: dict[str, dict[str, float]] = {}
  for method in _METHODS:
    method_rows = {row["reset_mode"]: row for row in rows if row["arm"] == method}
    for mode, row in method_rows.items():
      if (
        row.get("evaluation_schema_version") != 2
        or row.get("policy_seed") != expected_seed
        or row.get("seed") != 20260829
        or row.get("num_envs") != 512
        or row.get("steps") != 500
        or Path(row.get("checkpoint", "")).name != f"model_{gate}.pt"
      ):
        raise ValueError(f"{path} row protocol drifted for {method}/{mode}")
      successes = int(row["strict_successes"])
      rate = float(row["strict_success_rate"])
      if successes < 0 or successes > 512 or abs(rate - successes / 512) > 1.0e-12:
        raise ValueError(f"{path} success count/rate disagrees for {method}/{mode}")
    modes = {mode: float(method_rows[mode]["strict_success_rate"]) for mode in _MODES}
    fixed = [modes[mode] for mode in _FIXED_MODES]
    metrics = {
      **{f"mode_{mode}": value for mode, value in modes.items()},
      "native_gsi": modes["native_gsi"],
      "fixed_macro": sum(fixed) / len(fixed),
      "fixed_worst": min(fixed),
      "finite_action_rate": min(
        float(method_rows[mode]["finite_action_rate"]) for mode in _MODES
      ),
    }
    metrics.update(
      {
        metric: max(float(method_rows[mode][metric]) for mode in _MODES)
        for metric in _LOWER_SAFETY_METRICS
      }
    )
    by_method[method] = metrics
  return by_method, held_out_hash


def analyze(cfg: NativeEffectCfg) -> dict[str, Any]:
  registry = _load(cfg.registry)
  audit_registry(registry, cfg.registry)
  contract = registry.get("paired_analysis", {})
  replicates = int(contract.get("bootstrap_replicates", 0))
  bootstrap_seed = int(contract.get("bootstrap_seed", -1))
  if (
    tuple(contract.get("stable_gates", ())) != (15000, 25000, 29999)
    or contract.get("final_gate") != 29999
    or contract.get("late_regression_reference_gate") != 25000
    or replicates != 20000
    or bootstrap_seed != 20260829
  ):
    raise ValueError("paired analysis request differs from the frozen registry")

  sources = []
  data: dict[int, dict[int, dict[str, dict[str, float]]]] = {}
  held_out_hashes = set()
  for gate in _GATES:
    data[gate] = {}
    for seed in _SEEDS:
      path = cfg.evidence_dir / f"gate_{gate}" / f"seed_{seed}" / "summary.json"
      methods, held_out_hash = _load_summary(path, gate, seed)
      data[gate][seed] = methods
      held_out_hashes.add(held_out_hash)
      sources.append(
        {
          "checkpoint_step": gate,
          "policy_seed": seed,
          "path": str(path.resolve()),
          "sha256": _sha256(path),
        }
      )
  if len(held_out_hashes) != 1:
    raise ValueError("native matrices do not share one held-out manifest")

  effect_metrics = (
    *(f"mode_{mode}" for mode in _MODES),
    *_SUCCESS_METRICS,
    *_LOWER_SAFETY_METRICS,
  )
  contrasts: dict[str, dict[str, Any]] = {}
  for contrast_index, (name, numerator, denominator) in enumerate(_CONTRASTS):
    gate_results = {}
    for gate_index, gate in enumerate(_GATES):
      metrics = {}
      for metric_index, metric in enumerate(effect_metrics):
        deltas = [
          data[gate][seed][numerator][metric] - data[gate][seed][denominator][metric]
          for seed in _SEEDS
        ]
        metrics[metric] = _paired_interval(
          deltas,
          replicates,
          bootstrap_seed + 10000 * contrast_index + 100 * gate_index + metric_index,
        )
        metrics[metric]["direction"] = (
          "higher_is_better"
          if metric not in _LOWER_SAFETY_METRICS
          else "lower_is_better"
        )
      gate_results[str(gate)] = metrics
    contrasts[name] = {
      "numerator": numerator,
      "denominator": denominator,
      "gates": gate_results,
    }

  thresholds = contract["success_thresholds"]
  stable_gates = tuple(int(gate) for gate in contract["stable_gates"])
  reference = int(contract["late_regression_reference_gate"])
  final = int(contract["final_gate"])
  max_regression = float(contract["max_late_regression"])
  stability = {}
  for method in _METHODS:
    seeds = {}
    for seed in _SEEDS:
      stable_thresholds = all(
        data[gate][seed][method]["native_gsi"] >= thresholds["native_gsi"]
        and data[gate][seed][method]["fixed_macro"] >= thresholds["fixed_macro"]
        and data[gate][seed][method]["fixed_worst"] >= thresholds["fixed_worst"]
        and data[gate][seed][method]["finite_action_rate"]
        >= thresholds["finite_action_rate"]
        for gate in stable_gates
      )
      macro_regression = (
        data[reference][seed][method]["fixed_macro"]
        - data[final][seed][method]["fixed_macro"]
      )
      worst_regression = (
        data[reference][seed][method]["fixed_worst"]
        - data[final][seed][method]["fixed_worst"]
      )
      seeds[str(seed)] = {
        "stable_gate_thresholds": stable_thresholds,
        "macro_late_regression": macro_regression,
        "worst_late_regression": worst_regression,
        "passes": (
          stable_thresholds
          and macro_regression <= max_regression
          and worst_regression <= max_regression
        ),
      }
    stability[method] = {
      "policy_seeds": seeds,
      "all_policy_seeds_pass": all(row["passes"] for row in seeds.values()),
    }

  margin = float(contract["noninferiority_margin"])
  primary = contrasts[contract["primary_contrast"]]["gates"][str(final)]
  task = contrasts["proposed_smp_recovery_minus_task_only_ppo"]["gates"][str(final)]
  checks = {
    "primary_fixed_worst_superiority": primary["fixed_worst"]["ci95_low"] > 0.0,
    "primary_fixed_macro_noninferiority": primary["fixed_macro"]["ci95_low"] >= -margin,
    "task_only_fixed_worst_noninferiority": task["fixed_worst"]["ci95_low"] >= -margin,
    "proposed_checkpoint_stability": stability["proposed_smp_recovery"][
      "all_policy_seeds_pass"
    ],
  }
  supported = all(checks.values())
  return {
    "schema_version": 1,
    "status": (
      "PROPOSED_PAIRED_ADVANTAGE_SUPPORTED"
      if supported
      else "NATIVE_COMPARISON_COMPLETE_NO_ADVANTAGE"
    ),
    "sampling_unit": contract["sampling_unit"],
    "policy_seeds": list(_SEEDS),
    "checkpoint_gates": list(_GATES),
    "held_out_manifest_sha256": held_out_hashes.pop(),
    "registry": str(cfg.registry.resolve()),
    "registry_sha256": _sha256(cfg.registry),
    "contract": contract,
    "support_rule_checks": checks,
    "stability": stability,
    "contrasts": contrasts,
    "sources": sources,
    "next_action": (
      "Use the preregistered paired estimates in the native baseline table; external adapter baselines remain required."
      if supported
      else "Report the null native comparison honestly; do not tune thresholds post hoc, and use a preregistered follow-up if method revision is needed."
    ),
    "claim_boundary": (
      "This result covers only matched native Task-only, Original SMP, and Proposed SMP; it does not substitute for FIRM-R, tracking, terrain, plate, or hardware evidence."
    ),
  }


def _markdown(result: dict[str, Any]) -> str:
  final = str(result["contract"]["final_gate"])
  lines = [
    "# Native Tier-A paired baseline effects",
    "",
    f"Status: **{result['status']}**",
    "",
    "Paired sampling unit: independently trained policy seed.",
    "",
    "| Contrast | Fixed macro delta [95% CI] | Fixed worst delta [95% CI] |",
    "| --- | ---: | ---: |",
  ]
  for name, contrast in result["contrasts"].items():
    metrics = contrast["gates"][final]
    macro = metrics["fixed_macro"]
    worst = metrics["fixed_worst"]
    lines.append(
      f"| `{name}` | {macro['mean']:.3f} [{macro['ci95_low']:.3f}, {macro['ci95_high']:.3f}] | "
      f"{worst['mean']:.3f} [{worst['ci95_low']:.3f}, {worst['ci95_high']:.3f}] |"
    )
  lines.extend(("", "## Frozen support checks", ""))
  for name, passed in result["support_rule_checks"].items():
    lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
  lines.extend(("", result["next_action"], "", result["claim_boundary"], ""))
  return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def write_analysis(cfg: NativeEffectCfg) -> dict[str, Any]:
  result = analyze(cfg)
  markdown = cfg.output_markdown or cfg.output_json.with_suffix(".md")
  _atomic_write(cfg.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
  _atomic_write(markdown, _markdown(result))
  return result


def main(cfg: NativeEffectCfg) -> None:
  result = write_analysis(cfg)
  print(f"{result['status']}: {result['support_rule_checks']}")


if __name__ == "__main__":
  main(tyro.cli(NativeEffectCfg))
