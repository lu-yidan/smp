"""Aggregate final T or P evidence with independently trained policy as unit."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from aggregate_smp_policy_seeds import _bootstrap_mean


@dataclass(frozen=True)
class SpecialistAggregateCfg:
  analyses: tuple[Path, ...]
  output_json: Path
  protocol: Path = Path("docs/ral_terrain_plate_protocol.json")
  output_markdown: Path | None = None
  bootstrap_replicates: int = 20000
  bootstrap_seed: int = 20260910


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


def _numeric_metrics(analysis: dict[str, Any]) -> dict[str, float]:
  metrics = {
    key: float(value)
    for key, value in analysis["metrics"].items()
    if isinstance(value, (int, float)) and not isinstance(value, bool)
  }
  metrics["safety_ratio_max"] = float(analysis["safety_ratio_max"])
  if analysis["phase"] == "T":
    metrics["level1_each_terrain_worst"] = min(
      float(value)
      for value in analysis["metrics"]["level1_each_terrain_macro"].values()
    )
  else:
    metrics["plate_pose_worst"] = min(
      float(value) for value in analysis["metrics"]["plate_pose_macro"].values()
    )
  return metrics


def aggregate(cfg: SpecialistAggregateCfg) -> dict[str, Any]:
  protocol = _load(cfg.protocol)
  expected_seeds = sorted(
    int(seed) for seed in protocol["shared_training"]["policy_seeds"]
  )
  if len(cfg.analyses) != len(expected_seeds):
    raise ValueError("specialist aggregate requires exactly three analysis files")
  payloads = []
  sources = []
  for path in cfg.analyses:
    analysis = _load(path)
    if analysis.get("schema_version") != 1:
      raise ValueError(f"incompatible specialist analysis: {path}")
    if analysis.get("checkpoint_step") != 19999:
      raise ValueError("only the preregistered final checkpoint may promote a phase")
    if analysis.get("protocol") != str(cfg.protocol.resolve()) or analysis.get(
      "protocol_sha256"
    ) != _sha256(cfg.protocol):
      raise ValueError("specialist analysis protocol differs from aggregate protocol")
    summary = Path(analysis["summary"])
    manifest = Path(analysis["manifest"])
    if (
      not summary.is_file()
      or _sha256(summary) != analysis.get("summary_sha256")
      or not manifest.is_file()
      or _sha256(manifest) != analysis.get("manifest_sha256")
    ):
      raise ValueError(f"specialist analysis source changed: {path}")
    payloads.append(analysis)
    sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
  phases = {payload["phase"] for payload in payloads}
  arms = {payload["arm"] for payload in payloads}
  seeds = sorted(int(payload["policy_seed"]) for payload in payloads)
  if len(phases) != 1 or len(arms) != 1:
    raise ValueError("specialist aggregate must contain one matched phase and arm")
  if seeds != expected_seeds or len(set(seeds)) != len(seeds):
    raise ValueError("specialist aggregate policy seeds differ from protocol")
  phase = phases.pop()
  if phase not in ("T", "P"):
    raise ValueError(f"unexpected specialist phase: {phase}")
  metric_rows = [_numeric_metrics(payload) for payload in payloads]
  shared_metrics = set.intersection(*(set(row) for row in metric_rows))
  rng = random.Random(cfg.bootstrap_seed)
  metrics = {
    metric: _bootstrap_mean(
      [row[metric] for row in metric_rows], cfg.bootstrap_replicates, rng
    )
    for metric in sorted(shared_metrics)
  }
  each_seed_pass = all(payload["status"] == "PASS" for payload in payloads)
  result = {
    "schema_version": 1,
    "status": "PHASE_PASS" if each_seed_pass else "NO_PROMOTION",
    "phase": phase,
    "arm": arms.pop(),
    "checkpoint_step": 19999,
    "policy_seeds": seeds,
    "policy_seed_count": len(seeds),
    "sampling_unit": "independently trained policy seed",
    "checkpoint_rule": (
      "Final checkpoint only; 2k/5k/10k are diagnostic learning curves and "
      "cannot replace a failed final checkpoint."
    ),
    "each_seed_status": {
      str(payload["policy_seed"]): payload["status"] for payload in payloads
    },
    "metrics": metrics,
    "source_analyses": sources,
    "protocol": str(cfg.protocol.resolve()),
    "protocol_sha256": _sha256(cfg.protocol),
    "bootstrap_replicates": cfg.bootstrap_replicates,
    "bootstrap_seed": cfg.bootstrap_seed,
    "limitations": [
      "Three policy seeds provide only coarse policy-level uncertainty.",
      "A phase pass is simulation evidence and does not establish real-robot validity.",
      "Every policy seed must pass; an aggregate mean cannot hide a failed lineage.",
    ],
  }
  stable = {key: value for key, value in result.items() if key != "aggregate_id"}
  result["aggregate_id"] = hashlib.sha256(
    json.dumps(stable, sort_keys=True).encode()
  ).hexdigest()
  return result


def _markdown(result: dict[str, Any]) -> str:
  lines = [
    f"# Specialist {result['phase']} policy-seed aggregate",
    "",
    f"Status: **{result['status']}**",
    "",
    f"Policy seeds: `{result['policy_seeds']}`",
    "",
    "| Metric | Mean [policy-seed bootstrap 95% CI] |",
    "| --- | ---: |",
  ]
  for name, value in result["metrics"].items():
    lines.append(
      f"| `{name}` | {value['mean']:.3f} "
      f"[{value['ci95_low']:.3f}, {value['ci95_high']:.3f}] |"
    )
  lines.extend(("", result["checkpoint_rule"], ""))
  return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def write_aggregate(cfg: SpecialistAggregateCfg) -> dict[str, Any]:
  result = aggregate(cfg)
  output_markdown = cfg.output_markdown or cfg.output_json.with_suffix(".md")
  if cfg.output_json.exists():
    existing = _load(cfg.output_json)
    if existing.get("aggregate_id") != result["aggregate_id"]:
      raise ValueError("existing specialist aggregate conflicts with frozen sources")
    return existing
  _atomic_write(cfg.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
  _atomic_write(output_markdown, _markdown(result))
  return result


def main(cfg: SpecialistAggregateCfg) -> None:
  result = write_aggregate(cfg)
  print(f"{result['status']}: phase={result['phase']}, seeds={result['policy_seeds']}")


if __name__ == "__main__":
  main(tyro.cli(SpecialistAggregateCfg))
