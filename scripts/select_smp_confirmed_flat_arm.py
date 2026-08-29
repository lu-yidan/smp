"""Promote one three-seed-confirmed flat arm to the T/P specialist budget."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

from build_smp_causal_manifest import _ARMS


@dataclass(frozen=True)
class FlatPromotionCfg:
  aggregate: Path
  confirmation_manifest_index: Path
  protocol: Path = Path("docs/ral_terrain_plate_protocol.json")
  output: Path = Path("run_control/scratch_causal_policy_seed_eval/flat_promotion.json")


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


def _verify_source_hashes(aggregate: dict[str, Any]) -> None:
  sources = aggregate.get("source_summaries")
  if not isinstance(sources, list) or len(sources) != 3:
    raise ValueError("aggregate must freeze exactly three source summaries")
  for source in sources:
    path = Path(source["path"])
    if not path.is_file() or _sha256(path) != source.get("sha256"):
      raise ValueError(f"aggregate source changed: {path}")


def _checkpoint_sources(
  index: dict[str, Any], expected_seeds: tuple[int, ...]
) -> dict[str, dict[int, dict[str, Any]]]:
  rows = index.get("manifests")
  if index.get("status") != "READY" or not isinstance(rows, list):
    raise ValueError("confirmation manifest index is not READY")
  if tuple(sorted(index.get("policy_seeds", []))) != expected_seeds:
    raise ValueError("confirmation index policy seeds do not match protocol")
  by_arm: dict[str, dict[int, dict[str, Any]]] = {}
  for row in rows:
    path = Path(row["path"])
    if not path.is_file() or _sha256(path) != row.get("sha256"):
      raise ValueError(f"confirmation manifest changed: {path}")
    manifest = _load(path)
    seed = int(manifest["policy_seed"])
    if seed not in expected_seeds or int(manifest["environment_seed"]) != seed:
      raise ValueError(f"invalid seed provenance in {path}")
    for run in manifest.get("runs", []):
      checkpoint = Path(run["checkpoint"])
      if (
        int(run["policy_seed"]) != seed
        or int(run["environment_seed"]) != seed
        or not checkpoint.is_file()
        or _sha256(checkpoint) != run.get("checkpoint_sha256")
      ):
        raise ValueError(f"invalid confirmation checkpoint: {checkpoint}")
      by_arm.setdefault(run["name"], {})[seed] = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "run_dir": run["run_dir"],
      }
  for arm, seeds in by_arm.items():
    if tuple(sorted(seeds)) != expected_seeds:
      raise ValueError(f"arm {arm} lacks a matched checkpoint for every seed")
  return by_arm


def _eligibility(
  data: dict[str, Any], gates: dict[str, float]
) -> tuple[bool, list[str]]:
  metrics = data["metrics"]
  failures = []

  def mean_at_least(name: str, threshold_key: str) -> None:
    value = float(metrics[name]["mean"])
    threshold = float(gates[threshold_key])
    if value < threshold:
      failures.append(f"{name}.mean={value:.4f} < {threshold:.4f}")

  def each_at_least(name: str, threshold_key: str) -> None:
    threshold = float(gates[threshold_key])
    values = [float(value) for value in metrics[name]["policy_seed_values"]]
    if min(values) < threshold:
      failures.append(f"{name}.min_seed={min(values):.4f} < {threshold:.4f}")

  mean_at_least("gsi", "mean_native_gsi_success_min")
  mean_at_least("fixed_macro", "mean_fixed_macro_success_min")
  each_at_least("fixed_macro", "each_seed_fixed_macro_success_min")
  mean_at_least("fixed_worst", "mean_fixed_worst_success_min")
  each_at_least("fixed_worst", "each_seed_fixed_worst_success_min")
  finite = [
    float(value) for value in metrics["finite_action_rate"]["policy_seed_values"]
  ]
  required_finite = float(gates["finite_action_rate"])
  if min(finite) < required_finite:
    failures.append(
      f"finite_action_rate.min_seed={min(finite):.6f} < {required_finite:.6f}"
    )
  return not failures, failures


def _rank_key(item: tuple[str, dict[str, Any]]) -> tuple:
  arm, data = item
  metric = data["metrics"]
  return (
    -float(metric["fixed_worst"]["ci95_low"]),
    -float(metric["fixed_macro"]["ci95_low"]),
    float(metric["secondary_fall_rate_after_success"]["mean"]),
    float(metric["post_success_root_drift_p95_m"]["mean"]),
    float(metric["contact_foot_slip_p95_m_s"]["mean"]),
    float(metric["max_power_mean_w"]["mean"]),
    float(metric["max_joint_speed_p95_rad_s"]["mean"]),
    arm,
  )


def select(cfg: FlatPromotionCfg) -> dict[str, Any]:
  aggregate = _load(cfg.aggregate)
  protocol = _load(cfg.protocol)
  index = _load(cfg.confirmation_manifest_index)
  if aggregate.get("status") != "MINIMUM_POLICY_SEEDS_MET":
    raise ValueError("aggregate has not met the minimum policy-seed count")
  expected_seeds = tuple(
    sorted(int(seed) for seed in protocol["prerequisites"]["policy_seeds"])
  )
  if tuple(sorted(aggregate.get("policy_seeds", []))) != expected_seeds:
    raise ValueError("aggregate policy seeds do not match frozen protocol")
  if aggregate.get("policy_seed_count") != len(expected_seeds):
    raise ValueError("aggregate policy-seed count is inconsistent")
  _verify_source_hashes(aggregate)
  checkpoints = _checkpoint_sources(index, expected_seeds)
  catalog = {arm["name"]: arm for arm in _ARMS}
  gates = protocol["prerequisites"]["candidate_flat_gates"]

  audit = {}
  eligible = []
  for arm, data in aggregate.get("arms", {}).items():
    if arm not in catalog or arm not in checkpoints:
      raise ValueError(f"aggregate arm has no frozen checkpoint lineage: {arm}")
    passed, failures = _eligibility(data, gates)
    audit[arm] = {"eligible": passed, "failures": failures}
    if passed:
      eligible.append((arm, data))
  eligible.sort(key=_rank_key)
  winner = eligible[0][0] if eligible else None
  arm_index = int(winner[1]) if winner is not None else None
  status = "PROMOTE_TP_SPECIALISTS" if winner is not None else "NO_PROMOTION"
  result = {
    "schema_version": 1,
    "status": status,
    "selected_arm": winner,
    "selected_arm_index": arm_index,
    "policy_seeds": list(expected_seeds),
    "eligibility": audit,
    "eligible_ranking": [arm for arm, _ in eligible],
    "ranking_rule": protocol["prerequisites"]["selection_order"],
    "claim_boundary": protocol["prerequisites"]["selection_claim_boundary"],
    "matched_flat_checkpoints": checkpoints.get(winner, {}) if winner else {},
    "aggregate": str(cfg.aggregate.resolve()),
    "aggregate_sha256": _sha256(cfg.aggregate),
    "confirmation_manifest_index": str(cfg.confirmation_manifest_index.resolve()),
    "confirmation_manifest_index_sha256": _sha256(cfg.confirmation_manifest_index),
    "protocol": str(cfg.protocol.resolve()),
    "protocol_sha256": _sha256(cfg.protocol),
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
  }
  stable_material = {k: v for k, v in result.items() if k != "generated_at_utc"}
  result["promotion_id"] = hashlib.sha256(
    json.dumps(stable_material, sort_keys=True).encode()
  ).hexdigest()
  return result


def write_selection(cfg: FlatPromotionCfg) -> dict[str, Any]:
  result = select(cfg)
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  if cfg.output.exists():
    existing = _load(cfg.output)
    if existing.get("promotion_id") != result["promotion_id"]:
      raise ValueError("existing flat promotion conflicts with current frozen sources")
    return existing
  temporary = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  temporary.replace(cfg.output)
  return result


def main(cfg: FlatPromotionCfg) -> None:
  result = write_selection(cfg)
  print(f"{result['status']}: {result['selected_arm']}")


if __name__ == "__main__":
  main(tyro.cli(FlatPromotionCfg))
