"""Require matched three-seed T and P phase passes before any U experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro


@dataclass(frozen=True)
class UnifiedPrerequisiteCfg:
  terrain_aggregate: Path
  plate_aggregate: Path
  output: Path
  protocol: Path = Path("docs/ral_terrain_plate_protocol.json")


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


def _validate_aggregate(
  path: Path, phase: str, protocol: Path
) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
  aggregate = _load(path)
  if aggregate.get("phase") != phase or aggregate.get("checkpoint_step") != 19999:
    raise ValueError(f"{phase} aggregate is not the frozen final phase result")
  if aggregate.get("protocol") != str(protocol.resolve()) or aggregate.get(
    "protocol_sha256"
  ) != _sha256(protocol):
    raise ValueError(f"{phase} aggregate protocol changed")
  analyses = aggregate.get("source_analyses")
  if not isinstance(analyses, list) or len(analyses) != 3:
    raise ValueError(f"{phase} aggregate must freeze three source analyses")
  launch_ids = set()
  promotion_ids = set()
  flat_promotions = set()
  for source in analyses:
    analysis_path = Path(source["path"])
    if not analysis_path.is_file() or _sha256(analysis_path) != source.get("sha256"):
      raise ValueError(f"{phase} source analysis changed: {analysis_path}")
    analysis = _load(analysis_path)
    manifest_path = Path(analysis["manifest"])
    if not manifest_path.is_file() or _sha256(manifest_path) != analysis.get(
      "manifest_sha256"
    ):
      raise ValueError(f"{phase} source manifest changed: {manifest_path}")
    manifest = _load(manifest_path)
    launch_ids.add(str(manifest["launch_plan_id"]))
    promotion_ids.add(str(manifest["promotion_id"]))
    launch_path = Path(manifest["launch_manifest"])
    if not launch_path.is_file() or _sha256(launch_path) != manifest.get(
      "launch_manifest_sha256"
    ):
      raise ValueError(f"{phase} launch manifest changed: {launch_path}")
    launch = _load(launch_path)
    promotion_path = Path(launch["promotion"])
    if not promotion_path.is_file() or _sha256(promotion_path) != launch.get(
      "promotion_sha256"
    ):
      raise ValueError(f"{phase} flat promotion changed: {promotion_path}")
    flat_promotions.add(str(promotion_path.resolve()))
  return aggregate, launch_ids, promotion_ids, flat_promotions


def select(cfg: UnifiedPrerequisiteCfg) -> dict[str, Any]:
  protocol = _load(cfg.protocol)
  terrain, t_launch, t_promotion, t_flat = _validate_aggregate(
    cfg.terrain_aggregate, "T", cfg.protocol
  )
  plate, p_launch, p_promotion, p_flat = _validate_aggregate(
    cfg.plate_aggregate, "P", cfg.protocol
  )
  expected_seeds = sorted(
    int(seed) for seed in protocol["shared_training"]["policy_seeds"]
  )
  if (
    terrain.get("policy_seeds") != expected_seeds
    or plate.get("policy_seeds") != expected_seeds
  ):
    raise ValueError("T/P aggregates do not use frozen matched seeds")
  if terrain.get("arm") != plate.get("arm"):
    raise ValueError("T/P aggregates do not share one flat arm lineage")
  if (
    len(t_launch | p_launch) != 1
    or len(t_promotion | p_promotion) != 1
    or len(t_flat | p_flat) != 1
  ):
    raise ValueError("T/P aggregates do not share one launch and flat promotion")
  passed = terrain.get("status") == "PHASE_PASS" and plate.get("status") == "PHASE_PASS"
  result = {
    "schema_version": 1,
    "status": "PROMOTE_U" if passed else "NO_PROMOTION",
    "arm": terrain["arm"],
    "policy_seeds": expected_seeds,
    "terrain_aggregate": str(cfg.terrain_aggregate.resolve()),
    "terrain_aggregate_sha256": _sha256(cfg.terrain_aggregate),
    "plate_aggregate": str(cfg.plate_aggregate.resolve()),
    "plate_aggregate_sha256": _sha256(cfg.plate_aggregate),
    "specialist_launch_plan_id": next(iter(t_launch)),
    "flat_promotion_id": next(iter(t_promotion)),
    "flat_promotion": next(iter(t_flat)),
    "protocol": str(cfg.protocol.resolve()),
    "protocol_sha256": _sha256(cfg.protocol),
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "claim_boundary": (
      "PROMOTE_U authorizes only the preregistered U training budget. It is not "
      "evidence that U, hardware safety, or the RA-L paper claim has passed."
    ),
  }
  stable = {key: value for key, value in result.items() if key != "generated_at_utc"}
  result["decision_id"] = hashlib.sha256(
    json.dumps(stable, sort_keys=True).encode()
  ).hexdigest()
  return result


def write_selection(cfg: UnifiedPrerequisiteCfg) -> dict[str, Any]:
  result = select(cfg)
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  if cfg.output.exists():
    existing = _load(cfg.output)
    if existing.get("decision_id") != result["decision_id"]:
      raise ValueError("existing unified prerequisite decision conflicts")
    return existing
  temporary = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  temporary.replace(cfg.output)
  return result


def main(cfg: UnifiedPrerequisiteCfg) -> None:
  result = write_selection(cfg)
  print(f"{result['status']}: arm={result['arm']}")


if __name__ == "__main__":
  main(tyro.cli(UnifiedPrerequisiteCfg))
