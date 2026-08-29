"""Validate the frozen RA-L terrain/plate progression before launching jobs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro


@dataclass(frozen=True)
class ProtocolCfg:
  protocol: Path = Path("docs/ral_terrain_plate_protocol.json")
  output: Path | None = None


def _sum_one(values: dict[str, Any], label: str) -> None:
  total = sum(float(value) for value in values.values())
  if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError(f"{label} must sum to one, got {total}")
  if any(float(value) < 0.0 for value in values.values()):
    raise ValueError(f"{label} contains a negative probability")


def validate(payload: dict[str, Any]) -> dict[str, Any]:
  if payload.get("schema_version") != 1:
    raise ValueError("unsupported progression protocol schema")
  actor = payload["actor_contract"]
  if actor.get("dimension") != 93 or actor.get("history_frames") != 1:
    raise ValueError("progression actor must remain 93D and single-frame")
  forbidden = set(actor.get("forbidden", []))
  required_forbidden = {
    "true_base_linear_velocity",
    "terrain_type_or_level",
    "plate_pose_mass_or_geometry",
    "simulator_contact_force_or_label",
  }
  if not required_forbidden.issubset(forbidden):
    raise ValueError("deployable actor forbidden-observation audit is incomplete")

  prerequisite = payload["prerequisites"]
  seeds = prerequisite["policy_seeds"]
  shared = payload["shared_training"]
  if prerequisite.get("aggregate_status") != "MINIMUM_POLICY_SEEDS_MET":
    raise ValueError("terrain/plate progression must require the three-seed aggregate")
  if len(seeds) != 3 or len(set(seeds)) != 3 or seeds != shared["policy_seeds"]:
    raise ValueError("three unique policy seeds must be preserved across phases")
  expected_transitions = (
    int(shared["num_envs"])
    * int(shared["steps_per_update"])
    * int(shared["max_updates"])
  )
  if shared.get("transitions_per_run") != expected_transitions:
    raise ValueError("declared transition budget is inconsistent")

  phases = payload.get("phases", {})
  if set(phases) != {"T", "P", "U"}:
    raise ValueError("protocol must contain exactly T, P, and U phases")
  terrain = phases["T"]["training_distribution"]
  _sum_one(
    {name: terrain[name] for name in ("flat", "slope", "stairs", "rough")},
    "T terrain distribution",
  )
  _sum_one(terrain["stairs_edge_cohorts"], "T stair edge cohorts")
  _sum_one(terrain["terrain_levels"], "T terrain levels")
  _sum_one(terrain["fall_poses"], "T fall poses")
  if float(terrain.get("plate_probability", -1.0)) != 0.0:
    raise ValueError("T must not contain a plate")

  plate = phases["P"]["training_distribution"]
  _sum_one(
    {
      "flat_without_plate": plate["flat_without_plate"],
      "flat_with_plate": plate["flat_with_plate"],
    },
    "P plate presence",
  )
  _sum_one(plate["unpinned_fall_poses"], "P unpinned poses")
  _sum_one(plate["pinned_fall_poses"], "P pinned poses")
  _sum_one(plate["plate_mass_kg"], "P plate masses")
  if phases["P"]["physical_contract"].get("passive_motion") != (
    "vertical_prismatic_only"
  ):
    raise ValueError("P plate motion contract changed")

  unified = phases["U"]["training_distribution"]
  _sum_one(
    {
      name: unified[name]
      for name in (
        "flat_without_plate",
        "terrain_without_plate",
        "flat_with_plate",
        "stair_center_with_plate",
      )
    },
    "U cohort distribution",
  )
  if float(unified.get("plate_on_slope_rough_or_stair_edge", -1.0)) != 0.0:
    raise ValueError("U includes an unaudited plate/terrain combination")
  unified_gates = phases["U"]["promotion_gates"]
  if not (
    unified_gates.get("must_pass_all_T_gates")
    and unified_gates.get("must_pass_all_P_gates")
  ):
    raise ValueError("U must remain blocked on both specialist phases")
  if unified_gates.get("unmodeled_privileged_selector") is not False:
    raise ValueError("U may not use a privileged selector")

  return {
    "status": "VALID",
    "protocol_id": payload["protocol_id"],
    "actor_dimension": actor["dimension"],
    "history_frames": actor["history_frames"],
    "policy_seeds": seeds,
    "phases": list(phases),
    "transitions_per_run": expected_transitions,
    "claim_boundary": "Protocol validity is not experimental evidence.",
  }


def main(cfg: ProtocolCfg) -> None:
  result = validate(json.loads(cfg.protocol.read_text()))
  if cfg.output is not None:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(cfg.output)
  print(f"{result['status']}: {result['protocol_id']}")


if __name__ == "__main__":
  main(tyro.cli(ProtocolCfg))
