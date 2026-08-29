from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import evaluate_smp_baseline as evaluator


def _term(**params):
  return SimpleNamespace(params=params)


def _env_cfg():
  terrains = {
    name: SimpleNamespace(proportion=0.25) for name in evaluator._TERRAIN_TYPES
  }
  return SimpleNamespace(
    scene=SimpleNamespace(
      terrain=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains=terrains))
    ),
    events={
      "push_robot": object(),
      "mixed_fall_reset": _term(mode_weights=(0.25,) * 4),
      "sample_weighted_terrain_levels": _term(level_weights=(0.55, 0.30, 0.15, 0.0)),
      "sample_terrain_edge_reset": _term(cohort_weights=(0.4, 0.25, 0.2, 0.15)),
      "reset_escape_obstacle": _term(
        obstacle_probability_by_reset_type=(2 / 3, 2 / 3, 0.0, 0.0),
        plate_masses=(4.0, 8.0, 12.0),
        mass_weights=(0.25, 0.5, 0.25),
      ),
    },
  )


class SpecialistStratumConfigTest(unittest.TestCase):
  def test_forces_exact_terrain_level_pose_and_edge_cohort(self) -> None:
    env_cfg = _env_cfg()
    cfg = evaluator.EvalCfg(
      checkpoint=Path("model.pt"),
      reset_mode="right_side",
      evaluation_profile="terrain",
      terrain_type="stairs",
      terrain_level=1,
      stair_edge_cohort="straddle",
    )
    evaluator._configure_specialist_stratum(env_cfg, cfg)
    self.assertNotIn("push_robot", env_cfg.events)
    self.assertEqual(
      env_cfg.events["mixed_fall_reset"].params["mode_weights"],
      (0.0, 0.0, 0.0, 1.0),
    )
    proportions = {
      name: value.proportion
      for name, value in env_cfg.scene.terrain.terrain_generator.sub_terrains.items()
    }
    self.assertEqual(
      proportions, {"flat": 0.0, "slope": 0.0, "stairs": 1.0, "rough": 0.0}
    )
    self.assertEqual(
      env_cfg.events["sample_weighted_terrain_levels"].params["level_weights"],
      (0.0, 1.0, 0.0, 0.0),
    )
    self.assertEqual(
      env_cfg.events["sample_terrain_edge_reset"].params["cohort_weights"],
      (0.0, 0.0, 1.0, 0.0),
    )

  def test_forces_exact_pinned_pose_and_mass(self) -> None:
    env_cfg = _env_cfg()
    cfg = evaluator.EvalCfg(
      checkpoint=Path("model.pt"),
      reset_mode="supine",
      evaluation_profile="plate",
      plate_mode="pinned",
      plate_mass_kg=12.0,
    )
    evaluator._configure_specialist_stratum(env_cfg, cfg)
    term = env_cfg.events["reset_escape_obstacle"].params
    self.assertEqual(term["obstacle_probability_by_reset_type"], (1.0, 1.0, 0.0, 0.0))
    self.assertEqual(term["plate_masses"], (12.0,))
    self.assertEqual(term["mass_weights"], (1.0,))

  def test_rejects_unphysical_pinned_side_fixture(self) -> None:
    with self.assertRaisesRegex(ValueError, "only pins prone or supine"):
      evaluator._configure_specialist_stratum(
        _env_cfg(),
        evaluator.EvalCfg(
          checkpoint=Path("model.pt"),
          reset_mode="left_side",
          evaluation_profile="plate",
          plate_mode="pinned",
          plate_mass_kg=8.0,
        ),
      )


if __name__ == "__main__":
  unittest.main()
