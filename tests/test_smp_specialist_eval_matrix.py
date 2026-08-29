from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import run_smp_specialist_eval_matrix as matrix


class SpecialistMatrixTest(unittest.TestCase):
  def test_frozen_stratum_catalog_sizes_and_uniqueness(self) -> None:
    terrain = matrix.terrain_strata()
    plate = matrix.plate_strata()
    self.assertEqual(len(terrain), 76)
    self.assertEqual(len(plate), 10)
    self.assertEqual(len({matrix._stratum_id(row) for row in terrain}), 76)
    self.assertEqual(len({matrix._stratum_id(row) for row in plate}), 10)
    self.assertEqual(
      len([row for row in terrain if row["terrain_type"] == "stairs"]),
      48,
    )
    self.assertEqual(
      len([row for row in plate if row["plate_present"]]),
      6,
    )

  def test_dry_run_builds_one_command_per_terrain_stratum(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      checkpoint = root / "model_2000.pt"
      checkpoint.write_bytes(b"checkpoint")
      manifest = root / "manifest.json"
      manifest.write_text(
        json.dumps(
          {
            "phase": "T",
            "checkpoint_step": 2000,
            "runs": [
              {
                "phase": "T",
                "task": "Task-T",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": matrix._sha256(checkpoint),
                "policy_seed": 11,
              }
            ],
          }
        )
      )
      with redirect_stdout(StringIO()):
        result = matrix.run_matrix(
          matrix.SpecialistMatrixCfg(
            manifest=manifest,
            output_dir=root / "output",
            devices=("cuda:0", "cuda:1"),
            dry_run=True,
          )
        )
      self.assertEqual(result["status"], "DRY_RUN")
      self.assertEqual(result["stratum_count"], 76)
      self.assertEqual(result["command_count"], 76)

  def test_rejects_changed_checkpoint(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      checkpoint = root / "model.pt"
      checkpoint.write_bytes(b"changed")
      manifest = root / "manifest.json"
      manifest.write_text(
        json.dumps(
          {
            "phase": "P",
            "runs": [
              {
                "phase": "P",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": "wrong",
              }
            ],
          }
        )
      )
      with self.assertRaisesRegex(ValueError, "checkpoint changed"):
        matrix._load_manifest(manifest)


if __name__ == "__main__":
  unittest.main()
