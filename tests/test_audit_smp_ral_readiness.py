from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from audit_smp_ral_readiness import audit


def _ledger(evidence: list[dict]) -> dict:
  return {
    "schema_version": 1,
    "target": "RA-L",
    "policy_scope": "deployable",
    "criteria": [
      {
        "id": "C01",
        "name": "criterion",
        "priority": 0,
        "required": True,
        "status": "met",
        "evidence": evidence,
        "missing": "",
      }
    ],
  }


class ReadinessEvidenceTest(unittest.TestCase):
  def test_implementation_file_alone_cannot_prove_criterion(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger_path = root / "docs/ledger.json"
      implementation = root / "scripts/evaluator.py"
      ledger_path.parent.mkdir()
      implementation.parent.mkdir()
      implementation.write_text("pass\n")
      report = audit(
        _ledger(
          [
            {
              "type": "file",
              "target": "scripts/evaluator.py",
              "description": "evaluation implementation",
            }
          ]
        ),
        ledger_path,
      )
      criterion = report["criteria"][0]
      self.assertTrue(criterion["evidence_valid"])
      self.assertFalse(criterion["result_evidence_valid"])
      self.assertFalse(criterion["proven"])
      self.assertIn("path exists", criterion["evidence"][0]["detail"])

  def test_nonempty_valid_json_result_can_prove_criterion(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger_path = root / "docs/ledger.json"
      result = root / "results/frozen_matrix.json"
      ledger_path.parent.mkdir()
      result.parent.mkdir()
      result.write_text(json.dumps({"status": "complete"}))
      report = audit(
        _ledger(
          [
            {
              "type": "result",
              "target": "results/frozen_matrix.json",
              "description": "frozen evaluation result",
            }
          ]
        ),
        ledger_path,
      )
      criterion = report["criteria"][0]
      self.assertTrue(criterion["result_evidence_valid"])
      self.assertTrue(criterion["proven"])
      self.assertEqual(report["status"], "RAL_READY")

  def test_missing_or_invalid_runtime_result_is_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      ledger_path = root / "docs/ledger.json"
      ledger_path.parent.mkdir()
      missing = audit(
        _ledger(
          [
            {
              "type": "runtime",
              "target": "results/missing.json",
              "description": "runtime output",
            }
          ]
        ),
        ledger_path,
      )
      self.assertFalse(missing["criteria"][0]["evidence_valid"])

      invalid = root / "results/invalid.json"
      invalid.parent.mkdir()
      invalid.write_text("not-json")
      invalid_report = audit(
        _ledger(
          [
            {
              "type": "runtime",
              "target": "results/invalid.json",
              "description": "runtime output",
            }
          ]
        ),
        ledger_path,
      )
      self.assertFalse(invalid_report["criteria"][0]["evidence_valid"])


if __name__ == "__main__":
  unittest.main()
