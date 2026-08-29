"""Audit whether every required RA-L evidence criterion is actually proven."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_VALID_STATUSES = {"not_started", "in_progress", "met", "failed"}
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class AuditCfg:
  ledger: Path = Path("docs/ral_evidence_matrix.json")
  output_json: Path = Path("run_control/ral_readiness/latest.json")
  output_markdown: Path = Path("run_control/ral_readiness/latest.md")


def _evidence_valid(evidence: dict[str, Any], repo_root: Path) -> tuple[bool, str]:
  evidence_type = evidence.get("type")
  target = evidence.get("target")
  description = evidence.get("description")
  if not isinstance(target, str) or not target:
    return False, "missing target"
  if not isinstance(description, str) or not description:
    return False, "missing description"
  if evidence_type in {"file", "result", "runtime"}:
    path = Path(target)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
      return False, f"missing path: {resolved}"
    if evidence_type == "file":
      return True, f"path exists: {resolved}"
    if not resolved.is_file() or resolved.stat().st_size == 0:
      return False, f"result must be a non-empty file: {resolved}"
    if resolved.suffix == ".json":
      try:
        json.loads(resolved.read_text())
      except (json.JSONDecodeError, OSError) as error:
        return False, f"invalid result JSON {resolved}: {error}"
    return True, f"non-empty result exists: {resolved}"
  if evidence_type == "url":
    valid = target.startswith(("https://", "http://"))
    detail = (
      f"URL recorded: {target}" if valid else "URL must start with http:// or https://"
    )
    return valid, detail
  if evidence_type == "git_commit":
    valid = bool(_COMMIT.fullmatch(target))
    return valid, f"commit recorded: {target}" if valid else "invalid Git commit"
  return False, f"unknown evidence type: {evidence_type}"


def audit(ledger: dict[str, Any], ledger_path: Path) -> dict[str, Any]:
  if ledger.get("schema_version") != 1:
    raise ValueError("unsupported RA-L evidence ledger schema")
  criteria = ledger.get("criteria")
  if not isinstance(criteria, list) or not criteria:
    raise ValueError("RA-L evidence ledger contains no criteria")
  repo_root = ledger_path.resolve().parent.parent
  seen_ids = set()
  audited = []
  for criterion in criteria:
    criterion_id = criterion.get("id")
    if not isinstance(criterion_id, str) or not criterion_id:
      raise ValueError("criterion is missing an id")
    if criterion_id in seen_ids:
      raise ValueError(f"duplicate criterion id: {criterion_id}")
    seen_ids.add(criterion_id)
    status = criterion.get("status")
    if status not in _VALID_STATUSES:
      raise ValueError(f"{criterion_id} has invalid status: {status}")
    evidence_results = []
    for evidence in criterion.get("evidence", []):
      valid, detail = _evidence_valid(evidence, repo_root)
      evidence_results.append({"valid": valid, "detail": detail, "evidence": evidence})
    evidence_valid = bool(evidence_results) and all(
      result["valid"] for result in evidence_results
    )
    result_evidence_valid = any(
      result["valid"] and result["evidence"].get("type") in {"result", "runtime"}
      for result in evidence_results
    )
    proven = status == "met" and evidence_valid and result_evidence_valid
    audited.append(
      {
        "id": criterion_id,
        "name": criterion.get("name"),
        "priority": int(criterion.get("priority", 99)),
        "required": bool(criterion.get("required", True)),
        "declared_status": status,
        "evidence_valid": evidence_valid,
        "result_evidence_valid": result_evidence_valid,
        "proven": proven,
        "missing": criterion.get("missing", ""),
        "evidence": evidence_results,
      }
    )

  required = [item for item in audited if item["required"]]
  unresolved = [item for item in required if not item["proven"]]
  unresolved.sort(key=lambda item: (item["priority"], item["id"]))
  return {
    "target": ledger.get("target"),
    "policy_scope": ledger.get("policy_scope"),
    "status": "RAL_READY" if not unresolved else "NOT_RAL_READY",
    "required_count": len(required),
    "proven_required_count": len(required) - len(unresolved),
    "unresolved_required_ids": [item["id"] for item in unresolved],
    "next_priority": unresolved[0] if unresolved else None,
    "criteria": audited,
    "completion_rule": (
      "RAL_READY requires every required criterion to be declared met, all "
      "referenced evidence to exist, and at least one non-empty result/runtime "
      "artifact per criterion. In-progress, implementation-only, or historical "
      "proxy evidence does not pass."
    ),
  }


def _markdown(report: dict[str, Any]) -> str:
  lines = [
    "# RA-L evidence readiness audit",
    "",
    f"Status: **{report['status']}**",
    "",
    f"Proven required criteria: {report['proven_required_count']}/{report['required_count']}",
    "",
    "| ID | Criterion | Declared | Evidence valid | Result artifact | Proven | Missing |",
    "| --- | --- | --- | :---: | :---: | :---: | --- |",
  ]
  for item in report["criteria"]:
    lines.append(
      f"| {item['id']} | {item['name']} | {item['declared_status']} | "
      f"{'yes' if item['evidence_valid'] else 'no'} | "
      f"{'yes' if item['result_evidence_valid'] else 'no'} | "
      f"{'yes' if item['proven'] else 'no'} | {item['missing']} |"
    )
  lines.extend(("", report["completion_rule"], ""))
  return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(content)
  temporary.replace(path)


def main(cfg: AuditCfg) -> None:
  ledger = json.loads(cfg.ledger.read_text())
  report = audit(ledger, cfg.ledger)
  _atomic_write(cfg.output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
  _atomic_write(cfg.output_markdown, _markdown(report))
  print(
    f"{report['status']}: {report['proven_required_count']}/"
    f"{report['required_count']} required criteria proven"
  )


if __name__ == "__main__":
  main(tyro.cli(AuditCfg))
