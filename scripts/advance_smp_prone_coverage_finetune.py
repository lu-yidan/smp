"""Read-only health monitor for the A12 prone-coverage fine-tune."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import tyro

from launch_smp_prone_coverage_finetune import _load_json, _pid_alive, _sha256

_ERROR = re.compile(
  r"traceback|cuda out of memory|outofmemoryerror|fatal|physical_reset_alert|"
  r"(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)",
  re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Cfg:
  control_dir: Path = Path("run_control/prone_coverage_finetune_v1/training")
  state: Path = Path("run_control/automation_state/prone_coverage_finetune_latest.json")


def _atomic(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def _checkpoint_integrity(path: Path) -> dict[str, Any]:
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  tensors: list[torch.Tensor] = []

  def collect(value: Any) -> None:
    if torch.is_tensor(value):
      tensors.append(value)
    elif isinstance(value, dict):
      for item in value.values():
        collect(item)
    elif isinstance(value, (tuple, list)):
      for item in value:
        collect(item)

  collect(checkpoint)
  return {
    "sha256": _sha256(path),
    "embedded_iteration": checkpoint.get("iter"),
    "tensor_count": len(tensors),
    "tensor_elements": sum(t.numel() for t in tensors),
    "all_tensors_finite": all(bool(torch.isfinite(t).all()) for t in tensors),
    "actor_input_dim": checkpoint["actor_state_dict"]["mlp.0.weight"].shape[1],
    "critic_input_dim": checkpoint["critic_state_dict"]["mlp.0.weight"].shape[1],
  }


def advance(cfg: Cfg) -> dict[str, Any]:
  repo_root = Path(__file__).resolve().parents[1]
  control_dir = (
    cfg.control_dir if cfg.control_dir.is_absolute() else repo_root / cfg.control_dir
  )
  state_path = cfg.state if cfg.state.is_absolute() else repo_root / cfg.state
  launch_path = control_dir / "launch_manifest.json"
  now = datetime.now(timezone.utc)
  if not launch_path.is_file():
    result = {
      "schema_version": 1,
      "status": "WAITING_FOR_LAUNCH",
      "observed_at_utc": now.isoformat(),
    }
    _atomic(state_path, result)
    return result
  launch = _load_json(launch_path)
  log_path = Path(launch["log"])
  text = log_path.read_text(errors="replace") if log_path.is_file() else ""
  pid = int(launch["pid"])
  alive = _pid_alive(pid)
  matches = re.findall(r"Learning iteration\s+(\d+)/5000", text)
  throughput = re.findall(r"Steps per second:\s+(\d+)", text)
  age_minutes = (
    (now.timestamp() - log_path.stat().st_mtime) / 60 if log_path.is_file() else None
  )
  checkpoints = sorted(
    Path(launch["experiment_root"]).glob(f"*_{launch['run_name']}/model_*.pt")
  )
  iterations = []
  for path in checkpoints:
    try:
      iterations.append(int(path.stem.split("_")[1]))
    except (IndexError, ValueError):
      pass
  final_matches = [p for p in checkpoints if p.name == "model_4999.pt"]
  status = "PRONE_COVERAGE_TRAINING_ACTIVE"
  alert = None
  integrity = None
  if _ERROR.search(text):
    status = "PRONE_COVERAGE_TRAINING_ALERT"
    alert = "fatal/OOM/NaN/Inf/physical reset error in log"
  elif alive and age_minutes is not None and age_minutes > 45:
    status = "PRONE_COVERAGE_TRAINING_ALERT"
    alert = "training log stale for more than 45 minutes"
  elif not alive and len(final_matches) != 1:
    status = "PRONE_COVERAGE_TRAINING_ALERT"
    alert = "worker died before exactly one model_4999.pt"
  elif not alive:
    integrity = _checkpoint_integrity(final_matches[0])
    expected_integrity = {
      "embedded_iteration": 4999,
      "tensor_count": 73,
      "tensor_elements": 2620640,
      "all_tensors_finite": True,
      "actor_input_dim": 93,
      "critic_input_dim": 960,
    }
    if any(integrity.get(key) != value for key, value in expected_integrity.items()):
      status = "PRONE_COVERAGE_TRAINING_ALERT"
      alert = "final checkpoint integrity mismatch"
    else:
      status = "PRONE_COVERAGE_TRAINING_COMPLETE_READY_FOR_FROZEN_EVAL"
  result = {
    "schema_version": 1,
    "status": status,
    "observed_at_utc": now.isoformat(),
    "plan_id": launch["plan_id"],
    "code_commit": launch["code_commit"],
    "pid": pid,
    "process_alive": alive,
    "log": str(log_path),
    "log_age_minutes": age_minutes,
    "latest_log_iteration": int(matches[-1]) if matches else None,
    "latest_checkpoint_iteration": max(iterations) if iterations else None,
    "progress_iteration_lower_bound": max(iterations)
    if iterations
    else (int(matches[-1]) if matches else None),
    "latest_throughput_steps_s": int(throughput[-1]) if throughput else None,
    "checkpoint_count": len(checkpoints),
    "final_checkpoint": str(final_matches[0]) if len(final_matches) == 1 else None,
    "final_checkpoint_integrity": integrity,
    "alert": alert,
  }
  _atomic(state_path, result)
  return result


def main(cfg: Cfg) -> None:
  result = advance(cfg)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main(tyro.cli(Cfg))
