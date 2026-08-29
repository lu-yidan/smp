"""Build an all-or-nothing manifest for the eight-arm SMP causal screen."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

_ARMS: tuple[dict[str, str], ...] = (
  {
    "name": "a0_f2s2_gsi",
    "task": "Smp-Getup-Scratch-A0-F2S2-GSI-G1",
    "log_dir": "smp_scratch_a0_f2s2_gsi_g1",
    "run_suffix": "scratch_a0_f2s2_gsi_30k_seed20260830",
    "wandb_run_id": "zqrc8jmf",
  },
  {
    "name": "a1_v7_gsi",
    "task": "Smp-Getup-Scratch-A1-V7-GSI-G1",
    "log_dir": "smp_scratch_a1_v7_gsi_g1",
    "run_suffix": "scratch_a1_v7_gsi_30k_seed20260830",
    "wandb_run_id": "5xcp7tru",
  },
  {
    "name": "a2_f2s2_mix_strict",
    "task": "Smp-Getup-Scratch-A2-F2S2-Mix-Strict-G1",
    "log_dir": "smp_scratch_a2_f2s2_mix_strict_g1",
    "run_suffix": "scratch_a2_f2s2_mix_strict_30k_seed20260830",
    "wandb_run_id": "qpo8vd2a",
  },
  {
    "name": "a3_v7_mix_strict",
    "task": "Smp-Getup-Scratch-A3-V7-Mix-Strict-G1",
    "log_dir": "smp_scratch_a3_v7_mix_strict_g1",
    "run_suffix": "scratch_a3_v7_mix_strict_30k_seed20260830",
    "wandb_run_id": "a6az0q91",
  },
  {
    "name": "a4_f2s2_mix_reset_aware",
    "task": "Smp-Getup-Scratch-A4-F2S2-Mix-ResetAware-G1",
    "log_dir": "smp_scratch_a4_f2s2_mix_reset_aware_g1",
    "run_suffix": "scratch_a4_f2s2_mix_resetaware_30k_seed20260830",
    "wandb_run_id": "6ok4oe7g",
  },
  {
    "name": "a5_v7_mix_reset_aware",
    "task": "Smp-Getup-Scratch-A5-V7-Mix-ResetAware-G1",
    "log_dir": "smp_scratch_a5_v7_mix_reset_aware_g1",
    "run_suffix": "scratch_a5_v7_mix_resetaware_30k_seed20260830",
    "wandb_run_id": "adg5qrxg",
  },
  {
    "name": "a6_f2s2_mix_bridge",
    "task": "Smp-Getup-Scratch-A6-F2S2-Mix-Bridge-G1",
    "log_dir": "smp_scratch_a6_f2s2_mix_bridge_g1",
    "run_suffix": "scratch_a6_f2s2_mix_bridge_30k_seed20260830",
    "wandb_run_id": "hk30sstb",
  },
  {
    "name": "a7_v7_mix_bridge",
    "task": "Smp-Getup-Scratch-A7-V7-Mix-Bridge-G1",
    "log_dir": "smp_scratch_a7_v7_mix_bridge_g1",
    "run_suffix": "scratch_a7_v7_mix_bridge_30k_seed20260830",
    "wandb_run_id": "32nhutcb",
  },
)


@dataclass(frozen=True)
class ManifestCfg:
  checkpoint_step: int
  output: Path
  logs_root: Path = Path("logs/rsl_rl")
  policy_seed: int = 42
  environment_seed: int = 42
  project: str = "tabletennis/smp"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_commit() -> str:
  result = subprocess.run(
    ("git", "rev-parse", "HEAD"),
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _discover_run(
  logs_root: Path, arm: dict[str, str], checkpoint_step: int
) -> Path:
  task_dir = logs_root / arm["log_dir"]
  if not task_dir.is_dir():
    raise FileNotFoundError(f"missing task log directory: {task_dir}")
  checkpoint_name = f"model_{checkpoint_step}.pt"
  candidates = sorted(
    (
      path
      for path in task_dir.iterdir()
      if path.is_dir()
      and path.name.endswith(arm["run_suffix"])
      and (path / checkpoint_name).is_file()
    ),
    key=lambda path: path.stat().st_mtime,
  )
  if not candidates:
    raise FileNotFoundError(
      f"no run ending in {arm['run_suffix']!r} with {checkpoint_name} "
      f"under {task_dir}"
    )
  return candidates[-1]


def _recorded_seed(run_dir: Path, config_name: str) -> int:
  path = run_dir / "params" / config_name
  if not path.is_file():
    raise FileNotFoundError(f"missing seed provenance: {path}")
  match = re.search(r"^seed:\s*(-?\d+)\s*$", path.read_text(), flags=re.MULTILINE)
  if match is None:
    raise ValueError(f"top-level seed missing from {path}")
  return int(match.group(1))


def build_manifest(cfg: ManifestCfg) -> dict[str, Any]:
  discovered: list[tuple[dict[str, str], Path, Path, int, int]] = []
  missing: list[str] = []
  for arm in _ARMS:
    try:
      run_dir = _discover_run(cfg.logs_root, arm, cfg.checkpoint_step)
    except FileNotFoundError as error:
      missing.append(f"{arm['name']}: {error}")
      continue
    checkpoint = run_dir / f"model_{cfg.checkpoint_step}.pt"
    try:
      policy_seed = _recorded_seed(run_dir, "agent.yaml")
      environment_seed = _recorded_seed(run_dir, "env.yaml")
    except (FileNotFoundError, ValueError) as error:
      missing.append(f"{arm['name']}: {error}")
      continue
    if policy_seed != cfg.policy_seed or environment_seed != cfg.environment_seed:
      missing.append(
        f"{arm['name']}: recorded policy/environment seeds "
        f"{policy_seed}/{environment_seed}, expected "
        f"{cfg.policy_seed}/{cfg.environment_seed}"
      )
      continue
    discovered.append(
      (arm, run_dir.resolve(), checkpoint.resolve(), policy_seed, environment_seed)
    )

  if missing:
    details = "\n".join(f"- {item}" for item in missing)
    raise FileNotFoundError(
      "refusing to create a partial frozen manifest; missing:\n" + details
    )

  runs = []
  for arm, run_dir, checkpoint, policy_seed, environment_seed in discovered:
    runs.append(
      {
        "name": arm["name"],
        "task": arm["task"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "policy_seed": policy_seed,
        "environment_seed": environment_seed,
        "seed_provenance": {
          "agent_config": str(run_dir / "params" / "agent.yaml"),
          "environment_config": str(run_dir / "params" / "env.yaml"),
        },
        "run_dir": str(run_dir),
        "wandb_run_id": arm["wandb_run_id"],
        "wandb_url": (f"https://wandb.ai/{cfg.project}/runs/{arm['wandb_run_id']}"),
      }
    )

  return {
    "experiment": f"scratch-causal-{cfg.checkpoint_step}",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "code_commit": _git_commit(),
    "checkpoint_step": cfg.checkpoint_step,
    "policy_seed": cfg.policy_seed,
    "environment_seed": cfg.environment_seed,
    "historical_run_name_seed_label": 20260830,
    "seed_label_warning": (
      "The historical run suffix says seed20260830, but the saved agent and "
      "environment configs prove that both effective seeds were 42."
    ),
    "evaluation_protocol": {
      "reset_modes": [
        "native_gsi",
        "prone",
        "supine",
        "left_side",
        "right_side",
      ],
      "num_envs": 512,
      "steps": 500,
      "evaluation_seed": 20260829,
    },
    "runs": runs,
  }


def main(cfg: ManifestCfg) -> None:
  manifest = build_manifest(cfg)
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
  temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  temporary.replace(cfg.output)
  print(f"wrote {len(manifest['runs'])} arms to {cfg.output}")


if __name__ == "__main__":
  main(tyro.cli(ManifestCfg))
