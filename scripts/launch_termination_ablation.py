"""Launch one independent scratch experiment per GPU, with durable run records."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--preflight", action="store_true")
  p.add_argument("--seed", type=int, default=20260911)
  a = p.parse_args()
  root = Path(__file__).resolve().parents[1]
  suffix = "smoke" if a.preflight else "10000"
  control = root / "run_control" / f"scratch93_termination_seed{a.seed}_{suffix}"
  control.mkdir(parents=True, exist_ok=False)
  launches = []
  for gpu in range(6):
    arm = f"B{gpu}"
    logdir = root / "logs/rsl_rl/scratch93_termination" / f"{arm}_seed{a.seed}_{suffix}"
    if logdir.exists():
      raise FileExistsError(logdir)
    command = [str(root / ".venv/bin/python"), "-u", "scripts/train_termination_ablation.py",
               "--arm", arm, "--seed", str(a.seed), "--num-envs", "16" if a.preflight else "4096",
               "--iterations", "2" if a.preflight else "10000", "--log-dir", str(logdir)]
    if a.preflight:
      command.append("--preflight")
    run_id = f"scratch93-termination-{arm.lower()}-seed{a.seed}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH="src:scripts:.",
               MUJOCO_GL="egl", OMP_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="2",
               WANDB_PROJECT="smp", WANDB_ENTITY="tabletennis", WANDB_MODE="online",
               WANDB_RUN_ID=run_id, WANDB_NAME=f"scratch93_{arm}_seed{a.seed}",
               WANDB_RUN_GROUP=f"scratch93_termination_seed{a.seed}", WANDB_RESUME="never")
    with (control / f"{arm}.log").open("w") as stream:
      proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                              stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    launches.append(dict(arm=arm, gpu=gpu, pid=proc.pid, command=command,
                         log=str(control / f"{arm}.log"), log_dir=str(logdir),
                         wandb_url=None if a.preflight else f"https://wandb.ai/tabletennis/smp/runs/{run_id}"))
    (control / "launches.json").write_text(json.dumps(launches, indent=2))
  print(json.dumps(launches, indent=2))


if __name__ == "__main__":
  main()
