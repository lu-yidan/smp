"""Start R1 and R2 on GPU2/3; keep B0/B1 intact."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--preflight", action="store_true")
  a = p.parse_args()
  root = Path(__file__).resolve().parents[1]
  suffix = "smoke" if a.preflight else "10000"
  control = root / "run_control" / f"natural_low_seed20260911_{suffix}"
  control.mkdir(parents=True, exist_ok=False)
  records = []
  for arm, gpu in (("R1", 2), ("R2", 3)):
    logdir = root / "logs/rsl_rl/scratch93_natural_low" / f"{arm}_seed20260911_{suffix}"
    assert not logdir.exists()
    cmd = [str(root/".venv/bin/python"), "-u", "scripts/train_natural_low_ablation.py",
           "--arm", arm, "--num-envs", "128" if a.preflight else "4096",
           "--iterations", "4" if a.preflight else "10000", "--log-dir", str(logdir)]
    if a.preflight: cmd.append("--preflight")
    run_id = f"scratch93-natural-low-{arm.lower()}-seed20260911"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH="src:scripts:.",
      OMP_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="2", MUJOCO_GL="egl",
      WANDB_ENTITY="tabletennis", WANDB_PROJECT="smp", WANDB_MODE="online",
      WANDB_RUN_ID=run_id, WANDB_NAME=f"scratch93_{arm}_natural_low_seed20260911",
      WANDB_RUN_GROUP="scratch93_natural_low_seed20260911", WANDB_RESUME="never")
    with (control/f"{arm}.log").open("w") as stream:
      proc = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
                              stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    records.append(dict(arm=arm, gpu=gpu, pid=proc.pid, command=cmd,
      log_dir=str(logdir), log=str(control/f"{arm}.log"),
      wandb_url=None if a.preflight else f"https://wandb.ai/tabletennis/smp/runs/{run_id}"))
    (control/"launches.json").write_text(json.dumps(records, indent=2))
  print(json.dumps(records, indent=2))


if __name__ == "__main__":
  main()
