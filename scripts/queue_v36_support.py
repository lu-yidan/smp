"""Launch the approved short refinement once one server GPU is unoccupied.

Never kills other tasks. A finite queue job, not a recurring scheduler.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--control-dir", type=Path, required=True)
  parser.add_argument("--max-wait-hours", type=float, default=24.0)
  args = parser.parse_args()
  args.control_dir.mkdir(parents=True, exist_ok=False)
  status_path = args.control_dir / "status.json"

  def status(state, **kw):
    status_path.write_text(
      json.dumps(dict(state=state, time=time.time(), **kw), indent=2)
    )

  try:
    assert args.checkpoint.is_file() and not args.output.exists()
    deadline = time.time() + args.max_wait_hours * 3600
    previous, streak = None, 0
    while time.time() < deadline:
      lines = subprocess.check_output(
        [
          "nvidia-smi",
          "--query-gpu=index,memory.used,utilization.gpu",
          "--format=csv,noheader,nounits",
        ],
        text=True,
      ).splitlines()
      rows = [[int(x.strip()) for x in line.split(",")] for line in lines]
      free = [i for i, memory, usage in rows if memory < 256 and usage < 10]
      gpu = free[0] if free else None
      streak = streak + 1 if gpu is not None and gpu == previous else 0
      previous = gpu
      status("waiting_for_free_gpu", gpus=rows, consecutive_free_checks=streak)
      if streak >= 2:
        break
      time.sleep(30)
    else:
      status("expired_without_launch")
      return
    base = dict(
      os.environ,
      CUDA_VISIBLE_DEVICES=str(gpu),
      MUJOCO_GL="egl",
      OMP_NUM_THREADS="6",
      PYTHONPATH="src",
      WANDB_MODE="offline",
    )
    task = "Smp-Getup-Stairs-V36-93D-Support-G1"
    command = [
      sys.executable,
      "-u",
      "scripts/train_v36_93d.py",
      "--task",
      task,
      "--checkpoint",
      str(args.checkpoint),
      "--num-envs",
      "4096",
    ]
    smoke_dir = args.control_dir / "smoke4096"
    status("preflight", gpu=gpu)
    with (args.control_dir / "preflight.log").open("x") as log:
      subprocess.run(
        command + ["--smoke", "--iterations", "2", "--log-dir", str(smoke_dir)],
        env=base,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
      )
    import torch

    ckpt = torch.load(smoke_dir / "model_1.pt", map_location="cpu", weights_only=False)
    assert all(
      torch.isfinite(v).all()
      for name in ("actor_state_dict", "critic_state_dict")
      for v in ckpt[name].values()
      if torch.is_tensor(v)
    )
    assert all(g["lr"] == 3e-5 for g in ckpt["optimizer_state_dict"]["param_groups"])
    audit = json.loads((smoke_dir / "preflight.json").read_text())
    assert (
      audit["fresh_optimizer"]
      and audit["source_weights_exact"]
      and audit["support_initial_ground_loss_zero"]
    )
    train = command + ["--iterations", "3000", "--log-dir", str(args.output)]
    with (args.control_dir / "train.log").open("x") as log:
      proc = subprocess.Popen(
        train, env=base, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
      )
    status("training_started", gpu=gpu, training_pid=proc.pid, training_command=train)
    evaluation = [
      sys.executable,
      "-u",
      "scripts/watch_v36_stair_evaluation.py",
      "--training-pid",
      str(proc.pid),
      "--training-dir",
      str(args.output),
      "--output-dir",
      str(args.control_dir / "evaluation"),
      "--task",
      task,
      "--baseline",
      str(args.checkpoint),
      "--checkpoints",
      "500",
      "1000",
      "1500",
      "2000",
      "2500",
      "2999",
      "--num-envs",
      "64",
      "--stop-on-regression",
    ]
    with (args.control_dir / "evaluation.log").open("x") as log:
      ev = subprocess.Popen(
        evaluation,
        env=base,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
    status(
      "launched",
      gpu=gpu,
      training_pid=proc.pid,
      evaluation_pid=ev.pid,
      training_command=train,
      evaluation_command=evaluation,
      note="Training and the small evaluation batch share the reserved GPU.",
    )
  except BaseException as exc:
    status("failed", error=repr(exc))
    raise


if __name__ == "__main__":
  main()
