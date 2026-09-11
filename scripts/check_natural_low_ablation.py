"""Semantic and configuration checks for R1/R2 before expensive training."""
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import torch
from train_natural_low_ablation import build_config
from smp.rl.tasks.getup.natural_low_reset import natural_persistent_low_smp


def main():
  path = "datasets/reset_banks/natural_low_right_v1.npz"
  a, aa = build_config("R1", path)
  b, bb = build_config("R2", path)
  x, y = asdict(a), asdict(b)
  for z in (x, y):
    z["scene"]["terrain"]["spec_fn"] = z["scene"]["terrain"]["spec_fn"].__qualname__
    z["terminations"].pop("smp_too_low")
  assert x == y and asdict(aa) == asdict(bb)
  assert a.observations["actor"].enable_corruption
  assert all(k in a.events for k in ("foot_friction", "encoder_bias", "base_com", "push_robot", "gsi_refresh"))
  env = SimpleNamespace(step_dt=0.02, _natural_type=torch.tensor([0, -1]),
    _natural_bad_count=torch.zeros(2, dtype=torch.long), _smp_raw_err=torch.ones(2),
    episode_length_buf=torch.zeros(2, dtype=torch.long))
  for step in range(1, 51):
    env.episode_length_buf[:] = step
    out = natural_persistent_low_smp(env)
    assert bool(out[0]) == (step == 50), step
    assert bool(out[1]) == (step >= 5), step
  env._smp_raw_err.zero_()
  assert not natural_persistent_low_smp(env).any()
  assert not env._natural_bad_count.any()
  env._smp_raw_err.fill_(1)
  for _ in range(24):
    assert not natural_persistent_low_smp(env)[0]
  assert natural_persistent_low_smp(env)[0]
  env.episode_length_buf.zero_()
  assert not natural_persistent_low_smp(env).any()
  bank=np.load(path); names=("supine","prone","left_side_down","right_side_down")
  masses=[float(bank["sampling_weights"][bank["labels"]==n].sum()) for n in names]
  assert np.allclose(masses,[.1,.2,.2,.5])
  train_groups={s.replace("__mirror","") for s in bank["clips"]}
  test=np.load(path.replace(".npz","_heldout.npz"))
  assert not train_groups & {s.replace("__mirror","") for s in test["clips"]}
  print("PASS: R1/R2 differ only in low-SMP timing; noise/DR retained; grace, persistence, recovery reset and unchanged GSI gate; sampling weights; clip/mirror holdout")


if __name__ == "__main__":
  main()
