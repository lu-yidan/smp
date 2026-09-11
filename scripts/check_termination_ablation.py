"""Fast semantic checks before scheduling expensive B0--B5 training."""

from dataclasses import asdict
from types import SimpleNamespace

import torch

from train_termination_ablation import ARMS, build_config
from smp.rl.tasks.getup.termination_ablation import (
  mixed_smp_too_low, mixed_stood_up, mixed_time_out, sample_episode_mode,
)


def main():
  reference = None
  for arm in ARMS:
    cfg, agent = build_config(arm)
    state = asdict(cfg)
    state.pop("episode_length_s")
    state.pop("terminations")
    state["events"].pop("sample_episode_mode", None)
    # Terrain defaults allocate a fresh, otherwise identical lambda per config.
    spec_fn = state["scene"]["terrain"]["spec_fn"]
    state["scene"]["terrain"]["spec_fn"] = (spec_fn.__module__, spec_fn.__qualname__)
    invariant = (state, asdict(agent))
    if reference is None:
      reference = invariant
    else:
      assert invariant == reference, f"Unintended config change in {arm}"
    assert cfg.terminations["time_out"].time_out
    if "stood_up" in cfg.terminations:
      assert cfg.terminations["stood_up"].time_out
    if "smp_too_low" in cfg.terminations:
      assert not cfg.terminations["smp_too_low"].time_out
    assert not cfg.terminations["unstable_sim_state"].time_out
  env = SimpleNamespace(num_envs=100000, device="cpu", cfg=SimpleNamespace(seed=20260911))
  rng_before = torch.random.get_rng_state().clone()
  sample_episode_mode(env)
  assert torch.equal(torch.random.get_rng_state(), rng_before)
  assert 0.19 < env._ablation_long.float().mean() < 0.21
  old_mode = env._ablation_long.clone()
  sample_episode_mode(env, torch.tensor([0, 2, 7]))
  mask = torch.ones(env.num_envs, dtype=torch.bool)
  mask[[0, 2, 7]] = False
  assert torch.equal(env._ablation_long[mask], old_mode[mask])
  env.num_envs = 4
  env._ablation_long = torch.tensor([False, True, False, True])
  env.step_dt = 0.02
  env.episode_length_buf = torch.tensor([249, 999, 250, 1000])
  assert mixed_time_out(env).tolist() == [False, False, True, True]
  env._smp_raw_err = torch.ones(4)
  assert mixed_smp_too_low(env, grace_steps=5).tolist() == [True, False, True, False]
  env.episode_length_buf.zero_()
  assert not mixed_smp_too_low(env, grace_steps=5).any()
  robot = SimpleNamespace(
    find_sites=lambda *args, **kwargs: ([0], ["head"]),
    data=SimpleNamespace(site_pos_w=torch.tensor([[[0., 0., 1.3]]] * 4),
                         root_link_lin_vel_w=torch.zeros(4, 3)),
  )
  env.scene = {"robot": robot}
  for _ in range(24):
    assert not mixed_stood_up(env, hold_steps=25).any()
  assert mixed_stood_up(env, hold_steps=25).tolist() == [True, False, True, False]
  print("PASS: invariant configs, truncation flags, 80/20 sampling, private RNG, partial resets, 5/20s boundaries, masked low-SMP and standing terminations")


if __name__ == "__main__":
  main()
