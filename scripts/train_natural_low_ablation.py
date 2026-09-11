"""R1/R2: B1 from scratch + 10% natural low resets, with/without low-SMP patience."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml

from train_termination_ablation import build_config as baseline_config, AblationWrapper
from smp.rl.events import gsi_refresh
from smp.rl.tasks.getup.natural_low_reset import natural_mixed_reset, natural_persistent_low_smp


def build_config(arm, bank_path, num_envs=4096, seed=20260911):
  cfg, agent = baseline_config("B1", num_envs, seed)
  cfg.events["gsi_reset"] = EventTermCfg(func=natural_mixed_reset, mode="reset",
    params={"bank_path": str(Path(bank_path).resolve()), "probability": 0.1})
  if arm == "R2":
    cfg.terminations["smp_too_low"].func = natural_persistent_low_smp
    cfg.terminations["smp_too_low"].params.update(natural_grace_seconds=0.5, natural_hold_seconds=0.5)
  elif arm != "R1":
    raise ValueError(arm)
  return cfg, agent


class NaturalLoggedWrapper(AblationWrapper):
  def step(self, actions):
    env = self.unwrapped
    before = env._natural_type.clone()
    obs, reward, dones, extras = super().step(actions)
    natural = before >= 0
    logs = extras["log"]
    if not hasattr(self, "_natural_steps"):
      self._natural_steps = torch.zeros((), device=env.device, dtype=torch.long)
      self._all_steps = 0
    self._natural_steps += natural.sum()
    self._all_steps += env.num_envs
    counts = env._natural_draw_counts
    logs["Natural/episode_draw_fraction"] = counts[1:].sum() / counts.sum().clamp_min(1)
    logs["Natural/step_fraction"] = natural.float().mean()
    logs["Natural/step_fraction_cumulative"] = self._natural_steps / self._all_steps
    for i, name in enumerate(("supine", "prone", "left", "right")):
      logs["Natural/draw_fraction_" + name] = counts[i+1] / counts.sum().clamp_min(1)
    for source, mask in (("natural", natural), ("gsi", ~natural)):
      denom = mask.sum().clamp_min(1)
      logs[f"Natural/{source}_low_smp_end_rate"] = (
        env.termination_manager.get_term("smp_too_low") & mask).sum() / denom
      logs[f"Natural/{source}_stood_end_rate"] = (
        env.termination_manager.get_term("stood_up") & mask).sum() / denom
    return obs, reward, dones, extras


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--arm", choices=("R1", "R2"), required=True)
  p.add_argument("--bank", type=Path, default=Path("datasets/reset_banks/natural_low_right_v1.npz"))
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--iterations", type=int, default=10000)
  p.add_argument("--seed", type=int, default=20260911)
  p.add_argument("--log-dir", type=Path, required=True)
  p.add_argument("--preflight", action="store_true")
  a = p.parse_args()
  cfg, agent = build_config(a.arm, a.bank, a.num_envs, a.seed)
  random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
  prior = Path(cfg.events["init_smp_state"].params["ckpt_path"]).resolve()
  prior_hash = hashlib.sha256(prior.read_bytes()).hexdigest()
  assert prior_hash == "9439d9d9f58940f5472015a57da27c72e2305aafbd29cfd9be44f93da675cc59"
  cfg.events["init_smp_state"].params["ckpt_path"] = str(prior)
  agent.max_iterations = a.iterations; agent.save_interval = 500
  agent.logger = "tensorboard" if a.preflight else "wandb"
  agent.upload_model = False
  agent.run_name = f"scratch93_{a.arm}_natural_low_seed{a.seed}"
  a.log_dir.mkdir(parents=True, exist_ok=False)
  metadata = dict(arm=a.arm, baseline="B1", seed=a.seed, num_envs=a.num_envs,
    iterations=a.iterations, policy_checkpoint_loaded=False, episode_seconds=10,
    prior_sha256=prior_hash, bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(),
    natural_reset_probability=0.1, natural_direction_probability=dict(supine=0.1, prone=0.2, left=0.2, right=0.5),
    natural_velocity="zero; repeated static SMP history", low_smp_threshold=0.02,
    natural_grace_seconds=0.5 if a.arm == "R2" else 0.1,
    natural_bad_hold_seconds=0.5 if a.arm == "R2" else 0.0,
    gsi_termination="original 5-step grace, single low score",
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
  (a.log_dir / "launch.json").write_text(json.dumps(metadata, indent=2))
  dump_yaml(a.log_dir / "params/env.yaml", asdict(cfg))
  dump_yaml(a.log_dir / "params/agent.yaml", asdict(agent))
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    runner = MjlabOnPolicyRunner(NaturalLoggedWrapper(env, clip_actions=agent.clip_actions),
      asdict(agent), str(a.log_dir), "cuda:0")
    obs, _ = env.reset()
    assert obs["actor"].shape[-1] == 93 and obs["critic"].shape[-1] == 960
    assert env._smp_gsi_pool.shape[0] == 4096
    if a.preflight:
      natural = env._natural_type >= 0
      assert natural.any()
      assert env.sim.data.qvel[natural].abs().max() < 1e-6
      assert env.scene["robot"].data.root_link_pos_w[natural, 2].max() < 0.3
      print("NATURAL_RESET_VERIFIED", int(natural.sum()), flush=True)
      old_head = int(getattr(env, "_smp_gsi_head", 0))
      env.common_step_counter = 2400; gsi_refresh(env)
      assert int(env._smp_gsi_head) == (old_head + 1024) % 4096
      env.common_step_counter = 0
      print("GSI_REFRESH_VERIFIED", flush=True)
    runner.save(str(a.log_dir / "random_initial.pt"))
    print("CONFIG_VERIFIED", metadata, flush=True)
    runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=False)
  finally:
    env.close()


if __name__ == "__main__":
  main()
