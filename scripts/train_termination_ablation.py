"""B0--B5: controlled from-scratch deployment-93D recovery experiments."""

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml

from train_master_comparison import LoggedWrapper, mirror_foot_friction_to_pairs
from smp.rl.events import gsi_refresh
from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.getup.continuous_recovery import unstable_sim_state
from smp.rl.tasks.getup.getup_env_cfg import g1_getup_smp_env_cfg
from smp.rl.tasks.getup.master_deployment_contract import apply_deployment_contract
from smp.rl.tasks.getup.termination_ablation import (
  mixed_smp_too_low, mixed_stood_up, mixed_time_out, sample_episode_mode,
)

ARMS = {
  "B0": (5, True, True),
  "B1": (10, True, True),
  "B2": (10, False, True),
  "B3": (10, True, False),
  "B4": (10, False, False),
  "B5": (20, True, True),
}


def build_config(arm, num_envs=4096, seed=20260911):
  cfg = apply_deployment_contract(g1_getup_smp_env_cfg())
  agent = unitree_g1_smp_ppo_runner_cfg()
  cfg.seed = agent.seed = seed
  cfg.scene.num_envs = num_envs
  cfg.observations["actor"].terms.pop("base_lin_vel")
  cfg.events["deployment_pair_friction"] = EventTermCfg(
    func=mirror_foot_friction_to_pairs, mode="startup"
  )
  seconds, stand, low = ARMS[arm]
  cfg.episode_length_s = seconds
  if not stand:
    cfg.terminations.pop("stood_up")
  if not low:
    cfg.terminations.pop("smp_too_low")
  # Common guard for all six arms; reported separately from task terminations.
  cfg.terminations["unstable_sim_state"] = TerminationTermCfg(func=unstable_sim_state)
  if arm == "B5":
    cfg.events["sample_episode_mode"] = EventTermCfg(
      func=sample_episode_mode, mode="reset", params={"long_probability": 0.2}
    )
    cfg.terminations["time_out"].func = mixed_time_out
    cfg.terminations["time_out"].params = {"short_seconds": 5.0, "long_seconds": 20.0}
    cfg.terminations["stood_up"].func = mixed_stood_up
    cfg.terminations["smp_too_low"].func = mixed_smp_too_low
  return cfg, agent


class AblationWrapper(LoggedWrapper):
  def step(self, actions):
    env = self.unwrapped
    mode = getattr(env, "_ablation_long", None)
    if mode is not None:
      mode = mode.clone()  # Reset below samples the NEXT episode's mode.
    obs, rewards, dones, extras = super().step(actions)
    logs = extras.setdefault("log", {})
    robot = env.scene["robot"]
    head = robot.find_sites(["head"], preserve_order=True)[0][0]
    valid = ~dones.bool()
    denom = valid.sum().clamp_min(1)
    low = robot.data.site_pos_w[:, head, 2] < 0.85
    spinning = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=-1) > 2.0
    logs["Diagnostic/low_height_fraction_nonterminal"] = (low & valid).sum() / denom
    logs["Diagnostic/low_and_rotating_fraction_nonterminal"] = (
      low & spinning & valid
    ).sum() / denom
    logs["Diagnostic/nonterminal_fraction"] = valid.float().mean()
    if mode is not None:
      if not hasattr(self, "_long_steps"):
        self._long_steps = torch.zeros((), dtype=torch.long, device=env.device)
        self._total_steps = 0
      self._long_steps += mode.sum()
      self._total_steps += env.num_envs
      logs["Mixture/long_step_fraction"] = mode.float().mean()
      logs["Mixture/long_step_fraction_cumulative"] = self._long_steps / self._total_steps
      draws = env._ablation_draws
      logs["Mixture/long_episode_draw_fraction_cumulative"] = draws[1] / draws.sum().clamp_min(1)
      logs["Mixture/long_episode_end_rate"] = (dones.bool() & mode).float().mean()
      logs["Mixture/short_episode_end_rate"] = (dones.bool() & ~mode).float().mean()
    return obs, rewards, dones, extras


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--arm", choices=ARMS, required=True)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--iterations", type=int, default=10000)
  p.add_argument("--seed", type=int, default=20260911)
  p.add_argument("--log-dir", type=Path, required=True)
  p.add_argument("--preflight", action="store_true")
  a = p.parse_args()
  cfg, agent = build_config(a.arm, a.num_envs, a.seed)
  random.seed(a.seed)
  np.random.seed(a.seed)
  torch.manual_seed(a.seed)
  prior = Path(cfg.events["init_smp_state"].params["ckpt_path"]).resolve()
  prior_sha = hashlib.sha256(prior.read_bytes()).hexdigest()
  assert prior_sha == "9439d9d9f58940f5472015a57da27c72e2305aafbd29cfd9be44f93da675cc59"
  cfg.events["init_smp_state"].params["ckpt_path"] = str(prior)
  assert cfg.events["gsi_refresh"].params == {"num_samples": 1024, "step_interval": 2400}
  assert cfg.observations["actor"].enable_corruption
  assert agent.actor.distribution_cfg == {
    "class_name": "GaussianDistribution", "init_std": 0.3,
    "std_type": "scalar", "learn_std": False,
  }
  agent.max_iterations = a.iterations
  agent.save_interval = 500
  agent.logger = "tensorboard" if a.preflight else "wandb"
  agent.upload_model = False
  agent.run_name = f"scratch93_{a.arm}_seed{a.seed}"
  a.log_dir.mkdir(parents=True, exist_ok=False)
  metadata = dict(
    arm=a.arm, seed=a.seed, num_envs=a.num_envs, iterations=a.iterations,
    policy_checkpoint_loaded=False, prior_sha256=prior_sha,
    episode_seconds=cfg.episode_length_s, terminations=list(cfg.terminations),
    mixture={"short_probability": 0.8, "short_seconds": 5, "long_seconds": 20,
             "sampling": "independent per reset, private RNG", "mode_observed": False}
    if a.arm == "B5" else None,
    common_guard="unstable_sim_state (added to all arms, including B0)",
    reference="deployment93/5s, master 0e67286 + deployment contract",
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
  )
  (a.log_dir / "launch.json").write_text(json.dumps(metadata, indent=2))
  dump_yaml(a.log_dir / "params/env.yaml", asdict(cfg))
  dump_yaml(a.log_dir / "params/agent.yaml", asdict(agent))
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    runner = MjlabOnPolicyRunner(
      AblationWrapper(env, clip_actions=agent.clip_actions), asdict(agent),
      str(a.log_dir), "cuda:0",
    )
    obs, _ = env.reset()
    assert obs["actor"].shape[-1] == 93 and obs["critic"].shape[-1] == 960
    assert env._smp_gsi_pool.shape[0] == 4096
    if a.preflight:
      old_head = int(getattr(env, "_smp_gsi_head", 0))
      env.common_step_counter = 2400
      gsi_refresh(env)
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
