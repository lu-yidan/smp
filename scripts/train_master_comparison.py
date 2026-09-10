"""Master 0e67286 baseline versus a deployment-model/93D adaptation."""
import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from smp.rl.events import gsi_refresh
from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.getup.getup_env_cfg import g1_getup_smp_env_cfg
from smp.rl.tasks.getup.master_deployment_contract import apply_deployment_contract


@requires_model_fields('pair_friction')
def mirror_foot_friction_to_pairs(env, env_ids=None):
  """Explicit deployment pairs override geoms: propagate original foot DR values."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  m = env.sim.mj_model
  if m.npair != 14:
    raise ValueError('Expected 14 foot-ground pairs')
  # Each pair contains one foot geom and the terrain geom.
  foot = [int(a if m.geom_bodyid[a] != 0 else b) for a, b in zip(m.pair_geom1, m.pair_geom2, strict=True)]
  mu = env.sim.model.geom_friction[env_ids[:, None], torch.tensor(foot, device=env.device)[None, :], 0]
  env.sim.model.pair_friction[env_ids, :, 0] = mu
  env.sim.model.pair_friction[env_ids, :, 1] = mu


class LoggedWrapper(RslRlVecEnvWrapper):
  def step(self, actions):
    obs, rewards, dones, extras = super().step(actions)
    robot = self.unwrapped.scene['robot']
    env = self.unwrapped
    head = robot.find_sites(['head'], preserve_order=True)[0][0]
    extras.setdefault('log', {}).update({
      'Recovery/head_height': robot.data.site_pos_w[:, head, 2].mean().detach(),
      'Recovery/raw_smp': torch.exp(-6 * env._smp_raw_err).mean().detach(),
      'Recovery/gsi_pool_head': float(getattr(env, '_smp_gsi_head', 0)),
    })
    for name in ['smp_too_low', 'stood_up', 'time_out']:
      extras['log']['Termination/' + name] = env.termination_manager.get_term(name).float().mean().detach()
    return obs, rewards, dones, extras


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--arm', choices=['master96', 'deploy93'], required=True)
  p.add_argument('--num-envs', type=int, default=4096)
  p.add_argument('--iterations', type=int, default=10000)
  p.add_argument('--log-dir', type=Path, required=True)
  p.add_argument('--preflight', action='store_true')
  a = p.parse_args()
  cfg, agent = g1_getup_smp_env_cfg(), unitree_g1_smp_ppo_runner_cfg()
  cfg.seed = agent.seed = 20260910
  cfg.scene.num_envs = a.num_envs
  if a.arm == 'deploy93':
    cfg = apply_deployment_contract(cfg)
    cfg.observations['actor'].terms.pop('base_lin_vel')
    cfg.events['deployment_pair_friction'] = EventTermCfg(func=mirror_foot_friction_to_pairs, mode='startup')
  random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
  assert cfg.episode_length_s == 5
  assert cfg.events['gsi_refresh'].params == {'num_samples': 1024, 'step_interval': 2400}
  assert agent.actor.distribution_cfg['learn_std'] is False
  prior = Path(cfg.events['init_smp_state'].params['ckpt_path']).resolve()
  cfg.events['init_smp_state'].params['ckpt_path'] = str(prior)
  agent.max_iterations = a.iterations
  agent.save_interval = 500
  agent.logger = 'tensorboard' if a.preflight else 'wandb'
  agent.upload_model = False
  a.log_dir.mkdir(parents=True, exist_ok=False)
  metadata = dict(arm=a.arm, master_reference='0e67286', seed=cfg.seed, num_envs=a.num_envs, iterations=a.iterations, policy_checkpoint_loaded=False, prior_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(), intentional_overrides=['4096 envs', '10000 update budget', 'seed 20260910', 'W&B and observational diagnostics'], deployment_adaptations=['robot model/controller/contact solver', '93D actor', 'explicit-pair friction propagation'] if a.arm == 'deploy93' else [])
  (a.log_dir/'launch.json').write_text(json.dumps(metadata, indent=2))
  dump_yaml(a.log_dir/'params/env.yaml', asdict(cfg)); dump_yaml(a.log_dir/'params/agent.yaml', asdict(agent))
  env = ManagerBasedRlEnv(cfg, device='cuda:0')
  try:
    runner = MjlabOnPolicyRunner(LoggedWrapper(env, clip_actions=agent.clip_actions), asdict(agent), str(a.log_dir), 'cuda:0')
    obs, _ = env.reset()
    assert obs['actor'].shape[-1] == (96 if a.arm == 'master96' else 93)
    assert obs['critic'].shape[-1] == 960
    assert env._smp_gsi_pool.shape[0] == 4096
    if a.preflight:
      old_head = int(getattr(env, '_smp_gsi_head', 0))
      env.common_step_counter = 2400
      gsi_refresh(env)
      assert int(env._smp_gsi_head) == (old_head + 1024) % 4096
      env.common_step_counter = 0
      print('GSI_REFRESH_VERIFIED', flush=True)
    runner.save(str(a.log_dir/'random_initial.pt'))
    print('CONFIG_VERIFIED', metadata, flush=True)
    runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=False)
  finally:
    env.close()


if __name__ == '__main__':
  main()
