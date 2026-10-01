"""Four alpha=1 confirmatory interventions from R2; frozen A6 physics/reset."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml
import train_v33_path_ablation as legacy
from train_fixed_low_ablation import atomic_json
from train_r1_quality import verify_nested
from train_scratch_tradeoffs import tensor_hash
from smp.rl.tasks.getup import a6_confirmatory as ap

R2_SHA = '8b4889f80c6b7cc675f6e9b1e98f2d4a1886a15e372f86070f63c329ba9ca5d1'
REF_SHA = '870a1f7110d4cfd58a3d81d45e2ec728abbec7335a781032aedf3e36309e7df8'
ASSETS = {
  'outputs/multiterrain_bank/train.npz': '287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a',
  'outputs/multiterrain_bank/validation.npz': 'd0c4755474df9626a20167843eb453d00e76420771d29f34cd35d5a3a8d29b45',
  'datasets/reset_banks/natural_curriculum_v1/train.npz': 'e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9',
  'datasets/reset_banks/procedural_low_v1/train.npz': 'e289bed8d1c93e87fbdcf969fc5fc741f2d8500a0908ed145d2a4e02e7fe116d',
}


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_config(n, arm, evaluation=False, seed=20261013, stress_upper=1.):
  bank = f'outputs/multiterrain_bank/{"validation" if evaluation else "train"}.npz'
  cfg, agent = legacy.build_config(n, bank, evaluation, 'A6')
  # Historical runner seeds Python/NumPy/Torch explicitly from cfg.seed.
  # Preserve its agent config verbatim; every arm receives the same seed here.
  cfg.seed = seed
  ap.apply_intervention(cfg, arm)
  if evaluation:
    cfg.events['gsi_reset'].params['stress_upper'] = stress_upper
    cfg.metrics['paper_trial_loads'] = MetricsTermCfg(func=ap.sample_eval_loads, per_substep=True)
  return cfg, agent


def reward_contract(env, arm, out):
  opts = ap.settings(arm)
  params = env.cfg.rewards['task_smp_product'].params
  blocked = (env._mt_scene > 0) & ~env._mt_escaped
  actual = env.cfg.rewards['task_smp_product'].func(env, **params)
  expected = env._fixed_product * torch.where(blocked, opts['blocked_alpha'], 1.)
  assert torch.allclose(actual, expected, atol=1e-6)
  assert env.cfg.rewards['plate_geometry_progress'].weight == (.45 if opts['geometry'] else 0.)
  assert env.cfg.rewards['plate_clearance'].weight == (.08 if opts['geometry'] else 0.)
  assert env.cfg.rewards['plate_separation'].weight == (.01 if opts['geometry'] else 0.)
  assert env.cfg.rewards['path_quiet_feet'].weight == (-.03 if opts['quiet_feet'] else 0.)
  assert env.cfg.rewards['path_joint_stall'].weight == (-.20 if opts['joint_stall'] else 0.)
  assert env.cfg.rewards['plate_completion'].weight == .6
  assert env.cfg.rewards['plate_force'].weight == -.03
  assert env._pa_arm == 'A6'
  assert ('post_clear_anchor' in env.cfg.rewards) == opts['anchor']
  assert torch.equal(actual, env._fixed_product), 'Every confirmatory arm uses alpha=1'
  atomic_json(out/'reward_contract.json', {'pass':True, 'arm':arm, **opts})


class Runner(MjlabOnPolicyRunner):
  active = False

  def save(self, path, infos=None):
    env = self.env.unwrapped
    state = {'mean': env._smp_normalizer.mean.cpu(), 'count': env._smp_normalizer.count.cpu()}
    super().save(path, infos={**(infos or {}), 'a6_confirmatory': {'arm': self.args.arm, 'reference': state}})
    it = self.current_learning_iteration
    if not self.active or it == 0:
      return
    if it not in (5000, 10000, 15000, self.args.updates-1):
      return
    combined = {}
    for suite, mass in [('nominal', 1.), ('upper130', 1.3)]:
      out = self.args.out/'validation'/str(it)/suite
      out.mkdir(parents=True, exist_ok=True)
      cmd = [sys.executable, '-u', __file__, '--eval', '--arm', self.args.arm,
             '--checkpoint', str(Path(path).resolve()), '--out', str(out),
             '--num-envs', str(2048 if it in (10000, self.args.updates-1) else 192), '--steps', '1000', '--stress-upper', str(mass), '--seed', str(self.args.eval_seed)]
      if suite == 'nominal' and (it in (10000, self.args.updates-1) or it == self.args.updates-1):
        cmd += ['--video']
      with open('/tmp/smp-a6-paper-eval-gpu'+self.args.eval_gpu+'.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (out/'eval.log').open('w') as f:
          subprocess.run(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': self.args.eval_gpu},
                         stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, check=True)
      result = json.loads((out/'summary.json').read_text())
      combined[suite] = result
      for scene, vals in result.items():
        for key, val in vals.items():
          if isinstance(val, (int, float)):
            self.logger.writer.add_scalar(f'Validation/{suite}/{scene}/{key}', val, it)
    atomic_json(self.args.out/'progress.json', {'iteration': it, 'validation': combined})


def evaluate(env, wrapper, policy, obs, a):
  mt = legacy.mt
  ap.init_eval_loads(env)
  n, dev = env.num_envs, env.device
  hold = torch.zeros(n, device=dev); best = hold.clone()
  scene = env._mt_scene.clone(); direction = env._mt_direction.clone()
  first_clear = torch.full((n,), -1, device=dev, dtype=torch.long)
  first_stand = first_clear.clone(); refall = torch.zeros(n, device=dev, dtype=torch.bool)
  invalid = refall.clone(); seen = refall.clone()
  anchor = torch.zeros(n, 2, device=dev); drift = hold.clone(); samples = first_clear.clone().zero_()
  stand_anchor = anchor.clone(); stand_drift = hold.clone(); stand_samples = samples.clone()
  near_time = hold.clone(); near_before_stand = hold.clone(); near_after_stand = hold.clone()
  slip_before_stand = hold.clone(); slip_after_stand = hold.clone()
  contact_switches = hold.clone(); slip = hold.clone()
  last_contact = torch.zeros(n, 2, device=dev, dtype=torch.bool)
  selected = [int(torch.where((scene == s) & (direction == d))[0][0]) for s in range(3) for d in range(4)]
  writers = {}
  if a.video:
    import imageio.v2 as imageio
    for s, name in enumerate(('flat', 'guided_plate', 'free_plate')):
      writers[s] = imageio.get_writer(str(a.out/(name+'_20s.mp4')), fps=25, codec='libx264', quality=7)
  try:
    for step in range(a.steps):
      with torch.inference_mode():
        obs, _, done, _ = wrapper.step(policy(obs))
      alive_before = env._ap_alive.clone()
      env._ap_alive &= ~done.bool(); live = env._ap_alive
      invalid |= env._mt_invalid & alive_before
      clear = ((scene == 0) | env._mt_escaped) & live & ~invalid
      newly = clear & (first_clear < 0)
      first_clear[newly] = step
      xy = env.scene['robot'].data.root_link_pos_w[:, :2]
      anchor[newly] = xy[newly]
      window = live & (first_clear >= 0) & (step > first_clear) & (step <= first_clear+150)
      drift = torch.maximum(drift, (xy-anchor).norm(dim=-1)*window)
      samples += window.long()
      stable = env._r_stable & clear
      hold = torch.where(stable, hold+env.step_dt, 0.); best = torch.maximum(best, hold)
      first = (hold >= 1.-1e-4) & (first_stand < 0); first_stand[first] = step
      stand_anchor[first] = xy[first]
      stand_window = live & (first_stand >= 0) & (step > first_stand) & (step <= first_stand+150)
      stand_drift = torch.maximum(stand_drift, (xy-stand_anchor).norm(dim=-1)*stand_window)
      stand_samples += stand_window.long()
      seen |= hold >= 1.-1e-4
      r = env.scene['robot']; z = mt.height(env); u = -r.data.projected_gravity_b[:, 2]
      refall |= seen & live & ((z < .65) | (u < .5))
      near = live & (z > .78) & (u > .7)
      contacts = mt.ground_force(env, 'quality_feet')[..., 2].abs() > 20
      if step:
        contact_switches += (contacts != last_contact).sum(-1)*near
      slip_step = (r.data.body_link_lin_vel_w[:, env._r_feet, :2].norm(dim=-1)*contacts).sum(-1)*near*env.step_dt
      slip += slip_step
      near_time += near*env.step_dt
      near_before_stand += (near & ~seen)*env.step_dt
      near_after_stand += (near & seen)*env.step_dt
      slip_before_stand += slip_step * ~seen
      slip_after_stand += slip_step * seen
      last_contact = contacts.clone()
      if writers and step % 2 == 0:
        from PIL import Image, ImageDraw
        for s, writer in writers.items():
          tiles = []
          for j in selected[s*4:s*4+4]:
            env.cfg.viewer.env_idx = j
            im = Image.fromarray(env.render()) if live[j] else Image.new('RGB', (640, 480), 'black')
            d = ImageDraw.Draw(im); d.rectangle((0, 0, 640, 30), fill='black')
            d.text((5, 6), f'{a.arm} dir={int(direction[j])} t={step*.02:.2f} hold={float(hold[j]):.2f}', fill='white')
            tiles.append(np.asarray(im))
          writer.append_data(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0))
  finally:
    for writer in writers.values(): writer.close()
  summary = {}
  masks = {name: scene == i for i, name in enumerate(('flat', 'guided_plate', 'free_plate'))}
  for name, mask in list(masks.items()):
    for d, label in enumerate(('supine', 'prone', 'left', 'right')):
      masks[name+'/'+label] = mask & (direction == d)
  for name, mask in masks.items():
    vals = {'n': int(mask.sum()), 'cleared': float((first_clear[mask]>=0).float().mean()),
            'stable_1s': float((best[mask]>=1.-1e-4).float().mean()),
            'stable_10s': float((best[mask]>=10.-1e-4).float().mean()),
            'refall': float(refall[mask].float().mean()), 'terminated': float((~env._ap_alive[mask]).float().mean()),
            'near_upright_contact_switches_mean': float(contact_switches[mask].mean()),
            'near_upright_foot_slip_m_mean': float(slip[mask].mean())}
    for k, label in enumerate(('tau', 'speed', 'power')):
      peaks = env._ap_peaks[mask, :, k]
      vals[label+'_peak_p95'] = float(torch.quantile(peaks.amax(-1), .95))
      vals[label+'_joint_peak_p95'] = torch.quantile(peaks, .95, dim=0).cpu().tolist()
    vals['high_load_low_speed_duration_p95'] = float(torch.quantile(env._ap_stall_time[mask].amax(-1), .95))
    vals['longest_high_load_duration_p95'] = float(torch.quantile(env._ap_high_longest[mask].amax(-1), .95))
    valid = mask & (samples == 150); vals['post_clear_3s_n'] = int(valid.sum())
    vals['post_clear_max_offset_p95'] = float(torch.quantile(drift[valid], .95)) if valid.any() else None
    valid_stand = mask & (stand_samples == 150)
    vals['post_stand_3s_n'] = int(valid_stand.sum())
    vals['post_stand_max_offset_p95'] = float(torch.quantile(stand_drift[valid_stand], .95)) if valid_stand.any() else None
    vals['near_upright_exposure_s_mean'] = float(near_time[mask].mean())
    vals['near_upright_foot_travel_per_s'] = float(slip[mask].sum()/near_time[mask].sum()) if near_time[mask].sum() > 0 else None
    for key, steps in [('clear', first_clear), ('stand1s', first_stand)]:
      ok = mask & (steps >= 0); vals[key+'_time_n'] = int(ok.sum())
      vals[key+'_time_median_s'] = float(torch.quantile((steps[ok]+1).float()*.02, .5)) if ok.any() else None
    summary[name] = vals
  atomic_json(a.out/'summary.json', summary)
  np.savez_compressed(a.out/'per_trial.npz', scene=scene.cpu(), direction=direction.cpu(), best_hold=best.cpu(),
      first_clear=first_clear.cpu(), first_stand=first_stand.cpu(), refall=refall.cpu(), alive=env._ap_alive.cpu(),
      peaks=env._ap_peaks.cpu(), stall_time=env._ap_stall_time.cpu(), high_load_longest=env._ap_high_longest.cpu(),
      post_clear_max_offset=drift.cpu(), post_clear_samples=samples.cpu(), contact_switches=contact_switches.cpu(), foot_slip=slip.cpu(),
      post_stand_max_offset=stand_drift.cpu(), post_stand_samples=stand_samples.cpu(),
      near_time=near_time.cpu(), near_before_stand=near_before_stand.cpu(), near_after_stand=near_after_stand.cpu(),
      slip_before_stand=slip_before_stand.cpu(), slip_after_stand=slip_after_stand.cpu())
  print('EVALUATION_COMPLETE', flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--arm', choices=ap.ARMS, required=True); p.add_argument('--out', type=Path, required=True)
  p.add_argument('--checkpoint', type=Path, default=Path('datasets/checkpoints/R2_9000.pt'))
  p.add_argument('--num-envs', type=int, default=4096); p.add_argument('--updates', type=int, default=20000)
  p.add_argument('--seed', type=int, default=20261013); p.add_argument('--eval-gpu', default='7')
  p.add_argument('--eval-seed', type=int, default=20261013)
  p.add_argument('--eval', action='store_true'); p.add_argument('--video', action='store_true')
  p.add_argument('--preflight', action='store_true'); p.add_argument('--steps', type=int, default=1000)
  p.add_argument('--stress-upper', type=float, default=1.); a = p.parse_args()
  for path, expected in ASSETS.items(): assert sha(path) == expected, path
  reference = Path('configs/a6_paper/smp_reward_reference_v1.json'); assert sha(reference) == REF_SHA
  if not a.eval: assert sha(a.checkpoint) == R2_SHA
  a.out = a.out.resolve(); a.checkpoint = a.checkpoint.resolve(); a.out.mkdir(parents=True, exist_ok=a.eval)
  cfg, agent = build_config(a.num_envs, a.arm, a.eval, a.seed, a.stress_upper)
  random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
  agent.logger = 'tensorboard' if a.eval or a.preflight else 'wandb'
  agent.upload_model = False; agent.run_name = f'{a.arm}_s{a.seed}'; agent.wandb_project = 'smp'
  agent.save_interval = 500; agent.max_iterations = a.updates
  dump_yaml(a.out/'env.yaml', asdict(cfg)); dump_yaml(a.out/'agent.yaml', asdict(agent)); env = None
  try:
    env = ManagerBasedRlEnv(cfg, device='cuda:0', render_mode='rgb_array' if a.video else None)
    wrapper = legacy.Wrapper(env, clip_actions=agent.clip_actions)
    runner = Runner(wrapper, asdict(agent), str(a.out), 'cuda:0'); runner.args = a
    parent = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    critic = tensor_hash(runner.alg.save()['critic_state_dict'])
    runner.load(str(a.checkpoint), load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
    verify_nested(parent['actor_state_dict'], runner.alg.save()['actor_state_dict'])
    assert critic == tensor_hash(runner.alg.save()['critic_state_dict'])
    ref = json.loads(reference.read_text())
    env._smp_normalizer.mean.copy_(torch.tensor(ref['mean'], device=env.device))
    env._smp_normalizer.count.copy_(torch.tensor(ref['count'], device=env.device))
    env.common_step_counter = 0
    random.seed(cfg.seed+177); np.random.seed(cfg.seed+177); torch.manual_seed(cfg.seed+177)
    obs, _ = env.reset(); legacy.v.init(env); env._v_start_counter = -12000; env._v_ramp_updates = 500
    assert obs['actor'].shape[-1] == 93 and obs['critic'].shape[-1] == 960
    assert 'smp_too_low' not in cfg.terminations
    # Frozen A6 keeps this key for logging, but its function always returns false.
    stood = cfg.terminations.get('stood_up')
    if stood is not None:
      from smp.rl.tasks.getup.scratch_tradeoffs import no_stand_termination
      assert stood.func is no_stand_termination
      assert not stood.func(env, **stood.params).any()
    if not a.eval: assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
    legacy.audit_reset(env, a.out)
    if a.preflight:
      atomic_json(a.out/'dynamics_audit.json', legacy.bd.audit_dynamics(env))
      obs, _ = env.reset(); ids = torch.arange(0, env.num_envs, 8, device=env.device)
      mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device); mask[ids] = False
      before = env.sim.data.qpos[mask].clone(); mass = env.sim.model.body_mass[mask].clone()
      gain = env._bd_gain[mask].clone(); env._reset_idx(ids)
      assert torch.equal(before, env.sim.data.qpos[mask]) and torch.equal(mass, env.sim.model.body_mass[mask])
      assert torch.equal(gain, env._bd_gain[mask])
      # Reset clears only selected anchor states, never other worlds.
      env._ac_anchor_set[:] = True; env._ac_anchor[:] = 123.
      env._reset_idx(ids)
      assert not env._ac_anchor_set[ids].any() and env._ac_anchor_set[mask].all()
      assert (env._ac_anchor[mask] == 123.).all() and (env._ac_anchor[ids] == 0.).all()
      obs, _ = env.reset(); reward_contract(env, a.arm, a.out)
      atomic_json(a.out/'partial_reset_pass.json', {'pass':True})
    code = sorted([Path(__file__), *Path('src/smp').rglob('*.py')])
    meta = {'arm':a.arm,'historical_base_commit':'1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2',
      'code_sha256':hashlib.sha256(b''.join(f.read_bytes() for f in code)).hexdigest(),
      'checkpoint_sha256':sha(a.checkpoint),'reference_sha256':sha(reference),'assets':ASSETS,
      'actor_exact':True,'fresh_critic':True,'fresh_optimizer':True,
      'actor_sha':tensor_hash(runner.alg.save()['actor_state_dict']),'critic_sha':critic,
      'initial_qpos_sha':hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),
      'seed':cfg.seed,'updates':a.updates,'num_envs':a.num_envs,'episode_s':cfg.episode_length_s,
      'counts':legacy.reset_counts(env),'events':list(cfg.events),'actor_noise':cfg.observations['actor'].enable_corruption,
      'weights':{k:v.weight for k,v in cfg.rewards.items()}, 'intervention':ap.settings(a.arm),
      'hand_gate_disabled':False,'preflight':a.preflight,'evaluation':a.eval}
    atomic_json(a.out/'launch.json', meta); print('TRANSFER_AND_RESET_VERIFIED', json.dumps(meta), flush=True)
    if a.eval:
      evaluate(env, wrapper, runner.get_inference_policy(), obs, a)
    else:
      runner.save(str(a.out/'initial.pt')); runner.active = not a.preflight
      runner.learn(num_learning_iterations=a.updates, init_at_random_ep_len=False)
      runner.active = False; runner.save(str(a.out/'final.pt'))
      atomic_json(a.out/'completed.json', {'iteration':runner.current_learning_iteration})
  except BaseException as e:
    atomic_json(a.out/'failed.json', {'error':repr(e)}); raise
  finally:
    if env is not None: env.close()


if __name__ == '__main__': main()
