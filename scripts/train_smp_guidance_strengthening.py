"""Matched SMP adaptation with shared bounded anti-stagnation auxiliaries."""
import argparse
import contextlib
import io
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml
import train_a6_confirmatory as base
from train_scratch_tradeoffs import tensor_hash
from smp.rl.tasks.getup import guidance_strengthening as sg


def build_config(n, arm, seed):
    old_arm = 'C20_full' if arm == 'on' else 'C20_no_geometry'
    cfg, agent = base.build_config(n, old_arm, seed=seed)
    cfg.events['gsi_reset'].func = sg.reset
    cfg.rewards['aux_plate_no_progress'] = RewardTermCfg(func=sg.reward, params={'index':0}, weight=-.15)
    cfg.rewards['aux_prone_lateral'] = RewardTermCfg(func=sg.reward, params={'index':1}, weight=.20)
    assert 'smp_too_low' not in cfg.terminations
    assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
    return cfg, agent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Wrapper(base.legacy.Wrapper):
    def step(self, actions):
        obs, reward, done, extra = super().step(actions)
        env = self.unwrapped
        sg.update(env)
        log = extra.setdefault('log', {})
        blocked = (env._mt_scene > 0) & ~env._mt_escaped & ~env._mt_invalid
        if blocked.any():
            for label, values in [('smp', env._fixed_smp_score), ('task', env._fixed_task_score),
                                  ('stagnation_s', env._sg_timer), ('stall_cost', env._sg_stall),
                                  ('lateral_progress', env._sg_progress)]:
                log['Blocked/'+label] = values[blocked].mean().detach()
            log['Blocked/smp_below_002'] = (env._fixed_smp_score[blocked] < .02).float().mean()
        return obs, reward, done, extra


class Runner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        env = self.env.unwrapped
        state = {'mean': env._smp_normalizer.mean.cpu(), 'count': env._smp_normalizer.count.cpu()}
        super().save(path, infos={**(infos or {}), 'a6_confirmatory': {'arm': self.arm, 'reference': state}})
        base.atomic_json(self.out/'progress.json', {'iteration': self.current_learning_iteration,
                         'checkpoint': str(path), 'elapsed_s': time.time()-self.started})
        it = self.current_learning_iteration
        if self.args.preflight or Path(path).name != f'model_{it}.pt' or it not in (3000,5000,9999):
            return
        out = self.out/'validation'/str(it)
        out.parent.mkdir(parents=True,exist_ok=True)
        old_arm = 'C20_full' if self.arm == 'on' else 'C20_no_geometry'
        cmd = [sys.executable,'-u','scripts/evaluate_free_robustness.py','--family','confirmatory',
               '--case','mixed','--eval','--arm',old_arm,'--checkpoint',str(Path(path).resolve()),
               '--out',str(out.resolve()),'--num-envs','2048','--steps','1150','--seed','20261013']
        with (out.parent/(str(it)+'.log')).open('w') as log:
            rc = subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        base.atomic_json(out.parent/(str(it)+'.status.json'),dict(returncode=rc,command=cmd))
        if rc != 0:
            print('VALIDATION_FAILED',it,rc,flush=True)
            return
        results=json.loads((out/'sr3_summary.json').read_text())
        for scene in ('flat','guided_plate','free_plate'):
            self.logger.writer.add_scalar('Validation_SR3/'+scene,results[scene]['sr3'],it)
        print('VALIDATION_COMPLETE',it,json.dumps(results),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('off','on'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('datasets/checkpoints/R2_9000.pt'))
    p.add_argument('--seed', type=int, default=20261006)
    p.add_argument('--updates', type=int, default=10000)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--wandb', choices=('online','offline','disabled'), default='online')
    p.add_argument('--calibrate-steps', type=int, default=0)
    a = p.parse_args()
    assert a.updates > 0
    a.out.mkdir(parents=True, exist_ok=False)
    assert sha(a.checkpoint) == base.R2_SHA
    for path, expected in base.ASSETS.items(): assert sha(path) == expected, path
    reference = Path('configs/a6_paper/smp_reward_reference_v1.json')
    assert sha(reference) == base.REF_SHA
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg, agent = build_config(a.num_envs, a.arm, a.seed)
    agent.logger = 'tensorboard'; agent.upload_model = False
    agent.save_interval = 500; agent.max_iterations = a.updates
    agent.run_name = f'SG_{a.arm}_seed{a.seed}'
    dump_yaml(a.out/'env.yaml', asdict(cfg)); dump_yaml(a.out/'agent.yaml', asdict(agent))
    wb = None; env = None
    try:
        env = ManagerBasedRlEnv(cfg, device='cuda:0')
        wrapper = Wrapper(env, clip_actions=agent.clip_actions)
        runner = Runner(wrapper, asdict(agent), str(a.out), 'cuda:0')
        runner.arm = a.arm; runner.out = a.out; runner.started = time.time(); runner.args = a
        parent = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        critic = tensor_hash(runner.alg.save()['critic_state_dict'])
        runner.load(str(a.checkpoint), load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
        base.verify_nested(parent['actor_state_dict'], runner.alg.save()['actor_state_dict'])
        assert critic == tensor_hash(runner.alg.save()['critic_state_dict'])
        ref = json.loads(reference.read_text())
        env._smp_normalizer.mean.copy_(torch.tensor(ref['mean'], device=env.device))
        env._smp_normalizer.count.copy_(torch.tensor(ref['count'], device=env.device))
        env.common_step_counter = 0
        random.seed(a.seed+177); np.random.seed(a.seed+177); torch.manual_seed(a.seed+177)
        obs, _ = env.reset(); base.legacy.v.init(env)
        env._v_start_counter = -12000; env._v_ramp_updates = 500
        assert obs['actor'].shape[-1] == 93 and obs['critic'].shape[-1] == 960
        assert cfg.scale_rewards_by_dt and abs(env.step_dt-.02) < 1e-8
        base.legacy.audit_reset(env, a.out)
        scene, direction = sg.scene_layout(a.num_envs)
        assert np.array_equal(env._mt_scene.cpu().numpy(), scene)
        assert np.array_equal(env._mt_direction.cpu().numpy(), direction)
        launch = dict(arm=a.arm, seed=a.seed, updates=a.updates, num_envs=a.num_envs,
                      checkpoint_sha256=sha(a.checkpoint), reference_sha256=sha(reference),
                      actor_sha=tensor_hash(runner.alg.save()['actor_state_dict']), critic_sha=critic,
                      actor_exact=True, fresh_critic=True, fresh_optimizer=True,
                      scene_counts=np.bincount(scene).tolist(), scene_weights=[.3,.5,.2],
                      counts=base.legacy.reset_counts(env), control_dt=env.step_dt,
                      episode_s=cfg.episode_length_s, initial_qpos_sha=sha(a.out/'initial.npz'),
                      events=list(cfg.events), actor_noise=True, low_smp_termination=False,
                      weights={k:v.weight for k,v in cfg.rewards.items()},
                      core_geometry=a.arm=='on', shared_auxiliaries=True, alpha=1.,
                      git_commit=os.environ.get('SMP_SOURCE_COMMIT', subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),
                      runtime_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      script_sha256=sha(__file__), auxiliary_sha256=sha(sg.__file__),
                      packages={k:importlib.metadata.version(k) for k in ('torch','mjlab','mujoco','mujoco-warp','rsl-rl-lib')})
        base.atomic_json(a.out/'launch.json', launch)
        if a.preflight:
            base.atomic_json(a.out/'dynamics_audit.json', base.legacy.bd.audit_dynamics(env))
            # Preserve unrelated worlds and clear selected histories only.
            ids = torch.arange(0, env.num_envs, 8, device=env.device)
            untouched = torch.ones(env.num_envs,dtype=torch.bool,device=env.device); untouched[ids]=False
            env._sg_timer[:] = 7.; env._sg_lateral[:] = 3.
            q = env.sim.data.qpos.clone(); mass = env.sim.model.body_mass.clone(); gain = env._bd_gain.clone()
            env._reset_idx(ids)
            assert (env._sg_timer[ids]==0).all() and (env._sg_timer[untouched]==7).all()
            assert (env._sg_lateral[ids]==0).all() and (env._sg_lateral[untouched]==3).all()
            assert torch.equal(q[untouched],env.sim.data.qpos[untouched])
            assert torch.equal(mass[untouched],env.sim.model.body_mass[untouched])
            assert torch.equal(gain[untouched],env._bd_gain[untouched])
            obs, _ = env.reset()
            base.atomic_json(a.out/'partial_reset_pass.json', {'pass':True})
        if a.calibrate_steps:
            policy = runner.get_inference_policy(); total = torch.zeros(len(env.reward_manager.active_terms),device=env.device)
            for _ in range(a.calibrate_steps):
                with torch.no_grad(): obs, _, _, _ = wrapper.step(policy(obs))
                total += env.reward_manager._step_reward.mean(0)*env.step_dt
            base.atomic_json(a.out/'calibration.json', dict(seconds=a.calibrate_steps*env.step_dt,
                weighted_term_integrals=dict(zip(env.reward_manager.active_terms,total.tolist()))))
            env.common_step_counter=0; obs,_=env.reset()
        if a.wandb != 'disabled':
            import wandb
            wb=wandb.init(project='smp-guidance-strengthening',group='R2_shared_aux_10k_20261006',
                name=agent.run_name,config=launch,dir=str(a.out.resolve()),mode=a.wandb,sync_tensorboard=True)
            base.atomic_json(a.out/'wandb.json',dict(id=wb.id,url=wb.url))
        initial_actor = {k:v.detach().clone() for k,v in runner.alg.save()['actor_state_dict'].items()}
        native_log = runner.logger.log
        def log(**kw):
            it=kw['it']
            if it%25 == 0 or it == a.updates-1:
                native_log(**kw)
                losses={k:float(v) for k,v in kw['loss_dict'].items()}
                assert all(np.isfinite(v) for v in losses.values()),losses
                change=max(float((v-initial_actor[k]).abs().max()) for k,v in runner.alg.save()['actor_state_dict'].items())
                base.atomic_json(a.out/'progress.json',dict(completed_updates=it+1,target_updates=a.updates,
                    elapsed_s=time.time()-runner.started,actor_max_change=change,losses=losses))
            else:
                with contextlib.redirect_stdout(io.StringIO()):native_log(**kw)
        runner.logger.log=log
        runner.save(str(a.out/'initial.pt'))
        print('TRAINING_STARTED', json.dumps(launch), flush=True)
        runner.learn(num_learning_iterations=a.updates, init_at_random_ep_len=False)
        runner.save(str(a.out/'final.pt'))
        assert (a.out/f'model_{a.updates-1}.pt').exists()
        base.atomic_json(a.out/'completed.json',dict(updates=a.updates,elapsed_s=time.time()-runner.started))
    except BaseException as exc:
        base.atomic_json(a.out/'failed.json', {'error':repr(exc)})
        raise
    finally:
        if env is not None: env.close()
        if wb is not None: wb.finish()


if __name__ == '__main__': main()
