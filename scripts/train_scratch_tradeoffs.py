"""R0--R7: paired L4-derived from-scratch 20k reward/protocol ablations."""
import argparse,hashlib,json,os,random,subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor.contact_sensor import ContactMatch,ContactSensorCfg
from mjlab.utils.os import dump_yaml
from train_fixed_low_ablation import build_config as l4_config,FixedRunner,FixedWrapper,atomic_json
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_NAMES
from smp.rl.tasks.getup import scratch_tradeoffs as mdp

ARMS=tuple('R'+str(i) for i in range(8))
COST_NAMES=('effort_top3','speed_top3','target_slew','head_impact')

def build_config(arm,bank,num_envs=4096,seed=20260912):
    assert arm in ARMS
    cfg,agent=l4_config('L4',bank,num_envs,seed)
    if arm!='R0':cfg.terminations['stood_up'].func=mdp.no_stand_termination
    ground=ContactMatch(mode='geom',pattern='terrain')
    sensors=(
        ContactSensorCfg(name='quality_feet',primary=ContactMatch(mode='body',pattern=('left_ankle_roll_link','right_ankle_roll_link'),entity='robot'),secondary=ground,fields=('force',),reduce='netforce'),
        ContactSensorCfg(name='quality_other',primary=ContactMatch(mode='geom',pattern=tuple(n for n in COLLISION_NAMES if 'foot' not in n),entity='robot'),secondary=ground,fields=('force',),reduce='netforce'),
    )
    cfg.scene.sensors=tuple(cfg.scene.sensors)+sensors
    cfg.metrics['quality_substep']=MetricsTermCfg(func=mdp.sample_substep,per_substep=True)
    cfg.metrics['quality_cache']=MetricsTermCfg(func=mdp.cache_control)
    cfg.rewards['quiet_stand']=RewardTermCfg(func=mdp.quiet_reward,weight=.1 if arm not in ('R0','R1') else 0.)
    if arm in ('R3','R7'):
        terms=list(cfg.rewards['task_smp_product'].params['task_terms']);fn,w,p=terms[0];terms[0]=(mdp.slow_upward,w,p)
        cfg.rewards['task_smp_product'].params['task_terms']=tuple(terms)
    weights=[-.05 if arm in ('R4','R7') else 0.,-.1 if arm in ('R4','R7') else 0.,-.005 if arm in ('R5','R7') else 0.,-.1 if arm in ('R6','R7') else 0.]
    for i,(name,w) in enumerate(zip(COST_NAMES,weights)):cfg.rewards[name]=RewardTermCfg(func=mdp.safety_cost,params={'index':i},weight=w)
    return cfg,agent

class Wrapper(FixedWrapper):
    def step(self,actions):
        obs,r,d,e=super().step(actions);env=self.unwrapped
        groups=[('all',torch.ones(env.num_envs,device=env.device,dtype=torch.bool)),('low',env._fixed_group>=2)]+[(n,env._fixed_group==i+2) for i,n in enumerate(['supine','prone','left','right'])]
        for name,mask in groups:
            for i,key in enumerate(COST_NAMES):e['log'][f'Tradeoff/{name}/{key}']=env._r_costs[mask,i].mean().detach()
            for i,key in enumerate(['tau','dq','power','head_force']):e['log'][f'Tradeoff/{name}/peak_{key}']=env._r_peaks[mask,i].max().detach()
            e['log'][f'Tradeoff/{name}/quiet_score']=env._r_quiet[mask].mean().detach()
            e['log'][f'Recovery/{name}/stable_step_fraction']=env._r_stable[mask].float().mean().detach()
            e['log'][f'Recovery/{name}/hold_seconds']=env._r_hold[mask].mean().detach()
            weights=env._r_costs.new_tensor([abs(env.cfg.rewards[k].weight) for k in COST_NAMES])
            e['log'][f'Tradeoff/{name}/full_cost_to_task']=((env._r_costs[mask]*weights).sum(-1).mean()/env._fixed_product[mask].mean().clamp_min(1e-8)).detach()
        e['log']['Tradeoff/ramp']=torch.tensor(mdp.ramp(env),device=env.device)
        assert torch.isfinite(r).all() and torch.isfinite(env._r_costs).all()
        return obs,r,d,e

class Runner(FixedRunner):
    def save(self,path,infos=None):
        env=self.env.unwrapped
        state={'mean':env._smp_normalizer.mean.cpu(),'count':env._smp_normalizer.count.cpu(),'learning_rate':self.alg.learning_rate,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'reset_rng':env._fixed_rng.get_state().cpu(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'quality_ramp':mdp.ramp(env)}
        super().save(path,infos={**(infos or {}),'scratch_tradeoffs':state})
        if not self.active or (self.current_learning_iteration==0 and Path(path).name=='model_0.pt'):return
        # The same 128 initial poses at every checkpoint; load diagnostics at 2ms.
        a=self.args;ep=self.current_learning_iteration;out=a.log_dir/'validation'/f'{ep}_loads.json'
        cmd=[str(a.eval_workspace/'.venv/bin/python'),'-u',str(getattr(a,'load_audit_script',Path(__file__).with_name('audit_tradeoff_loads.py').resolve())),'--loads-output',str(out),'--policy-family','master','--checkpoint',str(Path(path).resolve()),'--num-envs','128','--steps','1000','--output',str(out.with_name(f'{ep}_regular.json'))]
        with open(out.with_suffix('.log'),'w') as f:subprocess.run(cmd,cwd=a.eval_workspace,env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=a.eval_gpu,MUJOCO_GL='egl',OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT,check=True)
        values=json.loads(out.read_text());summary={}
        for key in ['peak_tau','peak_dq','peak_power']:
            v=np.asarray(values[key]).max(1);summary[key]=float(v.max());summary[key+'_env_p95']=float(np.percentile(v,95))
        for key in ['speed_exceed_fraction','effort_exceed_fraction']:
            v=np.asarray(values[key]);summary[key+'_mean']=float(v.mean());summary[key+'_worst_joint_episode']=float(v.max())
        for sensor,names in values['contact_names'].items():
            ids=[i for i,n in enumerate(names) if 'head' in n]
            if ids:
                v=np.asarray(values['contact_peak'][sensor])[:,ids].max(1);summary['head_force_peak']=float(v.max());summary['head_force_env_p95']=float(np.percentile(v,95))
        regular=json.loads(out.with_name(f'{ep}_regular.json').read_text())['per_env']
        summary['regular_stable_10s']=float(np.mean(regular['success_10s']))
        times=np.asarray(regular['first_upright_s']);summary['ever_upright_fraction']=float(np.mean(times>=0))
        if np.any(times>=0):summary['first_upright_median_s']=float(np.median(times[times>=0]))
        if 'near_stand_time_s' in values:summary['near_stand_valid_fraction']=float(np.mean(np.asarray(values['near_stand_time_s'])>0))
        for key in ['near_stand_pose_error','near_stand_tau_ratio_rms','near_stand_effort_fraction','near_stand_time_s']:
            if key in values:
                v=np.asarray(values[key]);summary[key+'_mean']=float(v.mean());summary[key+'_max']=float(v.max())
        if 'phase' in values:
            phase=values['phase']
            for key in ['pre_gate_speed_peak','gate_active_speed_peak']:
                summary['phase_'+key+'_env_p95']=float(np.percentile(phase[key],95))
            summary['phase_peak_before_gate_fraction']=float(np.mean((np.asarray(phase['speed_peak_head_height'])<=1.)|(np.asarray(phase['speed_peak_upright'])<=.8)))
            summary['phase_peak_time_median_s']=float(np.median(phase['speed_peak_time_s']))
        atomic_json(a.log_dir/'load_progress.json',{'iteration':ep,'summary':summary})
        for k,v in summary.items():self.logger.writer.add_scalar('LoadValidation/'+k,v,ep)
        print('LOAD_VALIDATION_COMPLETE',ep,json.dumps(summary),flush=True)

def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--eval-gpu',required=True);p.add_argument('--iterations',type=int,default=20000);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--preflight',action='store_true');p.add_argument('--probe',action='store_true');a=p.parse_args()
    root=Path.cwd();a.bank_dir=root/'datasets/reset_banks/natural_curriculum_v1';a.eval_workspace=Path('/root/workplace/smp-flat93');a.log_dir=a.log_dir.resolve();a.log_dir.mkdir(parents=True,exist_ok=False)
    cfg,agent=build_config(a.arm,a.bank_dir/'train.npz',a.num_envs);agent.logger='tensorboard' if a.preflight or a.probe else 'wandb';agent.upload_model=False;agent.max_iterations=a.iterations;agent.save_interval=500;agent.run_name=a.arm+'-scratch20k'
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    dump_yaml(a.log_dir/'params/env.yaml',asdict(cfg));dump_yaml(a.log_dir/'params/agent.yaml',asdict(agent));env=None
    try:
        env=ManagerBasedRlEnv(cfg,device='cuda:0');w=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.log_dir),'cuda:0');runner.args=a
        obs,_=env.reset();assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
        assert env._smp_gsi_pool.shape[0]==0 and cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
        if a.probe:
            # Read-only cost calibration, never used to initialize a formal run.
            ck=root/'logs/rsl_rl/fixed_low/formal_20260913_112049/L4/model_9999.pt'
            runner.load(str(ck),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})
            ref=torch.load(root/'datasets/recovery_references/L4_initial_reference.pt',weights_only=False,map_location='cpu')
            env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.fill_(env._smp_normalizer.max_count+1)
            policy=runner.get_inference_policy();records=[]
            with torch.inference_mode():
                for i in range(500):
                    obs,_,_,_=w.step(policy(obs)+.3*torch.randn(env.num_envs,29,device=env.device))
                    if True:
                        mask=env._fixed_group>=2;records.append({'step':i,'costs':env._r_costs[mask].mean(0).cpu().tolist(),'task':float(env._fixed_product[mask].mean()),'quiet':float(env._r_quiet[mask].mean()),'head_force_peak':float(env._r_peaks[mask,3].max()),'cost_peaks':env._r_costs[mask].max(0).values.cpu().tolist()})
            atomic_json(a.log_dir/'probe.json',records);atomic_json(a.log_dir/'completed.json',{'probe_only':True});return
        assert env.sim.data.qvel.abs().max()<1e-6
        np.savez_compressed(a.log_dir/'initial_reset.npz',qpos=env.sim.data.qpos.cpu().numpy(),qvel=env.sim.data.qvel.cpu().numpy(),group=env._fixed_group.cpu().numpy(),source=env._fixed_source.cpu().numpy())
        state=runner.alg.save();meta={'arm':a.arm,'from_scratch':True,'checkpoint_loaded':False,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'seed':cfg.seed,'num_envs':a.num_envs,'iterations':a.iterations,'initial_lr':agent.algorithm.learning_rate,'actor_sha':tensor_hash(state['actor_state_dict']),'critic_sha':tensor_hash(state['critic_state_dict']),'initial_qpos_sha':hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),'quota_counts':env._fixed_quota_counts,'reset_sources':[[int(((env._fixed_group==g)&(env._fixed_source==s)).sum()) for s in (0,1)] for g in range(6)],'reward_weights':{k:v.weight for k,v in cfg.rewards.items()},'quality_ramp_updates':[5000,10000],'standing_termination':a.arm=='R0','episode_seconds':cfg.episode_length_s,'smp_reference':'fresh running estimator, saved in every checkpoint; no legacy reconstruction','train_bank_sha':hashlib.sha256((a.bank_dir/'train.npz').read_bytes()).hexdigest()}
        atomic_json(a.log_dir/'launch.json',meta);runner.save(str(a.log_dir/'initial.pt'));print('SCRATCH_VERIFIED',json.dumps(meta),flush=True)
        a.arm='L4' # Reuse both-bank validation dispatch without changing the run identity.
        runner.active=True;runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
        atomic_json(a.log_dir/'completed.json',{'iteration':runner.current_learning_iteration})
    except BaseException as e:atomic_json(a.log_dir/'failed.json',{'error':repr(e)});raise
    finally:
        if env is not None:env.close()
if __name__=='__main__':main()
