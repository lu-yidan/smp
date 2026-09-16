"""Matched V33-inspired reward migration: early L4 finetune and scratch."""
import argparse,hashlib,json,os,random,subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.os import dump_yaml
from train_scratch_tradeoffs import build_config as r_config,Runner as BaseRunner,Wrapper as BaseWrapper,tensor_hash
from train_fixed_low_ablation import atomic_json
from train_r1_quality import verify_nested
from smp.rl.tasks.getup import v33_reward_transfer as m


def build_config(bank,num_envs=4096):
    cfg,agent=r_config('R1',bank,num_envs)
    cfg.events['v33_stage_reset']=EventTermCfg(func=m.reset,mode='reset')
    cfg.metrics['v33_substep']=MetricsTermCfg(func=m.sample_substep,per_substep=True)
    cfg.metrics['v33_cache']=MetricsTermCfg(func=m.metric)
    cfg.rewards['task_smp_product'].params['task_terms']=tuple((m.task,w,{'index':i}) for i,w in enumerate(m.TASK_WEIGHTS))
    for i,(name,w) in enumerate(zip(m.COST_NAMES,m.COST_WEIGHTS)):
        cfg.rewards['v33_'+name]=RewardTermCfg(func=m.cost,params={'index':i},weight=w)
    # Restore legacy soft joint-limit cost without changing actuator limits.
    from mjlab.envs.mdp.rewards import joint_pos_limits
    cfg.rewards['v33_joint_limits']=RewardTermCfg(func=joint_pos_limits,weight=-.1)
    return cfg,agent

class Wrapper(BaseWrapper):
    def step(self,actions):
        o,r,d,e=super().step(actions);env=self.unwrapped
        groups=[('all',torch.ones(env.num_envs,device=env.device,dtype=torch.bool)),('low',env._fixed_group>=2)]+[(n,env._fixed_group==i+2) for i,n in enumerate(['supine','prone','left','right'])]
        for name,mask in groups:
            for i,key in enumerate(m.TASK_NAMES):e['log'][f'V33Task/{name}/{key}']=env._v_task[mask,i].mean().detach()
            for i,key in enumerate(m.COST_NAMES):e['log'][f'V33Cost/{name}/{key}']=env._v_cost[mask,i].mean().detach()
            for i in range(4):e['log'][f'V33Stage/{name}/{i}']=(env._v_stage[mask]==i).float().mean().detach()
            weights=env._v_cost.new_tensor(m.COST_WEIGHTS).abs()
            e['log'][f'V33Cost/{name}/full_cost_to_task']=(env._v_cost[mask].mul(weights).sum(-1).mean()/env._fixed_product[mask].mean().clamp_min(1e-8)).detach()
        e['log']['V33Cost/ramp']=torch.tensor(m.ramp(env),device=env.device)
        assert torch.isfinite(env._v_task).all() and torch.isfinite(env._v_cost).all()
        return o,r,d,e

class Runner(BaseRunner):
    def save(self,path,infos=None):
        if self.active and self.current_learning_iteration==self.args.start_iteration:return
        env=self.env.unwrapped
        super().save(path,infos={**(infos or {}),'v33_transfer':{'start_counter':env._v_start_counter,'ramp_updates':env._v_ramp_updates,'label':self.args.label}})
        if self.active:
            out=self.args.log_dir/'validation'/f'{self.current_learning_iteration}_loads.json'
            data=json.loads(out.read_text());phase=data['phase']
            for key,value in [('head_vz_env_p95',float(np.percentile(phase['head_vz_peak'],95))),('head_vz_above_02_fraction',float(np.mean(phase['head_vz_exceed_samples'])/data['physics_samples']))]:
                self.logger.writer.add_scalar('LoadValidation/'+key,value,self.current_learning_iteration)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['FT','Scratch'],required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--eval-gpu',required=True);p.add_argument('--updates',type=int);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--preflight',action='store_true');p.add_argument('--probe',action='store_true');a=p.parse_args()
    root=Path.cwd();a.bank_dir=root/'datasets/reset_banks/natural_curriculum_v1';a.eval_workspace=Path('/root/workplace/smp-flat93');a.log_dir=a.log_dir.resolve();a.log_dir.mkdir(parents=True,exist_ok=False);a.label='V33-'+a.mode;a.arm='L4';a.load_audit_script=Path(__file__).with_name('audit_v33_transfer_loads.py').resolve()
    if a.updates is None:a.updates=10000 if a.mode=='FT' else 20000
    parent=None
    if a.mode=='FT':
        if a.checkpoint is None:a.checkpoint=root/'logs/rsl_rl/fixed_low/formal_20260913_112049/L4/model_9999.pt'
        parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    a.start_iteration=int(parent['iter'])+1 if parent else 0;a.iterations=a.start_iteration+a.updates
    cfg,agent=build_config(a.bank_dir/'train.npz',a.num_envs);agent.logger='tensorboard' if a.preflight or a.probe else 'wandb';agent.upload_model=False;agent.max_iterations=a.iterations;agent.save_interval=500;agent.run_name=a.label
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    # Changed reward: transfer actor/observation statistics; train a fresh critic/optimizer.
    dump_yaml(a.log_dir/'params/env.yaml',asdict(cfg));dump_yaml(a.log_dir/'params/agent.yaml',asdict(agent));env=None
    try:
        env=ManagerBasedRlEnv(cfg,device='cuda:0');w=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.log_dir),'cuda:0');runner.args=a
        initial=runner.alg.save();critic_hash=tensor_hash(initial['critic_state_dict'])
        ref_path=root/'datasets/recovery_references/L4_initial_reference.pt'
        if parent:
            runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
            ref=torch.load(ref_path,weights_only=False,map_location='cpu')
            env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.fill_(env._smp_normalizer.max_count+1)
        runner.current_learning_iteration=a.start_iteration
        obs,_=env.reset();m.init(env);env._v_start_counter=env.common_step_counter;env._v_ramp_updates=500 if parent else 5000
        restored=runner.alg.save()
        assert tensor_hash(restored['critic_state_dict'])==critic_hash
        if parent:verify_nested(parent['actor_state_dict'],restored['actor_state_dict'])
        assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
        assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events and env._smp_gsi_pool.shape[0]==0
        assert env.sim.data.qvel.abs().max()<1e-6
        meta={'label':a.label,'from_scratch':parent is None,'source_checkpoint':str(a.checkpoint) if parent else None,'source_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if parent else None,'source_note':'early reset-ratio L4, not recent F4','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'start_iteration':a.start_iteration,'updates':a.updates,'end_iteration':a.iterations-1,'num_envs':a.num_envs,'actor_sha':tensor_hash(restored['actor_state_dict']),'critic_sha':critic_hash,'initial_qpos_sha':hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),'critic_and_optimizer':'fresh for changed reward','learning_rate':runner.alg.learning_rate,'smp_reference':'legacy reconstructed L4 reference (not exact historical reference)' if parent else 'fresh running estimator','reference_sha256':hashlib.sha256(ref_path.read_bytes()).hexdigest() if parent else None,'prior':'f2s2','smp_ws':cfg.rewards['task_smp_product'].params.get('ws',6),'weights':{k:v.weight for k,v in cfg.rewards.items()},'ramp_updates':env._v_ramp_updates,'episode_seconds':cfg.episode_length_s,'quota_counts':env._fixed_quota_counts,'actor_noise':cfg.observations['actor'].enable_corruption,'events':list(cfg.events)}
        atomic_json(a.log_dir/'launch.json',meta);np.savez_compressed(a.log_dir/'initial_reset.npz',qpos=env.sim.data.qpos.cpu().numpy(),qvel=env.sim.data.qvel.cpu().numpy(),group=env._fixed_group.cpu().numpy(),source=env._fixed_source.cpu().numpy())
        runner.save(str(a.log_dir/'initial.pt'));print('TRANSFER_VERIFIED',json.dumps(meta),flush=True)
        if a.probe:
            records=[];runner.alg.train_mode()
            with torch.inference_mode():
                for step in range(500):
                    obs,_,_,_=w.step(runner.alg.act(obs));runner.alg.transition.clear()
                    if step%5==0:
                        mask=env._fixed_group>=2
                        records.append({'step':step,'task':float(env._fixed_product.mean()),'task_low':float(env._fixed_product[mask].mean()),'costs':env._v_cost.mean(0).cpu().tolist(),'costs_low':env._v_cost[mask].mean(0).cpu().tolist(),'components_low':env._v_task[mask].mean(0).cpu().tolist()})
            atomic_json(a.log_dir/'probe.json',records);atomic_json(a.log_dir/'completed.json',{'probe_only':True});return
        runner.active=True;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False)
        atomic_json(a.log_dir/'completed.json',{'iteration':runner.current_learning_iteration})
    except BaseException as e:atomic_json(a.log_dir/'failed.json',{'error':repr(e)});raise
    finally:
        if env is not None:env.close()
if __name__=='__main__':main()
