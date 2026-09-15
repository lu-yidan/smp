"""Paired F0--F4 continuation from the same R1 20k checkpoint."""
import argparse,hashlib,json,os,random,subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.os import dump_yaml
from train_scratch_tradeoffs import build_config as r_config,Runner as RRunner,Wrapper as RWrapper,tensor_hash
from train_fixed_low_ablation import atomic_json
from smp.rl.tasks.getup import r1_quality as m

def build_config(arm,bank,num_envs=4096):
    cfg,agent=r_config('R1',bank,num_envs)
    cfg.metrics['f_sustained_substep']=MetricsTermCfg(func=m.sample_substep,per_substep=True)
    cfg.metrics['f_cache']=MetricsTermCfg(func=m.metric)
    for i,(name,w) in enumerate(zip(m.NAMES,m.WEIGHTS[arm])):
        cfg.rewards[name]=RewardTermCfg(func=m.reward,params={'index':i},weight=w)
    return cfg,agent

class Wrapper(RWrapper):
    def step(self,actions):
        o,r,d,e=super().step(actions);env=self.unwrapped
        groups=[('all',torch.ones(env.num_envs,device=env.device,dtype=torch.bool)),('low',env._fixed_group>=2)]+[(n,env._fixed_group==i+2) for i,n in enumerate(['supine','prone','left','right'])]
        for name,mask in groups:
            for i,key in enumerate(m.NAMES):e['log'][f'FineQuality/{name}/{key}']=env._f_values[mask,i].mean().detach()
            e['log'][f'FineQuality/{name}/near_stand_gate']=env._f_gate[mask].mean().detach()
            near=mask&(env._f_gate>.5)
            e['log'][f'FineQuality/{name}/near_stand_pose_error']=(env._f_pose_error*near).sum().detach()/near.sum().clamp_min(1)
            weights=env._f_values.new_tensor([abs(env.cfg.rewards[k].weight) for k in m.NAMES]);task=env._fixed_product[mask].mean().clamp_min(1e-8)
            e['log'][f'FineQuality/{name}/full_penalty_to_task']=((env._f_values[mask,1:]*weights[1:]).sum(-1).mean()/task).detach()
            e['log'][f'FineQuality/{name}/full_pose_bonus_to_task']=(env._f_values[mask,0].mean()*weights[0]/task).detach()
        e['log']['FineQuality/ramp']=torch.tensor(m.ramp(env),device=env.device)
        assert torch.isfinite(env._f_values).all()
        return o,r,d,e

class Runner(RRunner):
    def save(self,path,infos=None):
        if self.active and self.current_learning_iteration==self.args.start_iteration:return
        super().save(path,infos={**(infos or {}),'r1_quality':{'start_counter':self.env.unwrapped._f_start_counter,'ramp_updates':500,'arm':self.args.label}})

def verify_nested(a,b):
    if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:verify_nested(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):verify_nested(x,y)
    else:assert a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(m.WEIGHTS),required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--eval-gpu',required=True);p.add_argument('--additional',type=int,default=5000);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--preflight',action='store_true');p.add_argument('--probe',action='store_true');a=p.parse_args()
    root=Path.cwd();a.bank_dir=root/'datasets/reset_banks/natural_curriculum_v1';a.eval_workspace=Path('/root/workplace/smp-flat93');a.log_dir=a.log_dir.resolve();a.log_dir.mkdir(parents=True,exist_ok=False);a.label=a.arm;a.load_audit_script=Path(__file__).with_name('audit_r1_quality_loads.py').resolve()
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False);a.start_iteration=parent['iter']+1;a.iterations=a.start_iteration+a.additional
    cfg,agent=build_config(a.arm,a.bank_dir/'train.npz',a.num_envs);agent.logger='tensorboard' if a.preflight or a.probe else 'wandb';agent.upload_model=False;agent.algorithm.learning_rate=float(parent['infos']['scratch_tradeoffs']['learning_rate']);agent.max_iterations=a.iterations;agent.save_interval=500;agent.run_name=a.arm+'-r1-quality-5k'
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    dump_yaml(a.log_dir/'params/env.yaml',asdict(cfg));dump_yaml(a.log_dir/'params/agent.yaml',asdict(agent));env=None
    try:
        env=ManagerBasedRlEnv(cfg,device='cuda:0');w=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.log_dir),'cuda:0');runner.args=a
        runner.load(str(a.checkpoint));runner.current_learning_iteration=a.start_iteration
        state=parent['infos']['scratch_tradeoffs'];runner.alg.learning_rate=float(state['learning_rate'])
        for group in runner.alg.optimizer.param_groups:group['lr']=runner.alg.learning_rate
        env._smp_normalizer.mean.copy_(state['mean'].to(env.device));env._smp_normalizer.count.copy_(state['count'].to(env.device))
        # New paired episodes; source simulator/contact/event timers are not checkpointed.
        obs,_=env.reset();env._f_start_counter=env.common_step_counter
        verify_nested(state['mean'],env._smp_normalizer.mean);verify_nested(state['count'],env._smp_normalizer.count)
        restored=runner.alg.save()
        for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):verify_nested(parent[key],restored[key])
        assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
        assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events and env._smp_gsi_pool.shape[0]==0
        assert env.sim.data.qvel.abs().max()<1e-6 and a.start_iteration==20000
        meta={'arm':a.label,'from_scratch':False,'source_checkpoint':str(a.checkpoint),'source_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'start_iteration':a.start_iteration,'additional_updates':a.additional,'end_iteration':a.iterations-1,'num_envs':a.num_envs,'actor_sha':tensor_hash(restored['actor_state_dict']),'critic_sha':tensor_hash(restored['critic_state_dict']),'initial_qpos_sha':hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),'smp_mean_sha':hashlib.sha256(state['mean'].numpy().tobytes()).hexdigest(),'smp_count':state['count'].tolist(),'optimizer_exactly_restored':True,'learning_rate':runner.alg.learning_rate,'weights':dict(zip(m.NAMES,m.WEIGHTS[a.label])),'ramp_updates':500,'episode_seconds':cfg.episode_length_s,'quota_counts':env._fixed_quota_counts,'sources':[[int(((env._fixed_group==g)&(env._fixed_source==s)).sum()) for s in (0,1)] for g in range(6)],'actor_noise':cfg.observations['actor'].enable_corruption,'events':list(cfg.events),'reference_joint_names':list(env.scene['robot'].joint_names),'reference_joint_pos':m.reference_values(env.scene['robot'].joint_names),'resume_note':'Actor/critic/normalizers/optimizer/LR/SMP mean and count restored exactly. Same fresh seeded reset episodes across arms, not bitwise simulator continuation.'}
        atomic_json(a.log_dir/'launch.json',meta);np.savez_compressed(a.log_dir/'initial_reset.npz',qpos=env.sim.data.qpos.cpu().numpy(),qvel=env.sim.data.qvel.cpu().numpy(),group=env._fixed_group.cpu().numpy(),source=env._fixed_source.cpu().numpy())
        runner.save(str(a.log_dir/'initial.pt'));print('RESTORE_VERIFIED',json.dumps(meta),flush=True)
        if a.probe:
            records=[];runner.alg.train_mode()
            with torch.inference_mode():
                for step in range(500):
                    action=runner.alg.act(obs);obs,_,_,_=w.step(action)
                    if step%5==0:
                        records.append({'step':step,'ramp':m.ramp(env),'all':env._f_values.mean(0).cpu().tolist(),'low':env._f_values[env._fixed_group>=2].mean(0).cpu().tolist(),'task_all':float(env._fixed_product.mean()),'task_low':float(env._fixed_product[env._fixed_group>=2].mean()),'near_fraction':float((env._f_gate>.5).float().mean()),'max':env._f_values.max(0).values.cpu().tolist()})
                    # Probe does not collect/update PPO; clear its rollout storage cursor.
                    runner.alg.transition.clear()
            atomic_json(a.log_dir/'probe.json',records);atomic_json(a.log_dir/'completed.json',{'probe_only':True});return
        a.arm='L4';runner.active=True;runner.learn(num_learning_iterations=a.additional,init_at_random_ep_len=False)
        atomic_json(a.log_dir/'completed.json',{'iteration':runner.current_learning_iteration})
    except BaseException as e:atomic_json(a.log_dir/'failed.json',{'error':repr(e)});raise
    finally:
        if env is not None:env.close()
if __name__=='__main__':main()
