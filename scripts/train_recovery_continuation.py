"""Stage-one 2k continuation; save recoverable reward state from now on."""
import argparse,json,hashlib,random,subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.os import dump_yaml
from train_fixed_low_ablation import build_config as base_config,FixedRunner,FixedWrapper,atomic_json
from smp.rl.tasks.getup.recovery_quality import sample_quality,quality_cost

def build_config(arm,bank,num_envs=4096):
 source=arm.split('-')[0];cfg,agent=base_config(source,bank,num_envs)
 cfg.metrics['substep_speed_cost']=MetricsTermCfg(func=sample_quality,per_substep=True)
 cfg.metrics['quality_cache']=MetricsTermCfg(func=quality_cost,params={'index':0})
 weights=(0.,0.,0.) if '-Q' not in arm else ((-.2,-.2,0.) if arm.endswith('Q1') else (-.2,-.2,-.002))
 for i,(name,w) in enumerate(zip(['effort_excess','speed_excess','target_slew'],weights)):
  cfg.rewards[name]=RewardTermCfg(func=quality_cost,params={'index':i},weight=w)
 return cfg,agent

class Wrapper(FixedWrapper):
 def step(self,actions):
  obs,r,d,e=super().step(actions);env=self.unwrapped
  for group,mask in [('all',torch.ones(env.num_envs,device=env.device,dtype=torch.bool)),('low',env._course_type==2)]+[(name,env._fixed_group==i+2) for i,name in enumerate(['supine','prone','left','right'])]:
   for i,k in enumerate(['effort_excess','speed_excess','target_slew']):e['log'][f'Quality/{group}/{k}']=env._quality_costs[mask,i].mean().detach()
   e['log'][f'Quality/{group}/substep_peak_tau_mean']=env._quality_peaks[mask,0].mean().detach()
   e['log'][f'Quality/{group}/substep_peak_speed_mean']=env._quality_peaks[mask,1].mean().detach()
   weights=env._quality_costs.new_tensor([abs(env.cfg.rewards[k].weight) for k in ['effort_excess','speed_excess','target_slew']])
   e['log'][f'Quality/{group}/full_weight_cost_to_task_ratio']=((env._quality_costs[mask]*weights).sum(-1).mean()/env._fixed_product[mask].mean().clamp_min(1e-8)).detach()
  return obs,r,d,e

class Runner(FixedRunner):
 def save(self,path,infos=None):
  if self.active and self.current_learning_iteration==10000:return
  env=self.env.unwrapped
  state={'mean':env._smp_normalizer.mean.cpu(),'count':env._smp_normalizer.count.cpu(),'learning_rate':self.alg.learning_rate,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'reset_rng':env._fixed_rng.get_state().cpu()}
  super().save(path,infos={**(infos or {}),'recovery_continuation':state})
  if self.active:
   ep=self.current_learning_iteration
   if ep==self.args.iterations-1:
    a=self.args;out=a.log_dir/'validation'/f'{ep}_loads.json'
    cmd=[str(a.eval_workspace/'.venv/bin/python'),'-u','scripts/audit_motion_loads.py','--loads-output',str(out),'--policy-family','master','--checkpoint',str(Path(path).resolve()),'--num-envs','128','--steps','1000','--output',str(out.with_name(f'{ep}_regular.json'))]
    import os
    with open(out.with_suffix('.log'),'w') as f:subprocess.run(cmd,cwd=a.eval_workspace,env=dict(os.environ,PYTHONPATH='src:scripts:.',CUDA_VISIBLE_DEVICES=a.eval_gpu,MUJOCO_GL='egl',OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT,check=True)
    values=json.loads(out.read_text());summary={}
    for k in ['peak_tau','peak_dq','peak_power']:
     peaks=np.asarray(values[k]).max(axis=1)
     summary[k]=float(peaks.max());summary[k+'_env_p95']=float(np.percentile(peaks,95))
    for sensor,names in values['contact_names'].items():
     head=[i for i,n in enumerate(names) if 'head' in n]
     if head:
      peaks=np.asarray(values['contact_peak'][sensor])[:,head].max(axis=1)
      summary['head_net_force_peak']=float(peaks.max());summary['head_net_force_env_p95']=float(np.percentile(peaks,95))
    atomic_json(a.log_dir/'load_summary.json',summary)
    for k,v in summary.items():self.logger.writer.add_scalar('LoadValidation/'+k,v,ep)

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['L3','L4','L5','L4-Q1','L4-Q2'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--eval-gpu',required=True);p.add_argument('--additional',type=int,default=2000);p.add_argument('--calibrate',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
 root=Path.cwd();a.bank_dir=root/'datasets/reset_banks/natural_curriculum_v1';a.eval_workspace=Path('/root/workplace/smp-flat93');a.iterations=10000+a.additional;a.log_dir=a.log_dir.resolve();a.log_dir.mkdir(parents=True,exist_ok=False)
 cfg,agent=build_config(a.arm,a.bank_dir/'train.npz');agent.logger='tensorboard' if a.preflight or a.calibrate else 'wandb';agent.upload_model=False;agent.max_iterations=a.iterations;agent.save_interval=500;agent.run_name=a.arm+'-continuation'
 random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
 dump_yaml(a.log_dir/'params/env.yaml',asdict(cfg));dump_yaml(a.log_dir/'params/agent.yaml',asdict(agent))
 env=None
 try:
  env=ManagerBasedRlEnv(cfg,device='cuda:0');w=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.log_dir),'cuda:0');runner.args=a
  if a.calibrate:
   runner.load(str(a.checkpoint.parent/'initial.pt'),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})
   obs,_=env.reset();policy=runner.get_inference_policy();policy.obs_normalizer.train()
   with torch.inference_mode():
    for _ in range(500):
     act=policy(obs)+torch.randn((env.num_envs,29),device=env.device)*.3
     obs,_,_,_=w.step(act);policy.update_normalization(obs)
   state={'mean':env._smp_normalizer.mean.cpu(),'count':env._smp_normalizer.count.cpu(),'method':'500 steps of frozen initial-policy stochastic rollout, same reset/DR; reconstructed reference, not original missing state','source':str(a.checkpoint.parent/'initial.pt')}
   state['actor_normalizer_count']=float(policy.obs_normalizer.count);assert state['actor_normalizer_count']>0
   a.reference.parent.mkdir(parents=True,exist_ok=True);torch.save(state,a.reference);atomic_json(a.log_dir/'completed.json',{'reference':str(a.reference),'means':state['mean'][[8,15,22]].tolist()});return
  parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
  runner.load(str(a.checkpoint));runner.current_learning_iteration=parent['iter']+1
  runner.alg.learning_rate=float(runner.alg.optimizer.param_groups[0]['lr'])
  reference=torch.load(a.reference,map_location='cpu',weights_only=False)
  env._smp_normalizer.mean.copy_(reference['mean'].to(env.device));env._smp_normalizer.count.fill_(env._smp_normalizer.max_count+1)
  obs,_=env.reset();env._quality_start_counter=env.common_step_counter
  resumed=runner.alg.save()
  for key in ['actor_state_dict','critic_state_dict']:
   assert all(torch.equal(v.cpu(),resumed[key][k].cpu()) for k,v in parent[key].items())
  for key,state in parent['optimizer_state_dict']['state'].items():
   for k,v in state.items():
    got=resumed['optimizer_state_dict']['state'][key][k]
    assert torch.equal(v.cpu(),got.cpu()) if isinstance(v,torch.Tensor) else v==got
  assert runner.current_learning_iteration==10000 and env.common_step_counter==240000
  source=a.arm.split('-')[0];counts=[[int(((env._fixed_group==g)&(env._fixed_source==s)).sum()) for s in (0,1)] for g in range(6)]
  meta={'arm':a.arm,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'reward_weights':{k:cfg.rewards[k].weight for k in ['effort_excess','speed_excess','target_slew']},'ramp_updates':250,'reference_method':reference['method'],'source_arm':source,'source_checkpoint':str(a.checkpoint),'source_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'reference_means':reference['mean'][[8,15,22]].tolist(),'restored_optimizer':True,'restored_lr':runner.alg.learning_rate,'start_iteration':runner.current_learning_iteration,'additional_updates':a.additional,'end_iteration':a.iterations-1,'reset_groups':counts,'initial_qpos_sha':hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),'limitations':'Legacy checkpoint omitted SMP running reference and simulator/RNG states; reference rebuilt from initial policy and frozen. Fresh episodes, not bitwise continuation. L4 and Q1/Q2 share reference.'}
  atomic_json(a.log_dir/'launch.json',meta);runner.save(str(a.log_dir/'initial.pt'));print('RESTORE_VERIFIED',json.dumps(meta),flush=True)
  # FixedRunner dispatches both validation banks using source arm.
  a.arm=source;runner.active=True;runner.learn(num_learning_iterations=a.additional,init_at_random_ep_len=False)
  atomic_json(a.log_dir/'completed.json',{'iteration':runner.current_learning_iteration})
 except BaseException as e:atomic_json(a.log_dir/'failed.json',{'error':repr(e)});raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
