"""Flat L4 FT: reset focus and upward-speed comparison with shared dynamics DR."""
import argparse,json,random,hashlib,subprocess,os
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch,ContactSensorCfg
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from train_v33_reward_transfer import build_config as base_config
from train_fixed_low_ablation import atomic_json
from train_r1_quality import verify_nested
from train_scratch_tradeoffs import tensor_hash
from smp.rl.tasks.getup import multiterrain as mt,plate_transfer as p,v33_reward_transfer as v
from smp.rl.tasks.getup import balanced_dynamics as bd,ft_prone_speed as ft
from smp.rl.tasks.getup.multiterrain_geometry import terrain_spec,free_plate_spec,scene_spec,STRATA,DIRECTIONS
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN
from mjlab.managers.metrics_manager import MetricsTermCfg

def build_config(n=4096,bank='outputs/multiterrain_bank/train.npz',nominal=False,arm='FT_R0'):
 from train_d_series import build_config as d_config
 cfg,agent=d_config(n,bank,nominal,'D0')
 cfg.scene.spec_fn=bd.scene_spec
 cfg.events['gsi_reset']=EventTermCfg(func=ft.reset,mode='reset',params={'arm':arm,'bank_path':bank,'dynamics':not nominal,'evaluation':nominal})
 cfg.events.pop('multiterrain_phase',None);cfg.terminations.pop('invalid_plate',None);cfg.metrics.pop('d_progress',None)
 for name in list(cfg.rewards):
  if name.startswith('plate_'):cfg.rewards.pop(name)
 return cfg,agent

class Wrapper(RslRlVecEnvWrapper):
 def step(self,actions):
  obs,r,d,e=super().step(actions);env=self.unwrapped;log=e.setdefault('log',{})
  for g,name in enumerate(ft.GROUPS):
   mask=env._ft_group==g
   if mask.any():
    for key,value in [('stable',env._r_stable.float()),('task',env._fixed_task_score),('smp',env._fixed_smp_score),('product',env._plate_product)]:log[f'Recovery/{name}/{key}']=value[mask].mean().detach()
  for key,value in [('mass_upper',env._bd_mass[:,1]),('delay_ms',env._bd_lag*2),('nominal',env._bd_nominal.float())]:log['Dynamics/'+key]=value.float().mean().detach()
  assert torch.isfinite(r).all()
  return obs,r,d,e

class Runner(MjlabOnPolicyRunner):
 active=False
 def save(self,path,infos=None):
  env=self.env.unwrapped
  state={'mean':env._smp_normalizer.mean.cpu(),'count':env._smp_normalizer.count.cpu()}
  super().save(path,infos={**(infos or {}),'multiterrain':{'smp_reference':state,'source':str(self.args.checkpoint),'arm':self.args.arm,'bank_sha':self.bank_sha}})
  if not self.active:return
  ep=self.current_learning_iteration
  if ep==0:return
  combined={}
  for suite,mass in [('nominal',1.),('upper130',1.3)]:
   out=self.args.out/'validation'/str(ep)/suite
   cmd=[os.sys.executable,'-u',__file__,'--eval','--arm',self.args.arm,'--checkpoint',str(Path(path).resolve()),'--reference',str(self.args.reference),'--out',str(out),'--num-envs','256','--steps','1000','--stress-upper',str(mass)]
   if ep%1000==0 or ep==self.args.updates-1:cmd+=['--video']
   out.mkdir(parents=True,exist_ok=True)
   with open(out/'eval.log','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   result=json.loads((out/'summary.json').read_text());combined[suite]=result
   for case,vals in result.items():
    for key,value in vals.items():self.logger.writer.add_scalar(f'Validation/{suite}/{case}/{key}',value,ep)
  atomic_json(self.args.out/'progress.json',{'iteration':ep,'validation':combined})

def audit_reset(env,out):
 import mujoco
 cpu=mujoco.MjData(env.sim.mj_model);q=env.sim.data.qpos.cpu().numpy();mp=env.sim.data.mocap_pos.cpu().numpy();mq=env.sim.data.mocap_quat.cpu().numpy();depth=[]
 for i in range(env.num_envs):
  cpu.qpos[:]=q[i];cpu.mocap_pos[:]=mp[i];cpu.mocap_quat[:]=mq[i];mujoco.mj_forward(env.sim.mj_model,cpu);depth.append(min([float(c.dist) for c in cpu.contact]+[0.]))
 assert min(depth)>-.0021,min(depth)
 assert not env._mt_scene.any() and env._bd_flat_support
 assert torch.bincount(env._ft_group,minlength=6).tolist()==env._ft_counts
 assert env.sim.data.qvel.abs().max()<1e-6
 assert torch.equal(env._ft_source,env._mt_source)
 atomic_json(out/'reset_audit.json',{'counts':dict(zip(ft.GROUPS,env._ft_counts)),'procedural_count':int(env._ft_source.sum()),'minimum_contact_distance':min(depth),'relaxed_upward_speed':env._ft_relaxed})
 np.savez_compressed(out/'initial.npz',qpos=q,group=env._ft_group.cpu().numpy(),source=env._ft_source.cpu().numpy())

def evaluate(env,wrapper,policy,obs,a):
 hold=torch.zeros(env.num_envs,device=env.device);best=hold.clone();alive=torch.ones(env.num_envs,device=env.device,dtype=torch.bool);peak=torch.zeros(env.num_envs,4,device=env.device)
 path=hold.clone();switches=hold.clone();previous=torch.zeros(env.num_envs,2,device=env.device,dtype=torch.bool);seen=previous.clone();vz_peak=hold.clone()
 selected=[int(torch.where((env._ft_group==g)&(env._ft_source==src))[0][0]) for src in (0,1) for g in range(2,6)]
 writers={};traces=[]
 if a.video:
  import imageio.v2 as imageio
  for src,name in enumerate(['natural','procedural']):writers[src]=imageio.get_writer(str(a.out/f'{name}_20s.mp4'),fps=25,codec='libx264',quality=7)
 try:
  for step in range(a.steps):
   with torch.inference_mode():obs,_,done,_=wrapper.step(policy(obs))
   alive&=~done.bool();stable=env._r_stable&alive;hold=torch.where(stable,hold+env.step_dt,0);best=torch.maximum(best,hold);peak=torch.maximum(peak,env._r_peaks)
   robot=env.scene['robot'];late=(mt.height(env)>.78)&(-robot.data.projected_gravity_b[:,2]>.7)&alive
   contact=mt.ground_force(env,'quality_feet')[...,2].abs()>20
   switches+=((contact!=previous)&seen&late[:,None]).sum(-1);seen=late[:,None].expand_as(contact).clone();previous=contact.clone()
   path+=robot.data.root_link_lin_vel_w[:,:2].norm(dim=-1)*env.step_dt*late
   vz_peak=torch.maximum(vz_peak,robot.data.site_lin_vel_w[:,env._r_head,2].clamp_min(0))
   traces.append(env.sim.data.qpos[selected].cpu().numpy())
   if writers and step%2==0:
    from PIL import Image,ImageDraw
    for src,writer in writers.items():
     tiles=[]
     for dr in range(4):
      j=selected[src*4+dr];env.cfg.viewer.env_idx=j;im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,35),fill='black');draw.text((5,8),f'{DIRECTIONS[dr]} t={step*.02:.2f} hold={hold[j]:.2f}',fill='white');tiles.append(np.asarray(im))
     writer.append_data(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0))
 finally:
  for writer in writers.values():writer.close()
 masks={'all':torch.ones(env.num_envs,device=env.device,dtype=torch.bool)}
 for g,name in enumerate(DIRECTIONS,2):
  masks[name]=env._ft_group==g
  for src,source in enumerate(['natural','procedural']):masks[name+'/'+source]=(env._ft_group==g)&(env._ft_source==src)
 summary={}
 for name,mask in masks.items():
  vals={'n':int(mask.sum()),'stable_1s':float((best[mask]>=1-1e-4).float().mean()),'stable_10s':float((best[mask]>=10-1e-4).float().mean()),'alive':float(alive[mask].float().mean()),'late_contact_switches_mean':float(switches[mask].mean()),'late_base_xy_path_mean':float(path[mask].mean()),'head_upward_peak_p95':float(np.percentile(vz_peak[mask].cpu().numpy(),95))}
  for j,k in enumerate(['tau','speed','power','head_force']):vals[k+'_peak_p95']=float(np.percentile(peak[mask,j].cpu().numpy(),95))
  summary[name]=vals
 atomic_json(a.out/'summary.json',summary);np.savez_compressed(a.out/'trace.npz',qpos=np.asarray(traces),selected=selected)
 print('EVALUATION_COMPLETE',json.dumps(summary['all']),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['FT_R0','FT_R1','FT_R2'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=10000);p.add_argument('--calibrate',action='store_true');p.add_argument('--eval',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--stress-upper',type=float,default=1.);p.add_argument('--video',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
 assert hashlib.sha256(a.reference.read_bytes()).hexdigest()=='8f05543b644b0a1d11246e85778f99220460ef2a744417ea940296eed768c768'
 a.out=a.out.resolve();a.checkpoint=a.checkpoint.resolve();a.reference=a.reference.resolve();a.out.mkdir(parents=True,exist_ok=a.eval)
 bank=f'outputs/multiterrain_bank/{"validation" if a.eval else "train"}.npz';cfg,agent=build_config(a.num_envs,bank,a.eval,a.arm)
 if a.eval:cfg.events['gsi_reset'].params['stress_upper']=a.stress_upper
 random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
 agent.logger='tensorboard' if a.eval or a.preflight or a.calibrate else 'wandb';agent.upload_model=False;agent.run_name=a.arm;agent.save_interval=500;agent.max_iterations=a.updates
 dump_yaml(a.out/'env.yaml',asdict(cfg));dump_yaml(a.out/'agent.yaml',asdict(agent));env=None
 try:
  env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None)
  wrapper=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(wrapper,asdict(agent),str(a.out),'cuda:0');runner.args=a;runner.bank_sha=hashlib.sha256(Path(bank).read_bytes()).hexdigest()
  parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False);initialcritic=tensor_hash(runner.alg.save()['critic_state_dict'])
  runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});verify_nested(parent['actor_state_dict'],runner.alg.save()['actor_state_dict']);assert initialcritic==tensor_hash(runner.alg.save()['critic_state_dict'])
  if not a.eval:env.common_step_counter=0 # new curriculum, not inherited FT counter
  ref=torch.load(a.reference,map_location='cpu',weights_only=False)['infos']['scratch_tradeoffs'];env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.copy_(ref['count'].to(env.device))
  random.seed(cfg.seed+177);np.random.seed(cfg.seed+177);torch.manual_seed(cfg.seed+177)
  obs,_=env.reset();v.init(env);env._v_start_counter=-12000 if a.calibrate else 0;env._v_ramp_updates=500
  assert cfg.scale_rewards_by_dt and abs(env.step_dt-.02)<1e-8
  assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
  audit_reset(env,a.out)
  if a.preflight:
   atomic_json(a.out/'actual_dynamics_check.json',bd.audit_dynamics(env));obs,_=env.reset()
   assert env._bd_mass.min()>=.9-1e-5 and env._bd_mass.max()<=1.1+1e-5
   env.common_step_counter=48000;obs,_=env.reset();assert env._bd_mass.min()>=.8-1e-5 and env._bd_mass.max()<=1.2+1e-5
   env.common_step_counter=0;obs,_=env.reset()
   ids=torch.tensor([0,a.num_envs//4,a.num_envs//2,3*a.num_envs//4],device=env.device)
   mask=torch.ones(a.num_envs,dtype=torch.bool,device=env.device);mask[ids]=False
   before=env.sim.data.qpos[mask].clone();before_mass=env.sim.model.body_mass[mask].clone();before_gain=env._bd_gain[mask].clone();env._reset_idx(ids)
   assert torch.equal(before_mass,env.sim.model.body_mass[mask]) and torch.equal(before_gain,env._bd_gain[mask])
   assert torch.equal(before,env.sim.data.qpos[mask]),'partial reset polluted other worlds'
   obs,_=env.reset()
   atomic_json(a.out/'partial_reset_pass.json',{'untouched_worlds':int(mask.sum())})
  atomic_json(a.out/'launch.json',{'arm':a.arm,'checkpoint':str(a.checkpoint),'sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'actor_exact':True,'fresh_critic':True,'fresh_optimizer':True,'common_critic_sha':initialcritic,'num_envs':a.num_envs,'updates':a.updates,'bank_sha':runner.bank_sha,'seed':cfg.seed,'cost_ramp_updates':500,'low_smp_termination':False,'reset_counts':env._ft_counts,'relaxed_upward_speed':env._ft_relaxed,'reset_bank_sha':{name:hashlib.sha256(Path(f'datasets/reset_banks/{name}/train.npz').read_bytes()).hexdigest() for name in ['natural_curriculum_v1','procedural_low_v1']},'actor_noise':cfg.observations['actor'].enable_corruption,'events':list(cfg.events),'episode_seconds':cfg.episode_length_s,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()+Path(mt.__file__).read_bytes()+Path(bd.__file__).read_bytes()+Path(ft.__file__).read_bytes()+Path(v.__file__).read_bytes()).hexdigest()})
  if a.eval:evaluate(env,wrapper,runner.get_inference_policy(),obs,a)
  elif a.calibrate:
   policy=runner.get_inference_policy();terms=torch.zeros(len(env.reward_manager.active_terms),device=env.device)
   for _ in range(500):
    with torch.inference_mode():obs,_,_,_=wrapper.step(policy(obs))
    terms+=env.reward_manager._step_reward.mean(0)*env.step_dt
   atomic_json(a.out/'calibration.json',{'weighted_term_integrals':dict(zip(env.reward_manager.active_terms,terms.tolist()))})
  else:
   runner.save(str(a.out/'initial.pt'));runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;runner.save(str(a.out/'final.pt'))
   atomic_json(a.out/'completed.json',{'iteration':runner.current_learning_iteration})
 except BaseException as e:atomic_json(a.out/'failed.json',{'error':repr(e)});raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
