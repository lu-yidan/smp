"""Matched initialization, bounded local guidance, and terrain ablations."""
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
from smp.rl.tasks.getup import balanced_dynamics as bd,b_guidance as bg
from smp.rl.tasks.getup.multiterrain_geometry import terrain_spec,free_plate_spec,scene_spec,STRATA,DIRECTIONS
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN
GUIDANCE_WEIGHT=5.

def build_config(n=4096,bank='outputs/multiterrain_bank/train.npz',nominal=False,arm='B2_FT'):
 cfg,agent=base_config(Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),n)
 cfg.seed+=101
 multi=nominal or arm in ('B6_FT_terrain','B7_L4_terrain')
 cfg.scene.spec_fn=scene_spec if multi else bd.scene_spec;cfg.scene.env_spacing=0.
 cfg.scene.entities['escape_obstacle']=EntityCfg(spec_fn=p.plate_spec,init_state=EntityCfg.InitialStateCfg(pos=(20,20,.8),joint_pos={'escape_plate_slide':0.},joint_vel={'escape_plate_slide':0.}))
 cfg.scene.entities['free_obstacle']=EntityCfg(spec_fn=free_plate_spec,init_state=EntityCfg.InitialStateCfg(pos=(20,20,.1)))
 ground=ContactMatch(mode='body',pattern='surfaces')
 cfg.scene.sensors+=tuple(replace(s,name=s.name+'_terrain',secondary=ground) for s in cfg.scene.sensors if s.name in ('quality_feet','quality_other'))
 for name,entity,body in [('guided_contact','escape_obstacle','escape_plate'),('free_contact','free_obstacle','plate')]:
  cfg.scene.sensors+= (ContactSensorCfg(name=name,primary=ContactMatch(mode='geom',pattern=COLLISION_PATTERN,entity='robot'),secondary=ContactMatch(mode='body',pattern=body,entity=entity),fields=('found','force','dist'),reduce='mindist',num_slots=1),)
 cfg.events['gsi_reset']=EventTermCfg(func=bg.reset,mode='reset',params={'bank_path':bank,'dynamics':not nominal,'evaluation':nominal,'multiterrain':multi,'guidance':arm not in ('B2_FT','B3_L4') and not nominal})
 cfg.events['multiterrain_phase']=EventTermCfg(func=mt.update,mode='step')
 cfg.terminations.pop('smp_too_low',None);cfg.terminations['invalid_plate']=TerminationTermCfg(func=mt.invalid)
 cfg.rewards['task_smp_product'].func=mt.task
 for i,(name,w) in enumerate([('geometry_progress',.45),('clearance',.08),('completion',.60),('force',-.03),('separation',.01)]):
  cfg.rewards['plate_'+name]=RewardTermCfg(func=mt.escape_reward,params={'index':i},weight=w)
 if arm not in ('B2_FT','B3_L4') and not nominal:cfg.rewards['local_path_progress']=RewardTermCfg(func=bg.reward,weight=GUIDANCE_WEIGHT)
 cfg.sim.nconmax=256;cfg.sim.njmax=4000;cfg.episode_length_s=10.
 if nominal:
  cfg.observations['actor'].enable_corruption=False
  for key in list(cfg.events):
   if cfg.events[key].mode in ('startup','interval') and key!='init_smp_state':cfg.events.pop(key)
  cfg.terminations.pop('invalid_plate');cfg.episode_length_s=1000.
 action=cfg.actions['joint_pos'];cfg.actions['joint_pos']=bd.DelayedPositionActionCfg(**{f.name:getattr(action,f.name) for f in __import__('dataclasses').fields(action)})
 cfg.viewer.width=640;cfg.viewer.height=480;cfg.viewer.max_extra_envs=0
 return cfg,agent

class Wrapper(RslRlVecEnvWrapper):
 def step(self,actions):
  obs,r,d,e=super().step(actions);env=self.unwrapped;log=e.setdefault('log',{})
  for st,name in enumerate(STRATA):
   mask=env._mt_stratum==st
   if mask.any():
    for key,value in [('stable',env._r_stable.float()),('escaped',env._mt_escaped.float()),('invalid',env._mt_invalid.float()),('task',env._fixed_task_score),('smp',env._fixed_smp_score),('product',env._plate_product),('plate_force',env._mt_force),('relative_head_z',mt.height(env))]:log[f'Terrain/{name}/{key}']=value[mask].float().mean().detach()
  for key,value in [('eligible',env._bg_eligible.float()),('phase',env._bg_phase.float()),('credit',env._bg_credit),('pose_error',env._bg_error)]:log['Guidance/'+key]=value.float().mean().detach()
  for key,value in [('mass_legs',env._bd_mass[:,0]),('mass_upper',env._bd_mass[:,1]),('motor_waist',env._bd_gain[:,0]),('delay_ms',env._bd_lag*2),('nominal',env._bd_nominal.float())]:log['Dynamics/'+key]=value.float().mean().detach()
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
 cpu=mujoco.MjData(env.sim.mj_model);depth=[];q=env.sim.data.qpos.cpu().numpy();mp=env.sim.data.mocap_pos.cpu().numpy();mq=env.sim.data.mocap_quat.cpu().numpy()
 for i in range(env.num_envs):
  cpu.qpos[:]=q[i];cpu.mocap_pos[:]=mp[i];cpu.mocap_quat[:]=mq[i];mujoco.mj_forward(env.sim.mj_model,cpu)
  depth.append(min([float(c.dist) for c in cpu.contact]+[0.]))
 assert min(depth)>-.0011,min(depth)
 gravity=env.scene['robot'].data.projected_gravity_b
 actual=torch.where(gravity[:,0].abs()>=gravity[:,1].abs(),torch.where(gravity[:,0]>0,1,0),torch.where(gravity[:,1]>0,2,3))
 assert torch.equal(actual,env._mt_direction),torch.bincount(actual)
 assert (gravity[:,:2].norm(dim=-1)>.5).all() # total horizontal tilt, including diagonal side poses
 assert env.scene.env_origins.abs().max()==0
 scenes=2 if env._bd_flat_support else 4
 assert all(int((env._mt_scene==s).sum())==env.num_envs//scenes for s in range(scenes))
 assert all(int(((env._mt_scene==s)&(env._mt_direction==d)).sum())==env.num_envs//scenes//4 for s in range(scenes) for d in range(4))
 assert abs(int(env._mt_source.sum())-env.num_envs//4)<=16
 atomic_json(out/'dynamics_audit.json',{'mass_min':env._bd_mass.min(0).values.tolist(),'mass_max':env._bd_mass.max(0).values.tolist(),'gain_min':env._bd_gain.min(0).values.tolist(),'gain_max':env._bd_gain.max(0).values.tolist(),'delay_counts':torch.bincount(env._bd_lag,minlength=6).tolist(),'nominal_fraction':float(env._bd_nominal.float().mean())})
 np.savez_compressed(out/'initial.npz',qpos=q,mocap_pos=mp,mocap_quat=mq,scene=env._mt_scene.cpu().numpy(),direction=env._mt_direction.cpu().numpy(),stratum=env._mt_stratum.cpu().numpy())
 atomic_json(out/'reset_audit.json',{'min_dist':depth,'scene_counts':torch.bincount(env._mt_scene).tolist(),'stratum_counts':torch.bincount(env._mt_stratum).tolist(),'direction_counts':torch.bincount(env._mt_direction).tolist()})

def evaluate(env,wrapper,policy,obs,a):
 hold=torch.zeros(env.num_envs,device=env.device);best=hold.clone();escape=hold.bool();alive=torch.ones_like(escape);peak=torch.zeros(env.num_envs,4,device=env.device)
 ever_invalid=torch.zeros_like(escape)
 selected=[]
 for st in range(8):
  for dr in range(4):
   ids=torch.where((env._mt_stratum==st)&(env._mt_direction==dr))[0]
   if len(ids):selected.append(int(ids[0]))
 writers={}
 if a.video:
  import imageio.v2 as imageio
  for st in range(8):writers[st]=imageio.get_writer(str(a.out/f'{STRATA[st]}_20s.mp4'),fps=25,codec='libx264',quality=7)
 traces=[]
 try:
  for step in range(a.steps):
   with torch.inference_mode():obs,_,done,_=wrapper.step(policy(obs))
   alive&=~done.bool();plate=(env._mt_scene==1)|(env._mt_scene==2)
   ever_invalid|=env._mt_invalid
   stable=env._r_stable&alive&(~plate|(env._mt_escaped&~ever_invalid))
   hold=torch.where(stable,hold+env.step_dt,0);best=torch.maximum(best,hold);escape|=env._mt_escaped&alive;peak=torch.maximum(peak,env._r_peaks)
   traces.append(env.sim.data.qpos[selected].cpu().numpy())
   if writers and step%2==0:
    from PIL import Image,ImageDraw
    for st,writer in writers.items():
     tiles=[]
     for dr in range(4):
      ids=torch.where((env._mt_stratum==st)&(env._mt_direction==dr))[0];j=int(ids[0]);env.cfg.viewer.env_idx=j
      im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,35),fill='black');draw.text((5,8),f'{STRATA[st]} {DIRECTIONS[dr]} t={step*.02:.2f} hold={hold[j]:.2f}',fill='white');tiles.append(np.asarray(im))
     writer.append_data(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0))
 finally:
  for w in writers.values():w.close()
 summary={}
 for st,name in enumerate(STRATA):
  for dr in range(-1,4):
   mask=(env._mt_stratum==st)&((env._mt_direction==dr) if dr>=0 else True)
   if not mask.any():continue
   vals={'n':int(mask.sum()),'stable_1s':float((best[mask]>=1-1e-4).float().mean()),'stable_10s':float((best[mask]>=10-1e-4).float().mean()),'escaped':float(escape[mask].float().mean()),'alive':float(alive[mask].float().mean()),'invalid_plate':float(ever_invalid[mask].float().mean())}
   for j,k in enumerate(['tau','speed','power','head_force']):vals[k+'_peak_p95']=float(np.percentile(peak[mask,j].cpu().numpy(),95))
   summary[name+('/'+DIRECTIONS[dr] if dr>=0 else '')]=vals
 atomic_json(a.out/'summary.json',summary)
 np.savez_compressed(a.out/'trace.npz',qpos=np.asarray(traces),selected=selected)
 print('EVALUATION_COMPLETE',json.dumps({k:v for k,v in summary.items() if '/' not in k}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['B2_FT','B3_L4','B4_FT_guidance','B5_L4_guidance','B6_FT_terrain','B7_L4_terrain'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=10000);p.add_argument('--calibrate',action='store_true');p.add_argument('--eval',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--stress-upper',type=float,default=1.);p.add_argument('--video',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
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
  obs,_=env.reset();v.init(env);env._v_start_counter=-12000;env._v_ramp_updates=500
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
  atomic_json(a.out/'launch.json',{'arm':a.arm,'checkpoint':str(a.checkpoint),'sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'actor_exact':True,'fresh_critic':True,'fresh_optimizer':True,'common_critic_sha':initialcritic,'num_envs':a.num_envs,'updates':a.updates,'bank_sha':runner.bank_sha,'actor_noise':cfg.observations['actor'].enable_corruption,'events':list(cfg.events),'episode_seconds':cfg.episode_length_s,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()+Path(mt.__file__).read_bytes()+Path(bd.__file__).read_bytes()+Path(bg.__file__).read_bytes()).hexdigest()})
  if a.eval:evaluate(env,wrapper,runner.get_inference_policy(),obs,a)
  elif a.calibrate:
   policy=runner.get_inference_policy();eligible=env._bg_eligible.clone();integral=torch.zeros(env.num_envs,device=env.device);taskint=integral.clone();maxcredit=0.
   for _ in range(500):
    with torch.inference_mode():obs,reward,done,extra=wrapper.step(policy(obs))
    integral+=env._bg_reward*env.step_dt*GUIDANCE_WEIGHT;taskint+=env._plate_product*env.step_dt
    maxcredit=max(maxcredit,float(env._bg_credit.max()))
    assert not env._bg_eligible[env._mt_scene!=0].any()
    assert maxcredit<=1.20001
   mask=(env._mt_scene==0)&(env._mt_direction==0)
   atomic_json(a.out/'calibration.json',{'eligible_initial':int(eligible.sum()),'eligible_supine_fraction':float(eligible[mask].float().mean()),'guidance_integral_mean':float(integral.mean()),'task_integral_mean':float(taskint.mean()),'guidance_to_task':float(integral.sum()/taskint.sum().clamp_min(1e-8)),'max_episode_raw_credit':maxcredit,'theoretical_return_cap':1.2*GUIDANCE_WEIGHT,'scene_counts':torch.bincount(env._mt_scene).tolist()})
  else:
   runner.save(str(a.out/'initial.pt'));runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;runner.save(str(a.out/'final.pt'))
   atomic_json(a.out/'completed.json',{'iteration':runner.current_learning_iteration})
 except BaseException as e:atomic_json(a.out/'failed.json',{'error':repr(e)});raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
