"""R2 three-scene V33 path and simple-standing ablation."""
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
from smp.rl.tasks.getup import balanced_dynamics as bd,ft_prone_speed as ft,r2_ablation as ra
from smp.rl.tasks.getup.multiterrain_geometry import terrain_spec,free_plate_spec,scene_spec,STRATA,DIRECTIONS
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN
from mjlab.managers.metrics_manager import MetricsTermCfg
from smp.rl.tasks.getup import v33_path_ablation as pa

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=pa.ARMS,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=10000);p.add_argument('--calibrate',action='store_true');p.add_argument('--eval',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--stress-upper',type=float,default=1.);p.add_argument('--video',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
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
  atomic_json(a.out/'launch.json',{'arm':a.arm,'checkpoint':str(a.checkpoint),'sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'actor_exact':True,'fresh_critic':True,'fresh_optimizer':True,'common_critic_sha':initialcritic,'num_envs':a.num_envs,'updates':a.updates,'bank_sha':runner.bank_sha,'seed':cfg.seed,'cost_ramp_updates':0,'low_smp_termination':False,'reset_counts':reset_counts(env),'relaxed_upward_speed':env._ft_relaxed,'reset_bank_sha':{name:hashlib.sha256(Path(f'datasets/reset_banks/{name}/train.npz').read_bytes()).hexdigest() for name in ['natural_curriculum_v1','procedural_low_v1']},'actor_noise':cfg.observations['actor'].enable_corruption,'events':list(cfg.events),'episode_seconds':cfg.episode_length_s,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()+Path(mt.__file__).read_bytes()+Path(bd.__file__).read_bytes()+Path(ft.__file__).read_bytes()+Path(v.__file__).read_bytes()+Path(ra.__file__).read_bytes()+Path(pa.__file__).read_bytes()+Path(__import__('smp.rl.tasks.getup.multiterrain_geometry',fromlist=['BOXES']).__file__).read_bytes()).hexdigest()})
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


def build_config(n=4096,bank='outputs/multiterrain_bank/train.npz',nominal=False,arm='A0'):
 from train_d_series import build_config as d_config
 cfg,agent=d_config(n,bank,nominal,'D0')
 if arm=='A7' and not nominal:cfg.seed+=1009
 cfg.events['gsi_reset']=EventTermCfg(func=pa.reset,mode='reset',params={'arm':arm,'bank_path':bank,'dynamics':not nominal,'evaluation':nominal})
 cfg.scene.sensors+=(ContactSensorCfg(name='path_hands',primary=ContactMatch(mode='geom',pattern=r'(left|right)_hand_collision$',entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='maxforce',num_slots=1),)
 cfg.metrics.pop('d_progress',None)
 cfg.metrics['r2_load_substep']=MetricsTermCfg(func=pa.sample_substep,per_substep=True)
 cfg.metrics['r2_standing']=MetricsTermCfg(func=pa.metric)
 if arm=='A1':
  for i,(name,w) in enumerate(zip(ra.NAMES[:5],ra.WEIGHTS[:5])):
   cfg.rewards['r2_'+name]=RewardTermCfg(func=ra.reward,params={'index':i},weight=w)
 if arm in pa.G_ARMS:
  cfg.rewards['plate_geometry_progress']=RewardTermCfg(func=pa.reward,params={'index':0},weight=.45)
  cfg.rewards['plate_clearance']=RewardTermCfg(func=pa.reward,params={'index':1},weight=.08)
 if arm in pa.Q_ARMS:cfg.rewards['path_quiet_feet']=RewardTermCfg(func=pa.reward,params={'index':2},weight=-.03)
 if arm in pa.L_ARMS:cfg.rewards['path_joint_stall']=RewardTermCfg(func=pa.reward,params={'index':3},weight=-.20)
 return cfg,agent

class Wrapper(RslRlVecEnvWrapper):
 def step(self,actions):
  obs,r,d,e=super().step(actions);env=self.unwrapped;log=e.setdefault('log',{})
  for s,name in enumerate(('flat','vertical_plate','free_plate','stairs')):
   mask=env._mt_scene==s
   if mask.any():
    for key,value in [('stable',env._r_stable.float()),('escaped',env._mt_escaped.float()),('smp',env._fixed_smp_score),('task',env._fixed_task_score),('standing_gate',env._ra_gate),('stalled_load',env._ra_values[:,4]),('escape_credit',env._ra_credit),('path_support',env._pa_support),('path_progress',env._pa_progress),('quiet_gate',env._pa_gate),('joint_stall',env._pa_load)]:log[f'Recovery/{name}/{key}']=value[mask].mean().detach()
  for i,name in enumerate(('legs','upper','pelvis')):log['Dynamics/mass_'+name]=env._bd_mass[:,i].mean().detach()
  for i,name in enumerate(('waist','hip','knee','ankle','arm','wrist')):log['Dynamics/gain_'+name]=env._bd_gain[:,i].mean().detach()
  log['Dynamics/nominal_fraction']=env._bd_nominal.float().mean().detach();log['Dynamics/delay_ms']=env._bd_lag.float().mean().detach()*2
  assert torch.isfinite(r).all()
  return obs,r,d,e

class Runner(MjlabOnPolicyRunner):
 active=False
 def save(self,path,infos=None):
  env=self.env.unwrapped;state={'mean':env._smp_normalizer.mean.cpu(),'count':env._smp_normalizer.count.cpu()}
  super().save(path,infos={**(infos or {}),'multiterrain':{'smp_reference':state,'source':str(self.args.checkpoint),'arm':self.args.arm,'bank_sha':self.bank_sha}})
  if not self.active:return
  ep=self.current_learning_iteration
  if ep==0:return
  combined={}
  for suite,mass in [('nominal',1.),('upper130',1.3)]:
   out=self.args.out/'validation'/str(ep)/suite;out.mkdir(parents=True,exist_ok=True)
   cmd=[os.sys.executable,'-u',__file__,'--eval','--arm',self.args.arm,'--checkpoint',str(Path(path).resolve()),'--reference',str(self.args.reference),'--out',str(out),'--num-envs','192','--steps','1000','--stress-upper',str(mass)]
   if ep%1000==0 or ep==self.args.updates-1:cmd+=['--video']
   with open(out/'eval.log','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   result=json.loads((out/'summary.json').read_text());combined[suite]=result
   for case,vals in result.items():
    for key,value in vals.items():
     if isinstance(value,(int,float)):self.logger.writer.add_scalar(f'Validation/{suite}/{case}/{key}',value,ep)
  atomic_json(self.args.out/'progress.json',{'iteration':ep,'validation':combined})

def reset_counts(env):
 return {'scene':torch.bincount(env._mt_scene,minlength=4).tolist(),'stratum':torch.bincount(env._mt_stratum,minlength=8).tolist(),'kind_low_middle_late':torch.bincount(env._ra_kind,minlength=3).tolist(),'natural_procedural':torch.bincount(env._mt_source,minlength=2).tolist(),'group':torch.bincount(env._fixed_group,minlength=6).tolist()}

def audit_reset(env,out):
 import mujoco
 cpu=mujoco.MjData(env.sim.mj_model);q=env.sim.data.qpos.cpu().numpy();mp=env.sim.data.mocap_pos.cpu().numpy();mq=env.sim.data.mocap_quat.cpu().numpy();depth=[]
 for i in range(env.num_envs):
  cpu.qpos[:]=q[i];cpu.mocap_pos[:]=mp[i];cpu.mocap_quat[:]=mq[i];mujoco.mj_forward(env.sim.mj_model,cpu);depth.append(min([float(c.dist) for c in cpu.contact]+[0.]))
 assert min(depth)>-.0021,min(depth)
 assert not (env._mt_stratum>=3).any()
 assert len(__import__('smp.rl.tasks.getup.multiterrain_geometry',fromlist=['BOXES']).BOXES)==0
 assert env.sim.data.qvel.abs().max()<1e-6
 atomic_json(out/'reset_audit.json',{'counts':reset_counts(env),'minimum_contact_distance':min(depth),'relaxed_upward_speed':env._ft_relaxed})
 np.savez_compressed(out/'initial.npz',qpos=q,scene=env._mt_scene.cpu().numpy(),direction=env._mt_direction.cpu().numpy(),stratum=env._mt_stratum.cpu().numpy(),kind=env._ra_kind.cpu().numpy(),source=env._mt_source.cpu().numpy())

def evaluate(env,wrapper,policy,obs,a):
 hold=torch.zeros(env.num_envs,device=env.device);best=hold.clone();alive=torch.ones_like(hold,dtype=torch.bool);ever_invalid=~alive;escape=~alive
 seen_up=~alive;seen_hold=~alive;fall_time=hold.clone();refall=~alive;after_hold=~alive
 switches=hold.clone();path=hold.clone();previous=torch.zeros(env.num_envs,2,device=env.device,dtype=torch.bool);seen=previous.clone()
 selected=[int(torch.where((env._mt_stratum==st)&(env._mt_direction==dr))[0][0]) for st in range(3) for dr in range(4)]
 writers={};traces=[]
 phase_time=torch.zeros(env.num_envs,2,device=env.device);phase_switch=phase_time.clone();phase_slip=phase_time.clone();phase_body=phase_time.clone()
 pre_time=hold.clone();pre_support=hold.clone();pre_head=hold.clone();board_lift=hold.clone();board_tilt=hold.clone();stalltime=hold.clone();reversals=hold.clone();previous_dq=torch.zeros_like(env.scene['robot'].data.joint_vel)
 rows=torch.arange(env.num_envs,device=env.device);gids=torch.stack(env._mt_plate_ids)[(env._mt_scene==2).long()];board_start=env.sim.data.geom_xpos[rows,gids,2].clone()

 if a.video:
  import imageio.v2 as imageio
  for st in range(3):writers[st]=imageio.get_writer(str(a.out/f'{STRATA[st]}_20s.mp4'),fps=25,codec='libx264',quality=7)
 try:
  for step in range(a.steps):
   with torch.inference_mode():obs,_,done,_=wrapper.step(policy(obs))
   alive&=~done.bool();plate=(env._mt_scene==1)|(env._mt_scene==2);ever_invalid|=env._mt_invalid
   stable=env._r_stable&alive&(~plate|(env._mt_escaped&~ever_invalid))
   hold=torch.where(stable,hold+env.step_dt,0);best=torch.maximum(best,hold);escape|=env._mt_escaped&alive
   r=env.scene['robot'];z=mt.height(env);u=-r.data.projected_gravity_b[:,2]
   seen_up|=(z>=1.15)&(u>=.93);seen_hold|=hold>=1.
   fallen=(z<.65)|(u<.5);fall_time=torch.where(fallen,fall_time+env.step_dt,0.)
   refall|=seen_up&(fall_time>=.2);after_hold|=seen_hold&(fall_time>=.2)
   late=(z>.78)&(u>.7)&alive;contact=mt.ground_force(env,'quality_feet')[...,2].abs()>20
   changes=((contact!=previous)&seen&late[:,None]).sum(-1)
   switches+=changes
   transition=late&~seen_hold;standing=late&seen_hold
   foot_speed=r.data.body_link_lin_vel_w[:,env._r_feet,:2].norm(dim=-1)
   for k,phase in enumerate((transition,standing)):
    phase_time[:,k]+=phase*env.step_dt;phase_switch[:,k]+=changes*phase
    phase_slip[:,k]+=(foot_speed*contact).sum(-1)*phase*env.step_dt
    phase_body[:,k]+=r.data.root_link_ang_vel_w.norm(dim=-1)*phase*env.step_dt
   pre=plate&~escape&alive;pre_time+=pre*env.step_dt;pre_support+=env._pa_support*pre*env.step_dt;pre_head+=z*pre*env.step_dt
   board_lift=torch.maximum(board_lift,(env.sim.data.geom_xpos[rows,gids,2]-board_start).clamp_min(0)*plate)
   rot=env.sim.data.geom_xmat[rows,gids];board_tilt=torch.maximum(board_tilt,torch.acos(rot[:,2,2].abs().clamp(0,1))*plate)
   high=(r.data.qfrc_actuator.abs()>.7*env._r_limits)
   stalltime+=((high&(r.data.joint_vel.abs()<.3)).any(-1))*alive*env.step_dt
   reversals+=((r.data.joint_vel*previous_dq<0)&(r.data.joint_vel.abs()>.05)&(previous_dq.abs()>.05)&high).sum(-1)*alive
   previous_dq=r.data.joint_vel.clone()
   seen=late[:,None].expand_as(contact).clone();previous=contact.clone()
   path+=r.data.root_link_lin_vel_w[:,:2].norm(dim=-1)*env.step_dt*late
   traces.append(env.sim.data.qpos[selected].cpu().numpy())
   if writers and step%2==0:
    from PIL import Image,ImageDraw
    for st,w in writers.items():
     tiles=[]
     for dr in range(4):
      j=selected[st*4+dr];env.cfg.viewer.env_idx=j;im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,35),fill='black');draw.text((5,8),f'{a.arm} {STRATA[st]} {DIRECTIONS[dr]} t={step*.02:.2f} hold={hold[j]:.2f}',fill='white');tiles.append(np.asarray(im))
     w.append_data(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0))
 finally:
  for w in writers.values():w.close()
 masks={name:env._mt_scene==s for s,name in enumerate(('flat','vertical_plate','free_plate','stairs'))}
 masks.update({name:env._mt_stratum==s for s,name in enumerate(STRATA[:6])})
 for name,mask in list(masks.items()):
  for d,dr in enumerate(DIRECTIONS):masks[name+'/'+dr]=mask&(env._mt_direction==d)
 summary={}
 for name,mask in masks.items():
  if not mask.any():continue
  vals={'n':int(mask.sum()),'stable_1s':float((best[mask]>=1-1e-4).float().mean()),'stable_10s':float((best[mask]>=10-1e-4).float().mean()),'stable_10s_no_refall':float(((best[mask]>=10-1e-4)&~refall[mask]).float().mean()),'refall_after_upright':float(refall[mask].float().mean()),'refall_after_hold1s':float(after_hold[mask].float().mean()),'escaped':float(escape[mask].float().mean()),'invalid_plate':float(ever_invalid[mask].float().mean()),'alive':float(alive[mask].float().mean()),'late_contact_switches_mean':float(switches[mask].mean()),'late_base_xy_path_mean':float(path[mask].mean())}
  for key,value in [('tau',env._ra_taupeak),('speed',env._ra_speedpeak),('power',env._ra_powerpeak),('high_load_time',env._ra_hightime),('longest_high_load',env._ra_longest)]:
   vals[key+'_peak_p95']=float(np.percentile(value[mask].amax(-1).cpu().numpy(),95));vals[key+'_per_joint_p95']=np.percentile(value[mask].cpu().numpy(),95,axis=0).tolist()
  for k,phase in enumerate(('transition','post_hold1s')):
   vals[phase+'_time_mean']=float(phase_time[mask,k].mean())
   vals[phase+'_contact_switches_mean']=float(phase_switch[mask,k].mean())
   vals[phase+'_foot_slip_path_mean']=float(phase_slip[mask,k].mean())
   vals[phase+'_angular_path_mean']=float(phase_body[mask,k].mean())
  vals['pre_escape_hand_support_fraction']=float((pre_support[mask]/pre_time[mask].clamp_min(.02)).mean())
  vals['pre_escape_head_height_mean']=float((pre_head[mask]/pre_time[mask].clamp_min(.02)).mean())
  vals['board_lift_p95']=float(torch.quantile(board_lift[mask],.95))
  vals['board_tilt_rad_p95']=float(torch.quantile(board_tilt[mask],.95))
  vals['low_speed_high_load_time_p95']=float(torch.quantile(stalltime[mask],.95))
  vals['high_load_direction_changes_mean']=float(reversals[mask].mean())
  summary[name]=vals
 atomic_json(a.out/'summary.json',summary)
 np.savez_compressed(a.out/'trace.npz',qpos=np.asarray(traces),selected=selected)
 print('EVALUATION_COMPLETE',json.dumps({k:{x:v for x,v in val.items() if isinstance(v,(int,float))} for k,val in summary.items() if '/' not in k}),flush=True)

if __name__=='__main__':main()
