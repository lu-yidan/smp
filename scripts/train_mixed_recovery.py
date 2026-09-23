"""M0/M1/M2 mixed-scene experiment. Formal runs require checked banks and transfer."""
import argparse,hashlib,json,os,random,subprocess
from pathlib import Path
from dataclasses import asdict,replace,fields
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor.contact_sensor import ContactMatch,ContactSensorCfg
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from train_v33_reward_transfer import build_config as base_config
from train_fixed_low_ablation import atomic_json
from train_r1_quality import verify_nested
from train_scratch_tradeoffs import tensor_hash
from smp.recovery import mixed_task as mix
from smp.recovery.mixed_geometry import SCENES,scene_spec
from smp.rl.tasks.getup import balanced_dynamics as bd
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN

SOURCES={
 'R2':('/root/workplace/smp-ft-speed/logs/rsl_rl/ft_prone_speed/formal_20260918_ft_speed_v1/FT_R2/model_9000.pt','8b4889f80c6b7cc675f6e9b1e98f2d4a1886a15e372f86070f63c329ba9ca5d1'),
 'A6':('/root/workplace/smp-r2-v33-path/logs/rsl_rl/v33_path_ablation/formal_20260920_r2_v33path_v1/A6/model_9999.pt','4a2d4c8f0e710f760b0996237002824b4fd05872482ee96fb94f7330d09c8fd8')}

def build_config(n,arm,bank,evaluation=False,seed=20260923):
 cfg,agent=base_config(Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),n)
 cfg.seed=agent.seed=seed;cfg.scene.spec_fn=scene_spec;cfg.scene.env_spacing=0.
 cfg.events['gsi_reset']=EventTermCfg(func=mix.reset,mode='reset',params={'arm':arm,'bank_path':bank,'evaluation':evaluation})
 cfg.terminations.pop('smp_too_low',None);cfg.terminations.pop('stood_up',None)
 cfg.terminations['invalid_obstacle']=TerminationTermCfg(func=mix.invalid)
 cfg.rewards['task_smp_product'].func=mix.task
 for i,(name,w) in enumerate(zip(('path','clearance','completion','force','separation','quiet_feet','joint_stall'),(.45,.08,.60,-.03,.01,-.03,-.20))):
  cfg.rewards['mixed_'+name]=RewardTermCfg(func=mix.reward,params={'index':i},weight=w)
 sensors=list(cfg.scene.sensors)
 for suffix,body in (('support','mix_support'),('guide','mix_guide'),('free0','mix_free0'),('free1','mix_free1'),('fixed','mix_fixed')):
  sensors.extend(replace(s,name=s.name+'_'+suffix,secondary=ContactMatch(mode='body',pattern=body)) for s in cfg.scene.sensors if s.name in ('quality_feet','quality_other'))
  if suffix!='support':sensors.append(ContactSensorCfg(name='mix_contact_'+suffix,primary=ContactMatch(mode='geom',pattern=COLLISION_PATTERN,entity='robot'),secondary=ContactMatch(mode='body',pattern=body),fields=('found','force','dist'),reduce='mindist',num_slots=1))
 for name,secondary in (('mix_hands',ContactMatch(mode='geom',pattern='terrain')),('mix_hands_support',ContactMatch(mode='body',pattern='mix_support'))):
  sensors.append(ContactSensorCfg(name=name,primary=ContactMatch(mode='geom',pattern=r'(left|right)_hand_collision$',entity='robot'),secondary=secondary,fields=('found','force'),reduce='netforce',num_slots=1))
 cfg.scene.sensors=tuple(sensors);cfg.metrics['mixed_substep']=MetricsTermCfg(func=mix.sample_substep,per_substep=True)
 cfg.episode_length_s=10.;cfg.sim.nconmax=512;cfg.sim.njmax=4000
 action=cfg.actions['joint_pos'];cfg.actions['joint_pos']=bd.DelayedPositionActionCfg(**{f.name:getattr(action,f.name) for f in fields(action)})
 cfg.viewer.width=640;cfg.viewer.height=480;cfg.viewer.max_extra_envs=0
 if evaluation:
  cfg.observations['actor'].enable_corruption=False;cfg.episode_length_s=1000.
  for key in list(cfg.events):
   if cfg.events[key].mode in ('startup','interval') and key!='init_smp_state':cfg.events.pop(key)
  cfg.terminations.pop('invalid_obstacle',None)
 return cfg,agent

class Wrapper(RslRlVecEnvWrapper):
 def step(self,actions):
  obs,r,done,extras=super().step(actions);env=self.unwrapped;s=env._mix;log=extras.setdefault('log',{})
  st=mix.stable(env)
  for i,name in enumerate(SCENES):
   mask=s.tags[:,0]==i
   for key,val in (('stable',st),('escaped',s.escaped),('blocked',s.blocked.any(-1)),('smp',env._fixed_smp_score),('task',env._fixed_task_score)):
    if mask.any():log[f'Recovery/{name}/{key}']=val[mask].float().mean().detach()
  log['Mixed/invalid']=s.invalid.float().mean().detach();log['Mixed/height']=mix.height(env).mean().detach()
  for i,name in enumerate(('legs','upper','pelvis')):log['Dynamics/mass_'+name]=env._bd_mass[:,i].mean().detach()
  for i,name in enumerate(('waist','hip','knee','ankle','arm','wrist')):log['Dynamics/gain_'+name]=env._bd_gain[:,i].mean().detach()
  log['Dynamics/delay_ms']=env._bd_lag.float().mean().detach()*2
  assert torch.isfinite(r).all() and torch.isfinite(obs['actor']).all()
  return obs,r,done,extras

def audit(env,out):
 import mujoco
 s=env._mix;m=env.sim.mj_model;d=mujoco.MjData(m)
 saved={k:getattr(m,k).copy() for k in mix.FIELDS};depth=[];position_error=[];rotation_error=[]
 try:
  for i in range(env.num_envs):
   for k in mix.FIELDS:
    arr=getattr(env.sim.model,k)[i].cpu().numpy();getattr(m,k)[:]=arr.reshape(getattr(m,k).shape)
   d.qpos[:]=env.sim.data.qpos[i].cpu().numpy();d.qvel[:]=0;d.mocap_pos[:]=env.sim.data.mocap_pos[i].cpu().numpy();d.mocap_quat[:]=env.sim.data.mocap_quat[i].cpu().numpy();mujoco.mj_forward(m,d);depth.append(min([c.dist for c in d.contact]+[0.]))
   gids=s.gids.cpu().numpy();position_error.append(float(np.abs(d.geom_xpos[gids]-env.sim.data.geom_xpos[i,s.gids].cpu().numpy()).max()));rotation_error.append(float(np.abs(d.geom_xmat[gids].reshape(-1,3,3)-env.sim.data.geom_xmat[i,s.gids].cpu().numpy()).max()))
 finally:
  for k,value in saved.items():getattr(m,k)[:]=value
 assert min(depth)>=-.0021,min(depth)
 assert max(position_error)<1e-5 and max(rotation_error)<1e-5,(max(position_error),max(rotation_error))
 assert env.sim.data.qvel.abs().max()<1e-6
 mismatch=s.blocked.any(-1)!=s.active.any(-1)
 if mismatch.any():
  atomic_json(out/'unconstrained_initial.json',{'tags':s.tags[mismatch].tolist(),'rows':s.sample_indices[mismatch].tolist(),'active':s.active[mismatch].tolist(),'blocked':s.blocked[mismatch].tolist(),'root':env.scene['robot'].data.root_link_pos_w[mismatch].tolist(),'roof_model':env.sim.model.geom_pos[mismatch][:,s.gids[4]].tolist(),'roof_actual':env.sim.data.geom_xpos[mismatch][:,s.gids[4]].tolist()})
 assert not mismatch.any(), 'Overhead cohorts must start constrained; see unconstrained_initial.json'
 atomic_json(out/'reset_audit.json',{'minimum_depth':float(min(depth)),'cpu_gpu_position_error':max(position_error),'cpu_gpu_rotation_error':max(rotation_error),'counts':s.counts,'keys':s.keys,'actor_dim':93,'qvel_max':float(env.sim.data.qvel.abs().max())})
 np.savez_compressed(out/'initial.npz',qpos=env.sim.data.qpos.cpu().numpy(),rows=s.sample_indices.cpu().numpy(),tags=s.tags.cpu().numpy())

class Runner(MjlabOnPolicyRunner):
 active=False
 def save(self,path,infos=None):
  env=self.env.unwrapped
  super().save(path,infos={**(infos or {}),'mixed_recovery':{'arm':self.args.arm,'smp_reference':{'mean':env._smp_normalizer.mean.cpu(),'count':env._smp_normalizer.count.cpu()},'source':str(self.args.checkpoint),'bank_sha':self.bank_sha}})
  it=self.current_learning_iteration
  if not self.active or it==0:return
  out=self.args.out/'validation'/str(it);out.mkdir(parents=True,exist_ok=True)
  cmd=[os.sys.executable,'-u',__file__,'--arm',self.args.arm,'--checkpoint',str(Path(path).resolve()),'--out',str(out),'--eval','--num-envs','320','--steps','1000']
  if it%2000==0 or it==self.args.updates-1:cmd+=['--video']
  with (out/'eval.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  report=json.loads((out/'summary.json').read_text());atomic_json(self.args.out/'progress.json',{'iteration':it,'validation':report})
  for scene,vals in report.items():
   for key,val in vals.items():self.logger.writer.add_scalar(f'Validation/{scene}/{key}',val,it)

def evaluate(env,wrapper,policy,obs,a):
 s=env._mix;n=env.num_envs;hold=torch.zeros(n,device=env.device);best=hold.clone();alive=torch.ones(n,dtype=torch.bool,device=env.device);seen=~alive;refall=~alive
 peaks=torch.zeros(n,29,3,device=env.device);path=hold.clone();last=rroot=env.scene['robot'].data.root_link_pos_w[:,:2].clone();writers={};selected={}
 if a.video:
  import imageio.v2 as imageio
  for sc,name in enumerate(SCENES):
   if name not in a.video_scenes:continue
   ids=[torch.where((s.tags[:,0]==sc)&(s.tags[:,2]==dr))[0] for dr in range(4)]
   if all(len(x) for x in ids):selected[sc]=[int(x[0]) for x in ids];writers[sc]=imageio.get_writer(str(a.out/(name+'_20s.mp4')),fps=25,codec='libx264',quality=7)
 try:
  for step in range(a.steps):
   with torch.inference_mode():obs,_,done,_=wrapper.step(policy(obs))
   trial_peaks=torch.where(done.bool()[:,None,None],s.last_episode_peaks,s.peaks)
   peaks=torch.maximum(peaks,torch.where(alive[:,None,None],trial_peaks,peaks))
   alive&=~done.bool();stable=mix.stable(env)&alive;hold=torch.where(stable,hold+.02,0.);best=torch.maximum(best,hold)
   seen|=hold>=1;refall|=seen&alive&((mix.height(env)<.65)|(-env.scene['robot'].data.projected_gravity_b[:,2]<.5))
   xy=env.scene['robot'].data.root_link_pos_w[:,:2];path+=(xy-last).norm(dim=-1)*alive;last=xy.clone()
   if writers and step%2==0:
    from PIL import Image,ImageDraw
    for sc,w in writers.items():
     tiles=[]
     for j in selected[sc]:
      env.cfg.viewer.env_idx=j;im=Image.fromarray(env.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,640,30),fill='black');dr.text((5,7),f'{a.arm} {SCENES[sc]} dir={int(s.tags[j,2])} t={step*.02:.2f} hold={float(hold[j]):.2f}',fill='white');tiles.append(np.asarray(im))
     w.append_data(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0))
 finally:
  for w in writers.values():w.close()
 report={}
 for sc,name in enumerate(SCENES):
  mask=s.tags[:,0]==sc;vals={'n':int(mask.sum()),'stable_1s':float((best[mask]>=1-1e-4).float().mean()),'stable_10s':float((best[mask]>=10-1e-4).float().mean()),'refall':float(refall[mask].float().mean()),'path_mean':float(path[mask].mean())}
  for j,key in enumerate(('tau','speed','power')):vals[key+'_peak_p95']=float(torch.quantile(peaks[mask,:,j].amax(-1),.95))
  report[name]=vals
 np.savez_compressed(a.out/'per_trial.npz',peaks=peaks.cpu().numpy(),best_hold=best.cpu().numpy(),refall=refall.cpu().numpy(),path=path.cpu().numpy(),tags=s.tags.cpu().numpy())
 atomic_json(a.out/'summary.json',report);print('EVALUATION_COMPLETE',json.dumps(report),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=mix.ARMS,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=10000);p.add_argument('--preflight',action='store_true');p.add_argument('--eval',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--video-scenes',nargs='+',choices=SCENES,default=list(SCENES));p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
 source,sha=SOURCES['A6' if a.arm=='M2_A6_mix' else 'R2']
 if a.checkpoint is None:a.checkpoint=Path(source)
 if not a.eval:assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha
 a.checkpoint=a.checkpoint.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=a.eval)
 bank=f'outputs/mixed_bank_v4/{"validation" if a.eval else "train"}.npz';manifest=json.loads(Path(bank).with_suffix('.json').read_text());bank_sha=hashlib.sha256(Path(bank).read_bytes()).hexdigest();assert manifest['sha256']==bank_sha
 cfg,agent=build_config(a.num_envs,a.arm,bank,a.eval)
 random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
 agent.logger='tensorboard' if a.eval or a.preflight else 'wandb';agent.upload_model=False;agent.wandb_project='smp';agent.run_name=a.arm;agent.save_interval=500;agent.max_iterations=a.updates
 dump_yaml(a.out/'env.yaml',asdict(cfg));dump_yaml(a.out/'agent.yaml',asdict(agent));env=None
 try:
  env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None);wrapper=Wrapper(env,clip_actions=agent.clip_actions)
  runner=Runner(wrapper,asdict(agent),str(a.out),'cuda:0');runner.args=a;runner.bank_sha=bank_sha
  parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False);critic=tensor_hash(runner.alg.save()['critic_state_dict'])
  runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});verify_nested(parent['actor_state_dict'],runner.alg.save()['actor_state_dict']);assert critic==tensor_hash(runner.alg.save()['critic_state_dict'])
  ref=json.loads(Path('configs/recovery_study/smp_reward_reference_v1.json').read_text())
  env._smp_normalizer.mean.copy_(torch.tensor(ref['mean'],device=env.device));env._smp_normalizer.count.copy_(torch.tensor(ref['count'],device=env.device))
  env.common_step_counter=0;obs,_=env.reset();assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960;audit(env,a.out)
  if a.preflight:
   atomic_json(a.out/'dynamics_audit.json',bd.audit_dynamics(env));obs,_=env.reset()
   ids=torch.arange(0,env.num_envs,8,device=env.device);mask=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);mask[ids]=False
   before=env.sim.data.qpos[mask].clone();sz=env.sim.model.geom_size[mask].clone();mass=env.sim.model.body_mass[mask].clone();env._reset_idx(ids)
   assert torch.equal(before,env.sim.data.qpos[mask]) and torch.equal(sz,env.sim.model.geom_size[mask]) and torch.equal(mass,env.sim.model.body_mass[mask])
   obs,_=env.reset();atomic_json(a.out/'partial_reset_pass.json',{'pass':True})
  if not a.eval:assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
  meta={'arm':a.arm,'checkpoint':str(a.checkpoint),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'actor_exact':True,'actor_sha':tensor_hash(runner.alg.save()['actor_state_dict']),'initial_qpos_sha':hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),'fresh_critic':True,'fresh_optimizer':True,'critic_sha':critic,'seed':cfg.seed,'num_envs':a.num_envs,'updates':a.updates,'bank_sha256':bank_sha,'smp_reference_sha256':hashlib.sha256(Path('configs/recovery_study/smp_reward_reference_v1.json').read_bytes()).hexdigest(),'actor_noise':cfg.observations['actor'].enable_corruption,'episode_s':cfg.episode_length_s,'events':list(cfg.events),'reward_weights':{k:v.weight for k,v in cfg.rewards.items()},'cohort_keys':env._mix.keys,'cohort_counts':env._mix.counts,'geometry_curriculum_updates':[0,1500,3000],'dynamics_width_initial_final':[.1,.2],'delay_ms':[0,10],'nominal_dynamics_fraction':.25,'code_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sorted([Path(__file__),*Path('src/smp').rglob('*.py')]))).hexdigest()}
  atomic_json(a.out/'launch.json',meta);print('TRANSFER_AND_RESET_VERIFIED',json.dumps(meta),flush=True)
  if a.eval:evaluate(env,wrapper,runner.get_inference_policy(),obs,a)
  else:
   runner.save(str(a.out/'initial.pt'));runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;runner.save(str(a.out/'final.pt'));atomic_json(a.out/'completed.json',{'iteration':runner.current_learning_iteration})
 except BaseException as e:atomic_json(a.out/'failed.json',{'error':repr(e)});raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
