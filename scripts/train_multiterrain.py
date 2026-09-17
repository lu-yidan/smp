"""Matched P5 / FT12k / L4 continuation, with mandatory independent evaluation."""
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
from smp.rl.tasks.getup.multiterrain_geometry import terrain_spec,free_plate_spec,scene_spec,STRATA,DIRECTIONS
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN

def build_config(n=4096,bank='outputs/multiterrain_bank/train.npz',nominal=False):
 cfg,agent=base_config(Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),n)
 cfg.scene.spec_fn=scene_spec;cfg.scene.env_spacing=0.
 cfg.scene.entities['escape_obstacle']=EntityCfg(spec_fn=p.plate_spec,init_state=EntityCfg.InitialStateCfg(pos=(20,20,.8),joint_pos={'escape_plate_slide':0.},joint_vel={'escape_plate_slide':0.}))
 cfg.scene.entities['free_obstacle']=EntityCfg(spec_fn=free_plate_spec,init_state=EntityCfg.InitialStateCfg(pos=(20,20,.1)))
 ground=ContactMatch(mode='body',pattern='surfaces')
 cfg.scene.sensors+=tuple(replace(s,name=s.name+'_terrain',secondary=ground) for s in cfg.scene.sensors if s.name in ('quality_feet','quality_other'))
 for name,entity,body in [('guided_contact','escape_obstacle','escape_plate'),('free_contact','free_obstacle','plate')]:
  cfg.scene.sensors+= (ContactSensorCfg(name=name,primary=ContactMatch(mode='geom',pattern=COLLISION_PATTERN,entity='robot'),secondary=ContactMatch(mode='body',pattern=body,entity=entity),fields=('found','force','dist'),reduce='mindist',num_slots=1),)
 cfg.events['gsi_reset']=EventTermCfg(func=mt.reset,mode='reset',params={'bank_path':bank})
 cfg.events['multiterrain_phase']=EventTermCfg(func=mt.update,mode='step')
 cfg.terminations.pop('smp_too_low',None);cfg.terminations['invalid_plate']=TerminationTermCfg(func=mt.invalid)
 cfg.rewards['task_smp_product'].func=mt.task
 for i,(name,w) in enumerate([('geometry_progress',.45),('clearance',.08),('completion',.60),('force',-.03),('separation',.01)]):
  cfg.rewards['plate_'+name]=RewardTermCfg(func=mt.escape_reward,params={'index':i},weight=w)
 cfg.sim.nconmax=256;cfg.sim.njmax=4000;cfg.episode_length_s=10.
 if nominal:
  cfg.observations['actor'].enable_corruption=False
  for key in list(cfg.events):
   if cfg.events[key].mode in ('startup','interval') and key!='init_smp_state':cfg.events.pop(key)
  cfg.terminations.pop('invalid_plate');cfg.episode_length_s=1000.
 cfg.viewer.width=640;cfg.viewer.height=480;cfg.viewer.max_extra_envs=0
 return cfg,agent

class Wrapper(RslRlVecEnvWrapper):
 def step(self,actions):
  obs,r,d,e=super().step(actions);env=self.unwrapped;log=e.setdefault('log',{})
  for st,name in enumerate(STRATA):
   mask=env._mt_stratum==st
   if mask.any():
    for key,value in [('stable',env._r_stable.float()),('escaped',env._mt_escaped.float()),('invalid',env._mt_invalid.float()),('task',env._fixed_task_score),('smp',env._fixed_smp_score),('product',env._plate_product),('plate_force',env._mt_force),('relative_head_z',mt.height(env))]:log[f'Terrain/{name}/{key}']=value[mask].float().mean().detach()
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
  out=self.args.out/'validation'/str(ep);cmd=[os.sys.executable,'-u',__file__,'--eval','--arm',self.args.arm,'--checkpoint',str(Path(path).resolve()),'--reference',str(self.args.reference),'--out',str(out),'--num-envs','256','--steps','1000']
  if ep%1000==0 or ep==self.args.updates-1:cmd+=['--video']
  out.mkdir(parents=True,exist_ok=True)
  with open(out/'eval.log','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  result=json.loads((out/'summary.json').read_text())
  for case,vals in result.items():
   for key,value in vals.items():self.logger.writer.add_scalar(f'Validation/{case}/{key}',value,ep)
  atomic_json(self.args.out/'progress.json',{'iteration':ep,'validation':result})

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
 assert (gravity[:,:2].abs().amax(-1)>.5).all()
 assert env.scene.env_origins.abs().max()==0
 assert all(int((env._mt_scene==s).sum())==env.num_envs//4 for s in range(4))
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
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['T_P5','T_FT12k','T_L4'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=10000);p.add_argument('--eval',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--video',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
 a.out=a.out.resolve();a.checkpoint=a.checkpoint.resolve();a.reference=a.reference.resolve();a.out.mkdir(parents=True,exist_ok=a.eval)
 bank=f'outputs/multiterrain_bank/{"validation" if a.eval else "train"}.npz';cfg,agent=build_config(a.num_envs,bank,a.eval)
 random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
 agent.logger='tensorboard' if a.eval or a.preflight else 'wandb';agent.upload_model=False;agent.run_name=a.arm;agent.save_interval=500;agent.max_iterations=a.updates
 dump_yaml(a.out/'env.yaml',asdict(cfg));dump_yaml(a.out/'agent.yaml',asdict(agent));env=None
 try:
  env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None)
  wrapper=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(wrapper,asdict(agent),str(a.out),'cuda:0');runner.args=a;runner.bank_sha=hashlib.sha256(Path(bank).read_bytes()).hexdigest()
  parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False);initialcritic=tensor_hash(runner.alg.save()['critic_state_dict'])
  runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});verify_nested(parent['actor_state_dict'],runner.alg.save()['actor_state_dict']);assert initialcritic==tensor_hash(runner.alg.save()['critic_state_dict'])
  ref=torch.load(a.reference,map_location='cpu',weights_only=False)['infos']['scratch_tradeoffs'];env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.copy_(ref['count'].to(env.device))
  random.seed(cfg.seed+177);np.random.seed(cfg.seed+177);torch.manual_seed(cfg.seed+177)
  obs,_=env.reset();v.init(env);env._v_start_counter=-12000;env._v_ramp_updates=500
  assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
  audit_reset(env,a.out)
  if a.preflight:
   ids=torch.tensor([0,a.num_envs//4,a.num_envs//2,3*a.num_envs//4],device=env.device)
   mask=torch.ones(a.num_envs,dtype=torch.bool,device=env.device);mask[ids]=False
   before=env.sim.data.qpos[mask].clone();env._reset_idx(ids)
   assert torch.equal(before,env.sim.data.qpos[mask]),'partial reset polluted other worlds'
   obs,_=env.reset()
   atomic_json(a.out/'partial_reset_pass.json',{'untouched_worlds':int(mask.sum())})
  atomic_json(a.out/'launch.json',{'arm':a.arm,'checkpoint':str(a.checkpoint),'sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'actor_exact':True,'fresh_critic':True,'fresh_optimizer':True,'common_critic_sha':initialcritic,'num_envs':a.num_envs,'updates':a.updates,'bank_sha':runner.bank_sha,'actor_noise':cfg.observations['actor'].enable_corruption,'events':list(cfg.events),'episode_seconds':cfg.episode_length_s,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()+Path(mt.__file__).read_bytes()).hexdigest()})
  if a.eval:evaluate(env,wrapper,runner.get_inference_policy(),obs,a)
  else:
   runner.save(str(a.out/'initial.pt'));runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;runner.save(str(a.out/'final.pt'))
   atomic_json(a.out/'completed.json',{'iteration':runner.current_learning_iteration})
 except BaseException as e:atomic_json(a.out/'failed.json',{'error':repr(e)});raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
