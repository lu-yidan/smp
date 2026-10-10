"""Archived V34 93D task with robot spec-only replacement; no A6 task imports."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,copy,json,random,hashlib,subprocess,sys,fcntl
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.sensor.contact_sensor import ContactSensorCfg,ContactMatch
from mjlab.utils.os import dump_yaml
import smp.rl.tasks
from smp.rl.tasks.getup import master_deployment_contract as dc
TASK='Smp-Getup-Escape-Plate-V34-93D-G1'
PARENT='/root/workplace/smp-v34-table-mix/artifacts/v34_gate6000.pt'
SHA='471fb41444a5d8f2bce1f93b0e2d65a613e86c1d417ef4cba36ba48b9779474b'
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,default=str))
def configure(n,seed,model,evaluation):
 cfg=load_env_cfg(TASK,play=False);cfg.scene.num_envs=n;cfg.seed=seed
 if model=='deploy':
  robot=copy.deepcopy(cfg.scene.entities['robot']);robot.spec_fn=dc.deployment_robot_spec;robot.collisions=();cfg.scene.entities['robot']=robot
  cfg.scene.sensors=tuple(replace(s,primary=replace(s.primary,pattern=dc.COLLISION_PATTERN)) if s.name=='robot_obstacle_contact' else s for s in cfg.scene.sensors)
  for term in list(cfg.events.values())+list(cfg.rewards.values()):
   if 'collision_geom_pattern' in term.params:term.params['collision_geom_pattern']=dc.COLLISION_PATTERN
 # Compilation is an execution optimization, not a task change.
 cfg.events['init_smp_state'].params['compile_model']=False
 cfg.sim.contact_sensor_maxmatch=256
 cfg.viewer.width=640;cfg.viewer.height=480;cfg.viewer.max_extra_envs=0
 cfg.scene.sensors+=tuple(ContactSensorCfg(name='sr3_'+s,primary=ContactMatch(mode='geom',pattern='^'+s+'_foot[1-7]_collision$',entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force','normal','tangent'),reduce='maxforce',num_slots=1,global_frame=True) for s in ['left','right'])
 if evaluation:
  cfg.episode_length_s=1000;cfg.observations['actor'].enable_corruption=False
  for k in ['foot_friction','encoder_bias','base_com']:cfg.events.pop(k,None)
 return cfg

def evaluate(env,w,policy,obs,out,video):
 live=torch.ones(env.num_envs,device=env.device,dtype=torch.bool);passed=~live;hold=torch.zeros(env.num_envs,device=env.device);best=hold.clone();maxz=hold.clone();escaped=~live
 pose=env._robust_reset_type.clone();r=env.scene['robot'];head=r.find_sites(['head'])[0][0];writers={}
 if video:
  import imageio.v2 as imageio
  for p in (1,2):
   ids=torch.where(pose==p)[0]
   if len(ids):writers[int(ids[0])]=imageio.get_writer(str(out/f'pose{p}.mp4'),fps=25,codec='libx264',ffmpeg_params=['-threads','1'])
 try:
  for k in range(1150):
   with torch.inference_mode():obs,_,done,_=w.step(policy(obs))
   live&=~done.bool();z=r.data.site_pos_w[:,head,2]-env.scene.env_origins[:,2];u=-r.data.projected_gravity_b[:,2]
   force=torch.stack([env.scene['sr3_'+s].data.force[...,2].abs().amax(-1) for s in ['left','right']]).amin(0)
   good=live&(z>=1.15)&(u>=.9)&(force>20);hold=torch.where(good,hold+1,0);best=torch.maximum(best,hold);passed|=(hold>=150)&((k+1-hold)<=1000);maxz=torch.maximum(maxz,torch.where(live,z,0));escaped|=live&(env._escape_phase==3)
   if video and k%2==1:
    from PIL import Image,ImageDraw
    for j in list(writers):
     if not live[j]:writers.pop(j).close();continue
     env.cfg.viewer.env_idx=j;cam=env._offline_renderer._cam;root=r.data.root_link_pos_w[j].cpu().numpy();cam.lookat[:]=[root[0],root[1],.45];cam.elevation=-12;cam.azimuth=55;cam.distance=3
     im=Image.fromarray(env.render());ImageDraw.Draw(im).text((8,8),f'V34 model-only pose={int(pose[j])} t={(k+1)*.02:.2f}s',fill='white');writers[j].append_data(np.asarray(im))
 finally:
  for wr in writers.values():wr.close()
 result={}
 for p,name in [(None,'all'),(1,'prone'),(2,'supine')]:
  m=torch.ones_like(live) if p is None else pose==p
  result[name]=dict(n=int(m.sum()),sr3=float(passed[m].float().mean()),terminated=float((~live[m]).float().mean()),escaped=float(escaped[m].float().mean()),max_head_mean=float(maxz[m].mean()))
 write(out/'summary.json',result)
 np.savez_compressed(out/'trials.npz',pose=pose.cpu(),sr3=passed.cpu(),alive=live.cpu(),max_head=maxz.cpu())

class Runner(MjlabOnPolicyRunner):
 active=False
 def save(self,path,infos=None):
  e=self.env.unwrapped
  super().save(path,infos={**(infos or {}),'model_only':dict(smp_mean=e._smp_normalizer.mean.cpu(),smp_count=e._smp_normalizer.count.cpu(),common_step_counter=e.common_step_counter)})
  if not self.active:return
  i=self.current_learning_iteration;write(self.args.out/'progress.json',dict(iteration=i))
  if i==0 or (i%500 and i!=self.args.updates-1):return
  out=self.args.out/'validation'/str(i);out.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u',__file__,'--eval','--model',self.args.model,'--checkpoint',str(Path(path).resolve()),'--out',str(out),'--num-envs','512','--seed','20261071']
  if i%1000==0 or i==self.args.updates-1:cmd+=['--video']
  with open('/tmp/a6-escape-scale-eval-gpu5.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   with open(out/'console.log','w') as f:subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',MUJOCO_EGL_DEVICE_ID='5'),stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads((out/'summary.json').read_text());write(self.args.out/'progress.json',dict(iteration=i,validation=d))
  for key,val in d.items():self.logger.writer.add_scalar('Validation/'+key+'/SR3',val['sr3'],i)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',default=PARENT);p.add_argument('--model',choices=['original','deploy'],default='deploy');p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=6000);p.add_argument('--seed',type=int,default=20261008);p.add_argument('--eval',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=a.eval)
 if not a.eval:assert hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()==SHA
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 cfg=configure(a.num_envs,a.seed,a.model,a.eval);agent=load_rl_cfg(TASK);agent.seed=a.seed;agent.max_iterations=a.updates;agent.save_interval=500;agent.upload_model=False;agent.logger='tensorboard' if a.eval or a.preflight else 'wandb';agent.run_name='V34_model_only_'+a.model
 # Preserve archived PPO settings; this is not the 1e-5 A6 fine-tune recipe.
 dump_yaml(a.out/'env.yaml',asdict(cfg));dump_yaml(a.out/'agent.yaml',asdict(agent));env=None
 try:
  env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None);w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.out),'cuda:0');runner.args=a
  if not hasattr(runner.logger,'writer'):runner.logger.writer=None
  runner.load(a.checkpoint,load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False});parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
  for group in ['actor_state_dict','critic_state_dict']:
   for k,v in parent[group].items():
    if torch.is_tensor(v):assert torch.equal(v,runner.alg.save()[group][k].cpu()),(group,k)
  ref=(parent.get('infos') or {}).get('model_only')
  if ref:
   env._smp_normalizer.mean.copy_(ref['smp_mean'].to(env.device));env._smp_normalizer.count.copy_(ref['smp_count'].to(env.device))
  # Continue archived curriculum age rather than silently restarting its easy phase.
  env.common_step_counter=int(ref['common_step_counter'] if ref else (parent.get('infos') or {}).get('env_state',{}).get('common_step_counter',0))
  obs,_=env.reset();assert obs['actor'].shape[-1]==93
  write(a.out/'launch.json',dict(model=a.model,parent=a.checkpoint,parent_sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),seed=a.seed,updates=a.updates,lr=agent.algorithm.learning_rate,schedule=agent.algorithm.schedule,events=list(cfg.events),terminations=list(cfg.terminations),weights={k:v.weight for k,v in cfg.rewards.items()},normalizer='original update behavior; V7 statistic absent from historical checkpoint, freshly initialized',optimizer='fresh',curriculum_counter=env.common_step_counter,physics_dt=env.step_dt/cfg.decimation,control_dt=env.step_dt,pose_counts={str(i):int((env._robust_reset_type==i).sum()) for i in [1,2]}))
  if a.eval:evaluate(env,w,runner.get_inference_policy(),obs,a.out,a.video);return
  runner.save(str(a.out/'initial.pt'));runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;runner.save(str(a.out/'final.pt'));write(a.out/'completed.json',dict(iteration=runner.current_learning_iteration))
 except BaseException as e:write(a.out/'failed.json',dict(error=repr(e)));raise
 finally:
  if env:env.close()
if __name__=='__main__':main()
