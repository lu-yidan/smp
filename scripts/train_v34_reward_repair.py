"""Run with archived V34 src on PYTHONPATH; deploy adaptation is explicit."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MUJOCO_EGL_DEVICE_ID',os.environ.get('CUDA_VISIBLE_DEVICES','0'))
import argparse,json,random,hashlib,subprocess,sys,copy
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch,mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.sensor.contact_sensor import ContactSensorCfg,ContactMatch
from mjlab.utils.os import dump_yaml
import smp.rl.tasks
import v34_plate_transfer_adapter as adapt
import v34_reward_repair as repair
PARENT='/root/workplace/smp-v34-table-mix/artifacts/v34_gate6000.pt'
SHA='471fb41444a5d8f2bce1f93b0e2d65a613e86c1d417ef4cba36ba48b9779474b'
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,default=str))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evaluate(env,w,policy,obs,out,video=False):
 group=env._vp_group.clone();alive=torch.ones(env.num_envs,device=env.device,dtype=torch.bool);passed=alive.clone().fill_(False);escaped=passed.clone();hold=torch.zeros(env.num_envs,device=env.device);best=hold.clone();max_z=hold.clone();pose_hold=hold.clone();best_pose=hold.clone();r=env.scene['robot'];head=r.find_sites(['head'])[0][0];writers={}
 if video:
  import imageio.v2 as imageio
  for g in range(4):
   ids=torch.where(group==g)[0]
   if len(ids):writers[int(ids[0])]=imageio.get_writer(str(out/f'group{g}.mp4'),fps=25,codec='libx264',ffmpeg_params=['-threads','1'])
 reasons={}
 try:
  for k in range(1150):
   with torch.inference_mode():obs,_,done,_=w.step(policy(obs))
   alive&=~done.bool();z=r.data.site_pos_w[:,head,2]-env.scene.env_origins[:,2];u=-r.data.projected_gravity_b[:,2]
   feet=torch.stack([env.scene[n].data.force[...,2].abs().amax(-1) for n in ['sr3_left','sr3_right']]).amin(0)
   max_z=torch.maximum(max_z,torch.where(alive,z,0));pose_hold=torch.where(alive&(z>=1.15)&(u>=.9),pose_hold+1,0);best_pose=torch.maximum(best_pose,pose_hold)
   good=alive&(z>=1.15)&(u>=.9)&(feet>20);hold=torch.where(good,hold+1,0);best=torch.maximum(best,hold);passed|=(hold>=150)&((k+1-hold)<=1000);escaped|=alive&(env._escape_phase==3)
   if writers and k%2==1:
    from PIL import Image,ImageDraw
    for j in list(writers):
     if not alive[j]:writers.pop(j).close();reasons[str(j)]='terminated before '+str((k+1)*.02);continue
     env.cfg.viewer.env_idx=j;cam=env._offline_renderer._cam;root=r.data.root_link_pos_w[j].cpu().numpy();cam.lookat[:]=[root[0],root[1],max(.35,float(root[2])*.5+.2)];cam.azimuth=55;cam.elevation=-12;cam.distance=3
     im=Image.fromarray(env.render());ImageDraw.Draw(im).text((8,8),f'V34 reward repair group={int(group[j])} t={(k+1)*.02:.2f}s SR3={int(passed[j])}',fill='white');writers[j].append_data(np.asarray(im))
 finally:
  for wr in writers.values():wr.close()
 result={}
 for g,label in enumerate(['plate_supine','plate_prone','flat_supine','flat_prone']):
  m=group==g;result[label]=dict(n=int(m.sum()),sr3=float(passed[m].float().mean()),escaped=float(escaped[m].float().mean()),terminated=float((~alive[m]).float().mean()),max_hold_mean_s=float(best[m].mean()*.02),max_head_mean=float(max_z[m].mean()),max_pose_hold_mean_s=float(best_pose[m].mean()*.02))
 write(out/'summary.json',result);write(out/'video_end_reasons.json',reasons);np.savez_compressed(out/'trials.npz',group=group.cpu(),sr3=passed.cpu(),escaped=escaped.cpu(),alive=alive.cpu(),best_hold=best.cpu(),max_head=max_z.cpu(),best_pose_hold=best_pose.cpu());return result

class Wrapper(RslRlVecEnvWrapper):
 def step(self,actions):
  obs,reward,done,extra=super().step(actions);e=self.unwrapped;log=extra.setdefault('log',{})
  r=e.scene['robot'];head=r.find_sites(['head'])[0][0]
  for g,name in enumerate(['plate_supine','plate_prone','flat_supine','flat_prone']):
   mask=e._vp_group==g
   if mask.any():
    log['Repair/'+name+'/head_height']=(r.data.site_pos_w[:,head,2]-e.scene.env_origins[:,2])[mask].mean()
    log['Repair/'+name+'/escape']=(e._escape_phase[mask]==3).float().mean()
    if hasattr(e,'_fixed_product'):
     for key,attr in [('task','_fixed_task_score'),('smp','_fixed_smp_score'),('product','_fixed_product')]:log['Repair/'+name+'/'+key]=getattr(e,attr)[mask].mean()
  assert torch.isfinite(reward).all()
  return obs,reward,done,extra

class Runner(MjlabOnPolicyRunner):
 active=False
 def save(self,path,infos=None):
  e=self.env.unwrapped;n=e._smp_normalizer
  super().save(path,infos={**(infos or {}),'plate_transfer':dict(smp_mean=n.mean.cpu(),smp_count=n.count.cpu())})
  if not self.active:return
  i=self.current_learning_iteration;write(self.args.out/'progress.json',dict(iteration=i))
  if i==0 or i%500:return
  out=self.args.out/'validation'/str(i);out.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u',__file__,'--eval','--arm',self.args.arm,'--checkpoint',str(Path(path).resolve()),'--out',str(out),'--num-envs','512','--seed','20261071']
  if i%1000==0:cmd+=['--video']
  with open(out/'console.log','w') as f:subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=self.args.eval_gpu,MUJOCO_EGL_DEVICE_ID=self.args.eval_gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
  result=json.loads((out/'summary.json').read_text());write(self.args.out/'progress.json',dict(iteration=i,validation=result))
  for label,values in result.items():self.logger.writer.add_scalar('Validation/'+label+'/SR3',values['sr3'],i)

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['R0','R1','R2'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',default=PARENT);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=6000);p.add_argument('--seed',type=int,default=20261008);p.add_argument('--eval-gpu',default='5');p.add_argument('--preflight',action='store_true');p.add_argument('--eval',action='store_true');p.add_argument('--video',action='store_true');a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=a.eval)
 if not a.eval:assert digest(a.checkpoint)==SHA
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 cfg=adapt.configure(load_env_cfg('Smp-Getup-Escape-Plate-V34-93D-G1',play=False),a.eval);cfg=repair.configure(cfg,a.arm);cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.viewer.width=640;cfg.viewer.height=480;cfg.viewer.max_extra_envs=0
 if a.eval:cfg.episode_length_s=1000
 cfg.scene.sensors=tuple(cfg.scene.sensors)+tuple(ContactSensorCfg(name='sr3_'+side,primary=ContactMatch(mode='geom',pattern='^'+side+'_foot[1-7]_collision$',entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force','normal','tangent'),reduce='maxforce',num_slots=1,global_frame=True) for side in ['left','right'])
 agent=load_rl_cfg('Smp-Getup-Escape-Plate-V34-93D-G1');agent.seed=a.seed;agent.max_iterations=a.updates;agent.save_interval=500;agent.algorithm.learning_rate=1e-5;agent.algorithm.schedule='fixed';agent.logger='tensorboard' if a.preflight or a.eval else 'wandb';agent.run_name='V34Reward_'+a.arm+'_6000'
 dump_yaml(a.out/'env.yaml',asdict(cfg));dump_yaml(a.out/'agent.yaml',asdict(agent));env=None
 try:
  env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;w=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.out),'cuda:0');runner.args=a
  if not hasattr(runner.logger,'writer'):runner.logger.writer=None
  runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});runner.alg.actor.update_normalization=lambda x:None;parent=torch.load(a.checkpoint,weights_only=False,map_location='cpu');repair.load_reference(env);obs,_=env.reset();assert obs['actor'].shape[-1]==93
  if a.eval and 'plate_transfer' in (parent.get('infos') or {}):
   ref=parent['infos']['plate_transfer'];env._smp_normalizer.mean.copy_(ref['smp_mean'].to(env.device));env._smp_normalizer.count.copy_(ref['smp_count'].to(env.device))
  if not a.eval:
   import onnxruntime as ort
   opt=ort.SessionOptions();opt.intra_op_num_threads=1;sess=ort.InferenceSession('/root/workplace/smp-v34-table-mix/artifacts/v34_gate6000.onnx',opt,providers=['CPUExecutionProvider']);raw=obs['actor'][:16].cpu().numpy();expected=np.concatenate([sess.run(None,{'obs':x[None]})[0] for x in raw]);policy=runner.get_inference_policy()
   with torch.inference_mode():actual=policy(obs)[:16].cpu().numpy()
   err=float(abs(actual-expected).max());assert err<1e-4,err;write(a.out/'parity.json',dict(max_error=err))
  bodies=env._vp_bodies;scale=env._vp_mass[:,env._vp_body_group]
  assert torch.allclose(env.sim.model.body_mass[:,bodies],env.sim.get_default_field('body_mass')[bodies][None]*scale)
  assert torch.allclose(env.sim.model.body_inertia[:,bodies],env.sim.get_default_field('body_inertia')[bodies][None]*scale[:,:,None])
  for act,g in zip(env.scene['robot'].actuators,env._vp_act_group):
   assert torch.allclose(act.stiffness,act.default_stiffness*env._vp_gain[:,g,None])
   assert torch.allclose(act.damping,act.default_damping*env._vp_gain[:,g,None])
  counts=torch.bincount(env._vp_group,minlength=4).tolist();write(a.out/'reset_fingerprint.json',dict(qpos_sha=hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest(),groups_sha=hashlib.sha256(env._vp_group.cpu().numpy().tobytes()).hexdigest()));write(a.out/'launch.json',dict(parent=a.checkpoint,parent_sha=digest(a.checkpoint),counts=counts,steps=a.updates,seed=a.seed,arm=a.arm,prior='f2s2',ws=6.,critic_loaded=False,optimizer_fresh=True,actor_norm_frozen=True,source_sha=digest(__file__),adapter_sha=digest(adapt.__file__),terminations=list(cfg.terminations),events=list(cfg.events),lr=1e-5))
  if a.eval:evaluate(env,w,runner.get_inference_policy(),obs,a.out,a.video);return
  if a.preflight:
   m=copy.copy(env.sim.mj_model);d=mujoco.MjData(m);q=env.sim.data.qpos.cpu().numpy();mp=env.sim.data.mocap_pos.cpu().numpy();mq=env.sim.data.mocap_quat.cpu().numpy();bad=[]
   for i in range(min(a.num_envs,128)):
    d.qpos[:]=q[i];d.mocap_pos[:]=mp[i];d.mocap_quat[:]=mq[i];mujoco.mj_forward(m,d)
    bad.extend(dict(env=i,dist=float(c.dist),geoms=[m.geom(int(g)).name for g in [c.geom1,c.geom2]]) for c in d.contact if c.dist<-.0011)
   write(a.out/'contact_audit.json',bad);assert not bad,bad[:3]
   ids=torch.tensor([0,3,7],device=env.device);mask=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);mask[ids]=False;before=env.sim.data.qpos[mask].clone();env._reset_idx(ids);assert torch.equal(before,env.sim.data.qpos[mask])
  runner.save(str(a.out/'initial.pt'));runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;write(a.out/'completed.json',dict(iteration=runner.current_learning_iteration))
 except BaseException as e:write(a.out/'failed.json',dict(error=repr(e)));raise
 finally:
  if env:env.close()
if __name__=='__main__':main()
