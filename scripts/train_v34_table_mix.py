"""V34 gate6000 93D continuation on deployment tables/vertical plates/flat."""
import train_crawl_repair as base
from smp.rl.tasks.getup import v34_table_mix as mix
from dataclasses import asdict,replace
from pathlib import Path
import argparse,json,random,os,sys,subprocess
import torch,numpy as np
from mjlab.sensor.contact_sensor import ContactMatch
from tensordict import TensorDict
PARENT='/mnt/workspace/user/luyidan/smp-v38-unified/logs/rsl_rl/smp_getup_escape_plate_v34_93d_g1/2026-09-02_18-09-07_v34_93d_ablation_12k_seed20261701_offline/model_6000.pt'
PARENT_SHA='471fb41444a5d8f2bce1f93b0e2d65a613e86c1d417ef4cba36ba48b9779474b'
ONNX='/mnt/workspace/user/luyidan/smp-a6-paper/outputs/smp_v34_escape_gate6000_93d.onnx'

def config(n,evaluation,seed):
 cfg,agent=base.config(n,'CF0',evaluation,seed)
 cfg.sim.contact_sensor_maxmatch=256
 cfg.scene.sensors=tuple(replace(s,secondary=ContactMatch(mode='geom',pattern='escape_obstacle/escape_plate_geom')) if s.name=='free_contact' else s for s in cfg.scene.sensors)
 cfg.events['multiterrain_phase'].func=mix.update;cfg.metrics['crawl_contact'].func=mix.contact_sample
 cfg.events['init_smp_state'].params['ckpt_path']='datasets/pretrain_ckpt/pretrained_getup_lafan_route_v7.pt'
 cfg.rewards['task_smp_product'].params['ws']=4.
 agent.algorithm.learning_rate=3e-6;agent.run_name='V34M0_table60_guided30_flat10';agent.save_interval=500
 return cfg,agent

class Wrapper(base.Wrapper):
 def step(self,actions):
  obs,r,d,e=super().step(actions);env=self.unwrapped;log=e['log']
  for s,name in [(0,'flat'),(1,'table'),(2,'guided')]:
   m=env._mix_scene==s;log['Mix/'+name+'/SR3_episode']=env._tb_episode_sr3[m].float().mean();log['Mix/'+name+'/escape']=env._tb_done[m].float().mean()
  return obs,r,d,e


def evaluate(env,w,policy,obs,out,video=False):
 live=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);hold=torch.zeros(env.num_envs,device=env.device);passed=live.clone().fill_(False);escaped=passed.clone();sc=env._mix_scene.clone();po=env._tb_pose.clone();first=torch.full_like(hold,-1)
 writers={}
 if video:
  import imageio.v2 as imageio
  for s in (1,2):
   for p in (0,1):
    ids=torch.where((sc==s)&(po==p))[0]
    if len(ids):writers[int(ids[0])]=imageio.get_writer(str(out/f'{s}_{p}.mp4'),fps=25,codec='libx264',ffmpeg_params=['-threads','1'])
 try:
  for k in range(1150):
   with torch.inference_mode():obs,_,done,_=w.step(policy(obs))
   live&=~done.bool();r=env.scene['robot'];good=live&(base.mt.height(env)>=1.15)&(-r.data.projected_gravity_b[:,2]>=.9)&(base.mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)>20)
   hold=torch.where(good,hold+1,0.);passed|=(hold>=150)&((k+1-hold)<=1000);escaped|=env._tb_done&live
   if writers and k%2==1:
    from PIL import Image,ImageDraw
    for j,writer in writers.items():
     env.cfg.viewer.env_idx=j;cam=env._offline_renderer._cam;root=r.data.root_link_pos_w[j].cpu().numpy();cam.lookat[:]=[root[0],root[1],max(.4,float(root[2])*.5+.2)];cam.azimuth=55;cam.elevation=-12;cam.distance=3.5
     im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.text((8,8),f'V34M0 scene={int(sc[j])} pose={int(po[j])} t={(k+1)*.02:.2f}s SR3={int(passed[j])}',fill='white');writer.append_data(np.asarray(im))
 finally:
  for writer in writers.values():writer.close()
 result={}
 for s,name in [(0,'flat'),(1,'table'),(2,'guided')]:
  for p,label in [(None,name),(0,name+'/supine'),(1,name+'/prone')]:
   m=(sc==s) if p is None else (sc==s)&(po==p)
   result[label]=dict(n=int(m.sum()),sr3=float(passed[m].float().mean()),escaped=float(escaped[m].float().mean()))
 base.write(out/'summary.json',result);np.savez_compressed(out/'trials.npz',scene=sc.cpu(),pose=po.cpu(),sr3=passed.cpu(),escaped=escaped.cpu());return result

def audit_cpu(env,out):
 import copy,mujoco
 model=copy.copy(env.sim.mj_model);data=mujoco.MjData(model);bad=[];minimum=0.
 q=env.sim.data.qpos.cpu().numpy();mp=env.sim.data.mocap_pos.cpu().numpy();mq=env.sim.data.mocap_quat.cpu().numpy();sz=env.sim.model.geom_size[:,env._tb_gids].cpu().numpy()
 for i in range(min(env.num_envs,128)):
  for j,g in enumerate(env._tb_gids.cpu().numpy()):model.geom_size[g]=sz[i,j];model.geom_aabb.reshape(-1,2,3)[g]=[np.zeros(3),sz[i,j]];model.geom_rbound[g]=np.linalg.norm(sz[i,j])
  data.qpos[:]=q[i];data.qvel[:]=0;data.mocap_pos[:]=mp[i];data.mocap_quat[:]=mq[i];mujoco.mj_forward(model,data)
  for ct in data.contact:
   minimum=min(minimum,float(ct.dist))
   if ct.dist<-.0011:bad.append(dict(env=i,scene=int(env._mix_scene[i]),middle=bool(env._mix_middle[i]),dist=float(ct.dist),geoms=[model.geom(int(g)).name for g in [ct.geom1,ct.geom2]]))
 base.write(out/'cpu_contacts.json',dict(minimum=minimum,bad=bad))
 assert not bad,str(bad[:5])

class Runner(base.Runner):
 def save(self,path,infos=None):
  # Do not use crawl-only validation/absolute guards for a new mixture.
  base.MjlabOnPolicyRunner.save(self,path,infos={**(infos or {}),'mix':dict(smp_mean=self.env.unwrapped._smp_normalizer.mean.cpu(),smp_count=self.env.unwrapped._smp_normalizer.count.cpu())})
  if not self.active:return
  it=self.current_learning_iteration;base.write(self.args.out/'progress.json',dict(iteration=it))
  if it==0 or it%500:return
  out=self.args.out/'validation'/str(it);out.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u',__file__,'--eval','--checkpoint',str(Path(path).resolve()),'--out',str(out),'--num-envs','256','--seed','20261071']
  if it%1000==0:cmd+=['--video']
  with open(out/'eval.log','w') as f:subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',MUJOCO_EGL_DEVICE_ID='7'),stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads((out/'summary.json').read_text());b=json.loads((self.args.out/'baseline/summary.json').read_text())
  for name,v in d.items():self.logger.writer.add_scalar('Validation/'+name+'/SR3',v['sr3'],it)
  base.write(self.args.out/'progress.json',dict(iteration=it,validation=d))
  lost=[s for s in ['flat','guided'] if b[s]['sr3']>=.25 and d[s]['sr3']<b[s]['sr3']-.20]
  if lost:base.write(self.args.out/'retention_guard.json',dict(iteration=it,lost=lost,baseline=b,current=d));raise RuntimeError('Relative retention guard')


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=10000);p.add_argument('--seed',type=int,default=20261008);p.add_argument('--checkpoint',type=Path,default=Path(PARENT));p.add_argument('--preflight',action='store_true');p.add_argument('--eval',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--baseline',type=Path);p.add_argument('--retention-bank',type=Path);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=a.eval)
 if not a.eval:assert base.sha(a.checkpoint)==PARENT_SHA
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;mix.install();cfg,agent=config(a.num_envs,a.eval,a.seed);agent.logger='tensorboard' if a.eval or a.preflight else 'wandb';agent.max_iterations=a.updates
 base.dump_yaml(a.out/'env.yaml',asdict(cfg));base.dump_yaml(a.out/'agent.yaml',asdict(agent))
 env=None
 try:
  env=base.ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;w=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.out),'cuda:0');runner.args=a
  runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});parent=torch.load(a.checkpoint,weights_only=False,map_location='cpu');base.verify_nested(parent['actor_state_dict'],runner.alg.save()['actor_state_dict']);runner.alg.actor.update_normalization=lambda obs:None
  if not a.eval:env.common_step_counter=0
  obs,_=env.reset();env._v_start_counter=0;env._v_ramp_updates=500
  if a.eval and 'mix' in (parent.get('infos') or {}):
   ref=parent['infos']['mix'];env._smp_normalizer.mean.copy_(ref['smp_mean'].to(env.device));env._smp_normalizer.count.copy_(ref['smp_count'].to(env.device))
  assert obs['actor'].shape[-1]==93
  if not a.eval:
   import onnxruntime as ort
   opt=ort.SessionOptions();opt.intra_op_num_threads=1;session=ort.InferenceSession(ONNX,opt,providers=['CPUExecutionProvider']);policy=runner.get_inference_policy()
   print('PARITY_SETTINGS',torch.get_float32_matmul_precision(),runner.alg.actor.obs_normalizer.eps,flush=True)
   torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
   with torch.inference_mode():actual=policy(obs)[:16].cpu().numpy()
   raw=obs['actor'][:16].cpu().numpy();expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in raw]);error=float(np.max(np.abs(actual-expected)));assert error<1e-4,error
   base.write(a.out/'onnx_parity.json',dict(max_abs_error=error,checkpoint_sha=base.sha(a.checkpoint)))
  counts={str(i):int((env._mix_scene==i).sum()) for i in range(3)}
  base.write(a.out/'launch.json',dict(parent=str(a.checkpoint),parent_sha=base.sha(a.checkpoint),code_sha=base.sha(Path(__file__)),reset_code_sha=base.sha(Path(mix.__file__)),counts=counts,flat_middle=int(env._mix_middle.sum()),num_envs=a.num_envs,updates=a.updates,seed=a.seed,prior='V7',smp_ws=4.,smp_reference='fresh running estimator; no f2s2 statistics reused',actor_noise=cfg.observations['actor'].enable_corruption,events=list(cfg.events),terminations=list(cfg.terminations),weights={k:v.weight for k,v in cfg.rewards.items()},lr=3e-6,critic_warmup=50,normalizer_frozen=True))
  # Initial clearance uses all robot collision bounds for both plate and table.
  pos,ext=base.mt.robot_bounds(env);nonflat=env._mix_scene!=0;gids=torch.stack(env._mt_plate_ids)[(env._mix_scene==2).long()];rows=torch.arange(env.num_envs,device=env.device);pp=env.sim.data.geom_xpos[rows,gids];ss=env.sim.model.geom_size[rows,gids]
  gap=((pos-pp[:,None]).abs()-ext-ss[:,None]).amax(-1).amin(-1)
  badids=torch.where(nonflat&(gap<.0009))[0];base.write(a.out/'gap_details.json',[dict(env=int(i),scene=int(env._mix_scene[i]),gap=float(gap[i])) for i in badids]);assert not len(badids),str(json.loads((a.out/'gap_details.json').read_text())[:5])
  base.write(a.out/'reset_audit.json',dict(scene_counts=counts,min_obstacle_gap=float(gap[nonflat].min()),dynamics_mass_range=[env._bd_mass.min(0).values.tolist(),env._bd_mass.max(0).values.tolist()],gain_range=[float(env._bd_gain.min()),float(env._bd_gain.max())],delay_counts=torch.bincount(env._bd_lag,minlength=6).tolist()))
  np.savez_compressed(a.out/'initial.npz',qpos=env.sim.data.qpos.cpu(),scene=env._mix_scene.cpu(),pose=env._tb_pose.cpu())
  if a.eval:evaluate(env,w,runner.get_inference_policy(),obs,a.out,a.video);return
  assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
  if a.preflight:
   audit_cpu(env,a.out)
   base.bd.audit_dynamics(env)
   with torch.inference_mode():obs,_=env.reset()
   ids=torch.tensor([0,3,7],device=env.device);mask=torch.ones(env.num_envs,device=env.device,dtype=torch.bool);mask[ids]=False;before=env.sim.data.qpos[mask].clone()
   with torch.inference_mode():env._reset_idx(ids)
   assert torch.equal(before,env.sim.data.qpos[mask]),'partial reset pollution'
  if a.retention_bank:bank=torch.load(a.retention_bank,map_location=env.device,weights_only=False);ro,rt=bank['obs'],bank['targets']
  else:
   keepo=[];keept=[];policy=runner.get_inference_policy()
   for _ in range(500):
    with torch.inference_mode():
     act=policy(obs);ids=torch.where((env._mix_scene==0)|((env._mix_scene==2)&env._cr_clear))[0][:32]
     if len(ids):keepo.append(obs['actor'][ids].clone());keept.append(act[ids].clone())
     obs,_,_,_=w.step(act)
   ro=torch.cat(keepo).clone();rt=torch.cat(keept).clone();assert len(ro)>1000
  runner.alg.retention_obs=ro.clone();runner.alg.retention_targets=rt.clone();torch.save(dict(obs=ro.cpu(),targets=rt.cpu()),a.out/'retention_bank.pt')
  env.common_step_counter=0
  with torch.inference_mode():obs,_=env.reset()
  runner.save(str(a.out/'initial.pt'))
  if not a.preflight:
   assert a.baseline and (a.baseline/'summary.json').exists();import shutil;shutil.copytree(a.baseline,a.out/'baseline')
  runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;runner.save(str(a.out/'final.pt'));base.write(a.out/'completed.json',dict(iteration=runner.current_learning_iteration))
 except BaseException as e:base.write(a.out/'failed.json',dict(error=repr(e)));raise
 finally:
  if env:env.close()
if __name__=='__main__':main()
