"""A6-no-alpha fixed-table fine-tuning: matched T0/T1/T2, deployable noisy 93D actor."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID',os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0])
import argparse,json,hashlib,random,subprocess,sys,copy
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np,torch,mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import EntityCfg
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor.contact_sensor import ContactMatch
from mjlab.utils.os import dump_yaml
from smp.rl.tasks.getup import crawl_first as t,balanced_dynamics as bd,multiterrain as mt,r2_ablation as ra,v33_path_ablation as pa,v33_reward_transfer as v
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_PATTERN
import train_a6_paper_ablation as base
import train_crawl_first as legacy
from smp.rl.tasks.getup import crawl_repair as repair
from train_r1_quality import verify_nested
from train_scratch_tradeoffs import tensor_hash

DEFAULT_PARENT='/mnt/workspace/user/luyidan/smp-a6-paper/logs/rsl_rl/a6_paper/formal_20260929_seed20261013/A6_no_alpha/model_9999.pt'
T4_PARENT='/mnt/workspace/user/luyidan/smp-table-specialist/logs/table_specialist_20261007/T4/model_6000.pt'
T4_SHA='18c65dde50a18c060ab6be5e89d611a2559c8d64ae27ba6359906e4c3a414f22'
PARENT_SHA='24f1a4513cf5cabb9bd0b9ef54e90a3d50e625a87774878379af5da5dcf8fbd1'
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def config(n,arm,evaluation,seed):
 cfg,agent=legacy.config(n,'CR2',evaluation,seed)
 cfg.events['gsi_reset'].func=repair.reset;cfg.events['gsi_reset'].params.pop('arm',None)
 cfg.events['multiterrain_phase'].func=repair.update
 cfg.rewards['task_smp_product'].func=repair.task
 cfg.rewards['plate_completion'].func=repair.completion;cfg.rewards['plate_completion'].weight=.30
 cfg.rewards['table_post_clear_position'].func=repair.position_cost;cfg.rewards['table_post_clear_position'].params={}
 cfg.rewards['crawl_local_clearance'].weight=-.05
 cfg.rewards['repair_stand']=RewardTermCfg(func=repair.stand_reward,weight=.30)
 if arm=='CF1':cfg.rewards['crawl_gentle_contact']=RewardTermCfg(func=t.contact_cost,weight=-.01)
 agent.algorithm.schedule='fixed';agent.algorithm.learning_rate=1e-5;agent.algorithm.gamma=.995
 agent.algorithm.class_name='smp.rl.tasks.getup.crawl_repair_ppo:CrawlRepairPPO'
 agent.run_name='crawl_repair_'+arm;agent.seed=seed;agent.save_interval=250
 return cfg,agent


class Wrapper(RslRlVecEnvWrapper):
 def step(self,actions):
  obs,r,d,e=super().step(actions);env=self.unwrapped;log=e.setdefault('log',{})
  for f,name in enumerate(('square','rectangle','bench')):
   mask=env._tb_family==f
   for p,pose in enumerate(('supine','prone')):
    m=mask&(env._tb_pose==p)
    if m.any():
     for key,value in [('sr3_episode',env._tb_episode_sr3.float()),('escaped_once',env._tb_done.float()),('hold_s',env._tb_hold),('max_post_clear_drift_m',env._tb_drift),('smp',env._fixed_smp_score),('plate_force',env._mt_force)]:log[f'Table/{name}/{pose}/{key}']=value[m].mean().detach()
  log['Crawl/rise_blend']=env._cr_beta.mean().detach()
  for kind,name in enumerate(('deep','near','outside')):
   mask=env._cr_kind==kind
   log[f'Crawl/reset_fraction/{name}']=mask.float().mean().detach()
   if mask.any():log[f'Crawl/SR3_episode/{name}']=env._tb_episode_sr3[mask].float().mean().detach()
  log['Crawl/table_contact_peak_N']=env._cr_force_peak.mean().detach()
  for j,name in enumerate(('legs','upper','pelvis')):log['Dynamics/mass_'+name]=env._bd_mass[:,j].mean().detach()
  log['Dynamics/gain_mean']=env._bd_gain.mean().detach();log['Dynamics/delay_ms']=(env._bd_lag.float()*2).mean().detach()
  assert torch.isfinite(r).all();return obs,r,d,e

class Runner(MjlabOnPolicyRunner):
 active=False
 def save(self,path,infos=None):
  env=self.env.unwrapped
  if self.active and self.current_learning_iteration>=250:self.cfg['save_interval']=500
  super().save(path,infos={**(infos or {}),'crawl_first':dict(arm=self.args.arm,height_range_m=[.32,.40],smp_mean=env._smp_normalizer.mean.cpu(),smp_count=env._smp_normalizer.count.cpu())})
  if not self.active:return
  it=self.current_learning_iteration;write(self.args.out/'progress.json',dict(iteration=it,height_range_m=[.32,.40]))
  if it==0 or (it%1000!=0 and it not in (250,500)):return
  out=self.args.out/'validation'/str(it);out.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u',__file__,'--eval','--arm',self.args.arm,'--checkpoint',str(Path(path).resolve()),'--out',str(out),'--num-envs','768','--seed','20261071']
  with (out/'eval.log').open('w') as f:
   result=subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':self.args.eval_gpu,'MUJOCO_EGL_DEVICE_ID':self.args.eval_gpu},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode!=0:raise RuntimeError(f'Evaluation failed: {out}')
  data=json.loads((out/'summary.json').read_text())
  if data['reset/outside']['sr3']<.50 or data['table_only']['sr3']<.15:
   write(self.args.out/'retention_guard.json',dict(iteration=it,reason='Recovery retention below guard threshold',validation=data));raise RuntimeError('Retention guard: stop before further forgetting')
  for key,value in data.items():
   for metric in ['sr3','escaped','drift_p95','drift_escaped_p95','drift3_p95']:
    if value.get(metric) is not None:self.logger.writer.add_scalar(f'Validation/{key}/{metric}',value[metric],it)
  write(self.args.out/'progress.json',dict(iteration=it,height_range_m=[.32,.40],validation=data))

def audit(env,out):
 ids=np.arange(min(env.num_envs,128));model=copy.copy(env.sim.mj_model);data=mujoco.MjData(model)
 q=env.sim.data.qpos.cpu().numpy();mp=env.sim.data.mocap_pos.cpu().numpy();mq=env.sim.data.mocap_quat.cpu().numpy();gids=env._tb_gids.cpu().numpy();sizes=env.sim.model.geom_size[:,env._tb_gids].cpu().numpy();minimum=[]
 for i in ids:
  for j,g in enumerate(gids):model.geom_size[g]=sizes[i,j];model.geom_aabb.reshape(-1,2,3)[g]=[np.zeros(3),sizes[i,j]];model.geom_rbound[g]=np.linalg.norm(sizes[i,j])
  data.qpos[:]=q[i];data.qvel[:]=0;data.mocap_pos[:]=mp[i];data.mocap_quat[:]=mq[i];mujoco.mj_forward(model,data)
  minimum.append(min([0.]+[float(c.dist) for c in data.contact]))
 assert min(minimum)>-.0011,min(minimum)
 pos,ext=mt.robot_bounds(env);pp=env.sim.data.geom_xpos[:,env._tb_gids];ss=env.sim.model.geom_size[:,env._tb_gids]
 gap=((pos[:,:,None,:]-pp[:,None,:,:]).abs()-ext[:,:,None,:]-ss[:,None,:,:]).amax(-1).amin(dim=(1,2))
 assert gap.min()>=.0029,float(gap.min())
 # Broadphase probe: deliberately overlap top with a robot collision geom in a copied CPU state.
 robotgid=int(env._mt_geoms[0]);top=int(gids[0]);probe=env.scene['table_top'].indexing.mocap_id
 data.mocap_pos[probe]=data.geom_xpos[robotgid];mujoco.mj_forward(model,data)
 assert any(c.dist<0 and top in (int(c.geom1),int(c.geom2)) for c in data.contact),'CPU geometry resize collision missed'
 write(out/'reset_audit.json',dict(cpu_min_contact=min(minimum),gpu_aabb_min=float(gap.min()),family_counts=torch.bincount(env._tb_family).tolist(),pose_counts=torch.bincount(env._tb_pose).tolist(),source_counts=torch.bincount(env._tb_source).tolist(),dynamics_mass_range=[env._bd_mass.min(0).values.tolist(),env._bd_mass.max(0).values.tolist()],gain_range=[float(env._bd_gain.min()),float(env._bd_gain.max())],delay_counts=torch.bincount(env._bd_lag,minlength=6).tolist(),cpu_resized_collision_probe=True))
 np.savez_compressed(out/'initial.npz',qpos=q,mocap_pos=mp,mocap_quat=mq,bank_rows=env._tb_rows.cpu().numpy(),family=env._tb_family.cpu().numpy(),pose=env._tb_pose.cpu().numpy(),band=env._tb_band.cpu().numpy())

def evaluate(env,w,policy,obs,a):
 live=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);hold=torch.zeros(env.num_envs,device=env.device);passed=live.clone().fill_(False);escaped=passed.clone();peak=hold.clone();drift=hold.clone();initial_rows=env._tb_rows.clone();post_steps=hold.clone();drift3=hold.clone()
 from smp.rl.tasks.getup import a6_paper_ablation as ap
 ap.init_eval_loads(env);env._cr_eval_contact_peak=torch.zeros(env.num_envs,device=env.device)
 initial_kind=env._cr_kind.clone();initial_height_bin=env._cr_height_bin.clone()
 selected=[]
 for f in range(1):
  for p in range(2):
   for b in range(2):
    ii=torch.where((env._tb_family==f)&(env._tb_pose==p)&(env._tb_band==b))[0]
    if len(ii):selected.append((int(ii[0]),f,p,b))
 writers={}
 if a.video:
  import imageio.v2 as imageio
  for j,f,p,b in selected:
   if b==0 or a.video_low:writers[j]=imageio.get_writer(str(a.out/f'{f}_{p}_{b}.mp4'),fps=25,codec='libx264',ffmpeg_params=['-threads','1'])
 try:
  for k in range(a.eval_steps):
   with torch.inference_mode():obs,_,done,_=w.step(policy(obs))
   live &= ~done.bool();env._ap_alive &= live;r=env.scene['robot'];good=(mt.height(env)>=1.15)&(-r.data.projected_gravity_b[:,2]>=.9)&(mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)>20)&live
   hold=torch.where(good,hold+1,0);passed|=(hold>=150)&((k+1-hold)<=1000);escaped|=env._tb_done&live;peak=torch.maximum(peak,torch.where(live,(r.data.qfrc_actuator*r.data.joint_vel).abs().amax(-1),0));drift=torch.maximum(drift,torch.where(live,env._tb_drift,0))
   post_steps+=live&env._tb_done
   current_dist=(r.data.root_link_pos_w[:,:2]-env._tb_anchor).norm(dim=-1)
   drift3=torch.maximum(drift3,torch.where(live&env._tb_done&(post_steps<=150),current_dist,0))
   if writers and k%2==1:
    from PIL import Image,ImageDraw
    cam=env._offline_renderer._cam;cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=a.camera_azimuth;cam.elevation=a.camera_elevation
    for j,f,p,b in selected:
     if j not in writers:continue
     env.cfg.viewer.env_idx=j;root=r.data.root_link_pos_w[j].cpu().numpy();top=env.sim.data.geom_xpos[j,env._tb_gids[0]].cpu().numpy();cam.lookat[:]=[(root[0]+top[0])/2,(root[1]+top[1])/2,.32];cam.distance=3.6+.45*np.linalg.norm(root[:2]-top[:2]);im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);dims=env._tb_bank['dims'][initial_rows[j]].cpu().numpy();draw.text((8,28),f'L={dims[0]:.2f} W={dims[1]:.2f} underside H={dims[2]:.2f} m',fill='white');draw.text((8,8),f'{a.checkpoint.parent.name}@{a.checkpoint.stem} pose={p} t={(k+1)*.02:.2f}s SR3={int(passed[j])}',fill='white');writers[j].append_data(np.asarray(im))
     if k==1:im.save(a.out/f'initial_{f}_{p}_{b}.png')
 finally:
  for v in writers.values():v.close()
 summary={}
 for f,name in enumerate(('square','rectangle','bench')):
  for p,pose in enumerate(('supine','prone')):
   for b,band in enumerate(('high','low')):
    m=(env._tb_family==f)&(env._tb_pose==p)&(env._tb_band==b);n=int(m.sum())
    if n:summary[f'{name}/{pose}/{band}']=dict(n=n,sr3=float(passed[m].float().mean()),escaped=float(escaped[m].float().mean()),drift_p95=float(torch.quantile(drift[m],.95)),power_peak_p95_50hz=float(torch.quantile(peak[m],.95)))
    if n:
     em=m&escaped;pm=m&(post_steps>=150)
     summary[f'{name}/{pose}/{band}'].update(n_escaped=int(em.sum()),n_post3=int(pm.sum()),drift_escaped_p95=float(torch.quantile(drift[em],.95)) if em.any() else None,drift3_p95=float(torch.quantile(drift3[pm],.95)) if pm.any() else None)
 masks={'all':torch.ones_like(passed),'table_only':initial_kind!=2}
 masks.update({f'reset/{name}':initial_kind==i for i,name in enumerate(('deep','near','outside'))})
 if a.bank and '30cm' in a.bank:masks['height/30cm']=torch.ones_like(passed)
 else:masks.update({f'height/{32+i}-{33+i}cm':initial_height_bin==i for i in range(8)})
 for label,m in masks.items():
  if not m.any():continue
  pm=m&(post_steps>=150)
  summary[label]=dict(n=int(m.sum()),sr3=float(passed[m].float().mean()),escaped=float(escaped[m].float().mean()),drift3_p95=float(torch.quantile(drift3[pm],.95)) if pm.any() else None,contact_peak_p95_N=float(torch.quantile(env._cr_eval_contact_peak[m],.95)))
 for label,m in masks.items():
  if not m.any():continue
  for index,key in enumerate(('torque_peak_p95_Nm','joint_speed_peak_p95_rad_s','power_peak_p95_W')):summary[label][key]=float(torch.quantile(env._ap_peaks[:,:,index].amax(-1)[m],.95))
  summary[label]['high_load_duration_p95_s']=float(torch.quantile(env._ap_high_longest.amax(-1)[m],.95))
 write(a.out/'summary.json',summary)
 np.savez_compressed(a.out/'per_trial.npz',table_contact_peak=env._cr_eval_contact_peak.cpu(),joint_peaks=env._ap_peaks.cpu(),high_load_longest=env._ap_high_longest.cpu(),kind=initial_kind.cpu(),height_bin=initial_height_bin.cpu(),sr3=passed.cpu(),escaped=escaped.cpu(),drift=drift.cpu(),drift3=drift3.cpu(),post_steps=post_steps.cpu(),family=env._tb_family.cpu(),pose=env._tb_pose.cpu(),band=env._tb_band.cpu(),bank_rows=initial_rows.cpu())
 print('EVAL_COMPLETE',json.dumps(summary),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['CF0','CF1','CF2'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=None);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=10000);p.add_argument('--seed',type=int,default=20261008);p.add_argument('--eval-gpu',default='7');p.add_argument('--eval',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--video-low',action='store_true');p.add_argument('--camera-azimuth',type=float,default=55);p.add_argument('--camera-elevation',type=float,default=-8);p.add_argument('--eval-steps',type=int,default=1150);p.add_argument('--preflight',action='store_true');p.add_argument('--bank',default=None);p.add_argument('--retention-bank',default=None);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=a.eval)
 if a.checkpoint is None:a.checkpoint=Path(DEFAULT_PARENT if a.arm=='CF2' else T4_PARENT)
 if not a.eval:assert sha(a.checkpoint)==(PARENT_SHA if a.arm=='CF2' else T4_SHA)
 t.install();cfg,agent=config(a.num_envs,a.arm,a.eval,a.seed);agent.logger='tensorboard' if a.eval or a.preflight else 'wandb';agent.max_iterations=a.updates
 if a.bank:cfg.events['gsi_reset'].params['bank_path']=a.bank
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 dump_yaml(a.out/'env.yaml',asdict(cfg));dump_yaml(a.out/'agent.yaml',asdict(agent));env=None
 try:
  env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array' if a.video else None);w=Wrapper(env,clip_actions=agent.clip_actions);runner=Runner(w,asdict(agent),str(a.out),'cuda:0');runner.args=a
  parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False);critic=tensor_hash(runner.alg.save()['critic_state_dict']);runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});verify_nested(parent['actor_state_dict'],runner.alg.save()['actor_state_dict']);assert critic==tensor_hash(runner.alg.save()['critic_state_dict'])
  if not a.eval:env.common_step_counter=0 # restart table-specific dynamics curriculum
  ref=json.loads(Path('configs/a6_paper/smp_reward_reference_v1.json').read_text());env._smp_normalizer.mean.copy_(torch.tensor(ref['mean'],device=env.device));env._smp_normalizer.count.copy_(torch.tensor(ref['count'],device=env.device))
  obs,_=env.reset();env._v_start_counter=-12000;env._v_ramp_updates=500
  assert obs['actor'].shape[-1]==93 and obs['critic'].shape[-1]==960
  if a.bank and '30cm' in a.bank:env._tb_band.fill_(1)
  audit(env,a.out)
  if not a.eval:assert cfg.observations['actor'].enable_corruption and 'push_robot' in cfg.events
  meta=dict(arm=a.arm,parent=str(a.checkpoint),parent_sha=sha(a.checkpoint),actor_exact=True,fresh_critic_optimizer=True,common_step_counter=env.common_step_counter,eval_steps=a.eval_steps if a.eval else None,num_envs=a.num_envs,updates=a.updates,seed=a.seed,episode_s=cfg.episode_length_s,actor_noise=cfg.observations['actor'].enable_corruption,events=list(cfg.events),terminations=list(cfg.terminations),bank_sha=sha(env._tb_bank_path),weights={k:v.weight for k,v in cfg.rewards.items()},branch='codex/a6-crawl-repair',height_protocol=('fixed30cm independent evaluation' if a.bank and '30cm' in a.bank else 'stratified uniform underside32-40cm; no curriculum'),reset_protocol='60deep/30near/10outside; natural/procedural50/50; supine/prone50/50',repair=dict(fixed_lr=1e-5,gamma=.995,critic_warmup=50,retention_coefficient=.10,retention_bank=a.retention_bank,actor_normalizer_frozen=True,completion_bonus=.30,rise_smp_floor=.30))
  write(a.out/'launch.json',meta);print('TRANSFER_AND_RESET_VERIFIED',json.dumps(meta),flush=True)
  if a.eval:evaluate(env,w,runner.get_inference_policy(),obs,a)
  else:
   if a.preflight:
    with torch.inference_mode():
     mask=torch.arange(env.num_envs,device=env.device)%4!=0;before=env.sim.data.qpos[mask].clone();size=env.sim.model.geom_size[mask].clone();mass=env.sim.model.body_mass[mask].clone();env._reset_idx(torch.where(~mask)[0]);assert torch.equal(before,env.sim.data.qpos[mask]);assert torch.equal(size,env.sim.model.geom_size[mask]);assert torch.equal(mass,env.sim.model.body_mass[mask]);obs,_=env.reset()
     assert not repair.position_cost(env)[env._cr_kind==2].any(),'Outside low resets must not be anchored immediately'
     dynamics_check=bd.audit_dynamics(env)
     # Probe real GPU contacts with resized boxes, then restore the screened reset.
     j=env._tb_gids[0];pose=torch.zeros((1,7),device=env.device);pose[:,3]=1;pose[:,:3]=env.sim.data.geom_xpos[0,env._mt_geoms[0]]
     env.scene['table_top'].write_mocap_pose_to_sim(pose,env_ids=torch.tensor([0],device=env.device));env.sim.forward()
     with torch.inference_mode():w.step(runner.get_inference_policy()(obs))
     assert (env.scene['guided_contact'].data.found[0]>0).any(),'GPU resized table collision missed'
     env._tb_low_fraction=1.;env.reset();(a.out/'low_audit').mkdir();audit(env,a.out/'low_audit')
     env._tb_low_fraction=0.;obs,_=env.reset()
     # Post-clear progression is disabled irrespective of the current geometry.
     if True:
      env._tb_done[:]=True
      for index in (0,1):assert torch.equal(t.path_reward(env,index),torch.zeros(env.num_envs,device=env.device))
      assert torch.equal(t.escape_reward(env,4),torch.zeros(env.num_envs,device=env.device))
      env._tb_anchor[:]=env.scene['robot'].data.root_link_pos_w[:,:2]
      assert not t.post_clear_position(env).any()
      obs,_=env.reset()
     # Clear all table parts without altering the robot; require hold then smooth restoration.
     for name in t.PARTS:
      pose=torch.zeros((env.num_envs,7),device=env.device);pose[:,3]=1;pose[:,:3]=env.scene.env_origins+pose.new_tensor([10.,10.,.5]);env.scene[name].write_mocap_pose_to_sim(pose,env_ids=torch.arange(env.num_envs,device=env.device))
     bonus_count=torch.zeros(env.num_envs,device=env.device)
     for _ in range(45):
      obs,_,_,_=w.step(runner.get_inference_policy()(obs));bonus_count+=env._cf_bonus
     assert torch.equal(bonus_count,(env._cr_kind!=2).float()),'Completion bonus must occur exactly once for trapped trials'

     assert env._cr_clear.all() and torch.allclose(env._cr_beta,torch.ones_like(env._cr_beta))
     anchors=env._tb_anchor.clone()
     # Put the top shadow over each robot, high enough to avoid a physical impact.
     pose[:,:3]=env.scene['robot'].data.root_link_pos_w;pose[:,2]+=3.;env.scene['table_top'].write_mocap_pose_to_sim(pose,env_ids=torch.arange(env.num_envs,device=env.device))
     obs,_,_,_=w.step(runner.get_inference_policy()(obs))
     assert not env._cr_clear.any() and not env._cr_beta.any()
     assert torch.equal(anchors,env._tb_anchor),'Escape anchor reset on reentry'
     obs,_=env.reset()
     # Compare raw task components to selective masking on the same state.
     env._v_cache=-1;raw=t.ORIGINAL_COMPONENTS(env).clone()
     env._v_cache=-1;env._cr_component_tick=-1;env._cr_phase_tick=env.common_step_counter
     env._cr_clear[:]=False;env._cr_beta[:]=0
     masked=t.components(env).clone()
     if a.arm!='CR0':
      assert not masked[:,:4].any()
      assert torch.allclose(masked[:,4:8],raw[:,4:8],atol=1e-6)
      assert not env._v_stage.any()
      env._v_cache=-1;env._cr_component_tick=-1;env._cr_clear[:]=True;env._cr_beta[:]=1
      restored=t.components(env).clone();assert restored[:,:4].sum()>0
     obs,_=env.reset()
     initial_mocap=env.sim.data.mocap_pos.clone();policy=runner.get_inference_policy();terms=torch.zeros(len(env.reward_manager.active_terms),device=env.device)
     for k in range(250):
      with torch.inference_mode():obs,reward,done,extra=w.step(policy(obs))
      assert torch.isfinite(obs['actor']).all();assert not done.any(),'Unexpected termination during 5s preflight'
      assert torch.equal(env.sim.data.mocap_pos,initial_mocap),'Fixed table moved'
      terms+=env.reward_manager._step_reward.mean(0)
     write(a.out/'smoke_rollout.json',dict(steps=250,finite=True,fixed_tables=True,partial_reset_pass=True,gpu_collision_probe=True,low_height_audit=True,dynamics=dynamics_check,weighted_terms=dict(zip(env.reward_manager.active_terms,terms.tolist()))))
   # Freeze actor input scaling; regression targets come from the exact parent.
   runner.alg.actor.update_normalization=lambda observations:None
   if a.retention_bank:
    bank=torch.load(a.retention_bank,map_location=env.device,weights_only=False);ro=bank['obs'];rt=bank['targets']
   else:
    policy=runner.get_inference_policy();ret_obs=[];ret_targets=[]
    with torch.inference_mode():
     obs,_=env.reset()
     for k in range(500):
      actions=policy(obs)
      valid=env._cr_clear & ((env._cr_kind==2)|(mt.height(env)>.45))
      ids=torch.where(valid)[0]
      if len(ids):
       ids=ids[torch.randperm(len(ids),device=env.device)[:32]];ret_obs.append(obs['actor'][ids].clone());ret_targets.append(actions[ids].clone())
      obs,_,_,_=w.step(actions)
     assert len(ret_obs)>200,'Insufficient clear-state retention data'
     ro=torch.cat(ret_obs);rt=torch.cat(ret_targets)
   # Convert inference tensors to ordinary tensors for autograd training.
   runner.alg.retention_obs=ro.clone();runner.alg.retention_targets=rt.clone()
   torch.save(dict(obs=ro.cpu(),targets=rt.cpu()),a.out/'retention_bank.pt')
   env.common_step_counter=0
   with torch.inference_mode():obs,_=env.reset()
   env._v_start_counter=-12000
   print('RETENTION_BANK_READY',len(ro),flush=True)
   runner.save(str(a.out/'initial.pt'));runner.active=not a.preflight;runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False);runner.active=False;runner.save(str(a.out/'final.pt'));write(a.out/'completed.json',dict(iteration=runner.current_learning_iteration));print('TRAIN_COMPLETE',flush=True)
 except BaseException as e:write(a.out/'failed.json',dict(error=repr(e)));raise
 finally:
  if env:env.close()
if __name__=='__main__':main()
