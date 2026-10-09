"""Matched read-only A6 rollout audit, returns and low-camera table videos."""
import os,json,hashlib
from pathlib import Path
import numpy as np,torch
import train_a6_table_mix as run

def evaluate(env,w,policy,obs,out,video=False):
 import imageio.v2 as imageio
 from PIL import Image,ImageDraw
 n=env.num_envs;dev=env.device;scene=env._mix_scene.clone();pose=env._tb_pose.clone();r=env.scene['robot'];head=r.find_sites(['head'])[0][0]
 fingerprint=hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()).hexdigest()
 terms=list(env.reward_manager.active_terms);sums=torch.zeros(n,len(terms),device=dev);disc=sums.clone();actual=torch.zeros(n,device=dev);actual_disc=actual.clone()
 live=torch.ones(n,dtype=torch.bool,device=dev);hold=actual.clone();passed=~live;escaped=~live;maxz=actual.clone();first_clear=actual.clone().fill_(-1);first_up=first_clear.clone();low_smp=actual.clone();low_n=actual.clone();pose_hold=actual.clone();maxpose=actual.clone()
 low_terms=torch.zeros(len(terms),device=dev);low_count=0;trace=[];writers={}
 label=os.environ.get('AUDIT_LABEL','policy')
 noise_std=float(os.environ.get('AUDIT_ACTION_NOISE','0'))
 if noise_std:
  deterministic_policy=policy
  policy=lambda x:deterministic_policy(x)+noise_std*torch.randn((n,29),device=dev)
 if video:
  for p in (0,1):
   ids=torch.where((scene==1)&(pose==p))[0]
   if len(ids):writers[int(ids[0])]=imageio.get_writer(str(out/f'table_{["supine","prone"][p]}.mp4'),fps=25,codec='libx264',ffmpeg_params=['-threads','1'])
 for k in range(1150):
  with torch.inference_mode():obs,rew,done,_=w.step(policy(obs))
  before=live.clone();live &=~done.bool();z=run.base.mt.height(env);u=-r.data.projected_gravity_b[:,2];feet=run.base.mt.ground_force(env,'quality_feet')[...,2].abs().amin(-1)
  good=live&(z>=1.15)&(u>=.9)&(feet>20);hold=torch.where(good,hold+1,0.);passed|=(hold>=150)&((k+1-hold)<=1000)
  pose_hold=torch.where(live&(z>=1.15)&(u>=.9),pose_hold+1,0);maxpose=torch.maximum(maxpose,pose_hold)
  escaped|=live&env._tb_done;maxz=torch.maximum(maxz,torch.where(live,z,0))
  first_clear=torch.where((first_clear<0)&live&env._tb_done,torch.full_like(first_clear,(k+1)*.02),first_clear)
  first_up=torch.where((first_up<0)&good,torch.full_like(first_up,(k+1)*.02),first_up)
  tr=env.reward_manager._step_reward.clone()*before[:,None];sums+=tr*.02;disc+=tr*.02*(.99**k);actual+=rew*before;actual_disc+=rew*before*(.99**k)
  low=live&(z<.65);low_smp+=env._fixed_smp_score*low;low_n+=low
  mask=low&(scene==1);low_terms+=tr[mask].sum(0);low_count+=int(mask.sum())
  if k%50==49:
   trace.append(dict(t=(k+1)*.02,table_head=float(z[scene==1].mean()),table_smp=float(env._fixed_smp_score[scene==1].mean()),table_task=float(env._fixed_task_score[scene==1].mean())))
  if writers and k%2==1:
   import mujoco
   cam=env._offline_renderer._cam;cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=55;cam.elevation=-9;cam.distance=2.8
   for j,writer in writers.items():
    env.cfg.viewer.env_idx=j;root=r.data.root_link_pos_w[j].cpu().numpy();cam.lookat[:]=[root[0],root[1],.42]
    im=Image.fromarray(env.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,640,44),fill='black');dims=env._tb_bank['dims'][env._tb_rows[j]].cpu().numpy()
    dr.text((8,5),f'{label} | {["supine","prone"][int(pose[j])]} | t={(k+1)*.02:.1f}s | hold={hold[j]*.02:.1f}s',fill='white')
    dr.text((8,25),f'table L={dims[0]:.2f} W={dims[1]:.2f} underside={dims[2]:.2f}m',fill='white');writer.append_data(np.asarray(im))
 for writer in writers.values():writer.close()
 result={}
 for s,name in [(0,'flat'),(1,'table'),(2,'guided')]:
  for p,suffix in [(None,''),(0,'/supine'),(1,'/prone')]:
   m=scene==s
   if p is not None:m&=pose==p
   result[name+suffix]=dict(n=int(m.sum()),sr3=float(passed[m].float().mean()),escaped=float(escaped[m].float().mean()),return_mean=float(actual[m].mean()),discounted_return=float(actual_disc[m].mean()),max_head_mean=float(maxz[m].mean()),low_smp=float(low_smp[m].sum()/low_n[m].sum().clamp_min(1)),return_terms=dict(zip(terms,sums[m].mean(0).tolist())),discounted_terms=dict(zip(terms,disc[m].mean(0).tolist())))
 run.base.write(out/'diagnosis.json',dict(label=label,initial_qpos_sha=fingerprint,groups=result,table_low_reward_rate=dict(zip(terms,(low_terms/max(low_count,1)).tolist())),timeline=trace,term_integral_return_max_error=float((sums.sum(-1)-actual).abs().max())))
 np.savez_compressed(out/'diagnosis_trials.npz',scene=scene.cpu(),pose=pose.cpu(),sr3=passed.cpu(),escaped=escaped.cpu(),returns=actual.cpu(),discounted=actual_disc.cpu(),max_head=maxz.cpu(),first_clear=first_clear.cpu(),first_up=first_up.cpu(),terms=sums.cpu(),discounted_terms=disc.cpu())
run.evaluate=evaluate
if __name__=='__main__':run.main()
