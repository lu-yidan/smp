"""Run against the archived V34 source tree, not the current deployment task."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MUJOCO_EGL_DEVICE_ID',os.environ.get('CUDA_VISIBLE_DEVICES','0'))
import argparse,json,hashlib,random
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from mjlab.envs import ManagerBasedRlEnv,mdp
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
import smp.rl.tasks
p=argparse.ArgumentParser();p.add_argument('--pose',choices=['supine','prone'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
os.environ['SMP_PLAY_ESCAPE_RESET_POSE']=a.pose
seed=20260818;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
task='Smp-Getup-Escape-Plate-V34-G1';cfg=load_env_cfg(task,play=True);agent=load_rl_cfg(task);cfg.scene.num_envs=64;cfg.seed=seed;cfg.episode_length_s=1000;cfg.viewer.width=640;cfg.viewer.height=480;cfg.viewer.max_extra_envs=0
cfg.observations['actor'].enable_corruption=False
cfg.events['mixed_fall_reset'].params['mode_weights']=(0.,1.,0.,0.) if a.pose=='supine' else (1.,0.,0.,0.)
params=cfg.events['reset_escape_obstacle'].params;params.update(longitudinal_offset=-.10,xy_offset_range=.005,plate_mass_range=(8.,8.),initial_max_mass=8.)
ckpt=Path('/mnt/workspace/user/luyidan/baselines/G1_Recovery_Below_Block/v34/model_98000.pt')
env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array');writer=None
try:
 w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(w,asdict(agent),device='cuda:0');runner.load(str(ckpt),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});policy=runner.get_inference_policy();obs=w.get_observations();assert obs['actor'].shape[-1]==96
 np.savez_compressed(a.out/'initial.npz',qpos=env.sim.data.qpos.cpu(),qvel=env.sim.data.qvel.cpu())
 writer=imageio.get_writer(str(a.out/'video.mp4'),fps=25,codec='libx264',ffmpeg_params=['-threads','1']);vmax=0.;reason='23s observation completed';escaped=False;maxhead=0.;robot=env.scene['robot'];head=robot.find_sites(['head'],preserve_order=True)[0][0]
 for k in range(1150):
  expected=mdp.builtin_sensor(env,sensor_name='robot/imu_lin_vel');assert torch.allclose(obs['actor'][:,:3],expected);vmax=max(vmax,float(expected[0].abs().max()))
  with torch.inference_mode():obs,_,done,_=w.step(policy(obs))
  if bool(done[0]):reason='original task termination; stopped before showing a new reset';break
  phase=int(env._escape_phase[0]);escaped|=phase>=3;maxhead=max(maxhead,float(robot.data.site_pos_w[0,head,2]))
  if k%2==1:
   env.cfg.viewer.env_idx=0;cam=env._offline_renderer._cam;cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=55;cam.elevation=-12;root=robot.data.root_link_pos_w[0].cpu().numpy();cam.lookat[:]=[root[0],root[1],max(.35,float(root[2])*.5+.2)];cam.distance=3.5
   im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,48),fill='black');draw.text((8,6),f'V34 @98000 | {a.pose} | actual velocity | t={(k+1)*.02:.2f}s',fill='white');draw.text((8,26),'Vertical sliding plate | 0.90 x 0.64 m | 8 kg | 1x',fill='white');writer.append_data(np.asarray(im))
 d=dict(pose=a.pose,checkpoint=str(ckpt),sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),seed=seed,selected_env=0,selection='first env, no outcome selection',velocity='real imu_lin_vel,96D',max_abs_velocity=vmax,steps=k+1,reason=reason,max_head_height=maxhead,last_phase=int(env._escape_phase[0]),plate_params=params,original_task=task)
 (a.out/'manifest.json').write_text(json.dumps(d,indent=2,default=str));print('COMPLETE',json.dumps(d,default=str),flush=True)
finally:
 if writer:writer.close()
 env.close()
