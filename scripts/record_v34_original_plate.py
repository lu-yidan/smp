"""Run against the archived V34 source tree, not the current deployment task."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MUJOCO_EGL_DEVICE_ID',os.environ.get('CUDA_VISIBLE_DEVICES','0'))
import argparse,json,hashlib,random
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np,torch,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from mjlab.envs import ManagerBasedRlEnv,mdp
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
import smp.rl.tasks
p=argparse.ArgumentParser();p.add_argument('--pose',choices=['supine','prone'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--onnx',type=Path);p.add_argument('--deploy-contract',type=Path);p.add_argument('--diagnostic-no-invalid-reset',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
os.environ['SMP_PLAY_ESCAPE_RESET_POSE']=a.pose
seed=20260818;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
task='Smp-Getup-Escape-Plate-V34-G1';cfg=load_env_cfg(task,play=True);agent=load_rl_cfg(task);cfg.scene.num_envs=64;cfg.seed=seed;cfg.episode_length_s=1000;cfg.viewer.width=640;cfg.viewer.height=480;cfg.viewer.max_extra_envs=0
cfg.observations['actor'].enable_corruption=False
cfg.events['mixed_fall_reset'].params['mode_weights']=(0.,1.,0.,0.) if a.pose=='supine' else (1.,0.,0.,0.)
params=cfg.events['reset_escape_obstacle'].params;params.update(longitudinal_offset=-.10,xy_offset_range=.005,plate_mass_range=(8.,8.),initial_max_mass=8.)
if a.deploy_contract:
 import importlib.util,sys
 spec=importlib.util.spec_from_file_location('plate_deploy_contract',a.deploy_contract);contract=importlib.util.module_from_spec(spec);sys.modules[spec.name]=contract;spec.loader.exec_module(contract);cfg=contract.apply_deployment_contract(cfg)
 cfg.scene.sensors=tuple(replace(sensor,primary=replace(sensor.primary,pattern=contract.COLLISION_PATTERN)) if sensor.name=='robot_obstacle_contact' else sensor for sensor in cfg.scene.sensors)
 for term in list(cfg.events.values())+list(cfg.rewards.values()):
  if 'collision_geom_pattern' in term.params:term.params['collision_geom_pattern']=contract.COLLISION_PATTERN
if a.diagnostic_no_invalid_reset:cfg.terminations.pop('invalid_escape_episode',None)
ckpt=Path('/mnt/workspace/user/luyidan/baselines/G1_Recovery_Below_Block/v34/model_98000.pt')
env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array');writer=None
try:
 w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(w,asdict(agent),device='cuda:0');runner.load(str(ckpt),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False});policy=runner.get_inference_policy();obs=w.get_observations();assert obs['actor'].shape[-1]==96
 if a.onnx:
  import onnxruntime as ort
  opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
  session=ort.InferenceSession(str(a.onnx),opt,providers=['CPUExecutionProvider']);assert session.get_inputs()[0].shape[-1]==93
  def policy(observations):
   raw=observations['actor'][:,3:].cpu().numpy();assert raw.shape==(env.num_envs,93)
   return torch.tensor(np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0] for row in raw]),device=env.device)
 np.savez_compressed(a.out/'initial.npz',qpos=env.sim.data.qpos.cpu(),qvel=env.sim.data.qvel.cpu())
 writer=imageio.get_writer(str(a.out/'video.mp4'),fps=25,codec='libx264',ffmpeg_params=['-threads','1']);vmax=0.;reason='23s observation completed';escaped=False;maxhead=0.;invalid_at=None;robot=env.scene['robot'];head=robot.find_sites(['head'],preserve_order=True)[0][0]
 for k in range(1150):
  expected=mdp.builtin_sensor(env,sensor_name='robot/imu_lin_vel');assert torch.allclose(obs['actor'][:,:3],expected);vmax=max(vmax,float(expected[0].abs().max()))
  with torch.inference_mode():obs,_,done,_=w.step(policy(obs))
  if bool(done[0]):reason='original task termination; stopped before showing a new reset';break
  bad=bool(getattr(env,'_escape_invalid_contact',torch.zeros(1))[0]) or bool(getattr(env,'_escape_invalid_setup',torch.zeros(1))[0])
  if bad and invalid_at is None:invalid_at=(k+1)*.02
  phase=int(env._escape_phase[0]);escaped|=phase>=3;maxhead=max(maxhead,float(robot.data.site_pos_w[0,head,2]))
  if k%2==1:
   env.cfg.viewer.env_idx=0;cam=env._offline_renderer._cam;cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=55;cam.elevation=-12;root=robot.data.root_link_pos_w[0].cpu().numpy();cam.lookat[:]=[root[0],root[1],max(.35,float(root[2])*.5+.2)];cam.distance=3.5
   im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,48),fill='black');draw.text((8,6),f'{"V34 93D gate6000" if a.onnx else "V34 @98000"} | {a.pose} | t={(k+1)*.02:.2f}s',fill='white');draw.text((8,26),'Vertical sliding plate | 0.90 x 0.64 m | 8 kg | 1x',fill='white');
   if invalid_at is not None:draw.rectangle((0,49,640,74),fill='darkred');draw.text((8,54),f'DIAGNOSTIC: invalid contact/setup since {invalid_at:.2f}s; no reset',fill='white')
   writer.append_data(np.asarray(im))
 d=dict(pose=a.pose,checkpoint=str(ckpt),sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),seed=seed,selected_env=0,selection='first env, no outcome selection',velocity='real imu_lin_vel,96D',max_abs_velocity=vmax,steps=k+1,reason=reason,max_head_height=maxhead,last_phase=int(env._escape_phase[0]),plate_params=params,original_task=task,diagnostic_no_invalid_reset=a.diagnostic_no_invalid_reset,first_invalid_s=invalid_at,robot_model='RoboMimic deployment model and control' if a.deploy_contract else 'historical V34',physics_dt=env.cfg.sim.mujoco.timestep,control_dt=env.step_dt)
 if a.deploy_contract:d.update(deployment_contract=str(a.deploy_contract),source_xml_sha=hashlib.sha256((contract.ASSETS/'source.xml').read_bytes()).hexdigest(),control_sha=hashlib.sha256((contract.ASSETS/'control.yaml').read_bytes()).hexdigest())
 if a.onnx:d.update(checkpoint=str(a.onnx),sha256=hashlib.sha256(a.onnx.read_bytes()).hexdigest(),velocity='93D actor excludes linear velocity; raw obs slice3:',observation_dim=93,policy='v34_93d_gate6000')
 (a.out/'manifest.json').write_text(json.dumps(d,indent=2,default=str));print('COMPLETE',json.dumps(d,default=str),flush=True)
finally:
 if writer:writer.close()
 env.close()
