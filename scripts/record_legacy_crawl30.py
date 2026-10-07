"""Paired original 96D V33/V34, with real velocity observations, under 30cm tables."""
import train_crawl_repair as d
import argparse,json,random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.managers import ObservationTermCfg
from mjlab.envs import mdp

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--onnx',type=Path);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False);a.num_envs=256;a.seed=20261071;a.eval_steps=1150;a.bank='outputs/crawl_bank_30cm_v1/validation.npz';a.video=True;a.video_low=True;a.camera_azimuth=55;a.camera_elevation=-8
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
d.t.install();cfg,agent=d.config(a.num_envs,'CF0',True,a.seed)
cfg.events['gsi_reset'].params['bank_path']=a.bank
cfg.observations['actor'].terms={'base_lin_vel':ObservationTermCfg(func=mdp.builtin_sensor,params={'sensor_name':'robot/imu_lin_vel'}),**cfg.observations['actor'].terms}
agent.logger='tensorboard'
env=d.ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array')
try:
 w=d.Wrapper(env,clip_actions=agent.clip_actions);runner=d.Runner(w,asdict(agent),str(a.out),'cuda:0')
 runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
 parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False);d.verify_nested(parent['actor_state_dict'],runner.alg.save()['actor_state_dict'])
 obs,_=env.reset();env._tb_band.fill_(1);assert obs['actor'].shape[-1]==96
 d.audit(env,a.out)
 policy0=runner.get_inference_policy();vmax=0.
 def policy(obs):
  global vmax
  expected=mdp.builtin_sensor(env,sensor_name='robot/imu_lin_vel')
  assert torch.allclose(obs['actor'][:,:3],expected)
  vmax=max(vmax,float(expected.abs().max()))
  return policy0(obs)
 if a.onnx:
  import onnxruntime as ort
  opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
  session=ort.InferenceSession(str(a.onnx),opt,providers=['CPUExecutionProvider']);assert session.get_inputs()[0].shape[-1]==93
  def policy(obs):
   raw=obs['actor'][:,3:].cpu().numpy();assert raw.shape==(env.num_envs,93)
   return torch.tensor(np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0] for row in raw]),device=env.device)
  a.checkpoint=a.onnx
 d.evaluate(env,w,policy,obs,a)
 if not a.onnx:assert vmax>.01,'Velocity observations unexpectedly zero'
 d.write(a.out/'legacy_manifest.json',dict(policy=a.label,checkpoint=str(a.checkpoint),sha256=d.sha(a.checkpoint),observation_dim=93 if a.onnx else 96,linear_velocity='excluded from actor' if a.onnx else 'actual robot/imu_lin_vel, not zeroed',max_abs_velocity=vmax,table_underside_m=.30,seed=a.seed,actor_loaded_exact=not bool(a.onnx),controller='common deployment model/PD, entry blend and torque projection; not historical training physics',video='23 seconds, 25 fps, real time; first square supine/prone, no outcome selection'))
finally:env.close()
