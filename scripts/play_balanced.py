"""Local B1 play using the actual balanced plate environment; no hardware connection."""
import argparse,random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from train_balanced_dynamics import build_config
from smp.rl.tasks.getup import balanced_dynamics as bd,v33_reward_transfer as v
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--scene',choices=['flat','plate'],default='flat');p.add_argument('--pose',choices=['supine','prone','left','right'],default='supine')
 p.add_argument('--source',choices=['natural','procedural'],default='natural');p.add_argument('--sample',type=int,default=0)
 p.add_argument('--checkpoint',type=Path,default=Path('outputs/b1_play/B1_8500.pt'));p.add_argument('--speed',type=float,choices=[.125,.25,.5,1.,2.],default=.5)
 p.add_argument('--paused',action='store_true');p.add_argument('--headless',action='store_true');p.add_argument('--steps',type=int,default=100);p.add_argument('--upper-mass',type=float,default=1.)
 a=p.parse_args();assert a.checkpoint.is_file();assert 0<=a.sample<16,'validation pool sample must be 0..15'
 cfg,agent=build_config(32,'outputs/multiterrain_bank/validation.npz',True,'B1_dynamics')
 cfg.episode_length_s=20.;cfg.viewer.width=1280;cfg.viewer.height=900
 cfg.events['gsi_reset'].params['stress_upper']=a.upper_mass
 cfg.viewer.env_idx=['supine','prone','left','right'].index(a.pose)*8
 random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
 env=ManagerBasedRlEnv(cfg,device='cuda:0')
 try:
  bd.initialize(env,'outputs/multiterrain_bank/validation.npz')
  scene=0 if a.scene=='flat' else 1
  env._mt_scene.fill_(scene);env._mt_stratum.fill_(scene)
  env._mt_direction[:]=torch.arange(32,device=env.device)//8
  env._mt_source.fill_(0 if a.source=='natural' else 1)
  env._fixed_group=env._mt_direction+2;env._fixed_source=env._mt_source
  env._mt_pools={key:pool[a.sample:a.sample+1] for key,pool in env._mt_pools.items()}
  w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);agent.logger='tensorboard';agent.upload_model=False
  runner=MjlabOnPolicyRunner(w,asdict(agent),None,'cuda:0')
  runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})
  parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False);ref=parent['infos']['multiterrain']['smp_reference']
  env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.copy_(ref['count'].to(env.device))
  obs,_=env.reset();v.init(env);env._v_start_counter=-12000;env._v_ramp_updates=500
  assert obs['actor'].shape[-1]==93
  policy=runner.get_inference_policy(device='cuda:0')
  print(f'Checkpoint={a.checkpoint.resolve()} scene={a.scene} source={a.source} sample={a.sample} pose={a.pose}; nominal dynamics, no push/no observation noise, upper mass={a.upper_mass}; plate=6 kg, passive vertical slide only. 20s automatic reset. Space=pause; Enter=reset; Right=step; -/=speed; ,/.=environment (8 replicas per direction).',flush=True)
  if a.headless:
   with torch.inference_mode():
    for _ in range(a.steps):obs,reward,done,extra=w.step(policy(obs));assert torch.isfinite(reward).all()
   print('PLAY_PASS',a.scene,a.steps,flush=True)
  else:
   from mjlab.viewer import NativeMujocoViewer
   viewer=NativeMujocoViewer(w,policy)
   while viewer._time_multiplier>a.speed:viewer.decrease_speed()
   while viewer._time_multiplier<a.speed:viewer.increase_speed()
   if a.paused:viewer.pause()
   viewer.run()
 finally:env.close()
if __name__=='__main__':main()
