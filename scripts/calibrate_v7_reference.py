"""Calibrate V7 error scale on frozen FT12k trajectories; never reuse f2s2 MSE units."""
import argparse,json,random,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from train_prior_replay_transfer import build_config,SOURCE_SHA
from smp.rl.utils import load_denoiser
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);a=p.parse_args()
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==SOURCE_SHA
cfg,agent=build_config('P1_v7',Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),1024)
cfg.terminations.pop('smp_too_low',None)
random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
 w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(w,asdict(agent),None,'cuda:0')
 runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
 policy=runner.get_inference_policy();obs,_=env.reset()
 with torch.random.fork_rng(devices=[0]):
  f2=load_denoiser('datasets/pretrain_ckpt/pretrained_getup_f2s2.pt','cuda:0')
 priors=[f2,env._smp_bundle]
 sums=torch.zeros(2,3,device='cuda:0');samples=0
 noise_rng=torch.Generator(device='cuda:0').manual_seed(981347)
 source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['infos']['scratch_tradeoffs']
 with torch.inference_mode():
  for i in range(500):
   obs,_,_,_=w.step(policy(obs))
   if i%5==0:
    features=env._smp_buffer.compute_features()
    for j,ts in enumerate((8,15,22)):
     noise=torch.randn(features.shape,device='cuda:0',generator=noise_rng)
     t=torch.full((env.num_envs,),ts,device='cuda:0',dtype=torch.long)
     for k,(model,scheduler,lo,hi,*_) in enumerate(priors):
      x=2*(features-lo)/(hi-lo+1e-8)-1
      err=(model(scheduler.add_noise(x,noise,t),t)-noise).square().mean()
      sums[k,j]+=err
    samples+=1
 ratio=(sums[1]/sums[0].clamp_min(1e-8)).cpu()
 ref={'mean':source['mean'].cpu().clone(),'count':source['count'].cpu().clone(),'source_sha':SOURCE_SHA,'steps':500,'num_envs':1024,'matched_mse':(sums/samples).cpu(),'matched_ratio':ratio}
 ref['mean'][[8,15,22]]*=ratio
 assert torch.isfinite(ref['mean']).all() and (ratio>0).all()
 # Give the calibrated reference the same frozen treatment as the mature source reference.
 ref['count'][ref['count']>0]=env._smp_normalizer.max_count+1
 Path('outputs').mkdir(exist_ok=True);torch.save(ref,'outputs/v7_reference.pt')
 Path('outputs/v7_reference.json').write_text(json.dumps({'protocol':'matched prior MSE ratio times FT12k reference; common trajectories and noise','matched_mse':ref['matched_mse'].tolist(),'matched_ratio':ratio.tolist(),'steps':500,'num_envs':1024,'source_sha':SOURCE_SHA,'mean':ref['mean'].tolist(),'count':ref['count'].tolist(),'prior_sha':hashlib.sha256(Path(cfg.events['init_smp_state'].params['ckpt_path']).read_bytes()).hexdigest()},indent=2))
 print('V7_REFERENCE_READY',ref['mean'][[8,15,22]].tolist())
finally:env.close()
