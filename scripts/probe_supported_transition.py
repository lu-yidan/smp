"""Read-only calibration of bounded supported progress on the frozen FT12k actor."""
from pathlib import Path
from dataclasses import asdict
import json,torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from train_prior_replay_transfer import build_config
from smp.rl.tasks.getup import supported_transition as t,v33_reward_transfer as v
ckpt=Path('/root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt')
cfg,agent=build_config('P7_transition',Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),512)
env=ManagerBasedRlEnv(cfg,device='cuda:0');env._pr_disable_replay=True
try:
 w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(w,asdict(agent),None,'cuda:0')
 runner.load(str(ckpt),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
 ref=torch.load(ckpt,map_location='cpu',weights_only=False)['infos']['scratch_tradeoffs']
 env._smp_normalizer.mean.copy_(ref['mean'].to(env.device));env._smp_normalizer.count.copy_(ref['count'].to(env.device))
 policy=runner.get_inference_policy();obs,_=env.reset();v.init(env);env._v_start_counter=-12000
 sums=torch.zeros(2,2,device=env.device);peak=0.
 with torch.inference_mode():
  for _ in range(500):
   obs,_,_,_=w.step(policy(obs));credit=env._transition_credit*.20
   assert torch.isfinite(credit).all() and (credit>=0).all() and (credit<=.200001).all()
   peak=max(peak,float(credit.max()))
   for i,mask in enumerate((~env._plate_active,env._plate_active)):
    sums[i,0]+=credit[mask].mean();sums[i,1]+=env._plate_product[mask].mean()
 result={'passed':True,'weight':.20,'episode_credit_upper_bound':.20,'steps':500,'num_envs':512,'peak_reward_rate':peak,'flat_credit_to_task':float(sums[0,0]/sums[0,1].clamp_min(1e-9)),'plate_credit_to_task':float(sums[1,0]/sums[1,1].clamp_min(1e-9)),'integrated_credit_and_task':(sums*.02).tolist(),'note':'frozen actor; replay disabled; rewards cannot exceed 0.20 integrated credit per episode; no parameter updates'}
 assert result['flat_credit_to_task']<.30,result
 Path('outputs').mkdir(exist_ok=True);Path('outputs/transition_probe.json').write_text(json.dumps(result,indent=2));print(result)
finally:env.close()
