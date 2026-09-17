"""Calibrate a bounded supported-progress reward on frozen FT12k, without PPO."""
import argparse,json,random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from train_prior_replay_transfer import build_config
from smp.rl.tasks.getup import v33_reward_transfer as v,supported_transition as t
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
cfg,agent=build_config('P7_transition',Path('datasets/reset_banks/natural_curriculum_v1/train.npz'),1024)
random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
env=ManagerBasedRlEnv(cfg,device='cuda:0');env._pr_disable_replay=True
try:
 w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(w,asdict(agent),None,'cuda:0')
 runner.load(a.checkpoint,load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
 parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['infos']['scratch_tradeoffs']
 env._smp_normalizer.mean.copy_(parent['mean'].to(env.device));env._smp_normalizer.count.copy_(parent['count'].to(env.device))
 obs,_=env.reset();v.init(env);env._v_start_counter=-24*500;policy=runner.get_inference_policy()
 sums=torch.zeros(env.num_envs,device=env.device);max_credit=0.;bonus_total=0.;task_total=0.;constrained_violations=0
 with torch.inference_mode():
  for step in range(400):
   phase=env._escape_phase.clone()
   obs,_,done,_=w.step(policy(obs));credit=env._transition_credit*.20*env.step_dt
   assert torch.isfinite(credit).all() and (credit>=0).all()
   constrained_violations+=int(((phase==1)|(phase==2)) .logical_and(credit>0).sum())
   sums+=credit;max_credit=max(max_credit,float(sums.max()));sums[done.bool()]=0
   bonus_total+=float(credit.sum());task_total+=float(env._plate_product.sum()*env.step_dt)
   assert torch.equal(t.progress(env),t.progress(env)), 'Repeated reads must not consume progress'
 assert max_credit<=.20001 and constrained_violations==0,(max_credit,constrained_violations)
 result={'passed':True,'steps':400,'envs':1024,'weight':.20,'maximum_episode_progress_credit':max_credit,'theoretical_episode_bound':.20,'bonus_to_task_ratio':bonus_total/max(task_total,1e-8),'constrained_bonus_violations':constrained_violations,'policy':'frozen FT12k; no training','note':'Calibration ratio is not a guarantee under the learned policy'}
 Path('outputs/transition_probe.json').write_text(json.dumps(result,indent=2));print(result)
finally:env.close()
