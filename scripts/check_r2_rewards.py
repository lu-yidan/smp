"""Reward invariants and real-log stalled-joint signal calibration (not sim replay)."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from smp.rl.tasks.getup.r2_ablation import stalled_update

def integral(tau,dq,dt=.02):
 timer=torch.zeros(tau.shape[1]);values=[]
 for t,v in zip(torch.tensor(tau,dtype=torch.float32),torch.tensor(dq,dtype=torch.float32)):
  timer,cost=stalled_update(timer,t,v,dt);values.append(float(cost.max()))
 return np.array(values)

x=np.full((150,29),.95);zero=np.zeros_like(x)
stationary=integral(x,zero);moving=integral(x,np.ones_like(x));short=integral(x[:10],zero[:10]);low=integral(x*.5,zero)
assert stationary.sum()*.02>2 and moving.max()==0 and short.max()==0 and low.max()==0
# Confirmation+best-so-far escape credit cannot exceed .5 or reward repeated cycles.
for stream in [np.ones(100),np.tile([0.,1.],50),np.r_[np.linspace(0,1,40),np.zeros(20),np.linspace(0,1,40)]]:
 window=np.zeros(10);best=credit=0
 for i,v in enumerate(stream):
  window[i%10]=v;confirmed=window.min();credit+=.5*max(0,confirmed-best);best=max(best,confirmed)
 assert credit<=.5+1e-8
report={'stationary_3s_raw_integral':float(stationary.sum()*.02),'moving_integral':float(moving.sum()*.02),'short_200ms_integral':float(short.sum()*.02),'escape_credit_cap_passed':True}
if len(sys.argv)>1:
 root=Path(sys.argv[1]);sys.path.insert(0,str(root));from common.logger import Logger
 for tag,activation,joint in [('202137',1,11),('202905',10,11),('202905',10,14),('202905',11,14)]:
  d=Logger.load(str(root/f'logs/20260919_{tag}_freekick.bin'));mask=(d['executed_fsm_state']==18)&(d['smp_activation_id']==activation)
  tau=d['tau_cmd_est'][mask]/np.array(d['_meta']['smp_recovery']['config']['tau_limit']);dq=d['dq'][mask]
  # Sim uses physics substeps + 0.5s no-progress confirmation. Here validate only
  # the high-torque/low-speed temporal detector on available 50Hz PD estimates.
  c=integral(tau[:,joint:joint+1],dq[:,joint:joint+1]);report[f'{tag}_activation{activation}_joint{joint}']={'raw_integral':float(c.sum()*.02),'duration_s':len(c)*.02}
Path('outputs/reward_unit_calibration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
