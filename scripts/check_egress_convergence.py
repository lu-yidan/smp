"""Regression checks for egress latch, ascent isolation, and anchor loopholes."""
import json
from pathlib import Path
import torch,numpy as np
from smp.rl.tasks.getup import egress_convergence as ec,v33_reward_transfer as v
seen=torch.tensor([False,False]);anchor=torch.zeros(2,2);age=torch.zeros(2);xy=torch.tensor([[1.,2.],[4.,5.]])
seen,anchor,age,first=ec.latch_step(seen,torch.tensor([True,False]),anchor,xy,age,.02)
assert seen.tolist()==[True,False] and first.tolist()==[True,False]
saved=anchor.clone()
seen,anchor,age,first=ec.latch_step(seen,torch.tensor([False,False]),anchor,xy+1,age,.02)
assert seen.tolist()==[True,False] and torch.equal(anchor,saved) and age.eq(0).all()
seen,anchor,age,first=ec.latch_step(seen,torch.tensor([True,False]),anchor,xy+2,age,.02)
assert not first.any() and torch.equal(anchor,saved)
for arm in ec.ARMS:
 mask=ec.progress_mask(arm,seen)
 assert bool(mask[0])==(arm=='EC0') and bool(mask[1])
for arm,factor in [('EC0',1.),('EC3',1.5),('EC4',2.),('EC7',1.5)]:
 assert torch.allclose(ec.speed_multiplier(arm,seen,torch.ones(2)),torch.tensor([factor,1.]))
 assert ec.speed_multiplier(arm,seen,torch.zeros(2)).eq(1).all()
# Standing speed limit and downward penalty stay unchanged when ascent doubles.
z=torch.tensor([-.5,-.5,.4,.4]);stage=torch.tensor([0,3,0,3])
a=v.vertical_overspeed(z,stage,True,1.);b=v.vertical_overspeed(z,stage,True,2.)
assert torch.equal(a[:2],b[:2]) and a[3]==b[3] and b[2]<a[2]
assert ec.anchor_cost(torch.tensor([.1,.3,.6,2.]),torch.ones(4),torch.ones(4,dtype=torch.bool)).tolist()==[0.,0.,1.,4.]
assert ec.task_multiplier('EC5',torch.ones(2,dtype=torch.bool),torch.tensor([False,True]),torch.tensor([0.,.5])).tolist()[1]==1.
pools={}
for split in ['train','validation']:
 b=np.load(f'outputs/ceiling_bank/{split}.npz');stats={}
 for d in range(4):
  h=.42 if d<2 else .50
  for k in range(2):
   mask=(b['stratum']==0)&(b['direction']==d)&(b['source']==k)
   fit=int((b['robot_top'][mask]<=h-.01).sum());assert fit>0,(split,d,k)
   stats[f'{d}/{k}']={'min_height':h,'eligible':fit,'total':int(mask.sum())}
 pools[split]=stats
report={'reentry_does_not_rearm':True,'anchor_never_follows_robot':True,'speed_flat_and_trapped_unchanged':True,'standing_downward_penalty_unchanged':True,'height_pools':pools}
Path('outputs/egress_reward_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
