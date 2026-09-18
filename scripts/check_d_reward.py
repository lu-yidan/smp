"""Bounded-credit and one-shot regression checks independent of the simulator."""
from types import SimpleNamespace as N
import torch
from smp.rl.tasks.getup import d_recovery as d
n=3;e=N(num_envs=n,device='cpu',step_dt=.02,common_step_counter=0,cfg=N(seed=1))
d.init(e);e._mt_escaped=torch.zeros(n,dtype=torch.bool);e._mt_ever_contact=torch.ones(n,dtype=torch.bool);e._mt_invalid=torch.zeros(n,dtype=torch.bool)
e.scene={'robot':N(data=N(joint_vel=torch.zeros(n,29)))}
oldp,oldh=d.potential,d.mt.height
d.mt.height=lambda env:torch.zeros(n)
try:
 for i in range(200):
  vals=torch.tensor([min(i/100,1),0.,float(i%2)])
  d.potential=lambda env:vals
  e.common_step_counter+=1;d.update(e)
 assert torch.allclose(e._d_credit,torch.tensor([2.,0.,0.]),atol=1e-5),e._d_credit
 paid=0.
 for k in range(30):
  e._mt_escaped[:]=k%2==0;e._mt_invalid[2]=True
  e.common_step_counter+=1;d.update(e);paid+=e._d_escape_bonus*e.step_dt
 assert torch.equal(paid,torch.tensor([1.,1.,0.])),paid
 print('PASS: progress <=2; static and unconfirmed oscillation zero; escape exactly once; invalid excluded')
finally:d.potential=oldp;d.mt.height=oldh
