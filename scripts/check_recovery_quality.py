from dataclasses import asdict
import torch
from train_recovery_continuation import build_config
from smp.rl.tasks.getup.recovery_quality import cost_vectors,speed_reference
from smp.rl.tasks.getup.master_deployment_contract import JOINT_NAMES
b='datasets/reset_banks/natural_curriculum_v1/train.npz'
a,aa=build_config('L4',b)
for arm,expected in [('L4-Q1',[-.2,-.2,0.]),('L4-Q2',[-.2,-.2,-.002])]:
 c,cc=build_config(arm,b)
 for n,w in zip(['effort_excess','speed_excess','target_slew'],expected):
  assert c.rewards[n].weight==w;c.rewards[n].weight=0.
 da,dc=asdict(a),asdict(c);dc['scene']['terrain']['spec_fn']=da['scene']['terrain']['spec_fn']
 assert da==dc and asdict(aa)==asdict(cc)
limits=torch.tensor([10.,20.]);speeds=torch.tensor([20.,40.]);tau=torch.tensor([[0.,0.],[7.,14.],[10.,20.]]);dq=torch.tensor([[0.,0.],[10.,20.],[20.,40.]])
e,v=cost_vectors(tau,dq,limits,speeds)
assert torch.allclose(e,torch.tensor([0.,0.,1.]),atol=1e-6) and torch.allclose(v,torch.tensor([0.,0.,1.]))
assert len(speed_reference(JOINT_NAMES))==29
print('PASS: controls differ only in cost weights; soft thresholds and normalization verified')
