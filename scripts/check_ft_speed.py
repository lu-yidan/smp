import torch
from smp.rl.tasks.getup.v33_reward_transfer import speed_parameters,vertical_overspeed
s=torch.arange(4);like=torch.zeros(4)
t,u,d=speed_parameters(s,like,True)
assert torch.allclose(t,torch.tensor([.10,.15,.20,0.]))
assert torch.allclose(u,torch.tensor([.25,.30,.30,.12]))
vz=torch.linspace(-1,1,1001)
for stage in range(4):
 ss=torch.full((len(vz),),stage,dtype=torch.long)
 old=vertical_overspeed(vz,ss,False);new=vertical_overspeed(vz,ss,True)
 assert torch.equal(old,(vz.abs()-.2).clamp_min(0).square())
 assert torch.equal(old[vz<=0],new[vz<=0])
 if stage==3:assert torch.equal(old,new)
 else:assert (new[vz>0]<=old[vz>0]).all()
 target,upper,lower=speed_parameters(ss,vz,False)
 prior_limits=vz.new_tensor((.16,.18,.18,.12))[ss]
 excess=(vz-upper).clamp_min(0).square()+(-vz-lower).clamp_min(0).square()
 assert torch.equal(excess,(vz.abs()-prior_limits).clamp_min(0).square())
print('PASS: baseline exact thresholds; upward-only relaxation; descent and standing overspeed unchanged')
