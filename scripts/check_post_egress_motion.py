"""Check the factorial controls and no-cost-under-constraint invariants."""
import torch
from train_egress_convergence import build_config
from smp.rl.tasks.getup import egress_convergence as ec
one=torch.ones(5);enabled=torch.tensor([True,False,False,True,True])
gate,xy,yaw=ec.motion_costs(one,one,torch.tensor([.5,1.2,1.2,1.2,1.2]),one,torch.tensor([1.,1.,1.,0.,1.]),enabled)
assert gate.tolist()==[.25,0.,0.,0.,1.]
assert torch.all(xy[1:4]==0) and torch.all(yaw[1:4]==0)
assert torch.allclose(xy[4],4*xy[0])
_,xy,yaw=ec.motion_costs(one*.15,one*.3,one,one,one,enabled)
assert not xy.any() and not yaw.any()
configs={arm:build_config(192,arm=arm)[0] for arm in ('EC3','EC7','ES0','ES1','ES2','ES3')}
for arm in ('ES0','ES1','ES2','ES3'):
 c=configs[arm]
 assert ('egress_anchor' in c.rewards)==(arm in ('ES2','ES3'))
 assert ('egress_horizontal_motion' in c.rewards)==(arm in ('ES1','ES3'))
 assert ('egress_yaw_motion' in c.rewards)==(arm in ('ES1','ES3'))
 assert ec.speed_multiplier(arm,one.bool(),one).eq(1.5).all()
 assert not ec.progress_mask(arm,one.bool()).any()
 base={k:v for k,v in c.rewards.items() if k not in ('egress_anchor','egress_horizontal_motion','egress_yaw_motion')}
 assert base==configs['ES0'].rewards
assert configs['ES0'].rewards==configs['EC3'].rewards
assert configs['ES2'].rewards==configs['EC7'].rewards
print('PASS: factorial isolation; EC3/EC7 reward equivalence; no flat/constrained/reentry cost; posture ramp; deadbands; shared ascent factor')
