from dataclasses import asdict
from types import SimpleNamespace
import math,torch
from train_f2_task_tradeoffs import build_config
from train_r1_quality import build_config as f_config
from smp.rl.tasks.getup import f2_task_tradeoffs as c,r1_quality as f
bank='datasets/reset_banks/natural_curriculum_v1/train.npz'
base,agent=f_config('F2',bank)
for arm in c.ARMS:
 cfg,a=build_config(arm,bank);x,y=asdict(cfg),asdict(base)
 assert cfg.rewards['sustained_effort'].weight==-.05
 assert cfg.rewards['early_speed'].weight==(c.SPEED_WEIGHT if arm in ('C2','C3') else 0)
 assert (cfg.rewards['task_smp_product'].params['task_terms'][0][0]==c.quality_upward)==(arm in ('C1','C3'))
 x['metrics'].pop('c_speed_substep');x['metrics'].pop('c_quality');x['rewards'].pop('early_speed')
 if arm in ('C1','C3'):x['rewards']['task_smp_product']=y['rewards']['task_smp_product']
 x['scene']['terrain']['spec_fn']=y['scene']['terrain']['spec_fn'];assert x==y and asdict(a)==asdict(agent)
# Do not reintroduce R3's idle-reward boost. Smoothly cross the old .9m discontinuity.
z=torch.tensor([.3,.3,.3]);v=torch.tensor([0.,.25,1.]);q=torch.zeros(3)
assert torch.allclose(c.blend_upward(z,v,q),torch.tensor([math.exp(-6.25),1.,1.]))
left=c.blend_upward(torch.tensor([.9-1e-6]),torch.zeros(1),torch.ones(1)*.7)
right=c.blend_upward(torch.tensor([.9+1e-6]),torch.zeros(1),torch.ones(1)*.7)
assert abs(float(left-right))<1e-4
assert torch.allclose(c.blend_upward(torch.ones(3)*1.2,v,torch.tensor([0.,.5,1.])),torch.tensor([0.,.5,1.]))
# Previously unpenalized early-phase speed is now measured; moderate early motion is allowed.
dq=torch.zeros(3,29);dq[0,0]=8;dq[1,0]=20;dq[2,0]=20
cost=c.speed_cost(dq,torch.ones(29)*20,torch.tensor([0.,0.,1.]))
assert cost[0]==0 and 0<cost[1]<cost[2]
e=SimpleNamespace(common_step_counter=600000,_f_start_counter=480000,_c_start_counter=600000)
assert f.ramp(e)==1 and c.ramp(e)==0
e.common_step_counter+=6000;assert f.ramp(e)==1 and c.ramp(e)==.5
print('PASS: paired 2x2 configs; F2 fully retained; low idle unchanged; smooth height transition; early overspeed covered; independent ramp')
