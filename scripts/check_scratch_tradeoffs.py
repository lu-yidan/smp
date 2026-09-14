from dataclasses import asdict
from types import SimpleNamespace
import torch
from train_scratch_tradeoffs import build_config,COST_NAMES
from smp.rl.tasks.getup import scratch_tradeoffs as m
from train_fixed_low_ablation import build_config as original
bank='datasets/reset_banks/natural_curriculum_v1/train.npz'
r0,a=build_config('R0',bank);old,b=original('L4',bank)
# Original dynamics, observations, reset, algorithm and active rewards preserved.
x,y=asdict(r0),asdict(old)
x['scene']['sensors']=y['scene']['sensors'];x['metrics']=y['metrics']
for k in ['quiet_stand',*COST_NAMES]:assert x['rewards'].pop(k)['weight']==0
x['scene']['terrain']['spec_fn']=y['scene']['terrain']['spec_fn'];assert x==y and asdict(a)==asdict(b)
for i in range(1,8):
 cfg,agent=build_config(f'R{i}',bank);assert cfg.terminations['stood_up'].func==m.no_stand_termination
 assert asdict(agent)==asdict(a) and cfg.episode_length_s==10
 assert cfg.events['gsi_reset'].params==r0.events['gsi_reset'].params
 assert cfg.rewards['quiet_stand'].weight==(0 if i==1 else .1)
 expected=[-.05 if i in (4,7) else 0,-.1 if i in (4,7) else 0,-.005 if i in (5,7) else 0,-.02 if i in (6,7) else 0]
 assert [cfg.rewards[k].weight for k in COST_NAMES]==expected
 fn=cfg.rewards['task_smp_product'].params['task_terms'][0][0]
 assert (fn==m.slow_upward)==(i in (3,7))
for step,w in [(0,0),(12000,0),(36000,.5),(60000,1),(999999,1)]:assert m.ramp(SimpleNamespace(common_step_counter=step))==w
v=m.velocity_band(torch.tensor([-.1,0.,.1,.2,.3,.5,1.]))
assert torch.allclose(v[2:5],torch.ones(3)) and v[-1]<v[-2]<v[3] and v[0]<v[1]<v[2]
limits=torch.ones(29);speeds=torch.ones(29)*20;tau=torch.zeros(2,29);dq=torch.zeros_like(tau);tau[1,0]=1;dq[1,0]=20
e,s=m.load_costs(tau,dq,limits,speeds);assert torch.allclose(e,torch.tensor([0.,1/3]),atol=1e-6) and torch.allclose(s,e)
print('PASS: R0 original control, eight module assignments, fixed ramp, speed band and isolated-joint tail costs')
