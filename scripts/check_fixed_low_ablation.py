"""Check that only low-SMP termination differs and fixed quotas cover all directions."""
from dataclasses import asdict
from types import SimpleNamespace
import torch
from train_fixed_low_ablation import build_config
from train_natural_curriculum import build_config as natural_config
from smp.rl.tasks.getup.fixed_low_reset import fixed_low_smp,quota_counts
from smp.rl.tasks.getup.mdp.terminations import smp_too_low
bank='datasets/reset_banks/natural_curriculum_v1/train.npz'
a,aa=build_config('L0',bank);b,bb=build_config('L1',bank)
assert a.terminations['smp_too_low'].params['low_enabled'] is True
assert b.terminations['smp_too_low'].params['low_enabled'] is False
a.terminations['smp_too_low'].params['low_enabled']=False
da,db=asdict(a),asdict(b)
# Terrain's default factory creates distinct, equivalent zero-capture lambdas.
fa,fb=da['scene']['terrain']['spec_fn'],db['scene']['terrain']['spec_fn']
assert fa.__code__==fb.__code__ and fa.__defaults__==fb.__defaults__ and fa.__closure__==fb.__closure__
da['scene']['terrain']['spec_fn']=fb
assert da==db and asdict(aa)==asdict(bb)
c,ca=natural_config(bank)
for key in ['observations','actions','sim','decimation','episode_length_s']:assert asdict(b)[key]==asdict(c)[key],key
for key in ['time_out','stood_up','unstable_sim_state']:assert asdict(b.terminations[key])==asdict(c.terminations[key]),key
assert asdict(aa)==asdict(ca)
assert a.rewards['task_smp_product'].params==c.rewards['task_smp_product'].params and a.rewards['task_smp_product'].weight==c.rewards['task_smp_product'].weight
for key in c.events:
 if key!='gsi_reset':assert asdict(a.events[key])==asdict(c.events[key]),key
assert quota_counts(4096)==(2048,1640,102,102,102,102)
x=SimpleNamespace(num_envs=6,device='cpu',_fixed_group=torch.arange(6),_smp_raw_err=torch.ones(6),episode_length_buf=torch.full((6,),4),max_episode_length=500)
assert not fixed_low_smp(x,low_enabled=True).any() and not fixed_low_smp(x,low_enabled=False).any()
x.episode_length_buf[:]=5
assert torch.equal(fixed_low_smp(x,low_enabled=True),smp_too_low(x,grace_steps=5))
assert fixed_low_smp(x,low_enabled=False).tolist()==[True,True,False,False,False,False]
assert x._fixed_would_low.all()
x._smp_raw_err[:]=0
assert not fixed_low_smp(x,low_enabled=True).any()
x.episode_length_buf[:]=500
assert b.terminations['time_out'].func(x).all()
print('PASS: sole L0/L1 config difference is low_enabled; quota counts; original >=5 gate; only low groups bypass; timeout/noise/DR/PPO preserved')

# L2 changes only the low-group threshold, with the original timing retained.
l0,p0=build_config('L0',bank);l2,p2=build_config('L2',bank)
assert l2.terminations['smp_too_low'].params.pop('low_threshold')==.005
d0,d2=asdict(l0),asdict(l2)
d2['scene']['terrain']['spec_fn']=d0['scene']['terrain']['spec_fn']
assert d0==d2 and asdict(p0)==asdict(p2)
x._smp_raw_err[:]=.75
x.episode_length_buf[:]=4
assert not fixed_low_smp(x,low_threshold=.005).any()
x.episode_length_buf[:]=5
assert fixed_low_smp(x).all()
assert fixed_low_smp(x,low_threshold=.005).tolist()==[True,True,False,False,False,False]
x._smp_raw_err[:]=1
assert fixed_low_smp(x,low_threshold=.005).all()
x._smp_raw_err[:]=0
assert not fixed_low_smp(x,low_threshold=.005).any()
print('PASS: L2 changes only low threshold to .005; original timing and other groups preserved')

# L3 is L1 with a larger, direction-balanced low-state quota only.
l1,p1=build_config('L1',bank);l3,p3=build_config('L3',bank)
assert l3.events['gsi_reset'].params.pop('low_fraction')==.2
assert l3.events['gsi_reset'].params.pop('late_fraction')==.4
assert l3.events['gsi_reset'].params.pop('procedural_fraction')==0.
l3.events['gsi_reset'].params.pop('procedural_bank_path')
d1,d3=asdict(l1),asdict(l3)
d3['scene']['terrain']['spec_fn']=d1['scene']['terrain']['spec_fn']
assert d1==d3 and asdict(p1)==asdict(p3)
assert quota_counts(4096,.2,.4)==(1638,1642,204,204,204,204)
print('PASS: L3 changes only reset quota, 816 low environments equally split into four directions')

for arm,frac in [('L4',.05),('L5',.1)]:
 a,aa=build_config('L3',bank);b,bb=build_config(arm,bank)
 assert b.events['gsi_reset'].params['procedural_fraction']==frac
 b.events['gsi_reset'].params['procedural_fraction']=0.
 da,db=asdict(a),asdict(b);db['scene']['terrain']['spec_fn']=da['scene']['terrain']['spec_fn']
 assert da==db and asdict(aa)==asdict(bb)
print('PASS L3/L4/L5 differ only in fixed procedural source fraction')
