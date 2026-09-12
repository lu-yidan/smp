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
