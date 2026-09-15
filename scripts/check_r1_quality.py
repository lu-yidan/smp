from dataclasses import asdict
from types import SimpleNamespace
import torch
from train_r1_quality import build_config
from train_scratch_tradeoffs import build_config as base
from smp.rl.tasks.getup import r1_quality as m
bank='datasets/reset_banks/natural_curriculum_v1/train.npz'
r,a=base('R1',bank)
for arm in m.WEIGHTS:
 c,b=build_config(arm,bank);x,y=asdict(c),asdict(r)
 for key in ['f_sustained_substep','f_cache']:x['metrics'].pop(key)
 for key in m.NAMES:x['rewards'].pop(key)
 x['scene']['terrain']['spec_fn']=y['scene']['terrain']['spec_fn']
 assert x==y and asdict(a)==asdict(b)
 assert tuple(c.rewards[k].weight for k in m.NAMES)==m.WEIGHTS[arm]
# Short isolated torque spike is treated differently from sustained load.
z=torch.zeros(2,29);tau=z.clone();tau[:,0]=1
short=m.sustained_update(z,tau,.002)
long=z.clone()
for _ in range(500):long=m.sustained_update(long,tau,.002)
assert m.effort_cost(short).sum()==0 and (m.effort_cost(long)>.2).all()
assert m.effort_cost(torch.full_like(z,.4**2)).sum()==0
for s,v in [(480000,0),(486000,.5),(492000,1),(600000,1)]:assert m.ramp(SimpleNamespace(common_step_counter=s,_f_start_counter=480000))==v
print('PASS: only intended arm changes; no dynamics/noise/reset/PPO changes; transient vs sustained load; ramp')
# Sample callback clears only the newly reset environment's load memory.
e=SimpleNamespace(_r_accum=True,_f_effort_ms=torch.ones(2,29),_f_tick=10,cfg=SimpleNamespace(decimation=10),episode_length_buf=torch.tensor([0,10]),_f_speed_accum=torch.ones(2),scene={'robot':SimpleNamespace(data=SimpleNamespace(qfrc_actuator=torch.zeros(2,29),joint_vel=torch.zeros(2,29)))},_r_limits=torch.ones(29),_r_speeds=torch.ones(29)*20,physics_dt=.002)
m.sample_substep(e)
assert e._f_effort_ms[0].sum()==0 and (e._f_effort_ms[1]>.99).all()
print('PASS: per-environment episode reset clears sustained-load history without clearing peers')
