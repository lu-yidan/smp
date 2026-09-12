"""Meaningful invariants for curriculum sampling, holdouts and baseline controls."""
from dataclasses import asdict
import numpy as np
from train_natural_curriculum import build_config,passes
from train_termination_ablation import build_config as baseline
from smp.rl.tasks.getup.natural_curriculum import weights_for,MIXES,STAGES,DIRECTIONS
b=np.load('datasets/reset_banks/natural_curriculum_v1/train.npz')
for i,mix in enumerate(MIXES):
 w=weights_for(b,i)
 assert np.allclose([w[b['stages']==s].sum() for s in STAGES],mix)
 assert np.allclose([w[(b['stages']=='low')&(b['labels']==d)].sum() for d in DIRECTIONS],mix[2]/4)
sets=[]
for part in ['train','validation','test']:
 q=np.load('datasets/reset_banks/natural_curriculum_v1/'+part+'.npz');sets.append({x.replace('__mirror','') for x in q['clips']})
 for d in DIRECTIONS:assert ((q['stages']=='low')&(q['labels']==d)).any(),(part,d)
assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
cfg,agent=build_config('datasets/reset_banks/natural_curriculum_v1/train.npz',128)
ref,ra=baseline('B1',128,20260912)
for field in ['rewards','terminations','observations','actions','sim']:
 assert asdict(cfg)[field]==asdict(ref)[field],field
assert asdict(agent)==asdict(ra)
for key in ref.events:
 if key not in ('gsi_reset','gsi_refresh','init_smp_state'):assert asdict(cfg.events[key])==asdict(ref.events[key]),key
assert cfg.events['init_smp_state'].params['gsi_buffer_size']==0 and 'gsi_refresh' not in cfg.events
s={name:{'stable_1s':1.} for name in STAGES+DIRECTIONS};assert passes(s,0) and passes(s,1)
s['right_side_down']['stable_1s']=.14;assert not passes(s,0)
s['right_side_down']['stable_1s']=.2;assert passes(s,0) and not passes(s,1)
print('PASS sampling weights, clip-group separation, four-direction validation, gating and unchanged B1 reward/PPO/noise/DR/termination')
