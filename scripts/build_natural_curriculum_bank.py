"""Recheck static motion slices on deployment model and split by original clip."""
import argparse, hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np
import mujoco
from mjlab.scene import Scene
from train_termination_ablation import build_config
from smp.rl.tasks.getup.master_deployment_contract import JOINT_NAMES
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b=np.load(a.source);groups=np.array([s.replace('__mirror','') for s in b['clips']]);split=[]
for group in groups:
 test=int(hashlib.sha256(('reset-v1:'+group).encode()).hexdigest()[:8],16)%5==0
 val=int(hashlib.sha256(('curriculum-val-v1:'+group).encode()).hexdigest()[:8],16)%5==0
 split.append('test' if test else ('validation' if val else 'train'))
split=np.array(split); moved=[]
# Hash split alone can omit rare prone clips: move whole training clip groups.
for label in ('supine','prone','left_side_down','right_side_down'):
  mask=(b['stages']=='low')&(b['labels']==label)
  if not np.any(mask&(split=='validation')):
    candidates=sorted(set(groups[mask&(split=='train')]),key=lambda g:hashlib.sha256(g.encode()).hexdigest())
    candidates=[g for g in candidates if np.sum(mask&(groups==g))>=8]
    assert len(candidates)>=2, ('insufficient independent clips',label)
    group=candidates[0];split[groups==group]='validation';moved.append(group)
cfg,_=build_config('B1',1);m=Scene(cfg.scene,device='cpu').compile();d=mujoco.MjData(m)
joints=[int(m.joint('robot/'+n).qposadr[0]) for n in JOINT_NAMES];limits=np.array([m.joint('robot/'+n).range for n in JOINT_NAMES]);head=m.site('robot/head').id
mins=[];head_z=[]
for q in b['qpos']:
 assert np.isfinite(q).all() and ((q[7:]>=limits[:,0])&(q[7:]<=limits[:,1])).all()
 d.qpos[:7]=q[:7];d.qpos[joints]=q[7:];d.qvel[:]=0;mujoco.mj_forward(m,d)
 dist=min([0.]+[float(c.dist) for c in d.contact]);assert dist>=-.002,(dist,q)
 mins.append(dist);head_z.append(float(d.site_xpos[head,2]))
a.output.mkdir(parents=True,exist_ok=True);summary={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'split_rule':'test=reset-v1 hash mod5; validation=curriculum-val-v1 hash mod5 among remainder; mirror pairs stay together','static_history':'zero velocities, repeated deployment FK state','moved_whole_groups_for_validation_coverage':moved,'partitions':{}}
for name in ['train','validation','test']:
 ids=np.flatnonzero(split==name);assert len(ids)
 payload={k:b[k][ids] for k in ['qpos','labels','stages','clips','origins']};payload['source_indices']=ids
 path=a.output/(name+'.npz');np.savez_compressed(path,**payload)
 summary['partitions'][name]={'n':len(ids),'groups':len(set(groups[ids])),'stages':dict(Counter(b['stages'][ids].tolist())),'low_directions':dict(Counter(b['labels'][ids][b['stages'][ids]=='low'].tolist())),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
# Fixed validation: all held-out frames; no replacement or training membership.
summary['min_contact']=min(mins);summary['validation_used_for_curriculum']=True
(a.output/'manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
