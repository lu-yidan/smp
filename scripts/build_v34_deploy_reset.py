"""Project archived V34 procedural poses onto deployment collision constraints."""
import sys
from pathlib import Path
import argparse,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
a=argparse.ArgumentParser();a.add_argument("--input",required=True);a.add_argument("--out",required=True);args=a.parse_args();dest=Path(args.out);dest.mkdir(parents=True,exist_ok=False)
from build_table_specialist_bank import model,extents
import numpy as np,mujoco
from scipy.optimize import least_squares
m=model();d=mujoco.MjData(m);gids=np.array([i for i in range(m.ngeom) if m.geom_group[i]==3]);b=np.load(args.input);out=[]
for index in np.flatnonzero(b["scene"]==2):
 pose=int(b["pose"][index])
 q=b['qpos'][index][:m.nq].copy().astype(float);q[:2]=0
 ref=q[7:].copy();lo=m.jnt_range[1:,0]+.005;hi=m.jnt_range[1:,1]-.005
 def put(x):
  d.qpos[:]=q;d.qpos[7:]=x;d.qpos[2]=1;mujoco.mj_forward(m,d);ex=extents(m,d,gids);d.qpos[2]+=.004-np.min(d.geom_xpos[gids,2]-ex[:,2]);mujoco.mj_forward(m,d)
 def fun(x):
  put(x);neg=sorted([min(0,c.dist+.0005) for c in d.contact]);v=np.zeros(100);v[:len(neg)]=neg;return np.r_[(x-ref)*.01,v*100]
 res=least_squares(fun,np.clip(ref,lo,hi),bounds=(lo,hi),diff_step=1e-4,max_nfev=300)
 put(res.x);print(pose,'min',min([c.dist for c in d.contact]+[0]),'max_delta',max(abs(res.x-ref)),flush=True);out.append((pose,d.qpos.copy()))

for split,n,seed in [('train',1024,20261008),('validation',256,20271008)]:
 rng=np.random.default_rng(seed);records=[];labels=[];attempts=0
 for pose in (0,1):
  templates=[q for label,q in out if label==pose];accepted=0
  while accepted<n:
   attempts+=1
   if attempts>200000:raise RuntimeError('Not enough collision-valid reset poses')
   q=templates[rng.integers(len(templates))].copy();ref=q[7:].copy();x=np.clip(ref+rng.uniform(-.035,.035,len(ref)),lo,hi);yaw=rng.uniform(-np.pi,np.pi);rot=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]);old=q[3:7].copy();mujoco.mju_mulQuat(q[3:7],rot,old);put(x)
   if any(c.dist<-.001 for c in d.contact):continue
   records.append(d.qpos.copy());labels.append(pose);accepted+=1
 np.savez_compressed(dest/(split+'.npz'),qpos=np.asarray(records,dtype=np.float32),pose=np.array(labels))
 print(split,len(records),attempts,flush=True)
(dest/'manifest.json').write_text(json.dumps(dict(source_sha=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),adaptation='closest joint pose projection plus .035rad perturbations; deployment self-contact >=-1mm; grounded .004m; V34-derived, not exact historical reset'),indent=2))
