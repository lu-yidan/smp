"""Held-out, screened poses under six fixed-size purchase candidates."""
from pathlib import Path
import json,numpy as np,mujoco
from build_table_specialist_bank import model,extents
out=Path('outputs/table_purchase_20261007');out.mkdir(parents=True,exist_ok=True)
b=np.load('outputs/table_specialist_bank_v1/validation.npz');m=model();d=mujoco.MjData(m);gids=np.array([i for i in range(m.ngeom) if m.geom_group[i]==3]);pel=m.body('pelvis').id;tor=m.body('torso_link').id
poses=[]
for q in b['qpos']:
 d.qpos[:]=q;mujoco.mj_forward(m,d);poses.append((d.geom_xpos[gids].copy(),extents(m,d,gids),d.xpos[[pel,tor]].copy()))
rng=np.random.default_rng(20261007);rows=[];audit=[]
for f,(l,w) in enumerate([(1.,1.),(1.4,.7),(1.2,.38)]):
 for p in range(2):
  for s in range(2):
   for band in range(2):
    h=.40 if band==0 else (.37 if f==2 else .35);th=.04;leg=.05;inset=.02;pool=np.flatnonzero((b['pose']==p)&(b['source']==s));accepted=0;attempts=0
    while accepted<128 and attempts<100000:
     attempts+=1;i=int(rng.choice(pool));pos,ex,core=poses[i];center=core[rng.integers(2),:2]+rng.uniform(-.22,.22,2)*[l,w]
     if not np.any((np.abs(core[:,:2]-center)<np.array([l,w])/2-.015).all(1)&(core[:,2]<h)):continue
     pp=np.array([[*center,h+th/2]]+[[center[0]+sx*(l/2-leg/2-inset),center[1]+sy*(w/2-leg/2-inset),h/2] for sx in [-1,1] for sy in [-1,1]])
     sz=np.array([[l/2,w/2,th/2]]+[[leg/2,leg/2,h/2]]*4)
     gap=(np.abs(pos[:,None,:]-pp[None,:,:])-ex[:,None,:]-sz[None,:,:]).max(-1).min()
     if gap<.003:continue
     rows.append(dict(qpos=b['qpos'][i],positions=pp.astype('float32'),sizes=sz.astype('float32'),dims=np.array([l,w,h,th,leg,inset],dtype='float32'),friction=np.float32(1.1),family=f,pose=p,source=s,band=band,original_row=i,min_aabb_gap=gap));accepted+=1
    assert accepted==128,(f,p,s,band,accepted)
    audit.append(dict(family=f,pose=p,source=s,band=band,L=l,W=w,H=h,attempts=attempts,accepted=accepted))
np.savez_compressed(out/'bank.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
(out/'bank_manifest.json').write_text(json.dumps(audit,indent=2));print('PURCHASE_BANK_OK',len(rows))
