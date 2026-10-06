"""Deterministic, conservative collision-free reset bank for randomized tables."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco,json,hashlib,xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];OLD=Path('/home/luyd/workspace/smp-a6-paper');OUT=ROOT/'outputs/table_specialist_bank_v1'
SEED=20261007

def model():
 root=ET.parse(ROOT/'src/smp/assets/deploy_g1/source.xml').getroot();root.find('compiler').set('meshdir',str(ROOT/'src/smp/assets/deploy_g1/meshes'))
 for w in root.findall('worldbody'):
  for b in list(w.findall('body')):
   if b.get('name')!='pelvis':w.remove(b)
 return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
def extents(m,d,gids):
 r=d.geom_xmat[gids].reshape(-1,3,3);s=m.geom_size[gids];ex=np.abs(r)@s[...,None];ex=ex[...,0]
 for i,g in enumerate(gids):
  if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE:ex[i]=s[i,0]
  elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_CAPSULE:ex[i]=s[i,0]+np.abs(r[i,:,2])*s[i,1]
 return ex

def worker(job):
 split,family,pose,source,band,target=job;jobid=family*8+pose*4+source*2+band+(100 if split=='validation' else 0);rng=np.random.default_rng(SEED+jobid)
 name='natural_curriculum_v1' if source==0 else 'procedural_low_v1';b=np.load(OLD/f'datasets/reset_banks/{name}/train.npz');pool=np.flatnonzero((b['labels']==('supine' if pose==0 else 'prone'))&(b['stages']=='low'))
 splitrng=np.random.default_rng(SEED+source*7+pose);pool=splitrng.permutation(pool);nval=max(1,round(.2*len(pool)));pool=pool[:nval] if split=='validation' else pool[nval:]
 m=model();d=mujoco.MjData(m);gids=np.array([i for i in range(m.ngeom) if m.geom_group[i]==3]);torso=m.body('torso_link').id;pelvis=m.body('pelvis').id
 out=[];attempts=0
 while len(out)<target and attempts<200000:
  attempts+=1;q=b['qpos'][rng.choice(pool)].astype(float).copy();bank_index=int(np.where((b['qpos']==q).all(1))[0][0]) if False else -1
  # Explicit source index instead of ambiguous float matching.
  ix=int(rng.choice(pool));q=b['qpos'][ix].astype(float).copy();q[:2]=0;yaw=rng.uniform(-np.pi,np.pi);rot=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]);oldq=q[3:7].copy();mujoco.mju_mulQuat(q[3:7],rot,oldq)
  d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d);ex=extents(m,d,gids);q[2]+=.003-np.min(d.geom_xpos[gids,2]-ex[:,2]);d.qpos[:]=q;mujoco.mj_forward(m,d)
  chest=d.xmat[torso].reshape(3,3)[2,0]
  if (pose==0 and chest<.35) or (pose==1 and chest>-.35):continue
  if any(c.dist<-.001 for c in d.contact):continue
  if any(not (m.jnt_range[j,0]-.001<=q[m.jnt_qposadr[j]]<=m.jnt_range[j,1]+.001) for j in np.flatnonzero(m.jnt_limited)):continue
  if family==0:
   l=rng.uniform(.85,1.25);w=np.clip(l*rng.uniform(.9,1.1),.8,1.25)
  elif family==1:
   l=rng.uniform(1.1,1.7);w=rng.uniform(.55,.85)
   if not 1.5<=l/w<=2.8:continue
  else:l=rng.uniform(1.,1.4);w=rng.uniform(.35,.50)
  h=rng.uniform(.35,.4) if band==0 else rng.uniform(.30,.35);th=rng.uniform(.03,.06);leg=rng.uniform(.04,.07);inset=rng.uniform(0,.04)
  core=d.xpos[[pelvis,torso]];anchor=core[rng.integers(2),:2];center=anchor+rng.uniform(-.22,.22,2)*[l,w]
  if not np.any((np.abs(core[:,:2]-center)<np.array([l,w])/2-.015).all(1)&(core[:,2]<h)):continue
  positions=np.array([[*center,h+th/2]]+[[center[0]+sx*(l/2-leg/2-inset),center[1]+sy*(w/2-leg/2-inset),h/2] for sx in [-1,1] for sy in [-1,1]])
  sizes=np.array([[l/2,w/2,th/2]]+[[leg/2,leg/2,h/2]]*4)
  sep=np.max(np.abs(d.geom_xpos[gids,None,:]-positions[None,:,:])-ex[:,None,:]-sizes[None,:,:],axis=-1)
  if np.min(sep)<.003:continue
  out.append(dict(qpos=q.astype(np.float32),positions=positions.astype(np.float32),sizes=sizes.astype(np.float32),dims=np.array([l,w,h,th,leg,inset],np.float32),friction=np.float32(rng.uniform(.8,1.4)),family=family,pose=pose,source=source,band=band,bank_index=ix,min_aabb_gap=np.float32(sep.min())))
 if len(out)!=target:raise RuntimeError((job,'insufficient valid candidates',len(out),attempts))
 print('BANK_GROUP',job,'attempts',attempts,flush=True)
 return out,dict(split=split,family=family,pose=pose,source=source,band=band,accepted=len(out),attempts=attempts,unique_poses=len({x['bank_index'] for x in out}))
if __name__=='__main__':
 OUT.mkdir(exist_ok=True,parents=True);jobs=[(sp,f,p,s,b,256 if sp=='train' else 64) for sp in ['train','validation'] for f in range(3) for p in range(2) for s in range(2) for b in range(2)]
 with ProcessPoolExecutor(max_workers=6) as ex:results=list(ex.map(worker,jobs))
 for sp in ['train','validation']:
  records=[r for (rr,meta) in results if meta['split']==sp for r in rr];np.savez_compressed(OUT/f'{sp}.npz',**{k:np.stack([r[k] for r in records]) for k in records[0]})
 (OUT/'manifest.json').write_text(json.dumps(dict(seed=SEED,groups=[m for _,m in results],bank_sha256={sp:hashlib.sha256((OUT/f'{sp}.npz').read_bytes()).hexdigest() for sp in ['train','validation']},pose_splits_disjoint=True,selection='conservative robot collision AABB vs five table boxes, >=3mm; grounded; self/joint/pose screened'),indent=2)+'\n')
 print('BANK_COMPLETE',flush=True)
