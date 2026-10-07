"""Height-first, contact-checked crawl resets with settled natural/procedural poses."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco,yaml
from build_table_specialist_bank import model,extents
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/crawl_bank_30cm_v1';OLD=Path(os.environ.get('CRAWL_SOURCE_ROOT','/home/luyd/workspace/smp-a6-paper'));SEED=20261008

def prepare(job):
 split,p,s=job;rng=np.random.default_rng(SEED+1000*(split=='validation')+10*p+s);m=model();m.opt.timestep=.002;m.opt.iterations=100;m.opt.ls_iterations=50;d=mujoco.MjData(m);g=np.array([i for i in range(m.ngeom) if m.geom_group[i]==3]);tor=m.body('torso_link').id
 control=yaml.safe_load((ROOT/'src/smp/assets/deploy_g1/control.yaml').read_text());kp=np.array(control['kps']);kd=np.array(control['kds']);lim=np.array(control['tau_limit']);jids=m.actuator_trnid[:,0];adr=m.jnt_qposadr[jids];dofs=m.jnt_dofadr[jids];names=[m.joint(int(j)).name for j in jids]
 b=np.load(OLD/'datasets/reset_banks/natural_curriculum_v1/train.npz');pool=np.flatnonzero((b['labels']==('supine' if p==0 else 'prone'))&(b['stages']=='low'));sr=np.random.default_rng(SEED+p).permutation(pool);nv=max(1,round(.2*len(sr)));pool=sr[:nv] if split=='validation' else sr[nv:]
 target=(128 if s==0 else 256) if split=='train' else (64 if s==0 else 128);rows=[];attempts=0
 while len(rows)<target and attempts<target*100:
  attempts+=1;mujoco.mj_resetData(m,d)
  if s==0:
   ix=int(rng.choice(pool));q=b['qpos'][ix].astype(float).copy();q[7:]+=rng.normal(0,.015,29)
  else:
   ix=-attempts;q=m.qpos0.copy();q[7:]=0.;angle=(-1 if p==0 else 1)*np.pi/2+rng.uniform(-.08,.08);q[3:7]=[np.cos(angle/2),0,np.sin(angle/2),0]
   for k,name in enumerate(names):
    x=0.
    if 'hip_pitch' in name:x=rng.uniform(-.15,.20)
    elif 'hip_roll' in name:x=rng.uniform(-.10,.10)
    elif 'knee' in name:x=rng.uniform(.05,.45)
    elif 'ankle_pitch' in name:x=rng.uniform(-.18,.08)
    elif 'shoulder_pitch' in name:x=rng.uniform(-.35,.25)
    elif 'shoulder_roll' in name:x=(1 if 'left' in name else -1)*rng.uniform(.15,.8)
    elif 'elbow' in name:x=rng.uniform(.15,1.15)
    elif 'waist' in name:x=rng.uniform(-.05,.05)
    q[adr[k]]=x
  q[adr]=np.clip(q[adr],m.jnt_range[jids,0]+.01,m.jnt_range[jids,1]-.01);q[:2]=0;d.qpos[:]=q;mujoco.mj_forward(m,d);ex=extents(m,d,g);q[2]+=.003-(d.geom_xpos[g,2]-ex[:,2]).min();d.qpos[:]=q;mujoco.mj_forward(m,d)
  if any(c.dist<-.0015 for c in d.contact):continue
  targetq=q[adr].copy()
  for _ in range(300):
   d.ctrl[:]=np.clip(kp*(targetq-d.qpos[adr])-kd*d.qvel[dofs],-lim,lim);mujoco.mj_step(m,d)
  if not np.isfinite(d.qpos).all() or np.linalg.norm(d.qvel[:3])>.20 or np.max(np.abs(d.qvel[dofs]))>2.:continue
  if (1 if p==0 else -1)*d.xmat[tor].reshape(3,3)[2,0]<.5:continue
  q=d.qpos.copy();q[:2]=0;d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d);ex=extents(m,d,g);q[2]+=.003-(d.geom_xpos[g,2]-ex[:,2]).min();d.qpos[:]=q;mujoco.mj_forward(m,d)
  if any(c.dist<-.001 for c in d.contact):continue
  if (d.geom_xpos[g,2]+ex[:,2]).max()>.65:continue
  rows.append((q.astype('float32'),ix))
 assert len(rows)==target,(job,len(rows),attempts)
 path=OUT/f'poses_{split}_{p}_{s}.npz';np.savez_compressed(path,qpos=np.stack([x[0] for x in rows]),source_index=[x[1] for x in rows]);print('POSE_POOL',job,len(rows),attempts,flush=True)
 return dict(split=split,pose=p,source=s,n=len(rows),attempts=attempts,unique_source_indices=len(set(x[1] for x in rows)))

def geometry(job):
 split,f,p,s=job;rng=np.random.default_rng(SEED+5000+1000*(split=='validation')+f*100+p*10+s);pool=np.load(OUT/f'poses_{split}_{p}_{s}.npz');m=model();d=mujoco.MjData(m);g=np.array([i for i in range(m.ngeom) if m.geom_group[i]==3]);coreids=[m.body('pelvis').id,m.body('torso_link').id];records=[];attempts_total=0;target=8 if split=='train' else 4
 for kind in range(3):
  for side in range(4):
   for hb in range(8):
    for rep in range(target):
     # Freeze height AND dimensions first. Never redraw them after rejecting a pose.
     h=.30;th=rng.uniform(.03,.06);leg=rng.uniform(.04,.07);inset=rng.uniform(0,.04)
     if f==0:l=rng.uniform(.85,1.25);w=np.clip(l*rng.uniform(.9,1.1),.8,1.25)
     elif f==1:
      while True:
       l=rng.uniform(1.1,1.7);w=rng.uniform(.55,.85)
       if 1.5<=l/w<=2.8:break
     else:l=rng.uniform(1.,1.4);w=rng.uniform(.35,.5)
     axis=side//2;sgn=1 if side%2==0 else -1;half=np.array([l,w])/2
     for attempt in range(30000):
      attempts_total+=1;ix=int(rng.integers(len(pool['qpos'])));q=pool['qpos'][ix].astype(float).copy();yaw=rng.uniform(-np.pi,np.pi);mujoco.mju_mulQuat(q[3:7],np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]),q[3:7].copy());d.qpos[:]=q;mujoco.mj_forward(m,d);pos=d.geom_xpos[g];ex=extents(m,d,g);core=d.xpos[coreids];center=core[:,:2].mean(0).copy();center[1-axis]+=rng.uniform(-.15,.15)*2*half[1-axis]
      if kind==0:depth=rng.uniform(.35,.8)*half[axis];center[axis]-=sgn*(half[axis]-depth)
      elif kind==1:depth=rng.uniform(.01,.06);center[axis]-=sgn*(half[axis]-depth)
      else:
       extent=(ex[:,axis]+sgn*(center[axis]-pos[:,axis])).max();center[axis]-=sgn*(half[axis]+extent+rng.uniform(.08,.18))
      pp=np.array([[*center,h+th/2]]+[[center[0]+sx*(l/2-leg/2-inset),center[1]+sy*(w/2-leg/2-inset),h/2] for sx in [-1,1] for sy in [-1,1]]);sz=np.array([[l/2,w/2,th/2]]+[[leg/2,leg/2,h/2]]*4)
      gap=(np.abs(pos[:,None,:]-pp[None,:,:])-ex[:,None,:]-sz[None,:,:]).max(-1).min()
      coverage=((np.abs(core[:,:2]-center)<half-.005).all(-1)&(core[:,2]<h)).any()
      planar=(np.abs(pos[:,:2]-center)-ex[:,:2]-half).max(-1).min()
      if gap<.003 or (kind<2 and not coverage) or (kind==2 and planar<.04):continue
      if any(c.dist<-.001 for c in d.contact):continue
      records.append(dict(qpos=q.astype('float32'),positions=pp.astype('float32'),sizes=sz.astype('float32'),dims=np.array([l,w,h,th,leg,inset],dtype='float32'),friction=np.float32(rng.uniform(.8,1.4)),family=f,pose=p,source=s,band=int(h<.36),height_bin=hb,kind=kind,side=side,source_index=int(pool['source_index'][ix]),pose_index=ix,min_gap=float(gap)));break
     else:raise RuntimeError(('cannot fit frozen geometry',job,kind,side,hb,h,l,w))
 print('GEOMETRY_POOL',job,len(records),attempts_total,flush=True);return records
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 import shutil
 for source in (ROOT/'outputs/crawl_bank_v1').glob('poses_validation_*.npz'):shutil.copy2(source,OUT/source.name)
 audits=[]
 with ProcessPoolExecutor(max_workers=6) as ex:
  groups=list(ex.map(geometry,[('validation',f,p,s) for f in range(3) for p in range(2) for s in range(2)]))
 rr=sum(groups,[]);np.savez_compressed(OUT/'validation.npz',**{k:np.stack([r[k] for r in rr]) for k in rr[0]})
 manifest=dict(seed=SEED,pose_pools=audits,height='fixed underside0.30m; new collision-screened placements; no height rejection',settle='0.6s nominal PD, then grounded 3mm; root speed<0.2m/s, joint speed<2rad/s; reset velocities zero',source_split='natural source indices disjoint; procedural seeds disjoint; not a clip-disjoint claim',sha={sp:hashlib.sha256((OUT/f'{sp}.npz').read_bytes()).hexdigest() for sp in ['validation']})
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('CRAWL_BANK_COMPLETE',flush=True)
