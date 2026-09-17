"""Generate independently screened low poses per terrain, source and direction."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from smp.rl.tasks.getup.master_deployment_contract import deployment_robot_spec,COLLISION_NAMES,CONTROL
from smp.rl.tasks.getup.multiterrain_geometry import add_terrain,DIRECTIONS,STRATA,support_height

def main():
 p=argparse.ArgumentParser();p.add_argument('--per-cell',type=int,default=128);p.add_argument('--stratum',type=int);p.add_argument('--out',type=Path,default=Path('outputs/multiterrain_bank'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 s=deployment_robot_spec();s.worldbody.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1));add_terrain(s.worldbody);m=s.compile();m.opt.timestep=.002;d=mujoco.MjData(m)
 robot=set(m.geom(n).id for n in COLLISION_NAMES);ground=set(range(m.ngeom))-robot
 head=m.site('head').id;pelvis=m.body('pelvis').id
 kp=np.asarray(CONTROL['kps']);kd=np.asarray(CONTROL['kds']);lim=np.asarray(CONTROL['tau_limit'])
 if lim.ndim==0:lim=np.full(29,lim)
 reports={}
 for split,seed in [('train',202609171),('validation',202609172)]:
  rng=np.random.default_rng(seed);banks=[np.load(f'datasets/reset_banks/{name}/{split}.npz') for name in ('natural_curriculum_v1','procedural_low_v1')]
  rows=[];labels=[];strata=[];sources=[];depths=[];attempts={};n=a.per_cell if split=='train' else max(8,a.per_cell//4)
  for st in ([a.stratum] if a.stratum is not None else range(8)):
   for direction,label in enumerate(DIRECTIONS):
    for source in range(2):
     bank=banks[source];pool=np.flatnonzero((bank['labels']==label)&(bank['stages']=='low'));accepted=0;tried=0
     while accepted<n:
      tried+=1
      if tried>n*400:raise RuntimeError((split,st,label,source,accepted,tried))
      q=bank['qpos'][rng.choice(pool)].copy();q[:2]=0
      yaw=rng.uniform(-np.pi,np.pi);r=Rotation.from_euler('z',yaw)*Rotation.from_quat(q[3:7][[1,2,3,0]]);q[3:7]=r.as_quat()[[3,0,1,2]]
      if st<3:xy=rng.uniform(-.04,.04,2)
      elif st==3:xy=np.array([rng.uniform(4.65,5.12),rng.uniform(-.3,.3)])
      elif st==4:xy=np.array([rng.choice([3.6,4.4])+rng.uniform(-.12,.12),rng.uniform(-.35,.35)])
      elif st==5:xy=np.array([rng.choice([3.6,4.4])+rng.uniform(-.08,.08),rng.uniform(-.3,.3)])
      elif st==6:xy=np.array([rng.uniform(7.65,8.35),rng.uniform(-.25,.25)])
      else:xy=np.array([rng.uniform(7.6,8.4),rng.choice([-1,1])*(.8+rng.uniform(-.10,.03))])
      q[:2]=xy;d.qvel[:]=0;d.qpos[:]=q
      # Binary vertical placement uses actual MuJoCo primitive contacts.
      lo=-.3;hi=1.4
      for _ in range(18):
       d.qpos[2]=(lo+hi)/2;mujoco.mj_forward(m,d)
       coll=any(c.dist<0 and ((int(c.geom[0]) in robot) != (int(c.geom[1]) in robot)) for c in d.contact)
       if coll:lo=d.qpos[2]
       else:hi=d.qpos[2]
      d.qpos[2]=hi+.003;d.qvel[:]=0;mujoco.mj_forward(m,d)
      if any(c.dist<-.001 for c in d.contact):continue
      target=d.qpos[7:].copy()
      for _ in range(100):
       d.qfrc_applied[:]=0;d.qfrc_applied[6:]=np.clip(kp*(target-d.qpos[7:])-kd*d.qvel[6:],-lim,lim);mujoco.mj_step(m,d)
      mujoco.mj_forward(m,d)
      grav=-d.xmat[pelvis].reshape(3,3)[2];dr=(1 if grav[0]>0 else 0) if abs(grav[0])>=abs(grav[1]) else (2 if grav[1]>0 else 3)
      depth=min([float(c.dist) for c in d.contact]+[0.]);z=d.site_xpos[head,2]-support_height(d.qpos[:2])
      self_depth=min([float(c.dist) for c in d.contact if int(c.geom[0]) in robot and int(c.geom[1]) in robot]+[0.])
      if dr!=direction or self_depth<-.001 or depth<-.02 or not .08<z<.65 or np.max(abs(d.qvel))>15:continue
      contacts=[c for c in d.contact if c.dist<.003 and ((int(c.geom[0]) in robot)!=(int(c.geom[1]) in robot))]
      if not contacts:continue
      if st==5 and np.ptp([c.pos[2] for c in contacts])<.045:continue
      if st in (4,7):
       xyz=d.geom_xpos[list(robot)]
       if st==4:
        edge=3.6 if abs(xy[0]-3.6)<abs(xy[0]-4.4) else 4.4
        if not xyz[:,0].min()<edge<xyz[:,0].max():continue
       elif not xyz[:,1].min()<np.sign(xy[1])*.8<xyz[:,1].max():continue
      final=d.qpos.copy();final[2]+=.003-min(depth,0.) # de-compress ground contact; self collisions were separately rejected
      d.qpos[:]=final;d.qvel[:]=0;mujoco.mj_forward(m,d)
      if any(c.dist<-.001 for c in d.contact):continue
      rows.append(final);labels.append(direction);strata.append(st);sources.append(source);depths.append(depth);accepted+=1
     attempts[f'{st}/{direction}/{source}']=tried
    print(split,STRATA[st],label,'done',flush=True)
  np.savez_compressed(a.out/f'{split}.npz',qpos=np.array(rows),direction=labels,stratum=strata,source=sources)
  reports[split]={'n':len(rows),'attempts':attempts,'sha256':hashlib.sha256((a.out/f'{split}.npz').read_bytes()).hexdigest(),'max_settled_penetration':-min(depths)}
  (a.out/'manifest.json').write_text(json.dumps(reports,indent=2))
if __name__=='__main__':main()
