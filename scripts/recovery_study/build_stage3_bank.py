"""Build collision-screened training/validation banks for all mixed scene families."""
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from smp.recovery.stage3_geometry import SCENES,GEOMS,BODIES,OBJECT_GEOMS,draw,apply_cpu,cpu_model
from smp.rl.tasks.getup.master_deployment_contract import COLLISION_NAMES,CONTROL,JOINT_NAMES
DIRECTIONS=('supine','prone','left_side_down','right_side_down')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--per-cell',type=int,default=24);ap.add_argument('--split',choices=['train','validation'],required=True);a=ap.parse_args()
 a.out.mkdir(parents=True,exist_ok=True);path=a.out/(a.split+'.npz')
 if path.exists():raise FileExistsError(path)
 rng=np.random.default_rng(202609271 if a.split=='train' else 202609272)
 banks=[np.load(f'datasets/reset_banks/{b}/{a.split}.npz') for b in ('natural_curriculum_v1','procedural_low_v1')]
 m=cpu_model();d=mujoco.MjData(m);rids={m.geom(n).id for n in COLLISION_NAMES};support={m.geom('terrain').id}
 kp=np.array(CONTROL['kps']);kd=np.array(CONTROL['kds']);lim=np.array(CONTROL['tau_limit'])
 records=[];attempts={}
 for scene_id,scene in enumerate(SCENES):
  groups=[(0,dr,source) for dr in range(4) for source in range(2)]
  if scene=='flat':groups.extend([(1,-1,0),(2,-1,0)])
  for level in range(3):
   for kind,direction,source in groups:
    b=banks[source];stage=('low','middle','late')[kind]
    mask=b['stages']==stage
    if direction>=0:mask&=b['labels']==DIRECTIONS[direction]
    pool=np.flatnonzero(mask);assert len(pool),(scene,stage,direction,source)
    if source==0:
     clips=np.array([str(c).replace('__mirror','') for c in b['clips'][pool]])
     weights=np.array([1/np.count_nonzero(clips==c) for c in clips]);weights/=weights.sum()
    else:weights=None
    n=a.per_cell if a.split=='train' else max(4,a.per_cell//3);accepted=0;tries=0;reject=Counter()
    while accepted<n:
     tries+=1
     if tries>n*100:raise RuntimeError((scene,level,kind,direction,source,accepted,tries,dict(reject)))
     ix=int(rng.choice(pool,p=weights));q=b['qpos'][ix].copy();q[:2]=0
     params=draw(scene,rng,(level+1)/3);q[:2]=params['xy']
     quat=Rotation.from_euler('z',rng.uniform(-np.pi,np.pi))*Rotation.from_quat(q[3:7][[1,2,3,0]])
     q[3:7]=quat.as_quat()[[3,0,1,2]]
     mujoco.mj_resetData(m,d);d.qpos[:36]=q;apply_cpu(m,d,params)
     lo=-.5;hi=2.
     for _ in range(22):
      d.qpos[2]=(lo+hi)/2;mujoco.mj_forward(m,d)
      coll=any(c.dist<0 and ((int(c.geom1) in rids and int(c.geom2) in support) or (int(c.geom2) in rids and int(c.geom1) in support)) for c in d.contact)
      if coll:lo=d.qpos[2]
      else:hi=d.qpos[2]
     d.qpos[2]=hi+.002;mujoco.mj_forward(m,d)
     if any(c.dist<-.002 for c in d.contact):reject['fit_collision']+=1;continue
     # Brief support-only settling. Overhead dynamic objects are still parked.
     target=d.qpos[7:36].copy()
     for _ in range(50):
      d.qfrc_applied[:]=0;d.qfrc_applied[6:35]=np.clip(kp*(target-d.qpos[7:36])-kd*d.qvel[6:35],-lim,lim);mujoco.mj_step(m,d)
     mujoco.mj_forward(m,d)
     robot_depth=min([float(c.dist) for c in d.contact if int(c.geom1) in rids or int(c.geom2) in rids]+[0.])
     if robot_depth<-.02 or np.max(np.abs(d.qvel[:35]))>15:reject['settle_unstable']+=1;continue
     settled_support=any(c.dist<.005 and ((int(c.geom1) in rids and int(c.geom2) in support) or (int(c.geom2) in rids and int(c.geom1) in support)) for c in d.contact)
     # Remove compressed support contact; reject self-collision after decompression.
     d.qpos[2]+=max(0.,-robot_depth)+.001;d.qvel[:]=0;d.qfrc_applied[:]=0;mujoco.mj_forward(m,d)
     if direction>=0:
      grav=-d.xmat[m.body('pelvis').id].reshape(3,3)[2]
      dr=(1 if grav[0]>0 else 0) if abs(grav[0])>=abs(grav[1]) else (2 if grav[1]>0 else 3)
      if dr!=direction:reject['direction']+=1;continue
     xyz=d.geom_xpos[list(rids)];mat=d.geom_xmat[list(rids)].reshape(-1,3,3)
     half=np.einsum('nij,nj->ni',np.abs(mat),m.geom_aabb[list(rids),3:]);cent=xyz+np.einsum('nij,nj->ni',mat,m.geom_aabb[list(rids),:3])
     top=float((cent[:,2]+half[:,2]).max());rootxy=d.qpos[:2].copy()
     for slot in ([7,1,2] if scene=='ladder_boards' else range(len(BODIES))):
      if not params['active'][slot]:continue
      gi=OBJECT_GEOMS[slot][0];angle=rng.uniform(-np.pi,np.pi);offset=rng.uniform(-.2,.2,2)
      z=top+params['size'][gi,2]+.003
      params['poses'][slot]=[*list(rootxy+offset),z,np.cos(angle/2),0,0,np.sin(angle/2)]
      if slot==0:params['poses'][slot,3:]=[1,0,0,0]
      top=z+params['size'][gi,2]
     q=d.qpos[:36].copy();apply_cpu(m,d,params);d.qpos[:36]=q;mujoco.mj_forward(m,d)
     depth=min([float(c.dist) for c in d.contact]+[0.])
     if depth<-.002:reject['final_collision']+=1;continue
     active_support=[c for c in d.contact if c.dist<.008 and ((int(c.geom1) in rids and int(c.geom2) in support) or (int(c.geom2) in rids and int(c.geom1) in support))]
     if not settled_support:reject['no_support']+=1;continue
     records.append({**params,'qpos':q,'scene':scene_id,'kind':kind,'direction':direction,'source':source,'source_index':ix,'level':level,'depth':depth})
     accepted+=1
    attempts[f'{scene}/{level}/{kind}/{direction}/{source}']=tries
  print(scene,len(records),flush=True)
 keys=list(records[0]);np.savez_compressed(path,**{k:np.array([r[k] for r in records]) for k in keys})
 manifest={'split':a.split,'records':len(records),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'geometry_names':GEOMS,'joint_order':JOINT_NAMES,'source_sha256':{name:hashlib.sha256(Path(f'datasets/reset_banks/{name}/{a.split}.npz').read_bytes()).hexdigest() for name in ('natural_curriculum_v1','procedural_low_v1')},'attempts':attempts,'reset':'100ms support settling, zero final velocities, screened at -2mm; movable overhead objects placed after settling','not_final_test':True,'scene_names':SCENES,'independent_scene_rng_seed':202609271 if a.split=='train' else 202609272,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'geometry_sha256':hashlib.sha256(Path(__import__('smp.recovery.stage3_geometry',fromlist=['SCENES']).__file__).read_bytes()).hexdigest()}
 path.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print('BANK_COMPLETE',manifest['records'],manifest['sha256'],flush=True)
if __name__=='__main__':main()
