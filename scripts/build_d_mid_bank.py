"""Screen existing training-only LAFAN frames for supported intermediate resets."""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from smp.rl.tasks.getup.master_deployment_contract import deployment_robot_spec,COLLISION_NAMES,CONTROL
src=Path('datasets/reset_banks/natural_curriculum_v1/train.npz');b=np.load(src)
s=deployment_robot_spec();s.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1));m=s.compile();d=mujoco.MjData(m)
head=m.site('head').id;pelvis=m.body('pelvis').id;ground=m.geom('floor').id
qs=[];types=[];indices=[];mins=[]
m.opt.timestep=.002;kp=np.asarray(CONTROL['kps']);kd=np.asarray(CONTROL['kds']);lim=np.asarray(CONTROL['tau_limit'])
for i,q0 in enumerate(b['qpos']):
 q=q0.copy();q[:2]=0;d.qpos[:]=q
 lo=q[2]-.20;hi=q[2]+.20
 for _ in range(16):
  d.qpos[2]=(lo+hi)/2;mujoco.mj_forward(m,d)
  if any(ground in c.geom and c.dist<0 for c in d.contact):lo=d.qpos[2]
  else:hi=d.qpos[2]
 q[2]=lo-.0002;d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
 if not .45<d.site_xpos[head,2]<1.12 or d.xmat[pelvis].reshape(3,3)[2,2]<.4:continue
 target=q[7:].copy()
 for _ in range(150):
  d.qfrc_applied[:]=0;d.qfrc_applied[6:]=np.clip(kp*(target-d.qpos[7:])-kd*d.qvel[6:],-lim,lim);mujoco.mj_step(m,d)
 mujoco.mj_forward(m,d);q=d.qpos.copy()
 if abs(d.qvel).max()>5:continue
 depth=min([float(c.dist) for c in d.contact]+[0.]);z=d.site_xpos[head,2];u=d.xmat[pelvis].reshape(3,3)[2,2]
 contacts=[c for c in d.contact if ground in c.geom and c.dist<.005]
 names=[m.geom(int(c.geom[0] if c.geom[1]==ground else c.geom[1])).name for c in contacts]
 foot=sum(any(side in name and any(t in name for t in ['foot','ankle']) for name in names) for side in ['left','right'])
 knees=q[7:][[3,9]]
 typ=0 if .48<=z<.78 and u>.45 and len(contacts)>0 else 1 if .78<=z<1.05 and u>.70 and foot>=2 and knees.min()>.5 else -1
 selfdepth=min([float(c.dist) for c in d.contact if ground not in c.geom]+[0.])
 if typ<0 or depth<-.02 or selfdepth<-.001:continue
 excess=np.maximum(m.jnt_range[1:,0]-q[7:],q[7:]-m.jnt_range[1:,1]).max()
 if excess>.001:continue
 q[7:]=np.clip(q[7:],m.jnt_range[1:,0],m.jnt_range[1:,1])
 q[2]+=.001-min(depth,0.);d.qpos[:]=q;mujoco.mj_forward(m,d)
 if min([float(c.dist) for c in d.contact]+[0.])<-.001:continue
 qs.append(q);types.append(typ);indices.append(i);mins.append(depth)
assert all(types.count(t)>=16 for t in (0,1)),np.bincount(types)
out=Path('datasets/d_mid');out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'train.npz',qpos=qs,kind=types,source_indices=indices,clips=b['clips'][indices])
report={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'counts':np.bincount(types).tolist(),'min_original_contact':min(mins),'source':'training partition only; centered XY; PD settle .3s; decompress ground penetration+1mm','joint_limits':'reject settled excess >0.001 rad, clip residual <=0.001 rad then recheck contacts','selection':'transition head .48-.78,u>.45,ground contact; crouch head .78-1.05,u>.70,both feet in contact,knees>.5'}
(out/'audit.json').write_text(json.dumps(report,indent=2));print(report)
