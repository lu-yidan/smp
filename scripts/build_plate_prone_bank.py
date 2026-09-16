"""Deterministic contact-screened prone bank for the CURRENT deployment model.
Legacy shoulder-roll signs and root pitch cannot be copied into this model.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from smp.rl.tasks.getup.master_deployment_contract import deployment_robot_spec,JOINT_NAMES,COLLISION_NAMES,CONTROL

out=Path('datasets/reset_banks/plate_prone_v1');out.mkdir(parents=True,exist_ok=True)
spec=deployment_robot_spec();spec.worldbody.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1));m=spec.compile();d=mujoco.MjData(m)
ids=np.array([m.geom(n).id for n in COLLISION_NAMES]);hands=np.array([m.geom(s+'_hand_collision').id for s in ['left','right']]);head=m.site('head').id
base=np.array(CONTROL['default_joint_pos']);rng=np.random.default_rng(9132761);poses=[];heights=[];rejected=0
while len(poses)<2304:
 q=base+rng.uniform(-.06,.06,29)
 for start,sign in [(15,1),(22,-1)]:q[start:start+7]=np.array([-2.10,sign*.60,0,.744,0,0,0])+rng.uniform(-.035,.035,7)
 d.qpos[7:]=q;d.qpos[:3]=[0,0,.8];quat=Rotation.from_euler('xyz',[rng.uniform(-.025,.025),np.pi/2+rng.uniform(-.025,.025),rng.uniform(-np.pi,np.pi)]).as_quat();d.qpos[3:7]=quat[[3,0,1,2]];mujoco.mj_forward(m,d)
 mats=d.geom_xmat[ids].reshape(-1,3,3);size=m.geom_size[ids];typ=m.geom_type[ids]
 ext=(abs(mats[:,2,:])*size).sum(1)
 ext=np.where(typ==mujoco.mjtGeom.mjGEOM_SPHERE,size[:,0],ext)
 ext=np.where(typ==mujoco.mjtGeom.mjGEOM_CAPSULE,size[:,0]+size[:,1]*abs(mats[:,2,2]),ext)
 low=(d.geom_xpos[ids,2]-ext).min();d.qpos[2]+=.004-low;mujoco.mj_forward(m,d)
 limitbad=any(m.jnt_limited[j] and not(m.jnt_range[j,0]<=d.qpos[m.jnt_qposadr[j]]<=m.jnt_range[j,1]) for j in range(m.njnt))
 depth=min([c.dist for c in d.contact]+[0]);handbottom=d.geom_xpos[hands,2]-m.geom_size[hands,0]
 if depth<-.001 or limitbad or handbottom.min()>.04 or d.site_xpos[head,2]>.40:
  rejected+=1
  if rejected>20000:raise RuntimeError('No feasible contact-ready bank')
  continue
 poses.append(d.qpos.copy());heights.append([d.site_xpos[head,2],*handbottom])
for split,data in [('train',poses[:2048]),('validation',poses[2048:])]:np.savez_compressed(out/(split+'.npz'),qpos=np.asarray(data,dtype=np.float32))
report={'seed':9132761,'n_train':2048,'n_validation':256,'rejected':rejected,'gravity_x':'positive ~1 = prone, matching current four-direction bank','self_or_ground_penetration_limit_m':.001,'ground_gap_m':.004,'head_and_hand_bottom_ranges':np.stack([np.min(heights,axis=0),np.max(heights,axis=0)]).tolist(),'joint_order':JOINT_NAMES,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')}}
(out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
