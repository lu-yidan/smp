"""CPU invariants: exact quotas, local terrain heights and six-DOF plate physics."""
import json
from pathlib import Path
import numpy as np
import mujoco
from smp.rl.tasks.getup.multiterrain_geometry import quotas,support_height,add_terrain,free_plate_spec

def main():
 for n in (128,256,4096):
  scene,direction,st=quotas(n)
  for s in range(4):
   assert np.sum(scene==s)==n//4
   for d in range(4):assert np.sum((scene==s)&(direction==d))==n//16
  for k in range(8):assert len(set(np.bincount(direction[st==k],minlength=4)))==1
 s=mujoco.MjSpec();s.worldbody.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1));add_terrain(s.worldbody);m=s.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 rng=np.random.default_rng(771);xy=np.column_stack((rng.uniform(2.5,9.5,2000),rng.uniform(-1.3,1.3,2000)));analytic=support_height(xy);actual=[]
 for x,y in xy:
  gid=np.array([-1],dtype=np.int32);distance=mujoco.mj_ray(m,d,np.array([x,y,2.]),np.array([0.,0.,-1.]),None,1,-1,gid);actual.append(2.-distance)
 err=float(np.max(np.abs(np.asarray(actual)-analytic)));assert err<1e-6,err
 s=free_plate_spec();s.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1));m=s.compile();d=mujoco.MjData(m);assert m.nq==7 and m.nv==6
 d.qpos[:]=[0,0,.4,1,0,0,0];d.qvel[:]=[.3,.2,0,.4,.6,.2]
 start=d.qpos.copy()
 for _ in range(100):mujoco.mj_step(m,d)
 assert np.linalg.norm(d.qpos[:2]-start[:2])>.01 and abs(d.qpos[2]-start[2])>.01 and np.linalg.norm(d.qpos[4:])>.01
 out={'quota_pass':True,'height_ray_max_error':err,'free_plate_nq':m.nq,'free_plate_nv':m.nv,'free_plate_final_qpos':d.qpos.tolist()}
 Path('outputs/p5_multiterrain').mkdir(parents=True,exist_ok=True);Path('outputs/p5_multiterrain/geometry_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
