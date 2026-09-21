"""Attach conservative robot collision-top heights to an immutable reset bank."""
import numpy as np,mujoco,json,hashlib
from pathlib import Path
from smp.rl.tasks.getup.master_deployment_contract import ASSETS,COLLISION_NAMES
m=mujoco.MjModel.from_xml_path(str(ASSETS/'source.xml'));d=mujoco.MjData(m)
gids=np.array([m.geom(n).id for n in COLLISION_NAMES]);size=m.geom_size[gids];typ=m.geom_type[gids]
out=Path('outputs/ceiling_bank');out.mkdir(exist_ok=True)
report={}
for split in ['train','validation']:
 f=Path('outputs/multiterrain_bank')/(split+'.npz');b=dict(np.load(f));top=np.zeros(len(b['qpos']),np.float32)
 for i,q in enumerate(b['qpos']):
  mujoco.mj_resetData(m,d);d.qpos[:36]=q;mujoco.mj_forward(m,d)
  rot=d.geom_xmat[gids].reshape(-1,3,3);ext=np.einsum('gij,gj->gi',abs(rot),size)
  ext=np.where((typ==mujoco.mjtGeom.mjGEOM_SPHERE)[:,None],size[:,0,None],ext)
  ext=np.where((typ==mujoco.mjtGeom.mjGEOM_CAPSULE)[:,None],size[:,0,None]+size[:,1,None]*abs(rot[:,:,2]),ext)
  top[i]=(d.geom_xpos[gids,2]+ext[:,2]).max()
 b['robot_top']=top;np.savez_compressed(out/(split+'.npz'),**b)
 stats={}
 for dr in range(4):
  for source in range(2):
   mask=(b['stratum']==0)&(b['direction']==dr)&(b['source']==source)
   count=int((mask&(top<.49)).sum());assert count>0,(split,dr,source)
   stats[f'{dr}/{source}']={'total':int(mask.sum()),'fit_0.50m':count,'top_percentiles':np.percentile(top[mask],[0,50,100]).tolist()}
 report[split]={'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'pools':stats}
(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
