"""Physical fixture checks, not recovery-policy success tests."""
import json
from pathlib import Path
import mujoco
import numpy as np
from smp.recovery.scenes import ROOT,catalog,initialize,overhead_clearance
bank=ROOT/'datasets/recovery_benchmark/validation.npz';records=[]
for name,cfg in catalog().items():
 for site in cfg.get('sites',{'center':[0,0]}):
  for direction in range(4):
   spec,m,d,meta=initialize(name,bank,direction,site,20260923)
   assert meta['initial_min_contact_distance']>=-.002
   fixed=[];guided=[]
   for obj in meta['objects']:
    bid=m.body(obj['name']).id
    if obj['motion']=='fixed':assert m.body_jntnum[bid]==0;fixed.append((bid,d.xpos[bid].copy()))
    else:
     j=int(m.body_jntadr[bid]);v=int(m.jnt_dofadr[j])
     assert m.body_mass[bid]>0 and (m.body_inertia[bid]>0).all()
     assert abs(m.body_mass[bid]-obj['mass'])<1e-6
     if obj['motion']=='slide_z':
      assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE
      assert np.allclose(m.jnt_axis[j],[0,0,1]);guided.append((bid,d.xpos[bid,:2].copy()))
     else:assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE
   # Supports never appear in the overhead diagnostic, even when below the robot.
   assert set(overhead_clearance(m,d,meta))=={o['name'] for o in meta['objects']}
   if cfg['kind']=='support':assert not overhead_clearance(m,d,meta)
   # Passive 20ms sanity only; does not establish reset stability or recovery.
   for _ in range(10):mujoco.mj_step(m,d)
   assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
   for bid,pos in fixed:assert np.allclose(d.xpos[bid],pos)
   for bid,xy in guided:assert np.allclose(d.xpos[bid,:2],xy)
   records.append({k:meta[k] for k in ['scene','direction','site','bank_index','initial_min_contact_distance']})
   print(name,site,direction,'PASS',flush=True)
out=ROOT/'outputs/recovery_study';out.mkdir(parents=True,exist_ok=True)
(out/'scene_checks.json').write_text(json.dumps({'scope':'initial collision, object DOF/mass/inertia, role separation, 20ms finite passive physics; not policy performance','cases':records},indent=2)+'\n')
print('PASS',len(records),'scene/site/direction cases')
