"""Audit and extract a time-free supine-to-crouch pose manifold from reference motion."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from smp.rl.tasks.getup.master_deployment_contract import deployment_robot_spec,COLLISION_NAMES
parser=argparse.ArgumentParser()
parser.add_argument('--motion',type=Path,required=True)
parser.add_argument('--out',type=Path,default=Path('datasets/guidance'))
args=parser.parse_args();src=args.motion
b=np.load(src);s=deployment_robot_spec();s.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1));m=s.compile();d=mujoco.MjData(m);records=[];poses=[];features=[]
for i in range(625,776):
 d.qpos[:3]=b['body_pos_w'][i,0];d.qpos[3:7]=b['body_quat_w'][i,0];d.qpos[7:]=b['joint_pos'][i];mujoco.mj_forward(m,d)
 robot={m.geom(n).id for n in COLLISION_NAMES}
 selfdepth=min([c.dist for c in d.contact if c.geom1 in robot and c.geom2 in robot]+[0.])
 limit=float(np.maximum(m.jnt_range[1:,0]-d.qpos[7:],d.qpos[7:]-m.jnt_range[1:,1]).max())
 # Absolute root height is not used for guidance, but explicitly record invalid ground contacts.
 grounddepth=min([c.dist for c in d.contact if c.geom1 not in robot or c.geom2 not in robot]+[0.])
 err=float(np.max(abs(d.xpos[1:31]-b['body_pos_w'][i])))
 records.append({'frame':i,'seconds':i/50,'self_penetration':float(selfdepth),'ground_penetration':float(grounddepth),'limit_excess':max(limit,0),'body_fk_max_error':err})
 tracked_depth=min([c.dist for c in d.contact if c.geom1 in robot and c.geom2 in robot and not any(token in (m.geom(c.geom1).name+' '+m.geom(c.geom2).name) for token in ('hand','wrist','forearm','upper_arm','shoulder'))]+[0.])
 records[-1]['tracked_self_penetration']=float(tracked_depth)
 if tracked_depth<-.005 or limit>.01:continue
 # Guidance uses six leg flexion joints plus three waist joints and pelvis gravity.
 joints=[0,3,4,6,9,10,12,13,14]
 poses.append(d.qpos.copy());features.append(np.r_[d.qpos[7:][joints],-d.xmat[1].reshape(3,3)[2]])
out=args.out;out.mkdir(parents=True,exist_ok=True)
np.savez(out/'supine_to_crouch.npz',features=np.asarray(features,dtype=np.float32),qpos=np.asarray(poses),joint_indices=np.array([0,3,4,6,9,10,12,13,14]))
report={'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'window':[12.5,15.5],'accepted':len(poses),'total':len(records),'guidance_scope':'partial leg/waist/gravity only; raw hand collisions excluded, raw poses NOT reset-safe','records':records}
(out/'audit.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='records'});print('worst self',min(r['self_penetration'] for r in records),'ground',min(r['ground_penetration'] for r in records),'fk',max(r['body_fk_max_error'] for r in records))
