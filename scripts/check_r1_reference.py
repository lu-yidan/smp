"""Geometry-only check/render of the soft posture target in deployment assets."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from PIL import Image
from smp.rl.tasks.getup.master_deployment_contract import ASSETS,JOINT_NAMES
from smp.rl.tasks.getup.r1_quality import reference_values
out=Path('outputs/r1_quality_start');out.mkdir(parents=True,exist_ok=True)
root=ET.parse(ASSETS/'source.xml').getroot();root.find('compiler').set('meshdir',str(ASSETS/'meshes'))
if root.find(".//geom[@name='floor']") is None:ET.SubElement(root.find('worldbody'),'geom',name='floor',type='plane',size='0 0 .1',rgba='.22 .26 .3 1')
for world in root.findall('worldbody'):
 for child in list(world):
  if child.tag=='body' and child.get('name')!='pelvis':world.remove(child)
ET.SubElement(root.find('worldbody'),'light',pos='0 -2 3',dir='0 0 -1')
ET.SubElement(root.find(".//body[@name='torso_link']"),'site',name='head_check',pos='0 0 .43',size='.005')
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
data.qpos[:7]=[0,0,1,1,0,0,0];ref=reference_values(JOINT_NAMES)
for name,q in zip(JOINT_NAMES,ref):
 j=model.joint(name).id;data.qpos[model.jnt_qposadr[j]]=q
 assert not model.jnt_limited[j] or model.jnt_range[j,0]<=q<=model.jnt_range[j,1]
mujoco.mj_forward(model,data)
ids=[model.geom(g.attrib['name']).id for g in root.findall('.//geom') if g.get('class')=='collision']
mins=[]
for i in ids:
 R=data.geom_xmat[i].reshape(3,3);size=model.geom_size[i];kind=model.geom_type[i]
 if kind==mujoco.mjtGeom.mjGEOM_SPHERE:half=size[0]
 elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:half=size[0]+abs(R[2,2])*size[1]
 elif kind==mujoco.mjtGeom.mjGEOM_BOX:half=np.abs(R[2])@size
 else:raise ValueError(kind)
 mins.append(data.geom_xpos[i,2]-half)
data.qpos[2]+=.002-min(mins);mujoco.mj_forward(model,data)
feet=[data.xpos[model.body(n).id].copy() for n in ['left_ankle_roll_link','right_ankle_roll_link']]
contacts=[{'g1':model.geom(c.geom1).name,'g2':model.geom(c.geom2).name,'distance':float(c.dist)} for c in data.contact[:data.ncon]]
assert not any(c['distance']<-.001 for c in contacts)
report={'reference_joint_pos':ref,'head_height':float(data.site_xpos[model.site('head_check').id,2]),'foot_width':float(np.linalg.norm(feet[0][:2]-feet[1][:2])),'contacts':contacts,'root_height':float(data.qpos[2]),'minimum_robot_floor_clearance':.002,'scope':'Geometry/joint-limit check only; soft reward target, not proof of dynamic or hardware safety.'}
assert report['head_height']>1.15 and .12<report['foot_width']<.45
(out/'reference_check.json').write_text(json.dumps(report,indent=2))
model.vis.global_.offwidth=700;model.vis.global_.offheight=700;renderer=mujoco.Renderer(model,height=700,width=700);cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.7];cam.distance=2.7;cam.azimuth=135;cam.elevation=-12
renderer.update_scene(data,camera=cam);Image.fromarray(renderer.render()).save(out/'reference_pose.png');renderer.close();print(json.dumps(report,indent=2))
