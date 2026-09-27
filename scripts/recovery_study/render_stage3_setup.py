"""Render actual bank geometry and four-direction reset poses for visual audit."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from smp.recovery.stage3_geometry import SCENES,add_world,apply_cpu

def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 b=np.load(a.bank)
 from smp.rl.tasks.getup.master_deployment_contract import deployment_robot_spec
 spec=deployment_robot_spec()
 spec.add_texture(name='floor_grid',type=mujoco.mjtTexture.mjTEXTURE_2D,builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,width=512,height=512,rgb1=(.25,.29,.33),rgb2=(.35,.39,.43))
 spec.add_material(name='floor_mat',textures=['floor_grid'],texrepeat=(6,6),texuniform=True)
 spec.worldbody.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1),material='floor_mat')
 spec.worldbody.add_light(pos=(0,-3,4),dir=(0,.5,-1),diffuse=(.8,.8,.8),castshadow=True)
 add_world(spec);m=spec.compile();m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE);d=mujoco.MjData(m);m.vis.global_.offwidth=480;m.vis.global_.offheight=320
 cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=2.5;cam.azimuth=130;cam.elevation=-24
 canvas=Image.new('RGB',(4*480,7*350),(248,248,248));draw=ImageDraw.Draw(canvas)
 with mujoco.Renderer(m,height=320,width=480) as renderer:
  for sc,name in enumerate(SCENES):
   for dr,direction in enumerate(('Supine','Prone','Left side','Right side')):
    ix=np.flatnonzero((b['scene']==sc)&(b['direction']==dr)&(b['source']==0)&(b['level']==2))[0]
    mujoco.mj_resetData(m,d);d.qpos[:36]=b['qpos'][ix]
    params={k:b[k][ix] for k in ('size','pos','quat','poses','mass','inertia','ipos','iquat','active')};apply_cpu(m,d,params)
    cam.lookat[:]=[d.qpos[0],d.qpos[1],.45];renderer.update_scene(d,camera=cam)
    canvas.paste(Image.fromarray(renderer.render()),(dr*480,sc*350));draw.text((dr*480+10,sc*350+327),f'{name} | {direction} | bank row {ix}',fill='black')
 a.out.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.out)
 print(a.out)
if __name__=='__main__':main()
