from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw
from mjlab.scene import Scene
from train_multiterrain import build_config
from smp.rl.tasks.getup.multiterrain_geometry import STRATA,DIRECTIONS
b=np.load('outputs/verification/initial_256.npz');cfg,_=build_config(256,nominal=True)
scene=Scene(cfg.scene,device='cpu');m=scene.spec.compile();d=mujoco.MjData(m)
r=mujoco.Renderer(m,height=300,width=400);cam=mujoco.MjvCamera();cam.distance=2.5;cam.azimuth=125;cam.elevation=-25
canvas=Image.new('RGB',(1600,2400))
for st in range(8):
 for dr in range(4):
  i=np.flatnonzero((b['stratum']==st)&(b['direction']==dr))[0]
  d.qpos[:]=b['qpos'][i];d.mocap_pos[:]=b['mocap_pos'][i];d.mocap_quat[:]=b['mocap_quat'][i];mujoco.mj_forward(m,d)
  cam.lookat[:]=d.qpos[:3]+[0,0,.15];r.update_scene(d,camera=cam);im=Image.fromarray(r.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,400,30),fill='black');draw.text((5,8),f'{STRATA[st]} | {DIRECTIONS[dr]}',fill='white');canvas.paste(im,(400*dr,300*st))
canvas.save('outputs/verification/reset_grid.jpg');r.close()
print('RENDERED')
