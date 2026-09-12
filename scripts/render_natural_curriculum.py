"""Render audited training reset samples on the actual deployment model."""
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw
from mjlab.scene import Scene
from train_termination_ablation import build_config
b=np.load('datasets/reset_banks/natural_curriculum_v1/train.npz');cfg,_=build_config('B1',1);m=Scene(cfg.scene,device='cpu').compile();d=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=240,width=320)
camera=mujoco.MjvCamera();camera.distance=2.3;camera.elevation=-25;camera.azimuth=130
canvas=Image.new('RGB',(960,1440));names=['late','middle','supine','prone','left_side_down','right_side_down']
for row,name in enumerate(names):
 mask=(b['stages']==name) if row<2 else ((b['stages']=='low')&(b['labels']==name))
 ids=np.flatnonzero(mask);ids=ids[np.linspace(0,len(ids)-1,3,dtype=int)]
 for col,i in enumerate(ids):
  d.qpos[:]=b['qpos'][i];d.qvel[:]=0;mujoco.mj_forward(m,d);camera.lookat[:]=d.qpos[:3]
  renderer.update_scene(d,camera=camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,320,38),fill='black');draw.text((6,5),f'{name} | train row {i}',fill='white');draw.text((6,21),'Static reset sample, not policy rollout',fill='white');canvas.paste(im,(320*col,240*row))
out=Path('outputs/natural_curriculum');out.mkdir(exist_ok=True,parents=True);canvas.save(out/'reset_gallery.jpg');renderer.close();print(out/'reset_gallery.jpg')
