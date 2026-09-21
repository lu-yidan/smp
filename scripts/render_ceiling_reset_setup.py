"""Render audited, actual per-world training resets without advancing physics."""
import argparse
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw
from mjlab.scene import Scene
from train_v33_path_ablation import build_config
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=int,choices=[0,1],default=1);a=p.parse_args()
b=np.load(a.input);cfg,_=build_config(len(b['qpos']),arm='C2');scene=Scene(cfg.scene,device='cpu');model=scene.compile();data=mujoco.MjData(model)
renderer=mujoco.Renderer(model,height=480,width=640);camera=mujoco.MjvCamera();camera.distance=2.35;camera.azimuth=130;camera.elevation=-22
names=['flat','fixed ceiling','free rigid plate'];dirs=['supine','prone','left side','right side'];tiles=[]
for s in range(3):
 row=[]
 for d in range(4):
  j=np.flatnonzero((b['scene']==s)&(b['direction']==d)&(b['kind']==0)&(b['source']==a.source))[0]
  model.geom_size[:]=b['geom_size'][j];data.qpos[:]=b['qpos'][j];data.mocap_pos[:]=b['mocap_pos'][j];data.mocap_quat[:]=b['mocap_quat'][j]
  mujoco.mj_forward(model,data);camera.lookat[:]=[data.qpos[0],data.qpos[1],.45]
  renderer.update_scene(data,camera=camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,55),fill='black')
  source='procedural' if b['source'][j] else 'LAFAN'
  title=f'{names[s]} | {dirs[d]} | {source} | env {j}'
  draw.text((8,6),title,fill='white')
  if s:
   g=model.geom('escape_obstacle/escape_plate_geom' if s==1 else 'free_obstacle/plate_geom').id
   xyz=model.geom_size[g]*2
   draw.text((8,27),f'L/W/T={xyz[0]:.2f}/{xyz[1]:.2f}/{xyz[2]:.3f} m'+(f' bottom={b["ceiling_bottom"][j]:.2f} m' if s==1 else ''),fill='white')
  row.append(np.asarray(im))
 tiles.append(np.concatenate(row,axis=1))
a.out.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.concatenate(tiles,axis=0)).save(a.out);renderer.close();print(a.out)
