"""Four directions in a 2x2 video; archived layout1/variant0, not success-selected."""
import argparse,json,subprocess
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from smp.recovery.clutter_benchmark import load_case,DIRECTIONS


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--families',nargs='+',default=['crossed_timber','ladder_boards','hollow_container','fixed_c_space']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 w,h=640,400; font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);selection=[]
 for family in a.families:
  setup=[]
  for direction in DIRECTIONS:
   cid=family+'__'+direction+'__l1__v00';m,d,_=load_case(a.root/cid)
   with np.load(a.results/'run/A6_no_alpha_9999'/(cid+'.npz')) as f:tr={k:f[k] for k in ('state','initial_qpos','diagnostics')}
   tr['holds']=np.load(a.results/'review/A6_no_alpha_9999'/(cid+'.npz'))['holds'];tr['result']=json.loads((a.results/'review/A6_no_alpha_9999'/(cid+'.json')).read_text())
   renderer=mujoco.Renderer(m,height=h,width=w);cam=mujoco.MjvCamera();cam.distance=2.8;cam.azimuth=135;cam.elevation=-28
   setup.append((m,d,renderer,cam,tr,direction));selection.append(cid)
  dest=a.out/(family+'_four_directions_20s.mp4')
  cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w*2}x{h*2}','-framerate','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','21','-pix_fmt','yuv420p',str(dest)]
  with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
   for frame in range(500):
    im=Image.new('RGB',(w*2,h*2));t=frame*.04
    for k,(m,d,renderer,cam,tr,direction) in enumerate(setup):
     ix=min(frame*2-1,len(tr['state'])-1)
     if ix<0:d.qpos[:]=tr['initial_qpos'];d.qvel[:]=0;q=e=0.
     else:
      s=tr['state'][ix];d.qpos[:]=s[:m.nq];d.qvel[:]=s[m.nq:];q,e=tr['holds'][ix]
     mujoco.mj_forward(m,d);cam.lookat[:]=[d.qpos[0],d.qpos[1],.6];renderer.update_scene(d,camera=cam)
     tile=Image.fromarray(renderer.render());draw=ImageDraw.Draw(tile);draw.rectangle((0,0,w,60),fill='#132330')
     draw.text((8,4),f'{family} | {direction} | {t:.2f}s',font=font,fill='white')
     draw.text((8,23),f'Quiet hold: {q:.1f}s | Quiet + lift-clear: {e:.1f}s',font=font,fill='#f2dfa4')
     draw.text((8,42),f'Simulation | base XY ({d.qpos[0]:+.2f}, {d.qpos[1]:+.2f}) m',font=font,fill='#c9d4de')
     if tr['result']['unsafe'] and t>tr['result']['completed_sim_s']:draw.text((10,150),'NUMERICAL FAILURE',font=font,fill='red')
     im.paste(tile,((k%2)*w,(k//2)*h))
    proc.stdin.write(np.asarray(im).tobytes())
    if frame in (0,75,150,300,499):im.save(a.out/f'{family}_{frame:03d}.png')
   proc.stdin.close();assert proc.wait()==0
  for _,_,renderer,_,_,_ in setup:renderer.close()
  print('RENDERED',dest,flush=True)
 (a.out/'selection.json').write_text(json.dumps({'rule':'layout1 variant0 all directions, fixed before reviewing outcomes','cases':selection,'duration_s':20,'playback_speed':1,'policy':'A6_no_alpha_9999','scope':'simulation; no hardware'},indent=2)+'\n')

if __name__=='__main__':main()
