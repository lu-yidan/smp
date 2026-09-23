"""Fixed layout-1, four-direction, three-policy paired videos from saved states."""
import argparse,json,subprocess
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from smp.recovery.clutter_benchmark import FAMILIES,DIRECTIONS,load_case


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--review',type=Path);p.add_argument('--families',nargs='+',default=list(FAMILIES));p.add_argument('--profiles',nargs='+',default=['R2_9000','A6_9999','M2_2500']);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);w,h=480,320;cols=len(a.profiles);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for family in a.families:
        setups=[]
        for direction in DIRECTIONS:
            case=f'{family}__{direction}__l1__v00';m,d,meta=load_case(a.root/case)
            renderer=mujoco.Renderer(m,height=h,width=w)
            camera=mujoco.MjvCamera();camera.distance=2.9;camera.azimuth=135;camera.elevation=-24
            traces=[]
            for profile in a.profiles:
                path=a.run/profile/(case+'.npz');f=np.load(path)
                traces.append({'state':f['state'],'diag':f['diagnostics'],'nq':int(f['nq']),'nv':int(f['nv']),
                               'initial':f['initial_qpos'],'holds':np.load(a.review/profile/(case+'.npz'))['holds'] if a.review else None,'result':json.loads(path.with_suffix('.json').read_text())})
            setups.append((m,d,renderer,camera,traces,direction))
        output=a.out/(family+'_paired_20s.mp4')
        args=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w*cols}x{h*4}',
              '-framerate','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','21','-pix_fmt','yuv420p',str(output)]
        with subprocess.Popen(args,stdin=subprocess.PIPE) as proc:
            for frame in range(500):
                t=frame*.04;im=Image.new('RGB',(w*cols,h*4),'#172532')
                for row,(m,d,renderer,camera,traces,direction) in enumerate(setups):
                    for col,(profile,tr) in enumerate(zip(a.profiles,traces)):
                        idx=frame*2-1
                        if idx<0:
                            d.qpos[:]=tr['initial'];d.qvel[:]=0;hold=0;blocked=True
                        else:
                            idx=min(idx,len(tr['state'])-1);s=tr['state'][idx];d.qpos[:]=s[:tr['nq']];d.qvel[:]=s[tr['nq']:];hold=tr['diag'][idx,3];blocked=tr['diag'][idx,4]>.5
                        mujoco.mj_forward(m,d)
                        # Follow translation with the same rule in every tile; world grid retains scale.
                        camera.lookat[:]=[d.qpos[0],d.qpos[1],.65]
                        renderer.update_scene(d,camera=camera);tile=Image.fromarray(renderer.render());draw=ImageDraw.Draw(tile)
                        draw.rectangle((0,0,w,53),fill='#101d28');draw.text((6,3),f'{profile.replace("_","@")} | {direction.replace("_side_down"," side")} | {t:.2f}s',font=font,fill='white')
                        quiet=tr['holds'][idx,0] if idx>=0 and tr['holds'] is not None else 0.
                        draw.text((6,20),f'quiet {quiet:.1f}s | clear hold {hold:.1f}s | {"blocked" if blocked else "clear"}',font=font,fill='#e7ddad')
                        draw.text((6,36),f'base XY ({d.qpos[0]:+.2f}, {d.qpos[1]:+.2f}) m | nominal CPU',font=font,fill='#bbbbbb')
                        if tr['result']['unsafe'] and t>tr['result']['completed_sim_s']:
                            draw.rectangle((0,130,w,195),fill='black');draw.text((8,145),'NUMERICAL FAILURE - trial stopped',font=font,fill='red')
                        im.paste(tile,(col*w,row*h))
                proc.stdin.write(np.asarray(im).tobytes())
                if frame in (0,125,250,499):im.save(a.out/f'{family}_{frame:03d}.png')
            proc.stdin.close();assert proc.wait()==0
        for _,_,renderer,_,_,_ in setups:renderer.close()
        print('RENDERED',output,flush=True)
    (a.out/'selection.json').write_text(json.dumps({'selection':'all four directions, layout index 1, variant 0; not selected for success',
                                                  'profiles':a.profiles,'families':a.families,'playback_speed':1,'duration_s':20,'fps':25},indent=2)+'\n')

if __name__=='__main__':main()
