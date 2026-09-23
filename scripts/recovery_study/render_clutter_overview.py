"""Compose actual development-set MuJoCo images; no synthetic policy outcomes."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from smp.recovery.clutter_benchmark import FAMILIES,DIRECTIONS
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
labels=['Crossed timber: 4 independent objects','Ladder + board: open frame','Fixed C-space: top + side + base','Hollow container: 5-wall shell']
select=[('supine',2),('prone',1),('prone',0),('right_side_down',1)]
out=Image.new('RGB',(1600,1280),'#172532')
for i,(family,(direction,layout)) in enumerate(zip(FAMILIES,select)):
    im=Image.open(a.root/'development'/f'{family}__{direction}__l{layout}__v00'/'preview.png').convert('RGB')
    im=im.crop((0,48,960,720));im.thumbnail((780,560))
    x=(i%2)*800+10;y=(i//2)*640+55;out.paste(im,(x,y));ImageDraw.Draw(out).text((x,y-38),labels[i],font=font,fill='white')
ImageDraw.Draw(out).text((20,1240),'Initial-state fixtures only | no recovery success is claimed',font=font,fill='#e3e6e9')
out.save(a.root/'overview.png')
grid=Image.new('RGB',(1920,1520),'#172532');small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
for row,family in enumerate(FAMILIES):
    for col,direction in enumerate(DIRECTIONS):
        im=Image.open(a.root/'development'/f'{family}__{direction}__l1__v00'/'preview.png').convert('RGB');im.thumbnail((480,360))
        grid.paste(im,(col*480,row*380+20))
grid.save(a.root/'four_directions.png')
print(a.root/'overview.png')
