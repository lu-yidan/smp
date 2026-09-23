"""Shared MuJoCo topology for mixed recovery banks and GPU training.
All scene families share collision slots; inactive objects are parked on the floor.
"""
from pathlib import Path
import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
SCENES=('flat','free_plate','guided_plate','fixed_ceiling','table','pyramid_stairs','box_terrain','free_box','double_plate','mixed_clutter')
SHARES=np.array([.25,.20,.15,.05,.05,.075,.075,.05,.05,.05])
GEOMS=('mix_guide_g','mix_free0_a','mix_free0_b','mix_free1_g','mix_roof_g',*(f'mix_leg{i}' for i in range(4)),*(f'mix_support{i}' for i in range(4)))
BODIES=('mix_guide','mix_free0','mix_free1')
OBJECT_GEOMS=((0,),(1,2),(3,),(4,))

def add_world(spec):
 def box(body,name,size,pos=(0,0,0),mass=1.):
  # Nontrivial compiled frames prevent native MuJoCo same-frame shortcuts
  # from ignoring reset-time geometry/inertia transforms. Every bank row replaces these.
  initial_pos=(.001,.002,.003) if tuple(pos)==(0,0,0) else pos
  return body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,size=size,pos=initial_pos,quat=(math.cos(.017),0,0,math.sin(.017)),mass=mass,friction=(1.,.01,.001),solref=(.01,1.),rgba=(.6,.45,.2,1))
 inertial=dict(explicitinertial=True,mass=6.,inertia=(.2,.4,.6),ipos=(.004,.005,.006),iquat=(math.cos(.023),math.sin(.023),0,0))
 guide=spec.worldbody.add_body(name='mix_guide',pos=(20,20,.5),**inertial)
 guide.add_joint(name='mix_slide',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=(0,0,1),limited=True,range=(-.3,.6),damping=2.)
 box(guide,GEOMS[0],(.45,.32,.035),mass=6.)
 for i in range(2):
  body=spec.worldbody.add_body(name=f'mix_free{i}',pos=(22+2*i,20,.04),**inertial);body.add_freejoint(name=f'mix_free{i}_joint')
  box(body,GEOMS[1] if i==0 else GEOMS[3],(.45,.32,.035),mass=6.)
  if i==0:box(body,GEOMS[2],(1e-5,1e-5,1e-5),mass=1e-8)
 fixed=spec.worldbody.add_body(name='mix_fixed',mocap=True)
 for i in range(4,9):box(fixed,GEOMS[i],(.01,.01,.01),pos=(30+i,30,.02))
 support=spec.worldbody.add_body(name='mix_support',mocap=True)
 for i in range(9,13):box(support,GEOMS[i],(.01,.01,.01),pos=(30+i,30,.02))

def scene_spec(spec):
 from smp.rl.tasks.getup.master_deployment_contract import deployment_contacts
 deployment_contacts(spec);add_world(spec)

def cpu_model():
 from smp.rl.tasks.getup.master_deployment_contract import deployment_robot_spec,JOINT_NAMES
 s=deployment_robot_spec();s.worldbody.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1))
 assert tuple(j.name for j in s.joints if j.type!=mujoco.mjtJoint.mjJNT_FREE)==JOINT_NAMES
 add_world(s);m=s.compile();m.opt.timestep=.002
 return m

def inertia(parts,mass):
 volumes=np.array([np.prod(h) for h,p in parts]);weights=mass*volumes/volumes.sum()
 com=sum(w*np.array(p) for w,(h,p) in zip(weights,parts))/mass
 mat=np.zeros((3,3))
 for w,(half,pos) in zip(weights,parts):
  x,y,z=half;d=np.array(pos)-com
  mat+=np.diag(w/3*np.array([y*y+z*z,x*x+z*z,x*x+y*y]))+w*(d@d*np.eye(3)-np.outer(d,d))
 val,rot=np.linalg.eigh(mat)
 if np.linalg.det(rot)<0:rot[:,0]*=-1
 return val,com,Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]

def draw(scene,rng,difficulty=1.):
 """A geometry draw. All random dimensions/masses are archived in the bank."""
 sizes=np.full((13,3),.01);pos=np.zeros((13,3));quat=np.zeros((13,4));quat[:,0]=1
 pos[4:,0]=np.arange(9)+34;pos[4:,1]=30;pos[4:,2]=.02
 poses=np.array([[20,20,.5,1,0,0,0],[22,20,.04,1,0,0,0],[24,20,.04,1,0,0,0]],float)
 sizes[0]=sizes[1]=sizes[3]=[.45,.32,.035];sizes[2]=1e-5
 active=np.zeros(4,bool);mass=np.array([6.,6.,6.]);site=0
 lo=np.array([.6,.45,.03]);hi=np.array([1.2,.9,.08]);nom=np.array([.9,.64,.06])
 length,width,thick=nom+(rng.uniform(lo,hi)-nom)*difficulty
 if scene=='free_box':length,width,thick=rng.uniform([.55,.40,.10],[.95,.75,.24])
 if scene in ('guided_plate','free_plate','free_box','double_plate','mixed_clutter'):
  slot=0 if scene=='guided_plate' else 1;active[slot]=True;sizes[0 if slot==0 else 1]=np.array([length,width,thick])/2
  mass[slot]=6.+(rng.uniform(2.,12.)-6.)*difficulty
  if scene=='free_box':
   sizes[1]=[length/2,width/4,thick/2];pos[1]=[0,-width/4,0]
   sizes[2]=[length/4,width/4,thick/2];pos[2]=[-length/4,width/4,0]
   mass[1]=rng.uniform(4.,12.)
  if scene=='double_plate':
   active[2]=True;sizes[3]=rng.uniform([.3,.23,.015],[.6,.45,.035]);mass[2]=rng.uniform(2.,7.);mass[1]=rng.uniform(2.,7.)
 if scene in ('fixed_ceiling','table'):
  active[3]=True
  length,width=rng.uniform([1.,.75],[1.7,1.3]);bottom=rng.uniform(.55,.8);thick=rng.uniform(.04,.08)
  sizes[4]=[length/2,width/2,thick/2];pos[4]=[0,0,bottom+thick/2]
  if scene=='table':
   for j,(sx,sy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
    sizes[j+5]=[.035,.035,bottom/2];pos[j+5]=[sx*(length/2-.035),sy*(width/2-.035),bottom/2]
 xy=np.zeros(2)
 if scene=='pyramid_stairs':
  rise=.05+.10*difficulty*rng.random();tread=rng.uniform(.25,.35);platform=rng.uniform(.5,.7)
  for j in range(4):
   h=platform/2+(3-j)*tread;sizes[j+9]=[h,h,rise/2];pos[j+9]=[0,0,(j+.5)*rise]
  site=int(rng.integers(4));xy=np.array([(0,0),(platform/2+tread/2,0),(platform/2,0),(platform/2,platform/2)][site])+rng.uniform(-.04,.04,2)
 if scene in ('box_terrain','mixed_clutter'):
  for j,(x,y) in enumerate(((0,0),(.65,.1),(-.55,-.25),(.1,.6))):
   dims=rng.uniform([.3,.28,.04],[.65,.55,.20]);dims[2]=.04+(dims[2]-.04)*difficulty
   sizes[j+9]=dims/2;pos[j+9]=[x,y,dims[2]/2]
   angle=rng.uniform(-.4,.4);quat[j+9]=[math.cos(angle/2),0,0,math.sin(angle/2)]
  site=int(rng.integers(3));xy=np.array([(0,0),(sizes[9,0],0),(sizes[9,0],sizes[9,1])][site])+rng.uniform(-.04,.04,2)
 for body_index,indices in ((1,(1,2)),(2,(3,))):
  poses[body_index,2]=max(sizes[i,2]-pos[i,2] for i in indices)+.003
 inert=[];com=[];iquat=[]
 for indices,ms in zip(((0,),(1,2),(3,)),mass):
  a,b,c=inertia([(sizes[i],pos[i]) for i in indices],ms);inert.append(a);com.append(b);iquat.append(c)
 return dict(size=sizes,pos=pos,quat=quat,poses=poses,mass=mass,inertia=np.array(inert),ipos=np.array(com),iquat=np.array(iquat),active=active,xy=xy,site=site)

def apply_cpu(m,d,params):
 gids=np.array([m.geom(n).id for n in GEOMS]);bids=np.array([m.body(n).id for n in BODIES])
 m.geom_size[gids]=params['size'];m.geom_pos[gids]=params['pos'];m.geom_quat[gids]=params['quat']
 m.geom_aabb[gids,:3]=0;m.geom_aabb[gids,3:]=params['size'];m.geom_rbound[gids]=np.linalg.norm(params['size'],axis=1)
 for key in ('mass','inertia','ipos','iquat'):getattr(m,'body_'+key)[bids]=params[key]
 m.body_pos[bids[0]]=params['poses'][0,:3]
 for i in (1,2):
  j=m.body_jntadr[bids[i]];a=m.jnt_qposadr[j];d.qpos[a:a+7]=params['poses'][i]
 d.qpos[m.joint('mix_slide').qposadr[0]]=0;d.qvel[:]=0
 mujoco.mj_forward(m,d)
 return gids,bids
