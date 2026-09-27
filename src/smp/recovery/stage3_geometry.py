"""Fixed GPU topology for the planned R2 stage-three comparison.

Ladder rails and rungs are separate boxes on one free rigid body. Timbers
have independent free joints. No held-out benchmark geometries are loaded.
"""
import math
import mujoco
import numpy as np
from smp.recovery.mixed_geometry import inertia

SCENES=('flat','guided_plate','free_plate','fixed_ceiling','crossed_timber','ladder_boards','double_plate')
BODIES=('s3_guide','s3_free0','s3_free1',*(f's3_timber{i}' for i in range(4)),'s3_ladder')
OBJECT_NAMES=(*BODIES,'s3_fixed')
OBJECT_GEOMS=((0,),(1,),(2,),(3,),(4,),(5,),(6,),tuple(range(7,13)),(13,))
GEOMS=tuple(f's3_geom{i}' for i in range(14))

def add_world(spec):
    for j,name in enumerate(OBJECT_NAMES):
        kw=dict(explicitinertial=True,mass=6.,inertia=(.2,.4,.6),ipos=(.004,.005,.006),iquat=(math.cos(.023),math.sin(.023),0,0))
        if j==8:kw['mocap']=True
        body=spec.worldbody.add_body(name=name,pos=(20+2*j,20,.5) if j<8 else (0,0,0),**kw)
        if j==0:body.add_joint(name='s3_slide',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=(0,0,1),limited=True,range=(-.3,.6),damping=2.)
        elif j<8:body.add_freejoint(name=name+'_joint')
        for k in OBJECT_GEOMS[j]:
            body.add_geom(name=GEOMS[k],type=mujoco.mjtGeom.mjGEOM_BOX,size=(.05,.025,.02),pos=(.001,.002,.003),quat=(math.cos(.017),0,0,math.sin(.017)),mass=1.,friction=(1.,.01,.001),solref=(.01,1.),rgba=(.60,.43,.22,1.))

def scene_spec(spec):
    from smp.rl.tasks.getup.master_deployment_contract import deployment_contacts
    deployment_contacts(spec);add_world(spec)

def cpu_model():
    from smp.rl.tasks.getup.master_deployment_contract import deployment_robot_spec
    spec=deployment_robot_spec();spec.worldbody.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1))
    add_world(spec);m=spec.compile();m.opt.timestep=.002
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    return m

def draw(scene,rng,difficulty=1.):
    size=np.tile([.05,.025,.02],(len(GEOMS),1));pos=np.zeros_like(size)
    quat=np.zeros((len(GEOMS),4));quat[:,0]=1
    poses=np.array([[20+2*j,20,.1,1,0,0,0] for j in range(8)],float)
    active=np.zeros(9,bool);mass=np.full(8,2.);xy=np.zeros(2)
    pos[13]=[40,30,.1]
    # Non-overlapping default ladder parts, including in parked worlds.
    length,width,beam,thick=rng.uniform(0.9,1.3),rng.uniform(.44,.64),.045,rng.uniform(.035,.065)
    for k,sign in zip((7,8),(-1,1)):
        size[k]=[length/2,beam/2,thick/2];pos[k]=[0,sign*(width-beam)/2,0]
    for k,x in zip(range(9,13),np.linspace(-length/2+.11,length/2-.11,4)):
        size[k]=[.025,(width-2*beam)/2,thick/2];pos[k]=[x,0,0]
    mass[7]=rng.uniform(2.,3.5)
    if scene in ('guided_plate','free_plate','double_plate'):
        slots=(0,) if scene=='guided_plate' else (1,2) if scene=='double_plate' else (1,)
        for j in slots:
            nom=np.array([.9,.64,.06]);dims=nom+difficulty*(rng.uniform([.6,.45,.03],[1.2,.9,.08])-nom)
            size[j]=dims/2;mass[j]=6+difficulty*(rng.uniform(2.,12.)-6);active[j]=True
            if scene=='double_plate':mass[j]=rng.uniform(2.,7.)
    elif scene=='fixed_ceiling':
        active[8]=True;size[13]=rng.uniform([1.,.75,.04],[1.7,1.3,.08])/2
        pos[13]=[0,0,rng.uniform(.55,.8)+size[13,2]]
    elif scene=='crossed_timber':
        for j in range(3,3+int(rng.integers(2,5))):
            active[j]=True;size[j]=rng.uniform([.75,.055,.025],[1.25,.19,.045])/2
            mass[j]=rng.uniform(.7,2.5)
    elif scene=='ladder_boards':
        active[7]=active[1]=True;active[2]=bool(rng.integers(2))
        for j in (1,2):
            size[j]=rng.uniform([.65,.1,.025],[1.,.3,.045])/2;mass[j]=rng.uniform(.8,1.8)
    elif scene!='flat':raise ValueError(scene)
    inert=[];com=[];iquat=[]
    for j,indices in enumerate(OBJECT_GEOMS[:8]):
        a,b,c=inertia([(size[i],pos[i]) for i in indices],mass[j]);inert.append(a);com.append(b);iquat.append(c)
        poses[j,2]=max(size[i,2]-pos[i,2] for i in indices)+.003
    return dict(size=size,pos=pos,quat=quat,poses=poses,mass=mass,inertia=np.array(inert),ipos=np.array(com),iquat=np.array(iquat),active=active,xy=xy,site=0)

def apply_cpu(m,d,params):
    gids=np.array([m.geom(n).id for n in GEOMS]);bids=np.array([m.body(n).id for n in BODIES])
    for field,key in (('geom_size','size'),('geom_pos','pos'),('geom_quat','quat')):getattr(m,field)[gids]=params[key]
    m.geom_aabb[gids,:3]=0;m.geom_aabb[gids,3:]=params['size'];m.geom_rbound[gids]=np.linalg.norm(params['size'],axis=1)
    for key in ('mass','inertia','ipos','iquat'):getattr(m,'body_'+key)[bids]=params[key]
    m.body_pos[bids[0]]=params['poses'][0,:3]
    for j in range(1,len(BODIES)):
        adr=m.jnt_qposadr[m.body_jntadr[bids[j]]];d.qpos[adr:adr+7]=params['poses'][j]
    d.qpos[m.joint('s3_slide').qposadr[0]]=0;d.qvel[:]=0;mujoco.mj_forward(m,d)
    return gids,bids
