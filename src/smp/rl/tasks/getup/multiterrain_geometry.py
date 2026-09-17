"""Shared CPU/GPU geometry for the P5 initialization study. SI units."""
import math
import mujoco
import numpy as np

DIRECTIONS=('supine','prone','left_side_down','right_side_down')
STRATA=('flat','vertical_plate','free_plate','stair_interior','stair_edge','stair_straddle','slope_interior','slope_edge')
# Disjoint patches in each independent MuJoCo world. No inter-environment contact.
BOXES=[((3.2,.0,.05),(.4,1.0,.05),(1,0,0,0)),
       ((4.0,.0,.10),(.4,1.0,.10),(1,0,0,0)),
       ((4.9,.0,.15),(.5,1.0,.15),(1,0,0,0)),
       ((8.0,0.,.24),(1.1,.8,.08),(math.cos(math.radians(5)),0,math.sin(math.radians(5)),0))]

def add_terrain(body):
    for i,(pos,size,quat) in enumerate(BOXES):
        body.add_geom(name=f'surface_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,quat=quat,
                      friction=(1.,.005,.0001),rgba=(.35,.45,.55,1),solref=(.01,1.))

def terrain_spec():
    s=mujoco.MjSpec();add_terrain(s.worldbody.add_body(name='surfaces'));return s

def free_plate_spec():
    s=mujoco.MjSpec();b=s.worldbody.add_body(name='plate');b.add_freejoint(name='free_plate_joint')
    b.add_geom(name='plate_geom',type=mujoco.mjtGeom.mjGEOM_BOX,size=(.45,.32,.035),mass=8.,
               friction=(1.2,.01,.001),rgba=(.85,.45,.12,.8),solref=(.01,1.))
    return s

def quotas(n):
    assert n%16==0 and n>=16
    scene=np.repeat(np.arange(4),n//4);direction=np.tile(np.repeat(np.arange(4),n//16),4)
    strata=scene.copy()
    for d in range(4):
        ids=np.flatnonzero((scene==3)&(direction==d));raw=np.array([.2,.3,.2,.1,.2])*len(ids)
        counts=np.floor(raw).astype(int)
        for i in np.argsort(-(raw-counts))[:len(ids)-counts.sum()]:counts[i]+=1
        strata[ids]=np.repeat(np.arange(3,8),counts)
    return scene,direction,strata

def support_height(xy):
    """Top ray height for this union of boxes and plane; supports torch or numpy."""
    import torch
    is_t=torch.is_tensor(xy);out=torch.zeros_like(xy[...,0]) if is_t else np.zeros(xy.shape[:-1])
    for pos,size,q in BOXES:
        angle=2*math.atan2(q[2],q[0]);c=math.cos(angle);s=math.sin(angle)
        # Vertical intersection with the sloped top face. Side intersections use
        # the highest valid face; x side tops matter only in the thin edge strip.
        for sign in (-1,1):
            if abs(s)>1e-8:
                z=pos[2]+(c*(xy[...,0]-pos[0])-sign*size[0])/s
                local_z=s*(xy[...,0]-pos[0])+c*(z-pos[2])
                valid=(abs(xy[...,1]-pos[1])<=size[1])&(abs(local_z)<=size[2])&(z>=0)
                out=torch.where(valid,torch.maximum(out,z),out) if is_t else np.where(valid,np.maximum(out,z),out)
        z=pos[2]+(size[2]-s*(xy[...,0]-pos[0]))/c
        local_x=c*(xy[...,0]-pos[0])-s*(z-pos[2])
        valid=(abs(local_x)<=size[0])&(abs(xy[...,1]-pos[1])<=size[1])
        out=torch.where(valid,torch.maximum(out,z),out) if is_t else np.where(valid,np.maximum(out,z),out)
    return out


def scene_spec(spec):
    from smp.rl.tasks.getup.master_deployment_contract import deployment_contacts
    deployment_contacts(spec)
    add_terrain(spec.worldbody.add_body(name='surfaces'))
