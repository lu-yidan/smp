"""Shared CPU/GPU geometry for the P5 initialization study. SI units."""
import math
import mujoco
import numpy as np

DIRECTIONS=('supine','prone','left_side_down','right_side_down')
STRATA=('flat','vertical_plate','free_plate','stair_interior','stair_edge','stair_straddle','slope_interior','slope_edge')
# Disjoint patches in each independent MuJoCo world. No inter-environment contact.
BOXES=[]  # No stairs or slopes in this experiment.

def add_terrain(body):
    # Retain the existing terrain contact sensor target, physically out of reach.
    body.add_geom(name='sensor_target_parked',type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  pos=(100,100,-10),size=(.01,),rgba=(0,0,0,0))

def terrain_spec():
    s=mujoco.MjSpec();add_terrain(s.worldbody.add_body(name='surfaces'));return s

def free_plate_spec():
    s=mujoco.MjSpec();b=s.worldbody.add_body(name='plate');b.add_freejoint(name='free_plate_joint')
    b.add_geom(name='plate_geom',type=mujoco.mjtGeom.mjGEOM_BOX,size=(.45,.32,.035),mass=8.,
               friction=(1.2,.01,.001),rgba=(.85,.45,.12,.8),solref=(.01,1.))
    return s

def quotas(n):
    assert n%32==0 and n>=32
    counts=(n//2,n//4,n//4)
    scene=np.repeat(np.arange(3),counts)
    direction=np.concatenate([np.repeat(np.arange(4),c//4) for c in counts])
    return scene,direction,scene.copy()

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
