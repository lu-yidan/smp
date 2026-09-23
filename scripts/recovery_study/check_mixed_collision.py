"""Regression: runtime geometry must be checked beyond compiled native BVHs."""
import mujoco
import numpy as np
from smp.recovery.mixed_geometry import cpu_model,draw,apply_cpu,GEOMS
m=cpu_model();d=mujoco.MjData(m)
p=draw('flat',np.random.default_rng(1))
p['size'][9]=[.6,.6,.5];p['pos'][9]=[0,0,.5]
apply_cpu(m,d,p);d.qpos[:3]=[0,0,.5];mujoco.mj_forward(m,d)
g=m.geom(GEOMS[9]).id
hits=[c for c in d.contact if g in (c.geom1,c.geom2)]
assert hits and min(c.dist for c in hits)<-.02, 'Stale BVH hid deliberately intersecting support box'
assert all(m.geom_sameframe[m.geom(n).id]==0 for n in GEOMS)
d.qpos[2]=3.;mujoco.mj_forward(m,d)
assert not any(g in (c.geom1,c.geom2) for c in d.contact)
print('PASS: actual resized/moved support contacts detected; separated robot has no false contact.')
