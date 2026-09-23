"""CPU MuJoCo scene fixtures with deploy G1, explicit object roles and reset audits.
No policy, reward or privileged observation is silently attached by this module.
"""
from pathlib import Path
import copy
import json
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / 'src/smp/assets/deploy_g1'
CATALOG = ROOT / 'configs/recovery_study/scenes.json'
DIRECTIONS = ('supine', 'prone', 'left_side_down', 'right_side_down')


def catalog():
    return json.loads(CATALOG.read_text())['scenes']


def box(body, name, half, pos=(0, 0, 0), yaw=0., mass=None, color=(.72, .48, .22, 1)):
    args = dict(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos,
                quat=(math.cos(yaw/2), 0, 0, math.sin(yaw/2)), rgba=color,
                contype=1, conaffinity=1, friction=(1., .01, .001), solref=(.01, 1.))
    if mass is not None:
        args['mass'] = mass
    return body.add_geom(**args)


def build_spec(scene, seed=0, overrides=None):
    cfg = copy.deepcopy(catalog()[scene])
    overrides = overrides or {}
    if set(overrides)-set(cfg):
        raise ValueError(f'Unknown scene fields: {set(overrides)-set(cfg)}')
    cfg.update(overrides)
    for key in ('length', 'width', 'thickness', 'mass', 'bottom', 'step_height', 'step_width', 'platform_width', 'leg_width', 'travel', 'down_travel'):
        if key in cfg and (not np.isfinite(cfg[key]) or cfg[key] <= 0):
            raise ValueError(f'{key} must be finite and positive')
    if cfg.get('motion','fixed') not in ('fixed','slide_z','free'):
        raise ValueError('Unknown obstacle motion')
    if cfg.get('count',1)<1 or int(cfg.get('count',1))!=cfg.get('count',1):
        raise ValueError('count must be a positive integer')
    if 'leg_width' in cfg and 2*cfg['leg_width'] >= min(cfg['length'], cfg['width']):
        raise ValueError('Table legs leave no opening')
    xml = ET.parse(ASSETS/'source.xml').getroot()
    xml.find('compiler').set('meshdir', str(ASSETS/'meshes'))
    for world in xml.findall('worldbody'):
        for child in list(world):
            if child.tag == 'body' and child.get('name') == 'box':
                world.remove(child)  # unrelated deploy XML demonstration obstacle
    for k in xml.findall('keyframe'):
        xml.remove(k)
    spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding='unicode'))
    spec.option.timestep = .002
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = 720
    roles = {'floor': 'support'}
    objects = []
    rng = np.random.default_rng(seed)
    support_top = 0.
    if scene == 'pyramid_stairs':
        n = int(cfg['rings'])
        if n < 1 or n != cfg['rings']:
            raise ValueError('rings must be a positive integer')
        for i in range(n):
            half = cfg['platform_width']/2+(n-1-i)*cfg['step_width']
            name = f'stair_{i}'
            box(spec.worldbody, name, (half, half, cfg['step_height']/2),
                (0, 0, (i+.5)*cfg['step_height']), color=(.38, .48, .55, 1))
            roles[name] = 'support'
        support_top = n*cfg['step_height']
    if scene in ('box_terrain', 'mixed_clutter'):
        for i, (x, y) in enumerate([(0, 0), (.65, .1), (-.55, -.25), (.1, .6)]):
            length, width, height = rng.uniform([.28, .25, .05], [.50, .48, .20])
            name = f'support_box_{i}'
            box(spec.worldbody, name, (length/2, width/2, height/2), (x, y, height/2),
                yaw=rng.uniform(-.4, .4), color=(.45, .55, .35, 1))
            roles[name] = 'support'
            support_top = max(support_top, height)
    if cfg['kind'] in ('overhead', 'mixed'):
        count = int(cfg.get('count', 1))
        for i in range(count):
            name = f'obstacle_{i}'
            dynamic = cfg['motion'] != 'fixed'
            center_z = 3.+i if dynamic else cfg['bottom']+cfg['thickness']/2
            body = spec.worldbody.add_body(name=name, pos=((i-.5)*.2 if count>1 else 0, 0, center_z))
            if cfg['motion'] == 'free':
                body.add_freejoint(name=name+'_joint')
            elif cfg['motion'] == 'slide_z':
                body.add_joint(name=name+'_joint', type=mujoco.mjtJoint.mjJNT_SLIDE,
                               axis=(0, 0, 1), limited=True, range=(-cfg.get('down_travel',.30), cfg['travel']), damping=2.)
            l, w, t = cfg['length'], cfg['width'], cfg['thickness']
            mass = cfg.get('mass')
            names = []
            if cfg.get('shape') == 'L_compound':
                # Two touching, non-overlapping boxes, one rigid body. Volume split 2:1.
                for suffix, half, pos, fraction in [('_a', (l/2,w/4,t/2),(0,-w/4,0),2/3),('_b',(l/4,w/4,t/2),(-l/4,w/4,0),1/3)]:
                    g = name+suffix
                    box(body,g,half,pos,mass=mass*fraction);names.append(g)
            else:
                g = name+'_geom';box(body,g,(l/2,w/2,t/2),mass=mass);names.append(g)
            roles.update({g:'overhead' for g in names})
            objects.append({'name':name,'motion':cfg['motion'],'geoms':names,'mass':mass,'thickness':t})
        if scene == 'table':
            for i, (sx, sy) in enumerate([(-1,-1),(-1,1),(1,-1),(1,1)]):
                name = f'table_leg_{i}'
                h=cfg['bottom'];w=cfg['leg_width']
                box(spec.worldbody,name,(w/2,w/2,h/2),
                    (sx*(cfg['length']-w)/2,sy*(cfg['width']-w)/2,h/2),color=(.3,.5,.65,1))
                roles[name]='barrier'
    return spec, {'scene':scene,'seed':seed,'config':cfg,'roles':roles,'objects':objects,
                  'support_top':float(support_top),'robot_source':str(ASSETS/'source.xml')}


def role_ids(model, meta, role):
    return {model.geom(k).id for k,v in meta['roles'].items() if v==role}


def robot_geoms(model):
    pelvis=model.body('pelvis').id
    bodies={pelvis}
    for i in range(pelvis+1,model.nbody):
        if int(model.body_parentid[i]) in bodies:bodies.add(i)
    return [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies and (model.geom_contype[i] or model.geom_conaffinity[i])]


def bounds(model,data,ids):
    centers=data.geom_xpos[ids]
    extent=np.einsum('nij,nj->ni',np.abs(data.geom_xmat[ids].reshape(-1,3,3)),model.geom_aabb[ids,3:])
    # Local AABB centers need their own rotation; geom center is not always AABB center.
    centers=centers+np.einsum('nij,nj->ni',data.geom_xmat[ids].reshape(-1,3,3),model.geom_aabb[ids,:3])
    return centers,extent


def overhead_clearance(model,data,meta):
    """Per-object conservative XY clearance; support boxes are explicitly excluded.
    Diagnostic only: this is not yet a new multi-object task success/reward.
    """
    robot=robot_geoms(model);rp,re=bounds(model,data,robot);result={}
    for obj in meta['objects']:
        gs=[model.geom(g).id for g in obj['geoms']];op,oe=bounds(model,data,gs)
        delta=np.abs(rp[:,None,:2]-op[None,:,:2])-re[:,None,:2]-oe[None,:,:2]
        result[obj['name']]=float(delta.max(-1).min())
    return result


def initialize(scene,bank, direction=0,site='center',seed=0,overrides=None):
    spec,meta=build_spec(scene,seed,overrides)
    model=spec.compile();data=mujoco.MjData(model)
    ids=robot_geoms(model);support=role_ids(model,meta,'support')
    positions=meta['config'].get('sites',{'center':[0,0]})
    if site not in positions:raise ValueError(f'Unknown site {site}; expected {list(positions)}')
    b=np.load(bank);pool=np.flatnonzero((b['direction']==direction)&(b['stratum']==0))
    pool=np.random.default_rng(seed+direction).permutation(pool)
    for idx in pool:
        mujoco.mj_resetData(model,data);data.qpos[:36]=b['qpos'][idx]
        data.qpos[:2]=positions[site]
        data.qpos[2]+=meta['support_top']+.35
        # Descend vertically to the first valid support contact, not onto an overhead surface.
        landed=False
        for _ in range(300):
            mujoco.mj_forward(model,data)
            depths=[c.dist for c in data.contact if (int(c.geom1) in ids and int(c.geom2) in support) or (int(c.geom2) in ids and int(c.geom1) in support)]
            if depths and min(depths)<=0:
                data.qpos[2]+=-min(depths)+.001
                landed=True;break
            data.qpos[2]-=.005
        if not landed:continue
        mujoco.mj_forward(model,data)
        rp,re=bounds(model,data,ids);top=float((rp[:,2]+re[:,2]).max())
        # Place movable constraints just above the robot; stacked objects have positive gaps.
        for obj in meta['objects']:
            if obj['motion']=='fixed':continue
            bid=model.body(obj['name']).id;z=top+obj['thickness']/2+.002
            if obj['motion']=='free':
                adr=int(model.jnt_qposadr[model.body_jntadr[bid]])
                data.qpos[adr+2]=z
            else:
                model.body_pos[bid,2]=z
            top=z+obj['thickness']/2
        mujoco.mj_forward(model,data)
        depth=min([float(c.dist) for c in data.contact]+[0.])
        if depth < -.002:continue
        # Reset fitting is geometric; no policy rollout or proof of an escape route.
        meta.update({'direction':DIRECTIONS[direction],'site':site,'bank_index':int(idx),
                     'initial_min_contact_distance':depth,'initial_clearance':overhead_clearance(model,data,meta),
                     'qpos':data.qpos.tolist(),'reset_status':'geometrically_valid_not_policy_evaluated'})
        return spec,model,data,meta
    raise RuntimeError(f'No collision-valid reset for {scene}/{site}/{DIRECTIONS[direction]}')
