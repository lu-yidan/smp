"""Standalone CPU overhead-clutter benchmark. Does not alter training tasks.

Objects retain individual free joints and compound collision geometry. The
vertical-lift probe is an obstruction diagnostic, not a reachability oracle.
"""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from smp.recovery.scenes import ROOT, ASSETS, DIRECTIONS, build_spec, box, robot_geoms, bounds

FAMILIES = ('crossed_timber', 'ladder_boards', 'fixed_c_space', 'hollow_container')
LAYOUTS = {
    'crossed_timber': ('cross_two', 'fan_three', 'lattice_four'),
    'ladder_boards': ('transverse', 'diagonal', 'offset_two_boards'),
    'fixed_c_space': ('exit_front', 'exit_left', 'exit_right'),
    'hollow_container': ('opening_down', 'tilted_side', 'diagonal_shell'),
}
CONTROL = yaml.safe_load((ASSETS/'control.yaml').read_text())
JOINT_NAMES = tuple(x.get('joint') for x in ET.parse(ASSETS/'source.xml').getroot().find('actuator'))
VERSION = 'overhead_clutter_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direction_of(model, data):
    gravity = -data.xmat[model.body('pelvis').id].reshape(3, 3)[2]
    if abs(gravity[0]) >= abs(gravity[1]):
        return 1 if gravity[0] > 0 else 0
    return 2 if gravity[1] > 0 else 3


def parts_body(spec, name, parts, mass, xy, rotation, dynamic=True):
    # Distinct staging heights avoid object-object intersections during robot fitting.
    body = spec.worldbody.add_body(name=name, pos=(*xy, 3.+len(spec.bodies)), quat=rotation)
    if dynamic:
        body.add_freejoint(name=name+'_joint')
    volumes = np.array([np.prod(p['half']) for p in parts])
    for i, p in enumerate(parts):
        box(body, f'{name}_{i}', p['half'], p['pos'],
            mass=mass*volumes[i]/volumes.sum() if dynamic else None,
            color=p.get('color', (.64, .39, .17, 1)))
    return {'name': name, 'motion': 'free' if dynamic else 'fixed',
            'geoms': [f'{name}_{i}' for i in range(len(parts))],
            'mass_kg': mass if dynamic else None, 'parts': parts,
            'initial_xy': list(xy), 'initial_quat': list(rotation)}


def part(half, pos=(0., 0., 0.), color=None):
    p = {'half': list(half), 'pos': list(pos)}
    if color is not None:
        p['color'] = color
    return p


def quaternion(yaw=0., pitch=0., roll=0.):
    return Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()[[3, 0, 1, 2]].tolist()


def scene_parameters(family, layout, seed, split):
    rng = np.random.default_rng(seed)
    jitter = .035 if split == 'development' else .12
    return {'family': family, 'layout': LAYOUTS[family][layout], 'layout_index': layout,
            'geometry_seed': int(seed), 'split': split,
            'xy': rng.uniform(-jitter, jitter, 2).tolist(),
            'yaw_jitter': float(rng.uniform(-.10, .10)),
            'length_scale': float(rng.uniform(.94, 1.06) if split == 'development' else rng.uniform(.85, 1.15)),
            'friction': float(rng.uniform(.8, 1.) if split == 'development' else rng.uniform(.6, 1.2))}


def make_scene(params):
    family = params['family']; layout = params['layout_index']
    rng = np.random.default_rng(params['geometry_seed']+91)
    spec, base = build_spec('flat')
    # Native broadphase is not used as a shortcut in collision certification.
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    spec.option.iterations = 100
    spec.visual.global_.offwidth = 960; spec.visual.global_.offheight = 720
    objects = []; supports = ['floor']; barriers = []; floor_support = []
    scale = params['length_scale']; xy = np.array(params['xy']); yj = params['yaw_jitter']
    if family == 'crossed_timber':
        count = layout+2
        angles = [0., 1.28, -.7, .55]
        for i in range(count):
            # Alternate narrow board and batten; all are ABOVE the robot.
            length = (1.0 if i%2 == 0 else .85)*scale
            width = (.18 if i == 0 else .065)*rng.uniform(.85, 1.15)
            thick = rng.uniform(.025, .045)
            offset = xy + np.array([(-.10, .12, -.16, .18)[i], (.04, -.05, .08, -.10)[i]])
            color = (.68+.035*i, .43+.025*i, .21+.02*i, 1)
            objects.append(parts_body(spec, f'timber_{i}', [part((length/2, width/2, thick/2), color=color)],
                                      float(rng.uniform(.7, 1.5) if i else rng.uniform(1.2, 2.5)),
                                      offset, quaternion(angles[i]+yj)))
        # Separate predefined role: a minority of cases also have low support.
        if params.get('with_lower_support', False):
            name = 'lower_support'
            box(spec.worldbody, name, (.14, .13, .035), (0., .05, .035), color=(.40, .48, .31, 1))
            supports.append(name); floor_support.append(name)
    elif family == 'ladder_boards':
        length = 1.05*scale; width = .52*scale; beam = .045; thick = .055
        pieces = [part((length/2, beam/2, thick/2), (0., s*(width-beam)/2, 0.), (.69, .48, .24, 1)) for s in (-1, 1)]
        for x in np.linspace(-length/2+.11, length/2-.11, 4):
            pieces.append(part((.025, (width-2*beam)/2, thick/2), (float(x), 0., 0.), (.75, .54, .28, 1)))
        yaw = (0., .70, -1.0)[layout]+yj
        objects.append(parts_body(spec, 'ladder', pieces, float(rng.uniform(2., 3.5)), xy, quaternion(yaw)))
        for i in range(2 if layout == 2 else 1):
            objects.append(parts_body(spec, f'ladder_board_{i}', [part((.42*scale, .12 if i == 0 else .05, .017), color=(.48, .28, .12, 1))],
                                      float(rng.uniform(.8, 1.5)), xy+np.array([.10*(-1)**i, .08*i]), quaternion(yaw+1.15+i*.7)))
    elif family == 'fixed_c_space':
        length = 1.0*scale; width = .76*scale
        height = float(rng.uniform(.65, .75) if params['split'] == 'development' else rng.uniform(.55, .80))
        thick = .045; wall = .045
        yaw = (0., math.pi/2, -math.pi/2)[layout]+yj
        body = spec.worldbody.add_body(name='c_frame', pos=(*xy, 0.), quat=quaternion(yaw))
        pieces = [part((length/2, width/2, thick/2), (0., 0., height+thick/2), (.56, .36, .19, 1)),
                  part((wall/2, width/2, (height-thick)/2), (-length/2+wall/2, 0., (height+thick)/2), (.49, .30, .15, 1)),
                  part((length/2, width/2, thick/2), (0., 0., thick/2), (.43, .27, .14, 1))]
        names = []
        for i, p in enumerate(pieces):
            name = f'c_frame_{i}'; box(body, name, p['half'], p['pos'], color=p['color']); names.append(name)
        objects.append({'name': 'c_frame', 'motion': 'fixed', 'geoms': names,
                        'overhead_geoms': [names[0]], 'mass_kg': None, 'parts': pieces,
                        'clear_height_m': height-thick, 'table_bottom_z_m': height})
        supports.append(names[2]); barriers.append(names[1]); floor_support.append(names[2])
    elif family == 'hollow_container':
        length = .80*scale; width = .61*scale; height = .43*scale; shell = .014
        # Opening faces local -Z. Five non-overlapping walls, no solid interior.
        color = (.23, .60, .72, .87)
        pieces = [part((length/2, width/2, shell/2), (0., 0., (height-shell)/2), color)]
        for s in (-1, 1):
            pieces.append(part((shell/2, width/2, (height-shell)/2), (s*(length-shell)/2, 0., -shell/2), color))
            pieces.append(part(((length-2*shell)/2, shell/2, (height-shell)/2), (0., s*(width-shell)/2, -shell/2), color))
        pitch = (0., .30, -.22)[layout]; roll = (0., 0., .20)[layout]
        objects.append(parts_body(spec, 'container', pieces, float(rng.uniform(1.4, 3.2)), xy,
                                  quaternion((0., .4, 1.1)[layout]+yj, pitch, roll)))
    else:
        raise ValueError(family)
    for obj in objects:
        for name in obj['geoms']:
            spec.geom(name).friction = (params['friction'], .01, .001)
    return spec, {**params, 'objects': objects, 'supports': supports, 'barriers': barriers,
                  'lower_support_geoms': floor_support, 'robot_source': base['robot_source'],
                  'scope': 'collision-screened fixtures; not measured recovery performance'}


def penetrating(model, data, aset=None, bset=None, tol=0.):
    return any(c.dist < tol and (aset is None or ((int(c.geom1) in aset and (bset is None or int(c.geom2) in bset)) or
                                                 (int(c.geom2) in aset and (bset is None or int(c.geom1) in bset)))) for c in data.contact)


def min_depth(data):
    return min([float(c.dist) for c in data.contact]+[0.])


def lower_to_contact(model, data, adr, moving, targets, start, bottom=-.2):
    """Find FIRST contact in downward sweep; avoid a non-monotone broad bisection."""
    prev = start
    for z in np.arange(start, bottom-.005, -.01):
        data.qpos[adr] = z; mujoco.mj_forward(model, data)
        if penetrating(model, data, moving, targets):
            lo, hi = z, prev
            for _ in range(13):
                mid = (lo+hi)/2; data.qpos[adr] = mid; mujoco.mj_forward(model, data)
                if penetrating(model, data, moving, targets): lo = mid
                else: hi = mid
            data.qpos[adr] = hi+.001; mujoco.mj_forward(model, data)
            return True
        prev = z
    return False


def lift_obstruction(model, data, meta, maximum=.65):
    """Rigid upward translation test against INDIVIDUAL collision parts.
    A collision in this probe does NOT prove no articulated escape exists.
    """
    rd = mujoco.MjData(model); rd.qpos[:] = data.qpos
    rids = set(robot_geoms(model)); found = {}; original = data.qpos[2]
    names = {o['name']: {model.geom(g).id for g in o.get('overhead_geoms', o['geoms'])} for o in meta['objects']}
    for dz in np.arange(.01, maximum+.005, .01):
        rd.qpos[2] = original+dz; mujoco.mj_forward(model, rd)
        for name, ids in names.items():
            if name not in found and penetrating(model, rd, rids, ids, -.0005):found[name] = round(float(dz), 3)
        if len(found) == len(names):break
    return {'probe': 'rigid_vertical_translation_not_articulated_reachability', 'maximum_m': maximum,
            'first_collision_lift_m': found, 'all_objects_intersect_lift': len(found) == len(names)}


def state_digest(q):
    # Ignore global translation; equal underlying robot configurations stay together.
    return hashlib.sha256(np.round(np.asarray(q)[3:36], 6).tobytes()).hexdigest()


def source_pools(bank, direction, source, split):
    candidates = np.flatnonzero((bank['direction'] == direction)&(bank['stratum'] == 0)&(bank['source'] == source))
    groups = {}
    for i in candidates: groups.setdefault(state_digest(bank['qpos'][i]), []).append(int(i))
    ordered = sorted(groups, key=lambda k: hashlib.sha256(('split-v1'+k).encode()).hexdigest())
    cutoff = max(1, len(ordered)//4)
    chosen = ordered[:cutoff] if split == 'development' else ordered[cutoff:]
    if not chosen: raise ValueError('Not enough distinct reset states for disjoint split')
    return np.array([i for k in chosen for i in groups[k]], dtype=int)


def initialize(params, bank, direction, source, seed, max_attempts=60):
    spec, meta = make_scene(params); model = spec.compile(); data = mujoco.MjData(model)
    rids = set(robot_geoms(model)); supports = {model.geom(n).id for n in meta['supports']}
    pool = source_pools(bank, direction, source, params['split'])
    rng = np.random.default_rng(seed)
    for attempt in range(max_attempts):
        ix = int(pool[rng.integers(len(pool))]); mujoco.mj_resetData(model, data)
        q = bank['qpos'][ix].copy(); q[:2] = 0.
        # Small reproducible placement perturbation; keep semantic side labels intact.
        q[:2] += rng.uniform(-.025, .025, 2)
        data.qpos[:36] = q
        if params['family'] == 'fixed_c_space':
            # Fit against the back wall along its local outward direction. This
            # changes position, never pose labels, ceiling height, or geometry.
            mujoco.mj_forward(model, data)
            bid = model.body('c_frame').id; rot = data.xmat[bid].reshape(3, 3)
            rp, re = bounds(model, data, list(rids))
            local = (rp-data.xpos[bid])@rot; ext = re@np.abs(rot)
            xmin = float((local[:,0]-ext[:,0]).min())
            wall = meta['objects'][0]['parts'][1]
            inner = wall['pos'][0]+wall['half'][0]
            data.qpos[:2] += max(0., inner+.006-xmin)*rot[:2,0]
        if not lower_to_contact(model, data, 2, rids, supports, 1.2):continue
        if min_depth(data) < -.002:continue
        if direction_of(model, data) != direction:continue
        # Place each independent body onto robot/prior objects without mesh mutation.
        placed = set(supports)|rids
        for obj in meta['objects']:
            gids = {model.geom(n).id for n in obj['geoms']}
            if obj['motion'] == 'free':
                bid = model.body(obj['name']).id; j = model.body_jntadr[bid]; adr = int(model.jnt_qposadr[j])
                ok = lower_to_contact(model, data, adr+2, gids, placed, 1.7)
                if not ok:break
                placed |= gids
        else:
            mujoco.mj_forward(model, data)
            if min_depth(data) < -.002:continue
            probe = lift_obstruction(model, data, meta)
            if not probe['all_objects_intersect_lift']:continue
            limited = [model.joint(n).id for n in JOINT_NAMES if model.joint(n).limited]
            if any(not (model.jnt_range[j, 0]-.005 <= data.qpos[model.jnt_qposadr[j]] <= model.jnt_range[j, 1]+.005) for j in limited):continue
            meta.update({'direction': DIRECTIONS[direction], 'source': 'natural' if source == 0 else 'procedural',
                         'source_index': ix, 'source_pose_sha256': state_digest(bank['qpos'][ix]),
                         'reset_seed': int(seed), 'attempts': attempt+1,
                         'initial_min_contact_distance_m': min_depth(data), 'obstruction_probe': probe,
                         'reset_status': 'collision_screened_zero_velocity_no_policy',
                         'qpos': data.qpos.tolist()})
            data.qvel[:] = 0.; data.ctrl[:] = 0.
            return spec, model, data, meta
    raise RuntimeError(f"No acceptable reset: {params['family']}/{params['layout']}/{DIRECTIONS[direction]}/{source}")


def passive_check(model, data, meta, duration=.1):
    """Short zero-control sanity check from a copy; never alters certified reset."""
    test = mujoco.MjData(model); test.qpos[:] = data.qpos; mujoco.mj_forward(model, test)
    start = test.qpos.copy(); worst = min_depth(test)
    for _ in range(round(duration/model.opt.timestep)):
        mujoco.mj_step(model, test); worst = min(worst, min_depth(test))
    result = {'duration_s': duration, 'controls': 'zero', 'finite': bool(np.isfinite(test.qpos).all() and np.isfinite(test.qvel).all()),
              'minimum_contact_distance_m': worst, 'max_generalized_speed': float(abs(test.qvel).max()),
              'not_a_stable_reset_or_policy_test': True}
    for obj in meta['objects']:
        bid = model.body(obj['name']).id
        if obj['motion'] == 'fixed':
            assert model.body_jntnum[bid] == 0
        else:
            assert model.body_jntnum[bid] == 1 and model.jnt_type[model.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE
            # MuJoCo XML writer rounds decimal parameters; audit relative tolerance.
            assert np.isclose(model.body_mass[bid], obj['mass_kg'], rtol=1e-5, atol=1e-6)
            assert (model.body_inertia[bid] > 0).all()
    return result


def render(model, data, path, title='', azimuth=135):
    from PIL import Image, ImageDraw, ImageFont
    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera();cam.lookat[:] = [.02, 0., .40];cam.distance = 2.75;cam.azimuth = azimuth;cam.elevation = -32
    renderer.update_scene(data, cam);im = Image.fromarray(renderer.render());renderer.close()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    dr = ImageDraw.Draw(im);dr.rectangle((0, 0, 960, 48), fill='#172532');dr.text((14, 13), title, fill='white', font=font)
    im.save(path)


def export_case(spec, model, data, meta, folder, preview=False):
    folder = Path(folder);folder.mkdir(parents=True, exist_ok=False)
    spec.add_key(name='reset', qpos=data.qpos, qvel=np.zeros(model.nv), ctrl=np.zeros(model.nu))
    xml = folder/'scene.xml'
    root = ET.fromstring(spec.to_xml())
    root.find('compiler').set('meshdir', '../../assets/meshes')
    xml.write_text(ET.tostring(root, encoding='unicode'))
    np.savez_compressed(folder/'reset.npz', qpos=data.qpos, qvel=data.qvel)
    m = mujoco.MjModel.from_xml_path(str(xml));d = mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m, d, m.key('reset').id);mujoco.mj_forward(m, d)
    assert np.allclose(data.qpos, d.qpos, atol=1e-5, rtol=0.)
    assert min_depth(d) >= -.0021
    assert lift_obstruction(m, d, meta)['all_objects_intersect_lift']
    meta['roundtrip_verified'] = True; meta['model_sha256'] = sha(xml)
    meta['xml_roundtrip_qpos_max_error'] = float(abs(data.qpos-d.qpos).max())
    meta['exported_body_masses_kg'] = {o['name']: float(m.body_mass[m.body(o['name']).id]) for o in meta['objects'] if o['motion']=='free'}
    meta['reset_sha256'] = sha(folder/'reset.npz')
    meta['aperture_check'] = aperture_check(m, d, meta)
    meta['passive_check'] = passive_check(m, d, meta)
    assert meta['passive_check']['finite']
    assert meta['passive_check']['minimum_contact_distance_m'] > -.03
    (folder/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    if preview:render(m, d, folder/'preview.png', f"{meta['family']} | {meta['layout']} | {meta['direction']} | initial state")
    return meta


def load_case(folder):
    folder = Path(folder);meta = json.loads((folder/'manifest.json').read_text())
    if sha(folder/'scene.xml') != meta['model_sha256']:raise ValueError('Scene hash mismatch')
    model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('reset').id);mujoco.mj_forward(model, data)
    return model, data, meta


def aperture_check(model, data, meta):
    """Ray tests certify hollow/frame openings were not filled by a collider."""
    group_backup = model.geom_group.copy()
    model.geom_group[:] = 0
    result = {}
    try:
        for obj in meta['objects']:
            name = obj['name']
            if name not in ('container', 'ladder', 'c_frame'): continue
            model.geom_group[:] = 0
            for g in obj['geoms']: model.geom_group[model.geom(g).id] = 1
            bid = model.body(name).id; rot = data.xmat[bid].reshape(3, 3)
            def ray(local, direction):
                hit = np.array([-1], dtype=np.int32)
                return float(mujoco.mj_ray(model, data, data.xpos[bid]+rot@np.array(local),
                             rot@np.array(direction, dtype=float), np.array([0,1,0,0,0,0], dtype=np.uint8), True, -1, hit))
            if name == 'container':
                empty = ray([0,0,0], [0,0,-1]); wall = ray([0,0,0], [0,0,1])
            elif name == 'ladder':
                x = (obj['parts'][2]['pos'][0]+obj['parts'][3]['pos'][0])/2
                empty = ray([x,0,.15], [0,0,-1]); wall = ray([0,obj['parts'][0]['pos'][1],.15], [0,0,-1])
            else:
                z = obj['table_bottom_z_m']/2
                empty = ray([0,0,z], [1,0,0]); wall = ray([0,0,z], [-1,0,0])
            assert empty == -1. and wall > 0., (name,empty,wall)
            result[name] = {'opening_ray_clear': True, 'wall_ray_distance_m': wall}
    finally:
        model.geom_group[:] = group_backup
    return result
