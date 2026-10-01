"""Paired geometry/dynamics sweep using preselected clutter source poses, no policy."""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from smp.recovery.scenes import ASSETS, build_spec, box, robot_geoms
from smp.recovery.clutter_benchmark import lower_to_contact, min_depth, lift_obstruction, sha

CONDITIONS = {
    'nominal': (1.,6.,1.), 'size075':(.75,6.,1.), 'size125':(1.25,6.,1.),
    'mass3':(1.,3.,1.), 'mass9':(1.,9.,1.), 'friction05':(1.,6.,.5), 'friction15':(1.,6.,1.5),
}

def build_one(job):
    source,out,motion,condition,original=job
    scale,mass,friction=CONDITIONS[condition]
    old=json.loads((source/original['case_id']/'manifest.json').read_text())
    q=np.load(source/original['case_id']/'reset.npz')['qpos'][:36]
    spec,_=build_spec('flat');spec.option.iterations=100
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    body=spec.worldbody.add_body(name='plate',pos=(0,0,0))
    if motion=='free':body.add_freejoint(name='plate_joint')
    else:body.add_joint(name='plate_joint',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=(0,0,1),limited=False,damping=2.)
    half=(.90*scale/2,.64*scale/2,.07/2)
    box(body,'plate_geom',half,(0,0,0),mass=mass)
    # Explicit pairs vary plate friction while preserving robot-ground contacts.
    spec.geom('plate_geom').friction=(friction,.01,.001)
    m=spec.compile();rids=set(robot_geoms(m))
    for rid in sorted(rids):
        name=m.geom(rid).name
        spec.add_pair(name='sweep_plate_'+name,geomname1='plate_geom',geomname2=name,
                      condim=3,friction=(friction,friction,.01,.001,.001),solref=(.01,1.))
    m=spec.compile();d=mujoco.MjData(m);d.qpos[:36]=q
    adr=int(m.jnt_qposadr[m.joint('plate_joint').id]);gids={m.geom('plate_geom').id}
    if motion=='free':d.qpos[adr:adr+7]=[0,0,3,1,0,0,0];zadr=adr+2
    else:d.qpos[adr]=3.;zadr=adr
    assert lower_to_contact(m,d,2,rids,{m.geom('floor').id},1.2)
    assert lower_to_contact(m,d,zadr,gids,rids|{m.geom('floor').id},2.)
    if motion=='guided':
        robotq=d.qpos[:36].copy();z=float(d.qpos[adr])
        body.pos=(0,0,z);spec.joint('plate_joint').limited=True;spec.joint('plate_joint').range=(-.3,.6)
        m=spec.compile();d=mujoco.MjData(m);d.qpos[:36]=robotq;d.qpos[adr]=0.
    mujoco.mj_forward(m,d)
    assert min_depth(d)>=-.0021
    family=motion+'_plate';cid=family+'__'+condition+'__'+original['case_id'].split('__',1)[1]
    meta={**old,'case_id':cid,'family':family,'layout':condition,'condition':condition,'paired_source_case':original['case_id'],
          'split':'heldout','objects':[{'name':'plate','motion':motion,'geoms':['plate_geom'],'mass_kg':mass}],
          'supports':['floor'],'barriers':[],'lower_support_geoms':[],
          'plate_dimensions_m':[.9*scale,.64*scale,.07],'plate_mass_kg':mass,'plate_robot_sliding_friction':friction,
          'initial_min_contact_distance_m':min_depth(d),'robot_ground_contacts_unchanged':True,
          'source_status':'same preselected source robot states across every condition; re-fitted vertically to flat ground',
          'qpos':d.qpos.tolist()}
    meta['obstruction_probe']=lift_obstruction(m,d,meta)
    assert meta['obstruction_probe']['all_objects_intersect_lift'],cid
    folder=out/cid;folder.mkdir()
    spec.add_key(name='reset',qpos=d.qpos,qvel=np.zeros(m.nv),ctrl=np.zeros(m.nu))
    xml=ET.fromstring(spec.to_xml());xml.find('compiler').set('meshdir','../../assets/meshes')
    (folder/'scene.xml').write_text(ET.tostring(xml,encoding='unicode'))
    np.savez_compressed(folder/'reset.npz',qpos=d.qpos,qvel=np.zeros(m.nv))
    meta['model_sha256']=sha(folder/'scene.xml');meta['reset_sha256']=sha(folder/'reset.npz')
    check=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));cd=mujoco.MjData(check);mujoco.mj_resetDataKeyframe(check,cd,0);mujoco.mj_forward(check,cd)
    assert np.allclose(cd.qpos,d.qpos,atol=1e-5,rtol=0) and min_depth(cd)>=-.0021
    assert np.allclose(check.pair_friction[-len(rids):,:2],friction)
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return {**original,'case_id':cid,'family':family,'layout':condition,'condition':condition,'manifest_sha256':sha(folder/'manifest.json')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    assets=a.out.parent/'assets';shutil.copytree(ASSETS/'meshes',assets/'meshes',dirs_exist_ok=True)
    index=json.loads((a.source/'index.json').read_text())
    bases=[r for r in index['records'] if r['family']=='crossed_timber' and r['variant']<8 and '__l2__' not in r['case_id']]
    assert len(bases)==64
    jobs=[(a.source,a.out,motion,condition,original) for motion in ('guided','free') for condition in CONDITIONS for original in bases]
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(build_one,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%64==0:print('BUILT',len(rows),'/',len(jobs),flush=True)
    result={'split':'heldout','count':len(rows),'records':rows,'source_index_sha256':sha(a.source/'index.json'),
            'source_status':'preselected poses from archived clutter heldout split; not guaranteed unseen in prior training',
            'conditions':CONDITIONS,'family_counts':dict(Counter(r['family'] for r in rows)),
            'scope':'paired one-factor sensitivity, native CPU controller; not the training-simulator plate benchmark',
            'builder_sha256':sha(__file__)}
    (a.out/'index.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
