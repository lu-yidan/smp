"""Independent reload/geometry tests for the overhead evaluation collection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import hashlib
import mujoco
import numpy as np
from smp.recovery.clutter_benchmark import (ROOT, FAMILIES, DIRECTIONS, load_case,
    aperture_check, lift_obstruction, min_depth, direction_of, sha)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();summary={};source_sets={};seed_sets={};physical_sets={}
    for split in ('development','heldout'):
        root=a.root/split;index=json.loads((root/'index.json').read_text());records=index['records']
        assert index['count']==len(records)
        assert len(set(r['case_id'] for r in records))==len(records)
        byfamily=Counter();bydir=Counter();sources=Counter();depths=[];passive=[];masserr=[];lower=Counter();objects=Counter();physical=set()
        for record in records:
            case=root/record['case_id'];assert sha(case/'manifest.json')==record['manifest_sha256']
            m,d,meta=load_case(case);assert sha(case/'reset.npz')==meta['reset_sha256']
            assert meta['family']==record['family'] and meta['direction']==record['direction']
            assert min_depth(d)>=-.0021
            assert direction_of(m,d)==DIRECTIONS.index(meta['direction'])
            assert np.max(abs(d.qvel))==0
            assert lift_obstruction(m,d,meta)['all_objects_intersect_lift']
            aperture_check(m,d,meta)
            original=d.qpos.copy()
            # A separated robot must not be declared blocked by the original objects.
            d.qpos[0]+=4.;mujoco.mj_forward(m,d)
            assert not lift_obstruction(m,d,meta)['first_collision_lift_m']
            # Deliberate deep floor penetration must be seen by collision auditing.
            d.qpos[:]=original;d.qpos[2]-=.08;mujoco.mj_forward(m,d)
            assert min_depth(d)<-.01
            d.qpos[:]=original;mujoco.mj_forward(m,d)
            for obj in meta['objects']:
                bid=m.body(obj['name']).id
                if obj['motion']=='fixed':assert m.body_jntnum[bid]==0
                else:
                    assert m.jnt_type[m.body_jntadr[bid]]==mujoco.mjtJoint.mjJNT_FREE
                    assert np.isclose(m.body_mass[bid],obj['mass_kg'],rtol=1e-5,atol=1e-6)
                    assert (m.body_inertia[bid]>0).all()
                    masserr.append(abs(float(m.body_mass[bid])-obj['mass_kg']))
            assert meta['passive_check']['finite']
            byfamily[meta['family']]+=1;bydir[meta['direction']]+=1;sources[meta['source']]+=1
            depths.append(min_depth(d));passive.append(meta['passive_check']['minimum_contact_distance_m'])
            if meta['family']=='crossed_timber':lower[bool(meta['with_lower_support'])]+=1
            objects[f"{meta['family']}/{len(meta['objects'])}"]+=1
            # q and -q represent the same orientation; raw hashes alone miss that.
            pose=np.array(meta['qpos'][3:36]);pose[:4]/=np.linalg.norm(pose[:4])
            if pose[np.flatnonzero(abs(pose[:4])>1e-8)[0]]<0:pose[:4]*=-1
            physical.add(hashlib.sha256(np.round(pose,6).tobytes()).hexdigest())
        source_sets[split]={r['source_pose_sha256'] for r in records}
        physical_sets[split]=physical
        seed_sets[split]={json.loads((root/r['case_id']/'manifest.json').read_text())['geometry_seed'] for r in records}
        assert len(set(byfamily.values()))==1 and len(set(bydir.values()))==1
        summary[split]={'cases':len(records),'family_counts':dict(byfamily),'direction_counts':dict(bydir),
                        'sources':dict(sources),'distinct_source_poses':len(source_sets[split]),
                        'worst_initial_penetration_m':min(depths),'worst_100ms_passive_penetration_m':min(passive),
                        'max_xml_mass_rounding_kg':max(masserr),'timber_lower_support_counts':dict(lower),
                        'object_counts':dict(objects),'index_sha256':sha(root/'index.json')}
        print(split,'PASS',len(records),flush=True)
    assert not(source_sets['development']&source_sets['heldout'])
    assert not(seed_sets['development']&seed_sets['heldout'])
    assert not(physical_sets['development']&physical_sets['heldout'])
    result={'status':'passed','scope':'collision, directions, object DOF/inertia, hollow openings, negative counterexamples and split audit; not policy success',
            'source_pose_overlap':0,'geometry_seed_overlap':0,'mujoco_version':mujoco.__version__,
            'python_version':platform.python_version(),'splits':summary}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
