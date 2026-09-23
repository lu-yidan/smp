"""Generate frozen CPU scene/reset cases, with no policy evaluation or training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import shutil
import mujoco
import numpy as np
from smp.recovery.clutter_benchmark import (ROOT, ASSETS, VERSION, FAMILIES, LAYOUTS, DIRECTIONS,
    scene_parameters, initialize, export_case, sha)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bank', type=Path, default=ROOT/'datasets/recovery_benchmark/validation.npz')
    p.add_argument('--split', choices=['development','heldout'], required=True)
    p.add_argument('--variants', type=int, default=10)
    p.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    p.add_argument('--render', action='store_true')
    a=p.parse_args();assert a.variants>0
    a.out.mkdir(parents=True,exist_ok=False)
    assets=a.out.parent/'assets';assets.mkdir(exist_ok=True)
    if not (assets/'meshes').exists():shutil.copytree(ASSETS/'meshes',assets/'meshes')
    for name in ('source.xml','control.yaml'):
        target=assets/name
        if target.exists():assert sha(target)==sha(ASSETS/name)
        else:shutil.copy2(ASSETS/name,target)
    bank=np.load(a.bank);rows=[];count=0
    seedbase=92310000 if a.split=='development' else 92320000
    for family in a.families:
        fi=FAMILIES.index(family)
        for direction in range(4):
            for layout in range(3):
                for variant in range(a.variants):
                    source=int(count%4==3)
                    seed=seedbase+fi*100000+layout*1000+variant
                    params=scene_parameters(family,layout,seed,a.split)
                    params['with_lower_support']=family=='crossed_timber' and ((layout*a.variants+variant)%10 in (3,6,9))
                    cid=f'{family}__{DIRECTIONS[direction]}__l{layout}__v{variant:02d}'
                    spec,m,d,meta=initialize(params,bank,direction,source,seed+direction*10000)
                    meta.update(case_id=cid,source_bank_sha256=sha(a.bank),source_bank=str(a.bank.resolve()),
                                robot_sha256=sha(ASSETS/'source.xml'),control_sha256=sha(ASSETS/'control.yaml'))
                    meta=export_case(spec,m,d,meta,a.out/cid,a.render and variant==0)
                    rows.append({'case_id':cid,'family':family,'direction':DIRECTIONS[direction],
                                 'layout':params['layout'],'variant':variant,'source':meta['source'],
                                 'source_pose_sha256':meta['source_pose_sha256'],
                                 'min_depth_m':meta['initial_min_contact_distance_m'],
                                 'passive_min_depth_m':meta['passive_check']['minimum_contact_distance_m'],
                                 'objects':len(meta['objects']),'with_lower_support':params['with_lower_support'],
                                 'manifest_sha256':sha(a.out/cid/'manifest.json')})
                    count+=1
            print(f'{family}/{DIRECTIONS[direction]} {count} cases',flush=True)
    manifest={'version':VERSION,'mujoco_version':mujoco.__version__,'split':a.split,'records':rows,'count':len(rows),'variants_per_layout_direction':a.variants,
              'family_counts':dict(Counter(r['family'] for r in rows)),
              'source_counts':dict(Counter(r['source'] for r in rows)),
              'source_bank_sha256':sha(a.bank),'builder_sha256':sha(__file__),
              'geometry_sha256':sha(ROOT/'src/smp/recovery/clutter_benchmark.py'),
              'commit_at_build':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_status':'existing development reset assets; heldout only from this benchmark development split, not prior-training data',
              'scope':'geometry/reset and short passive physics audited; no policy success, no escape reachability proof',
              'evaluation_protocol':{'rollout_s':20,'hold_s':[1,10],'failures_in_load_metrics':True,
              'primary':'stable recovery after release from actual constraints; new object success semantics require rollout integration',
              'nominal_dynamics':'robot source XML; obstacle friction/mass archived per case',
              'training_unchanged':True}}
    (a.out/'index.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'out':str(a.out),'count':len(rows),'index_sha256':sha(a.out/'index.json')}),flush=True)

if __name__=='__main__':main()
