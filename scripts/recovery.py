"""Single read-only navigation and scene-fixture entry point (no training launch)."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from smp.recovery.scenes import ROOT,catalog,initialize,DIRECTIONS

p=argparse.ArgumentParser(description=__doc__)
s=p.add_subparsers(dest='command',required=True)
s.add_parser('list')
f=s.add_parser('find');f.add_argument('query')
b=s.add_parser('build');b.add_argument('scene',choices=list(catalog()));b.add_argument('--direction',choices=DIRECTIONS,default='prone');b.add_argument('--site',default='center');b.add_argument('--seed',type=int,default=20260923);b.add_argument('--overrides',type=Path);b.add_argument('--bank',type=Path,default=ROOT/'datasets/recovery_benchmark/validation.npz');b.add_argument('--out',type=Path,required=True);b.add_argument('--render',action='store_true')
a=p.parse_args()
if a.command=='list':
 print('SCENES');print('\n'.join(f'{k}: {v["kind"]}' for k,v in catalog().items()))
 print('\nEXPERIMENTS')
 for e in json.loads((ROOT/'configs/recovery_study/experiments.json').read_text())['experiments']:print(f"{e['id']:20} {e['role']} | {', '.join(e['scenes'])}")
 raise SystemExit
if a.command=='find':
 reg=json.loads((ROOT/'configs/recovery_study/experiments.json').read_text())
 found=[x for x in reg['experiments'] if a.query.lower() in json.dumps(x,ensure_ascii=False).lower()]
 docs=json.loads((ROOT/'configs/recovery_study/documents.json').read_text())['documents']
 matches=[x for x in docs if a.query.lower() in (x['title']+' '+' '.join(x['filenames'])).lower()]
 print(json.dumps({'experiments':found,'historical_documents':matches},ensure_ascii=False,indent=2));raise SystemExit
kwargs=json.loads(a.overrides.read_text()) if a.overrides else None
spec,m,d,meta=initialize(a.scene,a.bank,DIRECTIONS.index(a.direction),a.site,a.seed,kwargs)
a.out.mkdir(parents=True,exist_ok=False)
for obj in meta['objects']:
 if obj['motion']=='slide_z':spec.body(obj['name']).pos=m.body_pos[m.body(obj['name']).id]
spec.add_key(name='reset',qpos=d.qpos,qvel=np.zeros(m.nv),ctrl=np.zeros(m.nu))
(a.out/'scene.xml').write_text(spec.to_xml())
meta['bank_sha256']=hashlib.sha256(a.bank.read_bytes()).hexdigest()
meta['robot_sha256']=hashlib.sha256(Path(meta['robot_source']).read_bytes()).hexdigest()
meta['spec_sha256']=hashlib.sha256((a.out/'scene.xml').read_bytes()).hexdigest()
meta['config_sha256']=hashlib.sha256((ROOT/'configs/recovery_study/scenes.json').read_bytes()).hexdigest()
(a.out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
np.savez_compressed(a.out/'reset.npz',qpos=d.qpos,qvel=d.qvel)
# Verify the exported keyframe restores exactly the validated pose.
loaded=mujoco.MjModel.from_xml_path(str(a.out/'scene.xml'));ld=mujoco.MjData(loaded);mujoco.mj_resetDataKeyframe(loaded,ld,0);mujoco.mj_forward(loaded,ld)
assert np.allclose(ld.qpos,d.qpos,atol=1e-6)
assert min([float(c.dist) for c in ld.contact]+[0])>=-.0021
if a.render:
 from PIL import Image
 renderer=mujoco.Renderer(m,height=720,width=960)
 camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.45];camera.distance=3.7;camera.azimuth=130;camera.elevation=-27
 renderer.update_scene(d,camera);Image.fromarray(renderer.render()).save(a.out/'preview.png');renderer.close()
print(json.dumps({'scene':a.scene,'out':str(a.out),'status':meta['reset_status'],'depth':meta['initial_min_contact_distance']}))
