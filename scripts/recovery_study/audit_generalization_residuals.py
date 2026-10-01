"""Explain lift-probe failures after sustained quiet standing; no outcome relabeling."""
import argparse,json
from pathlib import Path
from collections import Counter
import mujoco
import numpy as np
from smp.recovery.clutter_benchmark import load_case
from smp.recovery.clutter_rollout import GeometryMetrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows=[]
 for p in sorted((a.results/'review/A6_no_alpha_9999').glob('*.json')):
  r=json.loads(p.read_text())
  if not r['quiet_stable_10s'] or r['success_10s']:continue
  cid=r['case_id'];m,d,meta=load_case(a.root/cid)
  holds=np.load(p.with_suffix('.npz'))['holds'];ix=int(np.argmax(holds[:,0]))
  with np.load(a.results/'run/A6_no_alpha_9999'/(cid+'.npz')) as f:
   s=f['state'][ix];d.qpos[:]=s[:m.nq];d.qvel[:]=s[m.nq:];d.ctrl[:]=f['tau'][(ix+1)*10-1]
  mujoco.mj_forward(m,d);gm=GeometryMetrics(m,meta);cond,v=gm.measure(d);blocked=gm.lift_blocked(d,v['head_height']);pairs=set();parts=set()
  if blocked:
   for c in gm.probe.contact:
    x,y=int(c.geom1),int(c.geom2)
    if c.dist<-.0005 and ((x in gm.robot and y in gm.object_geoms) or (y in gm.robot and x in gm.object_geoms)):
     robot=x if x in gm.robot else y;obj=y if x in gm.robot else x;rn=m.geom(robot).name
     parts.add('foot' if '_foot' in rn else 'hand' if '_hand' in rn else 'other_body')
     pairs.add((rn,m.geom(obj).name))
  rows.append({'case_id':cid,'family':r['family'],'sample_t_s':(ix+1)*.02,'quiet_hold_s':float(holds[ix,0]),'probe_blocked':blocked,'parts':sorted(parts),'pairs':sorted(pairs),'unchanged_E10':False,'Q10':True})
 result={'selection':'all Q10 true/E10 false; longest quiet-hold frame, no resampling or relabeling','n':len(rows),'part_categories':dict(Counter('+'.join(r['parts']) or 'unblocked_at_sample' for r in rows)), 'families':dict(Counter(r['family'] for r in rows)), 'rows':rows}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
