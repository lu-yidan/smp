"""Posthoc 50Hz collision audit of saved trajectories; never changes scores."""
import argparse,json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import mujoco
from smp.recovery.clutter_benchmark import load_case
from smp.recovery.scenes import robot_geoms


def check(job):
 policy,path,case=job;f=np.load(path);m,d,meta=load_case(case);robot=set(robot_geoms(m));worst=0.;pair=None;count=0;first=None;time=0.
 for i,s in enumerate(f['state']):
  d.qpos[:]=s[:m.nq];mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
  depth=0.
  for c in d.contact:
   a,b=int(c.geom1),int(c.geom2)
   if (a in robot)==(b in robot):continue
   depth=min(depth,float(c.dist))
   if c.dist<worst:worst=float(c.dist);pair=[m.geom(a).name,m.geom(b).name];time=(i+1)*.02
  if depth<-.02:
   count+=1
   if first is None:first=(i+1)*.02
 return policy,{'case_id':meta['case_id'],'family':meta['family'],'minimum_external_robot_contact_distance_m':worst,'worst_pair':pair,'worst_time_s':time,'sampled_frames_penetration_gt20mm':count,'first_gt20mm_s':first}


def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args();jobs=[];rows={}
 for root in a.runs:
  for policy in json.loads((root/'protocol.json').read_text())['policies']:
   rows[policy]=[]
   for file in (root/policy).glob('*.npz'):
    case=(a.calibration if file.stem.startswith('flat_calibration') else a.root)/file.stem
    jobs.append((policy,str(file),str(case)))
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
  for n,f in enumerate(as_completed([pool.submit(check,j) for j in jobs]),1):
   p,r=f.result();rows[p].append(r)
   if n%100==0:print(n,len(jobs),flush=True)
 summary={p:{'n':len(rs),'trials_with_sampled_penetration_gt20mm':sum(x['sampled_frames_penetration_gt20mm']>0 for x in rs),'worst_depth_m':min(x['minimum_external_robot_contact_distance_m'] for x in rs)} for p,rs in rows.items()}
 a.out.write_text(json.dumps({'scope':'50Hz saved states, external robot contacts; not continuous 500Hz penetration certification; no success-score changes','summary':summary,'per_trial':rows},indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
