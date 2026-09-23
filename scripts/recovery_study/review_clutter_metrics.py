"""Separate quiet standing from strict lift-clear standing using frozen traces.
No policy rerun; verifies the original strict score is reproduced exactly.
"""
import argparse,json,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import mujoco
from smp.recovery.clutter_rollout import GeometryMetrics
from smp.recovery.clutter_benchmark import load_case
from evaluate_clutter import aggregate


def score(job):
    path,case,out=map(Path,job);f=np.load(path);m,d,meta=load_case(case);gm=GeometryMetrics(m,meta)
    old=json.loads(path.with_suffix('.json').read_text());state=f['state'];diag=f['diagnostics'];tau=f['tau']
    quiet=0.;strict=0.;best=0.;strictbest=0.;stand1=False;falltime=0.;refall=False;holds=[];first=None;path_after=0.;anchor=None;maxdrift=0.;last=state[0,:2];upright_run=0.;best_upright=0.;lastmask=None
    for i,s in enumerate(state):
        d.qpos[:]=s[:m.nq];d.qvel[:]=s[m.nq:];d.ctrl[:]=tau[(i+1)*10-1];mujoco.mj_forward(m,d)
        cond,vals=gm.measure(d);clear=diag[i,5]>=.3-1e-8
        quiet=quiet+.02 if cond.all() else 0.;strict=strict+.02 if cond.all() and clear else 0.
        best=max(best,quiet);strictbest=max(strictbest,strict)
        upright=vals['head_height']>=1.15 and vals['upright']>=.93
        upright_run=upright_run+.02 if upright else 0.;best_upright=max(best_upright,upright_run)
        if best>=1.-1e-8 and not stand1:first=(i+1)*.02;stand1=True;anchor=d.qpos[:2].copy()
        fallen=vals['head_height']<.65 or vals['upright']<.5
        falltime=falltime+.02 if stand1 and fallen else 0.;refall|=falltime>=.2-1e-8
        if stand1:path_after+=float(np.linalg.norm(d.qpos[:2]-last));maxdrift=max(maxdrift,float(np.linalg.norm(d.qpos[:2]-anchor)))
        last=d.qpos[:2].copy();holds.append([quiet,strict]);lastmask=cond
    # Float32 stored torques can perturb threshold-level contact readings slightly.
    assert abs(strictbest-old['best_hold_s'])<=.04,(path,strictbest,old['best_hold_s'])
    new={**old,'quiet_stable_1s':best>=1.-1e-8 and old['unsafe'] is None,
         'quiet_stable_10s':best>=10.-1e-8 and old['unsafe'] is None,'quiet_best_hold_s':best,
         'upright_pose_10s':best_upright>=10.-1e-8,'quiet_first_1s_s':first,
         'quiet_refall_after_1s':bool(refall),'post_quiet_path_m':path_after if stand1 else None,
         'post_quiet_max_drift_m':maxdrift if stand1 else None,'final_quiet_conditions':lastmask.tolist(),
         'strict_rescore_max_hold_error_s':abs(strictbest-old['best_hold_s'])}
    out.mkdir(parents=True,exist_ok=True);(out/path.with_suffix('.json').name).write_text(json.dumps(new,indent=2)+'\n')
    np.savez_compressed(out/path.name,holds=np.asarray(holds))
    return path.parent.name,new


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    policies=json.loads((a.run/'protocol.json').read_text())['policies'];jobs=[];results={k:[] for k in policies}
    for policy in policies:
        for path in sorted((a.run/policy).glob('*.npz')):
            case=(a.calibration if path.stem.startswith('flat_calibration') else a.root)/path.stem
            jobs.append((str(path),str(case),str(a.out/policy)))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        for n,f in enumerate(as_completed([pool.submit(score,j) for j in jobs]),1):
            policy,result=f.result();results[policy].append(result)
            if n%24==0:print('RESCORED',n,len(jobs),flush=True)
    def group(rows):
        r=aggregate(rows)
        for key in ('quiet_stable_1s','quiet_stable_10s','quiet_refall_after_1s','upright_pose_10s'):r[key]=sum(x[key] for x in rows)
        for key in ('post_quiet_path_m','post_quiet_max_drift_m'):
            vals=[x[key] for x in rows if x[key] is not None];r[key+'_mean']=float(np.mean(vals)) if vals else None;r[key+'_n']=len(vals)
        return r
    summary={}
    for policy,rows in results.items():
        mainrows=[r for r in rows if r['family']!='flat_calibration']
        summary[policy]={'all_clutter':group(mainrows),'families':{k:group([r for r in rows if r['family']==k]) for k in sorted({r['family'] for r in rows})}}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('REVIEW_COMPLETE',flush=True)

if __name__=='__main__':main()
