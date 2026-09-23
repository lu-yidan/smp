"""Paired native-CPU development trials. Refuses heldout and existing run folders."""
import argparse,json,os,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from smp.recovery.clutter_rollout import actor_from_checkpoint,run_case
from smp.recovery.clutter_benchmark import sha,CONTROL

ACTORS={}
def worker(job):
    profile,checkpoint,case,out=job
    if checkpoint not in ACTORS:ACTORS[checkpoint]=actor_from_checkpoint(checkpoint)[0]
    return profile,run_case(Path(case),ACTORS[checkpoint],out)


def aggregate(rows):
    out={'n':len(rows),'stable_1s':sum(r['success_1s'] for r in rows),'stable_10s':sum(r['success_10s'] for r in rows),
         'numerical_failures':sum(r['unsafe'] is not None for r in rows),
         'upright_reached':sum(r['first_upright_s'] is not None for r in rows),
         'direct_lift_clear_reached':sum(r['first_direct_lift_clear_s'] is not None for r in rows),
         'refall_after_stable_1s':sum(r['refall_after_stable_1s'] for r in rows)}
    for key in ['peak_tau_nm','peak_dq_rad_s','peak_power_w','above_90pct_torque_s']:
        values=[max(r[key]) for r in rows]
        out[key+'_trial_max_p95']=float(np.percentile(values,95))
        out[key+'_max']=float(max(values))
    for key in ['post_clear_path_m','post_clear_max_drift_m','first_direct_lift_clear_s','first_upright_s']:
        values=[r[key] for r in rows if r[key] is not None]
        out[key+'_mean_reached_only']=float(np.mean(values)) if values else None
        out[key+'_denominator']=len(values)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--policies',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--calibration',type=Path);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();root=a.root.resolve();assert root.name=='development','Heldout is intentionally not enabled'
    index=json.loads((root/'index.json').read_text());policies=json.loads(a.policies.read_text());a.out.mkdir(parents=True,exist_ok=False)
    jobs=[];selection={}
    for name,info in policies.items():
        ckpt=Path(info['checkpoint']).resolve();actor,meta=actor_from_checkpoint(ckpt);del actor
        assert meta['checkpoint_sha256']==info['sha256'];selection[name]={**info,**meta}
        for row in index['records']:jobs.append((name,str(ckpt),str(root/row['case_id']),str(a.out/name)))
        if a.calibration:
            for case in sorted(a.calibration.glob('*/manifest.json')):jobs.append((name,str(ckpt),str(case.parent.resolve()),str(a.out/name)))
    manifest={'policies':selection,'development_index_sha256':sha(root/'index.json'),'seconds':20,'physics_dt':.002,'control_dt':.02,
              'warmup_steps':CONTROL['warmup_steps'],'entry_hold_s':.02,'controller':'93D native checkpoint, deployment target limiting at 50Hz, saturated PD at 500Hz',
              'robot_dynamics':'nominal','push':False,'observation_noise':False,'training_changed':False,
              'success':'continuous stable stand AND direct rigid-lift clearance; prototype geometric diagnostic, not reachability',
              'lift_probe_hz':10,'lift_probe_dz_m':.01,'clear_hold_s':.3,'head_height_reference':'mean loaded foot contact support height',
              'load_metrics':'actual actuator torque and joint velocity at 2ms; mechanical abs(tau*dq); p95 over trial maxima incl failures',
              'code_sha256':sha(Path(__file__).resolve()),'rollout_code_sha256':sha(Path(__file__).resolve().parents[2]/'src/smp/recovery/clutter_rollout.py'),
              'cases_per_policy':len(jobs)//len(policies),'scope':'development, not heldout or hardware evaluation'}
    (a.out/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results={k:[] for k in policies}
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for n,f in enumerate(as_completed(futures),1):
            profile,r=f.result();results[profile].append(r)
            print(n,'/',len(jobs),profile,r['case_id'],'hold',round(r['best_hold_s'],2),'unsafe',r['unsafe'],flush=True)
            (a.out/'progress.json').write_text(json.dumps({'completed':n,'total':len(jobs),'last':r['case_id']}))
    summary={}
    for profile,rows in results.items():
        mainrows=[r for r in rows if r['family']!='flat_calibration']
        summary[profile]={'all_clutter':aggregate(mainrows),
                          'families':{k:aggregate([r for r in rows if r['family']==k]) for k in sorted({r['family'] for r in rows})},
                          'directions':{k:aggregate([r for r in mainrows if r['direction']==k]) for k in sorted({r['direction'] for r in mainrows})}}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE',a.out,flush=True)

if __name__=='__main__':main()
