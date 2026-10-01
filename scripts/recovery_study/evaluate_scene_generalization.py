"""Frozen-policy evaluation on archived clutter cases; no training or model selection.

The heldout split is disjoint from clutter development fixtures, not a claim
that source poses are disjoint from all prior training. Native MuJoCo CPU.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import yaml

from smp.recovery.clutter_benchmark import CONTROL, load_case, sha
from smp.recovery.clutter_rollout import actor_from_checkpoint, run_case, GeometryMetrics
from evaluate_clutter import aggregate
from review_clutter_metrics import score

EXPECTED_SHA = '24f1a4513cf5cabb9bd0b9ef54e90a3d50e625a87774878379af5da5dcf8fbd1'
ACTORS = {}


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def worker(job):
    checkpoint, case, out = map(Path, job)
    if str(checkpoint) not in ACTORS:
        ACTORS[str(checkpoint)] = actor_from_checkpoint(checkpoint)[0]
    result = run_case(case, ACTORS[str(checkpoint)], out / 'run' / 'A6_no_alpha_9999')
    npz = out / 'run' / 'A6_no_alpha_9999' / (result['case_id'] + '.npz')
    _, result = score((str(npz), str(case), str(out / 'review' / 'A6_no_alpha_9999')))
    with np.load(npz) as f:
        xy = f['state'][:, :2]
        times = f['diagnostics'][:, 0]
        for key, start in [('clear', result['first_direct_lift_clear_s']),
                           ('stand', result['quiet_first_1s_s'])]:
            value = None
            if start is not None and times[-1] >= start + 3 - 1e-7:
                ids = np.flatnonzero((times >= start - 1e-7) & (times <= start + 3 + 1e-7))
                value = float(np.linalg.norm(xy[ids] - xy[ids[0]], axis=1).max())
            result[f'post_{key}_3s_max_displacement_m'] = value
    save(out / 'review' / 'A6_no_alpha_9999' / (result['case_id'] + '.json'), result)
    return result


def wilson(k, n):
    z = 1.959963984540054
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [float(c-h), float(c+h)]


def summarize(rows):
    result = aggregate(rows)
    for field in ('quiet_stable_1s', 'quiet_stable_10s', 'success_1s', 'success_10s', 'quiet_refall_after_1s'):
        k = sum(r[field] for r in rows)
        result[field] = {'count': k, 'n': len(rows), 'rate': k/len(rows),
                         'wilson95_descriptive': wilson(k, len(rows))}
    for phase in ('clear', 'stand'):
        key = f'post_{phase}_3s_max_displacement_m'
        v = [r[key] for r in rows if r[key] is not None]
        result[key] = {'n':len(v), 'median':float(np.median(v)) if v else None,
                      'p95':float(np.percentile(v,95)) if v else None}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--calibration', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--parity-fixture', type=Path, required=True)
    ap.add_argument('--deploy-config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    assert 1 <= args.workers <= 24
    args.out.mkdir(parents=True, exist_ok=False)
    assert sha(args.checkpoint) == EXPECTED_SHA
    actor, actor_meta = actor_from_checkpoint(args.checkpoint)
    fixture = np.load(args.parity_fixture)
    with torch.inference_mode():
        err = float(np.abs(actor(torch.from_numpy(fixture['obs'])).numpy()-fixture['actions']).max())
    assert err < 1e-5, err
    deploy = yaml.safe_load(args.deploy_config.read_text())
    for key in ('default_joint_pos','action_scale','kps','kds','tau_limit','clip_actions','warmup_steps','control_dt'):
        assert np.array_equal(deploy[key], CONTROL[key]), key
    index = json.loads((args.root / 'index.json').read_text())
    assert index['split'] == 'heldout'
    records = index['records']
    # Certify all archived geometry/reset hashes before revealing any policy outcome.
    for row in records:
        case = args.root / row['case_id']
        assert sha(case/'manifest.json') == row['manifest_sha256']
        meta = json.loads((case/'manifest.json').read_text())
        assert sha(case/'scene.xml') == meta['model_sha256']
        assert sha(case/'reset.npz') == meta['reset_sha256']
    probes = []
    for family in sorted({r['family'] for r in records}):
        case = args.root / next(r['case_id'] for r in records if r['family']==family and r['direction']=='prone')
        m,d,meta = load_case(case); gm = GeometryMetrics(m,meta)
        initial = d.qpos.copy()
        assert gm.lift_blocked(d,0.) and np.array_equal(initial,d.qpos)
        d.qpos[0] += 4.; mujoco.mj_forward(m,d)
        assert not gm.lift_blocked(d,0.)
        probes.append(family)
    code_root = Path(__file__).resolve().parents[2]
    code = [Path(__file__).resolve(), code_root/'src/smp/recovery/clutter_rollout.py',
            code_root/'src/smp/recovery/clutter_benchmark.py', Path(__file__).with_name('review_clutter_metrics.py')]
    protocol = {
        'created_unix_s':time.time(), 'policy':actor_meta,
        'deployment_fixture_max_error':err, 'control_contract_equal':True,
        'mujoco_version':mujoco.__version__, 'split':index['split'],
        'index_sha256':sha(args.root/'index.json'), 'code_sha256':{p.name:sha(p) for p in code},
        'geometry_negative_controls_passed':probes,
        'scene_counts':dict(Counter(r['family'] for r in records)),
        'source_counts':dict(Counter(r['source'] for r in records)),
        'direction_counts':dict(Counter(r['direction'] for r in records)),
        'distinct_source_poses':len({r['source_pose_sha256'] for r in records}),
        'seconds':20, 'physics_dt':.002, 'control_dt':.02,
        'entry_hold_s':.02, 'warmup_steps':10, 'robot_dynamics':'nominal',
        'obstacle_parameters':'frozen per-case dimensions, mass, friction, placement',
        'push':False, 'observation_noise':False, 'training_changed':False,
        'success':'E10: quiet standing AND direct rigid-lift clearance for 10 continuous seconds; Q10 reports quiet standing alone',
        'clearance_semantics':'conservative rigid vertical lift diagnostic, not articulated reachability; differs from plate phase evaluation',
        'source_status':index['source_status'],
        'scope':'simulation generalization; no hardware substitution, no checkpoint selection; fixed C reported separately from movable obstacles',
        'confidence_interval_scope':'descriptive Wilson over trials; repeated source poses/layouts mean these are not independent training seeds',
        'preflight_only':args.preflight_only,
    }
    save(args.out/'protocol.json', protocol)
    del actor
    cases = sorted(p.parent for p in args.calibration.glob('*/manifest.json'))
    assert len(cases)==4
    if not args.preflight_only:
        cases += [args.root/r['case_id'] for r in records]
    jobs=[(str(args.checkpoint.resolve()),str(p.resolve()),str(args.out.resolve())) for p in cases]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(futures):
            r=future.result(); rows.append(r)
            save(args.out/'progress.json',{'completed':len(rows),'total':len(jobs),'last':r['case_id']})
            if len(rows)%20==0 or len(rows)==len(jobs):
                print('COMPLETED',len(rows),'/',len(jobs),r['case_id'],flush=True)
    assert len({r['case_id'] for r in rows})==len(jobs)
    result={'families':{f:summarize([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})}}
    if not args.preflight_only:
        result['movable_obstacles']=summarize([r for r in rows if r['family'] not in ('flat_calibration','fixed_c_space')])
        result['by_condition']={c:summarize([r for r in rows if r['layout']==c]) for c in sorted({r['layout'] for r in rows if r['family']!='flat_calibration'})}
        result['by_family_condition']={f:{c:summarize([r for r in rows if r['family']==f and r['layout']==c]) for c in sorted({r['layout'] for r in rows if r['family']==f})} for f in result['families']}
        result['by_direction']={f:{d:summarize([r for r in rows if r['family']==f and r['direction']==d])
                                   for d in sorted({r['direction'] for r in rows if r['family']==f})}
                                for f in result['families']}
    save(args.out/'summary.json', result)
    print('COMPLETE',args.out,flush=True)

if __name__=='__main__':
    main()
